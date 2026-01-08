"""Embedding builder for LSTM link prediction.

Creates node-level embeddings that combine graph structural features,
transactional features (aggregated per time bucket), and historical fraud flags.

Primary outputs:
- `emb_map`: dict mapping node_id -> list of (timestamp, embedding_array)
- `feature_names`: ordered list of feature names for the embedding vector

Design / Schema (per-embedding vector):
  STATIC (graph-level / slow-changing):
    - in_degree: int
    - out_degree: int
    - total_degree: int
    - pagerank: float
    - clustering_coeff: float (undirected clustering)
    - reciprocal_count: int (nodes with bidirectional edges)
    - unique_counterparties: int (total unique neighbors)
    - baseline_avg_out_amount: float (from historical baseline)

  DYNAMIC (per-time-bucket transactional features):
    - tx_count_out
    - tx_count_in
    - total_out_amount
    - total_in_amount
    - avg_out_amount
    - median_out_amount
    - std_out_amount
    - unique_out_counterparties_in_bucket
    - time_since_last_tx (seconds)

  HISTORICAL FRAUD FLAGS / META:
    - fraud_score (0..1)

Notes:
- All features are concatenated (static + dynamic + fraud) to form the embedding vector.
- The builder returns time-ordered embeddings per node (ascending timestamps).
- Missing values are filled with zeros.

Usage example:
    emb_map, feature_names = build_time_series_node_embeddings(df, freq='1D', fraud_scores=fraud_dict)

"""
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import networkx as nx


def _ensure_timestamp_column(df: pd.DataFrame, ts_col: str = 'timestamp') -> pd.DataFrame:
    if ts_col in df.columns:
        df = df.copy()
        df[ts_col] = pd.to_datetime(df[ts_col], unit='s', errors='coerce')
    return df


def compute_structural_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute static structural features per node from transaction dataframe.

    Args:
        df: DataFrame with columns ['source', 'target', 'amount', 'timestamp']

    Returns:
        DataFrame indexed by node id with structural feature columns.
    """
    # Build directed graph
    G = nx.DiGraph()
    edges = df[['source', 'target']].dropna()
    G.add_edges_from(edges.itertuples(index=False, name=None))

    nodes = list(G.nodes())
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    deg = {n: in_deg.get(n, 0) + out_deg.get(n, 0) for n in nodes}

    # PageRank
    try:
        pr = nx.pagerank(G)
    except Exception:
        pr = {n: 0.0 for n in nodes}

    # clustering: use undirected version
    try:
        und = G.to_undirected()
        clustering = nx.clustering(und)
    except Exception:
        clustering = {n: 0.0 for n in nodes}

    # reciprocal count: number of neighbors with edges both ways
    reciprocal = {}
    for n in nodes:
        rec = 0
        for nbr in G.successors(n):
            if G.has_edge(nbr, n):
                rec += 1
        reciprocal[n] = rec

    # unique counterparty count
    unique_cps = {}
    for n in nodes:
        nbrs = set(G.predecessors(n)) | set(G.successors(n))
        unique_cps[n] = len(nbrs)

    data = {
        'node': nodes,
        'in_degree': [in_deg.get(n, 0) for n in nodes],
        'out_degree': [out_deg.get(n, 0) for n in nodes],
        'total_degree': [deg.get(n, 0) for n in nodes],
        'pagerank': [pr.get(n, 0.0) for n in nodes],
        'clustering_coeff': [clustering.get(n, 0.0) for n in nodes],
        'reciprocal_count': [reciprocal.get(n, 0) for n in nodes],
        'unique_counterparties': [unique_cps.get(n, 0) for n in nodes]
    }

    df_struct = pd.DataFrame(data).set_index('node')
    return df_struct


def compute_node_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Compute historical transaction aggregates per node (global baseline metrics).

    Returns DataFrame indexed by node with baseline statistics.
    """
    df = df.copy()
    df = _ensure_timestamp_column(df)

    # Outgoing aggregates
    out_agg = df.groupby('source').agg(
        avg_out_amount=('amount', 'mean'),
        median_out_amount=('amount', 'median'),
        std_out_amount=('amount', 'std'),
        total_out_volume=('amount', 'sum'),
        out_tx_count=('amount', 'count'),
        unique_out_targets=('target', pd.Series.nunique)
    )

    in_agg = df.groupby('target').agg(
        avg_in_amount=('amount', 'mean'),
        median_in_amount=('amount', 'median'),
        std_in_amount=('amount', 'std'),
        total_in_volume=('amount', 'sum'),
        in_tx_count=('amount', 'count'),
        unique_in_sources=('source', pd.Series.nunique)
    )

    agg = out_agg.join(in_agg, how='outer').fillna(0)
    return agg


def _bucket_transactions(df: pd.DataFrame, freq: str = '1D') -> pd.DataFrame:
    """Bucket transactions by time (floor) and return grouped dataframe.

    Returns a DataFrame with index (node, bucket_ts) and aggregated features.
    """
    df = _ensure_timestamp_column(df)
    df['bucket'] = df['timestamp'].dt.floor(freq)

    # Outgoing per bucket
    out = df.groupby(['source', 'bucket']).agg(
        tx_count_out=('amount', 'count'),
        total_out_amount=('amount', 'sum'),
        avg_out_amount=('amount', 'mean'),
        median_out_amount=('amount', 'median'),
        std_out_amount=('amount', 'std'),
        unique_out_targets=('target', pd.Series.nunique),
        last_out_ts=('timestamp', 'max')
    )
    out = out.fillna(0)

    # Incoming per bucket
    inc = df.groupby(['target', 'bucket']).agg(
        tx_count_in=('amount', 'count'),
        total_in_amount=('amount', 'sum'),
        avg_in_amount=('amount', 'mean'),
        median_in_amount=('amount', 'median'),
        std_in_amount=('amount', 'std'),
        unique_in_sources=('source', pd.Series.nunique),
        last_in_ts=('timestamp', 'max')
    )
    inc = inc.fillna(0)

    # normalize index names for join
    out.index.names = ['node', 'bucket']
    inc.index.names = ['node', 'bucket']

    merged = out.join(inc, how='outer').fillna(0)
    return merged.reset_index()


def build_time_series_node_embeddings(df: pd.DataFrame,
                                      freq: str = '1D',
                                      fraud_scores: Optional[Dict[str, float]] = None,
                                      normalize: bool = True) -> Tuple[Dict[str, List[Tuple[pd.Timestamp, np.ndarray]]], List[str]]:
    """Build time-ordered embeddings per node.

    Args:
        df: transactions DataFrame with columns ['source','target','amount','timestamp']
        freq: bucket frequency (pandas offset alias, e.g., '1D')
        fraud_scores: optional dict mapping node -> fraud score in [0,1]
        normalize: whether to z-score normalize features across all node-time rows

    Returns:
        emb_map: dict node -> list of (bucket_ts (pd.Timestamp), embedding np.ndarray)
        feature_names: list of feature names in embedding order
    """
    df = df.copy()
    df = _ensure_timestamp_column(df)

    fraud_scores = fraud_scores or {}

    # static structural features
    struct = compute_structural_features(df)
    agg = compute_node_aggregates(df)

    # merge static features into one DF
    static = struct.join(agg, how='outer').fillna(0)

    # bucketed dynamic features
    buckets = _bucket_transactions(df, freq=freq)

    # create rows: node, bucket, features
    rows = []
    for _, r in buckets.iterrows():
        node = r['node']
        bucket = r['bucket']
        static_row = static.loc[node] if node in static.index else pd.Series(0, index=static.columns)
        fraud = float(fraud_scores.get(node, 0.0))

        # dynamic features selection
        dyn_feats = [
            r.get('tx_count_out', 0.0),
            r.get('tx_count_in', 0.0),
            r.get('total_out_amount', 0.0),
            r.get('total_in_amount', 0.0),
            r.get('avg_out_amount', 0.0) if not pd.isna(r.get('avg_out_amount', np.nan)) else 0.0,
            r.get('median_out_amount', 0.0) if not pd.isna(r.get('median_out_amount', np.nan)) else 0.0,
            r.get('std_out_amount', 0.0) if not pd.isna(r.get('std_out_amount', np.nan)) else 0.0,
            r.get('unique_out_targets', 0.0),
        ]

        # time_since_last_tx: approximate seconds since last_out_ts or last_in_ts
        last_out = r.get('last_out_ts', pd.NaT)
        last_in = r.get('last_in_ts', pd.NaT)
        last_ts = None
        if pd.notna(last_out):
            last_ts = pd.to_datetime(last_out)
        if pd.notna(last_in):
            t_in = pd.to_datetime(last_in)
            if last_ts is None or t_in > last_ts:
                last_ts = t_in
        if last_ts is pd.NaT or last_ts is None:
            delta = 0.0
        else:
            delta = (pd.to_datetime(bucket) - last_ts).total_seconds()
            if delta < 0:
                delta = 0.0

        # baseline metrics from static
        baseline_avg_out = float(static_row.get('avg_out_amount', 0.0))

        # assemble feature vector: static + dynamic + fraud
        static_values = [
            float(static_row.get('in_degree', 0)),
            float(static_row.get('out_degree', 0)),
            float(static_row.get('total_degree', 0)),
            float(static_row.get('pagerank', 0.0)),
            float(static_row.get('clustering_coeff', 0.0)),
            float(static_row.get('reciprocal_count', 0)),
            float(static_row.get('unique_counterparties', 0)),
            baseline_avg_out
        ]

        feat_vector = np.array(static_values + dyn_feats + [delta, float(fraud)], dtype=np.float32)
        rows.append((node, pd.to_datetime(bucket), feat_vector))

    if len(rows) == 0:
        return {}, []

    # Optionally normalize across all dynamic rows (per-feature z-score)
    all_feats = np.stack([row[2] for row in rows], axis=0)
    if normalize:
        mean = np.mean(all_feats, axis=0)
        std = np.std(all_feats, axis=0) + 1e-9
        all_feats = (all_feats - mean) / std
        for i in range(len(rows)):
            rows[i] = (rows[i][0], rows[i][1], all_feats[i])

    # build emb_map grouping by node and sorted by timestamp
    emb_map: Dict[str, List[Tuple[pd.Timestamp, np.ndarray]]] = {}
    for node, bucket, vec in rows:
        emb_map.setdefault(node, []).append((bucket, vec))

    # sort each node's timeline
    for node in list(emb_map.keys()):
        emb_map[node] = sorted(emb_map[node], key=lambda x: x[0])

    # feature names list
    feature_names = [
        'in_degree', 'out_degree', 'total_degree', 'pagerank', 'clustering_coeff',
        'reciprocal_count', 'unique_counterparties', 'baseline_avg_out',
        'tx_count_out', 'tx_count_in', 'total_out_amount', 'total_in_amount',
        'avg_out_amount', 'median_out_amount', 'std_out_amount', 'unique_out_targets',
        'time_since_last_tx_seconds', 'fraud_score'
    ]

    return emb_map, feature_names


# simple utility to build pair-wise sequences using existing emb_map
# This will delegate to the LSTM helper but included for convenience
def build_pair_sequences_for_pairs(emb_map: Dict[str, List[Tuple[pd.Timestamp, np.ndarray]]],
                                   pair_list: List[Tuple[str, str]],
                                   seq_len: int,
                                   allow_padding: bool = True) -> Tuple[np.ndarray, List[Tuple[str, str]]]:
    """Create sequences for node pairs by aligning timestamps and concatenating node embeddings.

    Args:
        emb_map: dict mapping node -> list of (timestamp, embedding) sorted ascending
        pair_list: list of (u, v) pairs to build sequences for
        seq_len: sequence length
        allow_padding: if True, pad sequences shorter than seq_len with zeros at the front.
                       if False, drop pairs with < seq_len common timestamps.

    Returns (sequences, valid_pairs) where sequences shape (N, seq_len, feat_dim*2)
    """
    sequences = []
    valid_pairs = []
    for u, v in pair_list:
        u_tuples = emb_map.get(u, [])
        v_tuples = emb_map.get(v, [])
        # convert to dict by timestamp for quick lookup
        u_dict = {ts: emb for ts, emb in u_tuples}
        v_dict = {ts: emb for ts, emb in v_tuples}
        # find intersection timestamps
        common = sorted(set(u_dict.keys()) & set(v_dict.keys()))
        
        if len(common) == 0:
            continue
        
        # if not enough common timestamps and padding disabled, skip
        if len(common) < seq_len and not allow_padding:
            continue
        
        if allow_padding:
            # pad at the front if needed
            if len(common) < seq_len:
                # pad with zeros
                feat_dim_single = u_tuples[0][1].shape[0] if u_tuples else 1
                feat_dim = feat_dim_single * 2  # concatenated
                pad_len = seq_len - len(common)
                padding = np.zeros((pad_len, feat_dim), dtype=np.float32)
                seq_ts = common  # use all available timestamps
                arr_vals = np.stack([np.concatenate([u_dict[t], v_dict[t]], axis=0) for t in seq_ts], axis=0)
                arr = np.vstack([padding, arr_vals])
            else:
                # take last seq_len timestamps
                seq_ts = common[-seq_len:]
                arr = np.stack([np.concatenate([u_dict[t], v_dict[t]], axis=0) for t in seq_ts], axis=0)
        else:
            # original logic: take last seq_len if available
            seq_ts = common[-seq_len:]
            arr = np.stack([np.concatenate([u_dict[t], v_dict[t]], axis=0) for t in seq_ts], axis=0)
        
        sequences.append(arr)
        valid_pairs.append((u, v))

    if len(sequences) == 0:
        return np.zeros((0, seq_len, 0), dtype=np.float32), []
    return np.stack(sequences, axis=0), valid_pairs
