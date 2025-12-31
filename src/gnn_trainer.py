import os
import sys
import random
import time
from typing import Tuple, Dict

# Ensure workspace root is on path so `src` imports work when running this script directly.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import networkx as nx
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from src.graph_analyzer import AMLGraphAnalyzer


def generate_synthetic_graph(num_nodes: int = 1000, mule_fraction: float = 0.05,
                             avg_degree: float = 4.0, months: int = 6, seed: int = 42):
    """Generates a richer synthetic directed transaction graph with planted mule clusters,
    APP fraud events, synthetic identities, and mule personas. Returns (G, labels, profiles).

    - G: networkx.DiGraph with transaction edges annotated with metadata
    - labels: dict node->0/1 (1 = mule-related)
    - profiles: dict node->profile metadata (age, credit_history_years, typical_countries, accounts, devices)
    """
    random.seed(seed)
    np.random.seed(seed)

    G = nx.DiGraph()
    nodes = [f"A_{i:05d}" for i in range(num_nodes)]
    G.add_nodes_from(nodes)

    labels = {n: 0 for n in nodes}
    profiles = {}

    # helper: generate geolocation and ip
    def random_geo():
        # sample lat/lon roughly worldwide
        lat = random.uniform(-60, 60)
        lon = random.uniform(-180, 180)
        return lat, lon

    def random_ip():
        return f"{random.randint(1,255)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(0,255)}"

    now = int(time.time())
    seconds_per_month = 30 * 86400

    # create per-account profile metadata (age, credit history years, typical_countries, devices)
    country_pool = ['US','GB','NG','RU','CN','IN','JP','HK','BR','DE','FR','CA']
    for n in nodes:
        age = random.randint(18, 80)
        credit_hist = max(0, int((age - random.randint(16, 22)) * random.random()))
        typical_country = random.choice(country_pool)
        devices = [f"dev_{random.randint(1,10000)}" for _ in range(random.randint(1,3))]
        profiles[n] = {
            'age': age,
            'credit_history_years': credit_hist,
            'typical_countries': {typical_country},
            'devices': devices,
            'typical_hours': (8, 18)  # business hours default
        }

    # Plant mule hubs and smurfing rings with persona diversity
    num_mules = max(1, int(num_nodes * mule_fraction))
    mule_hubs = random.sample(nodes, num_mules)
    for hub in mule_hubs:
        labels[hub] = 1
        profiles[hub]['persona'] = 'mule_hub'
        hub_country = random.choice(list(profiles[hub]['typical_countries']))

        # create many inbound small-to-medium transactions over time (fan-in)
        spokes = random.sample([n for n in nodes if n != hub], k=min(200, num_nodes-1))
        for i, s in enumerate(spokes[:80]):
            t = now - random.randint(0, months * seconds_per_month)
            amount = float(np.random.uniform(500, 15000))
            ip = random_ip()
            lat, lon = random_geo()
            channel = random.choice(['bank_transfer','p2p','cash_withdrawal'])
            device = random.choice(profiles[s]['devices'])
            G.add_edge(s, hub, amount=amount, timestamp=t, channel=channel, ip=ip, device_id=device, lat=lat, lon=lon, target_country=hub_country)

        # connect hub to a set of mule accounts (multi-homing / mule herd)
        herd = random.sample([n for n in nodes if n != hub], k=min(100, num_nodes-1))
        for m in herd[:30]:
            labels[m] = 1
            profiles[m]['persona'] = 'mule_sub'
            t = now - random.randint(0, months * seconds_per_month)
            amount = float(np.random.uniform(200, 8000))
            G.add_edge(hub, m, amount=amount, timestamp=t, channel='p2p', ip=random_ip(), device_id=random.choice(profiles[hub]['devices']), lat=random_geo()[0], lon=random_geo()[1], target_country=random.choice(country_pool))

    # Add synthetic identity accounts: thin files, piggybacking, and bust-out patterns
    num_synthetic = max(1, int(0.02 * num_nodes))
    synth_accounts = random.sample([n for n in nodes if n not in mule_hubs], num_synthetic)
    for s in synth_accounts:
        profiles[s]['synthetic'] = True
        profiles[s]['age'] = random.randint(20, 60)
        profiles[s]['credit_history_years'] = random.choice([0,0,0,1,2])
        # small steady activity then sudden utilization spike
        base_time = now - months * seconds_per_month
        for i in range(random.randint(2,8)):
            t = base_time + i * (seconds_per_month // 4) + random.randint(0, 10000)
            amt = float(np.random.uniform(5, 200))
            G.add_edge(s, random.choice(nodes), amount=amt, timestamp=t, channel='card', ip=random_ip(), device_id=random.choice(profiles[s]['devices']), lat=random_geo()[0], lon=random_geo()[1], target_country=random.choice(country_pool))
        # burst (bust-out)
        burst_time = now - random.randint(0, seconds_per_month)
        for i in range(5):
            G.add_edge(s, random.choice(nodes), amount=float(np.random.uniform(5000, 15000)), timestamp=burst_time + i*3600, channel='card', ip=random_ip(), device_id=random.choice(profiles[s]['devices']), lat=random_geo()[0], lon=random_geo()[1], target_country=random.choice(country_pool))
        labels[s] = 1

    # APP fraud injections: authorized push payments with payee mismatch and bypass events
    num_app = max(1, int(0.01 * num_nodes))
    app_victims = random.sample(nodes, num_app)
    for v in app_victims:
        # choose a new payee in an unusual country
        payee = f"EXT_{random.randint(1000,9999)}"
        tgt_country = random.choice([c for c in country_pool if c not in profiles[v]['typical_countries']])
        t = now - random.randint(0, months * seconds_per_month)
        amount = float(np.random.uniform(1000, 50000))
        G.add_edge(v, payee, amount=amount, timestamp=t, channel='bank_transfer', ip=random_ip(), device_id=random.choice(profiles[v]['devices']), lat=random_geo()[0], lon=random_geo()[1], target_country=tgt_country, event_type='app_tx')
        # label the external payee as suspicious
        labels[v] = 1

    # add background organic edges with temporal spread and different channels
    num_edges = int(num_nodes * avg_degree)
    for _ in range(num_edges):
        a = random.choice(nodes)
        b = random.choice(nodes)
        if a == b:
            continue
        t = now - random.randint(0, months * seconds_per_month)
        channel = random.choice(['card','bank_transfer','p2p','wire'])
        amt = float(np.random.uniform(5, 1000))
        G.add_edge(a, b, amount=amt, timestamp=t, channel=channel, ip=random_ip(), device_id=random.choice(profiles[a]['devices']), lat=random_geo()[0], lon=random_geo()[1], target_country=random.choice(country_pool))

    # plant smurfing rings (cycles) with temporal sequencing
    for _ in range(max(2, num_mules // 2)):
        ring_len = random.randint(3, 6)
        ring_nodes = random.sample(nodes, ring_len)
        base_t = now - random.randint(0, months * seconds_per_month)
        for i in range(ring_len):
            src = ring_nodes[i]
            dst = ring_nodes[(i+1) % ring_len]
            labels[dst] = max(labels.get(dst,0), 1)
            G.add_edge(src, dst, amount=float(np.random.uniform(100, 2000)), timestamp=base_t + i * 3600, channel='p2p', ip=random_ip(), device_id=random.choice(profiles[src]['devices']), lat=random_geo()[0], lon=random_geo()[1], target_country=random.choice(country_pool))

    return G, labels, profiles


def graph_to_tx_df(G: nx.DiGraph) -> pd.DataFrame:
    rows = []
    for u, v, d in G.edges(data=True):
        row = {
            'source': u,
            'target': v,
            'amount': float(d.get('amount', 0.0)),
            'timestamp': int(d.get('timestamp', time.time())),
            'type': d.get('type','synthetic'),
            'channel': d.get('channel'),
            'ip': d.get('ip'),
            'device_id': d.get('device_id'),
            'lat': d.get('lat'),
            'lon': d.get('lon'),
            'target_country': d.get('target_country'),
            'event_type': d.get('event_type')
        }
        rows.append(row)
    return pd.DataFrame(rows)


def build_node_features(G: nx.DiGraph, df_tx: pd.DataFrame, profiles: Dict[str, Dict] = None) -> Tuple[np.ndarray, Dict[int,str]]:
    """Extracts per-node features using graph analytics (fan-in/out, cycles, pagerank, centrality).

    Returns features ndarray (N x F) and mapping idx->node.
    """
    analyzer = AMLGraphAnalyzer(df_tx)
    nodes = list(G.nodes())
    node_index = {n: i for i, n in enumerate(nodes)}

    N = len(nodes)
    feat_list = []
    # base structural features
    indeg = np.zeros(N, dtype=float)
    outdeg = np.zeros(N, dtype=float)
    # compute Pagerank with a lightweight power-iteration implementation to avoid scipy dependency
    def simple_pagerank(G, alpha=0.85, max_iter=100, tol=1e-6):
        nodes = list(G.nodes())
        N = len(nodes)
        idx = {n: i for i, n in enumerate(nodes)}
        M = np.zeros((N, N), dtype=float)
        for u, v in G.edges():
            if u in idx and v in idx:
                M[idx[v], idx[u]] += 1.0
        # normalize columns
        col_sum = M.sum(axis=0)
        for j in range(N):
            if col_sum[j] > 0:
                M[:, j] /= col_sum[j]
            else:
                M[:, j] = 1.0 / N

        pr = np.ones(N, dtype=float) / N
        for _ in range(max_iter):
            pr_new = (1 - alpha) / N + alpha * (M @ pr)
            if np.linalg.norm(pr_new - pr, 1) < tol:
                pr = pr_new
                break
            pr = pr_new
        return {nodes[i]: float(pr[i]) for i in range(N)}

    pagerank = simple_pagerank(nx.DiGraph(G))
    bet = nx.betweenness_centrality(nx.DiGraph(G))

    # cycle membership: simple cycles up to length 6
    cycles = analyzer.detect_cycles()
    in_cycle = {n: 0 for n in nodes}
    for cyc in cycles:
        for n in cyc:
            if n in in_cycle:
                in_cycle[n] = 1

    for n in nodes:
        i = node_index[n]
        indeg[i] = G.in_degree(n)
        outdeg[i] = G.out_degree(n)

    # temporal / behavioral features from df_tx and profiles
    profiles = profiles or {}
    avg_tx = np.zeros(N, dtype=float)
    recv_frac = np.zeros(N, dtype=float)
    uniq_devices = np.zeros(N, dtype=float)
    uniq_ips = np.zeros(N, dtype=float)
    avg_interarrival = np.zeros(N, dtype=float)

    # group transactions by source/target
    if df_tx is not None and len(df_tx) > 0:
        for n, g in df_tx.groupby('source'):
            if n not in node_index:
                continue
            i = node_index[n]
            avg_tx[i] = float(g['amount'].mean())
            uniq_devices[i] = g['device_id'].nunique() if 'device_id' in g else 0
            uniq_ips[i] = g['ip'].nunique() if 'ip' in g else 0
            times = np.sort(g['timestamp'].values)
            if len(times) > 1:
                avg_interarrival[i] = float(np.mean(np.diff(times)))

        for n, g in df_tx.groupby('target'):
            if n not in node_index:
                continue
            i = node_index[n]
            # fraction of incoming relative to total outgoing (simple proxy)
            in_sum = float(g['amount'].sum())
            # find outgoing total
            out_sum = float(df_tx[df_tx['source'] == n]['amount'].sum()) if 'amount' in df_tx else 0.0
            total = max(1.0, in_sum + out_sum)
            recv_frac[i] = in_sum / total

    # integrate profile-level signals (age, credit history)
    age_feat = np.zeros(N, dtype=float)
    credit_hist_feat = np.zeros(N, dtype=float)
    for n, prof in (profiles or {}).items():
        if n in node_index:
            i = node_index[n]
            age_feat[i] = prof.get('age', 0)
            credit_hist_feat[i] = prof.get('credit_history_years', 0)

    # assemble features
    features = np.vstack([
        indeg,
        outdeg,
        np.array([pagerank.get(n,0.0) for n in nodes]),
        np.array([bet.get(n,0.0) for n in nodes]),
        np.array([in_cycle.get(n,0) for n in nodes]),
        avg_tx,
        recv_frac,
        uniq_devices,
        uniq_ips,
        avg_interarrival,
        age_feat,
        credit_hist_feat
    ]).T

    # normalize features
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True) + 1e-6
    features = (features - mean) / std

    return features.astype(np.float32), node_index


class GraphSage(nn.Module):
    def __init__(self, in_feats: int, hidden: int = 64, out_feats: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(in_feats * 2, hidden)
        self.fc2 = nn.Linear(hidden * 2, out_feats)
        self.out = nn.Linear(out_feats, 1)

    def aggregate(self, x: torch.Tensor, adj: torch.sparse.FloatTensor) -> torch.Tensor:
        # simple mean aggregator: adj is sparse adjacency (N x N)
        deg = torch.sparse.sum(adj, dim=1).to_dense().unsqueeze(1) + 1e-6
        neigh_sum = torch.sparse.mm(adj, x)
        neigh_mean = neigh_sum / deg
        return neigh_mean

    def forward(self, x: torch.Tensor, adj: torch.sparse.FloatTensor):
        h_neigh = self.aggregate(x, adj)
        h = torch.cat([x, h_neigh], dim=1)
        h = torch.relu(self.fc1(h))

        h_neigh2 = self.aggregate(h, adj)
        h2 = torch.cat([h, h_neigh2], dim=1)
        h2 = torch.relu(self.fc2(h2))

        out = torch.sigmoid(self.out(h2)).squeeze(1)
        return out


def adjacency_sparse_from_nx(G: nx.DiGraph, node_index: Dict[str,int]) -> torch.sparse.FloatTensor:
    N = len(node_index)
    rows = []
    cols = []
    vals = []
    for u, v in G.edges():
        if u in node_index and v in node_index:
            i = node_index[u]
            j = node_index[v]
            rows.append(i)
            cols.append(j)
            vals.append(1.0)
    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.tensor(vals, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, (N, N))


def train_demo(num_nodes: int = 1000, epochs: int = 60):
    G, labels, profiles = generate_synthetic_graph(num_nodes=num_nodes)
    df_tx = graph_to_tx_df(G)
    X_np, node_index = build_node_features(G, df_tx, profiles)

    # prepare adjacency and tensors
    adj = adjacency_sparse_from_nx(G, node_index)
    X = torch.from_numpy(X_np)

    # fix node ordering: sorted by index mapping
    nodes_sorted = sorted(node_index, key=lambda x: node_index[x])
    y = torch.tensor([labels.get(nodestr, 0) for nodestr in nodes_sorted], dtype=torch.float32)

    model = GraphSage(in_feats=X.shape[1])
    opt = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    loss_fn = nn.BCELoss()

    # quick train/test split
    N = X.shape[0]
    idx = np.arange(N)
    np.random.shuffle(idx)
    split = int(0.8 * N)
    train_idx = torch.tensor(idx[:split], dtype=torch.long)
    test_idx = torch.tensor(idx[split:], dtype=torch.long)

    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(X, adj)
        loss = loss_fn(out[train_idx], y[train_idx])
        loss.backward()
        opt.step()

        if (ep + 1) % 10 == 0 or ep == 0:
            model.eval()
            with torch.no_grad():
                pred = (out[test_idx] > 0.5).float()
                acc = (pred == y[test_idx]).float().mean().item()
                print(f"Epoch {ep+1}/{epochs} loss={loss.item():.4f} test_acc={acc:.4f}")

    # final evaluation
    model.eval()
    with torch.no_grad():
        out = model(X, adj)
        pred = (out > 0.5).float()
        acc = (pred == y).float().mean().item()
        print(f"Final accuracy (all nodes): {acc:.4f}")

    # return model, node ordering, predictions
    return model, nodes_sorted, out.numpy(), y.numpy()


if __name__ == '__main__':
    print("Running GNN GraphSAGE demo with synthetic mule data...")
    model, nodes, scores, labels = train_demo(num_nodes=800, epochs=50)
    # print top suspicious by score
    idx_sorted = np.argsort(-scores)
    print("Top predicted mule-like nodes:")
    for i in idx_sorted[:10]:
        print(f"{nodes[i]} score={scores[i]:.3f} label={labels[i]}")
