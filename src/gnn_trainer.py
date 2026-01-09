import os
import sys
import random
import time
import json
from typing import Tuple, Dict, Any

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
try:
    import yaml
    _HAS_YAML = True
except Exception:
    yaml = None
    _HAS_YAML = False

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


def generate_hetero_graph(num_accounts: int = 800, num_customers: int = 400,
                          num_devices: int = 600, num_ips: int = 600,
                          num_merchants: int = 200, mule_fraction: float = 0.05, seed: int = 42):
    """Generate a heterogeneous graph with node types: Account, Customer, Device, IP, Merchant.
    Edges: sends_money (Account->Account or Account->Merchant), owns (Customer->Account),
    accessed_from (Account->IP), used_device (Account->Device).

    Returns: G (nx.MultiDiGraph), account_labels (dict account->0/1), profiles (dict)
    """
    random.seed(seed)
    np.random.seed(seed)
    G = nx.MultiDiGraph()

    accounts = [f'A_{i:05d}' for i in range(num_accounts)]
    customers = [f'C_{i:05d}' for i in range(num_customers)]
    devices = [f'D_{i:05d}' for i in range(num_devices)]
    ips = [f'IP_{i:05d}' for i in range(num_ips)]
    merchants = [f'M_{i:05d}' for i in range(num_merchants)]

    profiles = {}
    for a in accounts:
        G.add_node(a, ntype='account')
        profiles[a] = {'balance': float(np.random.exponential(3000.0)), 'age_days': random.randint(30, 3650), 'flags': 0}
    for c in customers:
        G.add_node(c, ntype='customer')
        profiles[c] = {'kyc_risk': random.random(), 'occupation': random.choice(['retail','tech','student','other'])}
    for d in devices:
        G.add_node(d, ntype='device')
        profiles[d] = {'os_version': random.choice(['android_10','android_11','ios_14','ios_15','win_10'])}
    for ip in ips:
        G.add_node(ip, ntype='ip')
        profiles[ip] = {'lat': random.uniform(-60,60), 'lon': random.uniform(-180,180), 'asn': random.randint(1000,65000)}
    for m in merchants:
        G.add_node(m, ntype='merchant')
        profiles[m] = {'mcc': random.choice(['5311','5411','6011','7995','4829','4814']), 'category': random.choice(['grocery','atm','retail','tech','utilities'])}

    # assign customers to accounts (owns)
    for a in accounts:
        owner = random.choice(customers)
        G.add_edge(owner, a, rel='owns')

    # account accessed_from ip and used_device
    for a in accounts:
        ip = random.choice(ips)
        dev = random.choice(devices)
        G.add_edge(a, ip, rel='accessed_from')
        G.add_edge(a, dev, rel='used_device')

    # transactions (sends_money) between accounts and merchants
    num_tx = int(len(accounts) * 4)
    now = int(time.time())
    for _ in range(num_tx):
        src = random.choice(accounts)
        if random.random() < 0.15:
            dst = random.choice(merchants)
        else:
            dst = random.choice(accounts)
        amount = float(np.random.uniform(1, 20000))
        ts = now - random.randint(0, 30*86400)
        G.add_edge(src, dst, rel='sends_money', amount=amount, timestamp=ts)

    # plant mule hubs among accounts
    account_labels = {a: 0 for a in accounts}
    num_mules = max(1, int(num_accounts * mule_fraction))
    mule_hubs = random.sample(accounts, num_mules)
    for hub in mule_hubs:
        account_labels[hub] = 1
        # connect many small incoming from random accounts
        spokes = random.sample([x for x in accounts if x != hub], k=min(200, len(accounts)-1))
        for s in spokes[:80]:
            G.add_edge(s, hub, rel='sends_money', amount=float(np.random.uniform(100,5000)), timestamp=now - random.randint(0,30*86400))
        # hub pays out to mule subs
        herd = random.sample([x for x in accounts if x != hub], k=min(100, len(accounts)-1))
        for m in herd[:30]:
            account_labels[m] = 1
            G.add_edge(hub, m, rel='sends_money', amount=float(np.random.uniform(50,8000)), timestamp=now - random.randint(0,30*86400))

    return G, account_labels, profiles


def build_hetero_node_features(G: nx.MultiDiGraph):
    """Return features per node-type and mappings.
    Returns:
      feats: dict ntype -> numpy array (N_type x D_type)
      idx: dict ntype -> mapping node->index
      nodes_sorted: dict ntype -> list of nodes in index order
    """
    from collections import defaultdict as _dd
    nodes_by_type = _dd(list)
    for n, d in G.nodes(data=True):
        ntype = d.get('ntype', 'account')
        nodes_by_type[ntype].append(n)

    feats = {}
    idx = {}
    nodes_sorted = {}
    for ntype, nodes in nodes_by_type.items():
        nodes_sorted[ntype] = nodes
        idx_map = {n: i for i, n in enumerate(nodes)}
        idx[ntype] = idx_map
        if ntype == 'account':
            arr = np.zeros((len(nodes), 3), dtype=float)
            for n in nodes:
                a = G.nodes[n]
                bal = a.get('balance', np.random.exponential(3000.0))
                age = a.get('age_days', random.randint(30,3650))
                flags = a.get('flags', 0)
                arr[idx_map[n], :] = [bal, age, flags]
            feats[ntype] = arr
        elif ntype == 'customer':
            arr = np.zeros((len(nodes), 2), dtype=float)
            for n in nodes:
                a = G.nodes[n]
                kyc = a.get('kyc_risk', random.random())
                occ = 0
                if a.get('occupation','') == 'tech':
                    occ = 1
                arr[idx_map[n], :] = [kyc, occ]
            feats[ntype] = arr
        elif ntype == 'device':
            arr = np.zeros((len(nodes), 2), dtype=float)
            for n in nodes:
                a = G.nodes[n]
                osv = a.get('os_version','android_10')
                arr[idx_map[n], 0] = hash(osv) % 1000 / 1000.0
                arr[idx_map[n], 1] = random.random()
            feats[ntype] = arr
        elif ntype == 'ip':
            arr = np.zeros((len(nodes), 3), dtype=float)
            for n in nodes:
                a = G.nodes[n]
                lat = a.get('lat', random.uniform(-60,60))
                lon = a.get('lon', random.uniform(-180,180))
                asn = a.get('asn', random.randint(1000,65000))
                arr[idx_map[n], :] = [lat/90.0, lon/180.0, asn/65000.0]
            feats[ntype] = arr
        elif ntype == 'merchant':
            arr = np.zeros((len(nodes), 2), dtype=float)
            for n in nodes:
                a = G.nodes[n]
                mcc = a.get('mcc','5311')
                cat = a.get('category','retail')
                arr[idx_map[n], 0] = int(mcc) % 1000 / 1000.0
                arr[idx_map[n], 1] = 1.0 if cat == 'retail' else 0.0
            feats[ntype] = arr
        else:
            feats[ntype] = np.zeros((len(nodes), 4), dtype=float)

    return feats, idx, nodes_sorted


def build_relation_adjs(G: nx.MultiDiGraph, idx_map: Dict[str, Dict[str,int]]):
    """Return dict of relation name -> sparse tensor of shape (N_dst, N_src).
    For each edge u->v with rel attribute, map to indices in idx_map.
    """
    from collections import defaultdict as _dd
    rel_edges = _dd(list)
    for u, v, k, d in G.edges(keys=True, data=True):
        rel = d.get('rel')
        if rel is None:
            continue
        src_type = G.nodes[u].get('ntype')
        dst_type = G.nodes[v].get('ntype')
        if src_type not in idx_map or dst_type not in idx_map:
            continue
        src_idx = idx_map[src_type].get(u)
        dst_idx = idx_map[dst_type].get(v)
        if src_idx is None or dst_idx is None:
            continue
        rel_edges[(src_type, rel, dst_type)].append((dst_idx, src_idx))

    rel_adjs = {}
    for (src_t, rel, dst_t), pairs in rel_edges.items():
        rows = [p[0] for p in pairs]
        cols = [p[1] for p in pairs]
        vals = [1.0] * len(rows)
        indices = torch.tensor([rows, cols], dtype=torch.long)
        shape = (len(idx_map[dst_t]), len(idx_map[src_t]))
        values = torch.tensor(vals, dtype=torch.float32)
        rel_adjs[(src_t, rel, dst_t)] = torch.sparse_coo_tensor(indices, values, shape)

    return rel_adjs


class HeteroGraphSage(nn.Module):
    def __init__(self, in_dims: Dict[str,int], hidden: int = 64, out_dim: int = 32):
        super().__init__()
        self.in_dims = in_dims
        self.hidden = hidden
        self.out_dim = out_dim
        self.rel_linears = nn.ModuleDict()
        self.type_proj = nn.ModuleDict({t: nn.Linear(d, hidden) for t, d in in_dims.items()})
        self.combine = nn.ModuleDict({t: nn.Sequential(nn.Linear(hidden*2, hidden), nn.ReLU()) for t in in_dims.keys()})
        self.account_out = nn.Linear(hidden, 1)

    def add_relation(self, name: str, src_dim: int):
        self.rel_linears[name] = nn.Linear(src_dim, self.hidden)

    def forward(self, feats: Dict[str, torch.Tensor], rel_adjs: Dict[tuple, torch.sparse.FloatTensor]):
        proj = {t: self.type_proj[t](feats[t]) for t in feats}
        neigh = {t: torch.zeros_like(proj[t]) for t in feats}
        for (src_t, rel, dst_t), adj in rel_adjs.items():
            key = f"{src_t}->{dst_t}:{rel}"
            if key not in self.rel_linears:
                self.add_relation(key, feats[src_t].shape[1])
            src_h = feats[src_t]
            lin = self.rel_linears[key](src_h)
            neigh_msg = torch.sparse.mm(adj, lin)
            neigh[dst_t] = neigh[dst_t] + neigh_msg

        out_h = {}
        for t in feats:
            h_self = proj[t]
            h_nei = neigh[t]
            h = torch.cat([h_self, h_nei], dim=1)
            out_h[t] = self.combine[t](h)

        acct_out = torch.sigmoid(self.account_out(out_h['account'])).squeeze(1)
        return acct_out


def train_demo_hetero(num_accounts: int = 800, epochs: int = 40, constraint_lambda: float = 1.0, save_model_flag: bool = True):
    G, labels, profiles = generate_hetero_graph(num_accounts=num_accounts)
    feats_dict, idx_map, nodes_sorted = build_hetero_node_features(G)
    # normalize per-type features to zero-mean/unit-variance to stabilize training
    feats_t = {}
    for t, arr in feats_dict.items():
        arr = np.array(arr, dtype=np.float32)
        if arr.shape[0] > 0:
            mu = arr.mean(axis=0, keepdims=True)
            sd = arr.std(axis=0, keepdims=True) + 1e-6
            norm = (arr - mu) / sd
        else:
            norm = arr
        feats_t[t] = torch.from_numpy(norm.astype(np.float32))
    rel_adjs = build_relation_adjs(G, idx_map)
    in_dims = {t: feats_dict[t].shape[1] for t in feats_dict}
    model = HeteroGraphSage(in_dims=in_dims, hidden=64)
    for (src_t, rel, dst_t), adj in rel_adjs.items():
        key = f"{src_t}->{dst_t}:{rel}"
        model.add_relation(key, in_dims[src_t])

    opt = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    loss_fn = nn.BCELoss()

    acct_nodes = nodes_sorted['account']
    N = len(acct_nodes)
    y_np = np.array([labels.get(n, 0) for n in acct_nodes], dtype=np.float32)
    y = torch.from_numpy(y_np).float()
    idx = np.arange(N)
    np.random.shuffle(idx)
    split = int(0.8 * N)
    train_idx = torch.tensor(idx[:split], dtype=torch.long)
    test_idx = torch.tensor(idx[split:], dtype=torch.long)

    raw_acct = feats_dict['account']
    rule_scores, per_rule = compute_rule_scores_from_registry(raw_acct, RULE_CONFIG)
    rule_scores_t = torch.from_numpy(rule_scores).float()

    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(feats_t, rel_adjs)
        cls_loss = loss_fn(out[train_idx], y[train_idx])
        rule_loss = nn.MSELoss()(out[train_idx], rule_scores_t[train_idx])
        loss = cls_loss + constraint_lambda * rule_loss
        loss.backward()
        opt.step()

        if (ep + 1) % 10 == 0 or ep == 0:
            model.eval()
            with torch.no_grad():
                pred = (out[test_idx] > 0.5).float()
                acc = (pred == y[test_idx]).float().mean().item()
                print(f"Epoch {ep+1}/{epochs} loss={loss.item():.4f} cls_loss={cls_loss.item():.4f} rule_loss={rule_loss.item():.4f} test_acc={acc:.4f}")

    model.eval()
    with torch.no_grad():
        out = model(feats_t, rel_adjs)
        pred = (out > 0.5).float()
        acc = (pred == y).float().mean().item()
        print(f"Final account accuracy: {acc:.4f}")

    try:
        per_rule_nodes = {k: v for k, v in per_rule.items()}
        out_csv = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'hetero_rule_explanations.csv')
        export_rule_explanations(acct_nodes, per_rule_nodes, rule_scores, out_csv)
    except Exception as e:
        print('Failed to export hetero rule explanations:', e)

    # save model and metadata if requested
    if save_model_flag:
        try:
            save_gnn_model(model, acct_nodes, num_accounts, epochs, constraint_lambda, model_type='hetero_graphsage')
        except Exception as e:
            print(f'Failed to save heterogeneous GNN model: {e}')

    return model, acct_nodes, out, y


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


def build_node_features(G: nx.DiGraph, df_tx: pd.DataFrame, profiles: Dict[str, Dict] = None) -> Tuple[np.ndarray, Dict[int,str], np.ndarray]:
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

    # assemble raw (unnormalized) features
    raw = np.vstack([
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

    # normalize features for model input
    mean = raw.mean(axis=0, keepdims=True)
    std = raw.std(axis=0, keepdims=True) + 1e-6
    features = (raw - mean) / std

    return features.astype(np.float32), node_index, raw.astype(np.float32)


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


def compute_rule_scores(raw_feats: np.ndarray) -> np.ndarray:
    """Compute rule scores via a configurable registry.

    This module exposes a `RULE_REGISTRY` (name -> function) and `RULE_CONFIG`
    mapping that controls which rules are enabled and their weights. Each rule
    function accepts `raw_feats` and returns a numpy array in [0,1].
    """
    raise RuntimeError("use compute_rule_scores_from_registry(raw_feats, rule_config) instead")


def _zscore(raw_feats: np.ndarray) -> np.ndarray:
    mu = raw_feats.mean(axis=0, keepdims=True)
    sigma = raw_feats.std(axis=0, keepdims=True) + 1e-6
    return (raw_feats - mu) / sigma


def rule_fan_in_smurf(raw_feats: np.ndarray) -> np.ndarray:
    z = _zscore(raw_feats)
    indeg_z = z[:, 0]
    recv_frac_z = z[:, 6]
    avg_tx_z = z[:, 5]
    in_cycle = raw_feats[:, 4]
    lin = 0.9 * indeg_z + 1.2 * recv_frac_z - 0.8 * avg_tx_z + 0.6 * in_cycle
    return 1.0 / (1.0 + np.exp(-np.clip(lin, -10, 10)))


def rule_mule_hub(raw_feats: np.ndarray) -> np.ndarray:
    z = _zscore(raw_feats)
    avg_tx_z = z[:, 5]
    outdeg_z = z[:, 1]
    credit_z = z[:, 11]
    lin = 0.7 * avg_tx_z + 0.8 * outdeg_z - 0.5 * credit_z
    return 1.0 / (1.0 + np.exp(-np.clip(lin, -10, 10)))


def rule_new_device_high_amount(raw_feats: np.ndarray) -> np.ndarray:
    z = _zscore(raw_feats)
    avg_tx_z = z[:, 5]
    uniq_devices_z = z[:, 7]
    recv_frac_z = z[:, 6]
    # treat uniq_devices_z low -> suspicious (new device)
    new_dev_score = np.exp(-uniq_devices_z)
    lin = 1.0 * avg_tx_z + 1.2 * new_dev_score + 0.6 * recv_frac_z
    return 1.0 / (1.0 + np.exp(-np.clip(lin, -10, 10)))


def rule_impossible_travel(raw_feats: np.ndarray) -> np.ndarray:
    # Placeholder: requires per-node last location/time; approximate with high interarrival and high uniq_ips
    z = _zscore(raw_feats)
    interarrival_z = z[:, 9]
    uniq_ips_z = z[:, 8]
    lin = 0.6 * uniq_ips_z + 0.8 * interarrival_z
    return 1.0 / (1.0 + np.exp(-np.clip(lin, -10, 10)))


def rule_device_ip_entropy(raw_feats: np.ndarray) -> np.ndarray:
    z = _zscore(raw_feats)
    uniq_devices_z = z[:, 7]
    uniq_ips_z = z[:, 8]
    lin = 0.7 * uniq_ips_z + 0.6 * uniq_devices_z
    return 1.0 / (1.0 + np.exp(-np.clip(lin, -10, 10)))


def rule_rapid_outgoing_spike(raw_feats: np.ndarray) -> np.ndarray:
    z = _zscore(raw_feats)
    outdeg_z = z[:, 1]
    avg_tx_z = z[:, 5]
    lin = 0.8 * outdeg_z + 0.9 * avg_tx_z
    return 1.0 / (1.0 + np.exp(-np.clip(lin, -10, 10)))


def rule_recurring_tester_tx(raw_feats: np.ndarray) -> np.ndarray:
    # small frequent transactions (tester) -> many small avg_tx and high indeg/outdeg
    z = _zscore(raw_feats)
    avg_tx_z = z[:, 5]
    indeg_z = z[:, 0]
    outdeg_z = z[:, 1]
    lin = -0.7 * avg_tx_z + 0.6 * (indeg_z + outdeg_z)
    return 1.0 / (1.0 + np.exp(-np.clip(lin, -10, 10)))


def rule_app_payee_mismatch(raw_feats: np.ndarray) -> np.ndarray:
    # Placeholder proxy: high avg_tx and high recv_frac indicating large outward push
    z = _zscore(raw_feats)
    avg_tx_z = z[:, 5]
    recv_frac_z = z[:, 6]
    lin = 1.0 * avg_tx_z + 0.8 * recv_frac_z
    return 1.0 / (1.0 + np.exp(-np.clip(lin, -10, 10)))


def rule_synthetic_identity_burst(raw_feats: np.ndarray) -> np.ndarray:
    z = _zscore(raw_feats)
    credit_z = z[:, 11]
    avg_tx_z = z[:, 5]
    lin = -0.9 * credit_z + 1.0 * avg_tx_z
    return 1.0 / (1.0 + np.exp(-np.clip(lin, -10, 10)))


# Registry of available rules
RULE_REGISTRY = {
    'fan_in_smurf': rule_fan_in_smurf,
    'mule_hub': rule_mule_hub,
    'new_device_high_amount': rule_new_device_high_amount,
    'impossible_travel': rule_impossible_travel,
    'device_ip_entropy': rule_device_ip_entropy,
    'rapid_outgoing_spike': rule_rapid_outgoing_spike,
    'recurring_tester_tx': rule_recurring_tester_tx,
    'app_payee_mismatch': rule_app_payee_mismatch,
    'synthetic_identity_burst': rule_synthetic_identity_burst,
}


# Default rule configuration (analyst-editable)
RULE_CONFIG = {name: {'enabled': True, 'weight': 1.0} for name in RULE_REGISTRY.keys()}


def compute_rule_scores_from_registry(raw_feats: np.ndarray, rule_config: Dict = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Compute aggregated rule scores using the registry and a config dict.

    Returns (scores, per_rule_scores)
    - scores: (N,) aggregated score in [0,1]
    - per_rule_scores: dict name->(N,) arrays
    """
    config = RULE_CONFIG.copy()
    if rule_config:
        # override defaults
        for k, v in rule_config.items():
            if k in config:
                config[k].update(v)
            else:
                config[k] = v

    per_rule = {}
    weights = []
    enabled_rules = []
    for name, fn in RULE_REGISTRY.items():
        conf = config.get(name, {'enabled': False, 'weight': 0.0})
        if not conf.get('enabled', False):
            continue
        try:
            scores = fn(raw_feats)
        except Exception:
            scores = np.zeros(raw_feats.shape[0], dtype=np.float32)
        per_rule[name] = scores
        enabled_rules.append(name)
        weights.append(float(conf.get('weight', 1.0)))

    if len(enabled_rules) == 0:
        return np.zeros(raw_feats.shape[0], dtype=np.float32), per_rule

    weights = np.array(weights, dtype=float)
    weights = weights / (weights.sum() + 1e-9)

    # weighted average aggregation
    stacked = np.vstack([per_rule[n] for n in enabled_rules])  # (R, N)
    agg = np.dot(weights, stacked)
    agg = np.clip(agg, 0.0, 1.0)
    return agg.astype(np.float32), per_rule


def load_rule_config(path: str) -> Dict[str, Any]:
    """Load rule config from JSON or YAML file.

    Returns a dict suitable for passing into compute_rule_scores_from_registry.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.yml', '.yaml'):
        if not _HAS_YAML:
            raise RuntimeError('PyYAML is required to load YAML rule configs. Install with `pip install pyyaml`.')
        with open(path, 'r', encoding='utf8') as f:
            cfg = yaml.safe_load(f)
    elif ext == '.json':
        with open(path, 'r', encoding='utf8') as f:
            cfg = json.load(f)
    else:
        raise RuntimeError('Unsupported rule config format: ' + ext)
    # validate simple shape
    if not isinstance(cfg, dict):
        raise RuntimeError('Rule config must be a mapping of rule_name -> {enabled,weight}')
    return cfg


def export_rule_explanations(nodes: list, per_rule: Dict[str, np.ndarray], agg_scores: np.ndarray, out_path: str):
    """Export per-node rule activations and aggregated score to CSV for analyst review."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df = pd.DataFrame({'node': nodes, 'agg_score': agg_scores.tolist()})
    # add per-rule columns
    for name, arr in per_rule.items():
        df[name] = arr.tolist()
    df.to_csv(out_path, index=False)
    print(f'Wrote rule explanations to {out_path}')


def save_gnn_model(model: nn.Module, nodes_sorted: list, num_nodes: int, epochs: int, constraint_lambda: float, model_type: str = 'graphsage'):
    """Save GNN model weights and metadata to disk."""
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Save model state dict
    model_path = os.path.join(models_dir, 'gnn_model.pt')
    torch.save(model.state_dict(), model_path)
    print(f'Saved GNN model weights to {model_path}')
    
    # Save metadata
    metadata = {
        'model_type': model_type,
        'num_nodes': num_nodes,
        'epochs_trained': epochs,
        'constraint_lambda': float(constraint_lambda),
        'timestamp': time.time(),
        'node_count': len(nodes_sorted),
        'nodes_sample': nodes_sorted[:100]  # store first 100 node IDs for reference
    }
    metadata_path = os.path.join(models_dir, 'gnn_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f'Saved GNN metadata to {metadata_path}')
    
    return model_path, metadata_path


def load_gnn_model(model_class=None):
    """Load GNN model weights and metadata from disk."""
    if model_class is None:
        model_class = GraphSage
    
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    model_path = os.path.join(models_dir, 'gnn_model.pt')
    metadata_path = os.path.join(models_dir, 'gnn_metadata.json')
    
    if not os.path.exists(model_path) or not os.path.exists(metadata_path):
        raise FileNotFoundError(f'GNN model files not found in {models_dir}')
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Infer input features from metadata (default 8 for now, can be enhanced)
    model = model_class(in_feats=8)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    print(f'Loaded GNN model from {model_path}')
    print(f'Model metadata: {metadata}')
    return model, metadata


def train_demo(num_nodes: int = 1000, epochs: int = 60, constraint_lambda: float = 1.0, save_model_flag: bool = True):
    G, labels, profiles = generate_synthetic_graph(num_nodes=num_nodes)
    df_tx = graph_to_tx_df(G)
    X_np, node_index, raw_feats = build_node_features(G, df_tx, profiles)

    # prepare adjacency and tensors
    adj = adjacency_sparse_from_nx(G, node_index)
    X = torch.from_numpy(X_np)

    # fix node ordering: sorted by index mapping
    nodes_sorted = sorted(node_index, key=lambda x: node_index[x])
    y = torch.tensor([labels.get(nodestr, 0) for nodestr in nodes_sorted], dtype=torch.float32)

    # precompute rule soft targets from raw features using registry
    rule_scores, per_rule = compute_rule_scores_from_registry(raw_feats, RULE_CONFIG)
    rule_scores_t = torch.from_numpy(rule_scores).float()

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
        cls_loss = loss_fn(out[train_idx], y[train_idx])
        # constraint loss: encourage model outputs to align with rule-derived soft targets
        rule_loss = nn.MSELoss()(out[train_idx], rule_scores_t[train_idx])
        loss = cls_loss + constraint_lambda * rule_loss
        loss.backward()
        opt.step()

        if (ep + 1) % 10 == 0 or ep == 0:
            model.eval()
            with torch.no_grad():
                pred = (out[test_idx] > 0.5).float()
                acc = (pred == y[test_idx]).float().mean().item()
                print(f"Epoch {ep+1}/{epochs} loss={loss.item():.4f} cls_loss={cls_loss.item():.4f} rule_loss={rule_loss.item():.4f} test_acc={acc:.4f}")

    # final evaluation
    model.eval()
    with torch.no_grad():
        out = model(X, adj)
        pred = (out > 0.5).float()
        acc = (pred == y).float().mean().item()
        print(f"Final accuracy (all nodes): {acc:.4f}")

    # export per-node rule explanations for analysts
    try:
        out_csv = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'rule_explanations.csv')
        export_rule_explanations(nodes_sorted, per_rule, rule_scores, out_csv)
    except Exception as e:
        print('Failed to export rule explanations:', e)

    # save model and metadata if requested
    if save_model_flag:
        try:
            save_gnn_model(model, nodes_sorted, num_nodes, epochs, constraint_lambda)
        except Exception as e:
            print(f'Failed to save GNN model: {e}')

    # return model, node ordering, predictions
    return model, nodes_sorted, out.numpy(), y.numpy()


if __name__ == '__main__':
    print("Running GNN GraphSAGE demo with synthetic mule data...")
    # example: disable a rule or change weight
    RULE_CONFIG['impossible_travel']['enabled'] = False
    RULE_CONFIG['new_device_high_amount']['weight'] = 1.2
    # if a config exists, load it (JSON/YAML) and override defaults
    cfg_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'rules.yml')
    if os.path.exists(cfg_path):
        try:
            cfg = load_rule_config(cfg_path)
            # cfg is expected as mapping rule->{enabled,weight}
            for k, v in cfg.items():
                if k in RULE_CONFIG:
                    RULE_CONFIG[k].update(v)
                else:
                    RULE_CONFIG[k] = v
            print(f'Loaded rule config from {cfg_path}')
        except Exception as e:
            print('Failed to load rule config:', e)

    model, nodes, scores, labels = train_demo(num_nodes=800, epochs=50, constraint_lambda=1.0)
    # print top suspicious by score
    idx_sorted = np.argsort(-scores)
    print("Top predicted mule-like nodes:")
    for i in idx_sorted[:10]:
        print(f"{nodes[i]} score={scores[i]:.3f} label={labels[i]}")
