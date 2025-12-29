import numpy as np
from collections import defaultdict


def build_csr(df):
    """Build CSR representation from transactions DataFrame.
    Returns: node_map (id->int), inv_map (int->id), indptr, indices
    """
    sources = df['source'].astype(str).tolist()
    targets = df['target'].astype(str).tolist()

    nodes = set(sources) | set(targets)
    node_list = sorted(nodes)
    node_map = {n: i for i, n in enumerate(node_list)}
    inv_map = {i: n for n, i in node_map.items()} if False else node_list

    # collect neighbors
    nbrs = defaultdict(list)
    for s, t in zip(sources, targets):
        nbrs[node_map[s]].append(node_map[t])

    n = len(node_list)
    indptr = np.zeros(n + 1, dtype=np.int64)
    indices_list = []
    for i in range(n):
        neigh = nbrs.get(i, [])
        # remove duplicates and sort for deterministic traversal
        if neigh:
            uniq = sorted(set(neigh))
        else:
            uniq = []
        indices_list.extend(uniq)
        indptr[i + 1] = indptr[i] + len(uniq)

    if indices_list:
        indices = np.array(indices_list, dtype=np.int32)
    else:
        indices = np.zeros(0, dtype=np.int32)

    return node_map, node_list, indptr, indices


def _dfs_collect(start, current, adj_indptr, adj_indices, visited, path, max_k, cycles, max_cycles):
    if len(cycles) >= max_cycles:
        return
    # explore neighbors
    s = start
    for idx in range(adj_indptr[current], adj_indptr[current + 1]):
        v = int(adj_indices[idx])
        if v == s:
            if len(path) + 1 > 2:
                cycles.append(path + [s])
                if len(cycles) >= max_cycles:
                    return
            continue
        if v < s:
            # enforce canonical ordering: only visit nodes with id >= start to avoid duplicates
            continue
        if v in visited:
            continue
        if len(path) + 1 >= max_k:
            continue
        visited.add(v)
        _dfs_collect(start, v, adj_indptr, adj_indices, visited, path + [v], max_k, cycles, max_cycles)
        visited.remove(v)
        if len(cycles) >= max_cycles:
            return


def detect_cycles_csr(indptr, indices, node_list, max_k=6, max_cycles=500):
    """Detect simple directed cycles up to length max_k using CSR adjacency.
    Returns cycles as lists of node identifiers (original ids).
    """
    n = len(node_list)
    cycles = []
    for s in range(n):
        # skip nodes with no outgoing edges
        if indptr[s] == indptr[s + 1]:
            continue
        visited = set([s])
        path = [s]
        _dfs_collect(s, s, indptr, indices, visited, path, max_k, cycles, max_cycles)
        if len(cycles) >= max_cycles:
            break
    # map back to original ids
    mapped = []
    for c in cycles:
        mapped.append([node_list[i] for i in c])
    return mapped


if __name__ == '__main__':
    # quick smoke test
    import pandas as pd
    df = pd.DataFrame({
        'source': ['A','B','C','D','C'],
        'target': ['B','C','A','C','A'],
        'amount': [1,2,3,4,5],
        'timestamp': [1,2,3,4,5],
        'type': ['x']*5
    })
    node_map, node_list, indptr, indices = build_csr(df)
    print('node_map len', len(node_map))
    cycles = detect_cycles_csr(indptr, indices, node_list, max_k=6, max_cycles=100)
    print('cycles', cycles)
