import networkx as nx
import pandas as pd
import numpy as np

class AMLGraphAnalyzer:
    def __init__(self, df_transactions):
        self.df = df_transactions
        self.G = self._build_graph()

    def _build_graph(self):
        """Builds a MultiDiGraph from the transaction dataframe."""
        G = nx.MultiDiGraph()
        for _, row in self.df.iterrows():
            G.add_edge(
                row['source'], 
                row['target'], 
                amount=row['amount'], 
                timestamp=row['timestamp'],
                type=row['type']
            )
        return G

    def detect_fan_in(self, threshold_indegree=5, time_window_seconds=3600):
        """
        Detects nodes with high indegree within short time windows (Fan-In).
        Returns a list of suspicious (node, count) tuples.
        """
        suspicious_nodes = []
        # Calculate static indegree first as a quick filter
        for node in self.G.nodes():
            in_edges = list(self.G.in_edges(node, data=True))
            if len(in_edges) < threshold_indegree:
                continue
            
            # Temporal clustering check
            timestamps = sorted([d['timestamp'] for _, _, d in in_edges])
            if not timestamps:
                continue

            # Sliding window to find max burst
            max_burst = 0
            for i in range(len(timestamps)):
                # Count how many tx happen within 'time_window_seconds' of timestamps[i]
                count = 0
                for t in timestamps[i:]:
                    if t - timestamps[i] <= time_window_seconds:
                        count += 1
                    else:
                        break
                max_burst = max(max_burst, count)
            
            if max_burst >= threshold_indegree:
                suspicious_nodes.append({'node': node, 'burst_size': max_burst, 'risk': 'Fan-In'})
        
        return suspicious_nodes

    def detect_fan_out(self, threshold_outdegree=5, time_window_seconds=3600):
        """
        Detects nodes with high outdegree within short time windows (Fan-Out).
        """
        suspicious_nodes = []
        for node in self.G.nodes():
            out_edges = list(self.G.out_edges(node, data=True))
            if len(out_edges) < threshold_outdegree:
                continue
            
            timestamps = sorted([d['timestamp'] for _, _, d in out_edges])
            max_burst = 0
            for i in range(len(timestamps)):
                count = 0
                for t in timestamps[i:]:
                    if t - timestamps[i] <= time_window_seconds:
                        count += 1
                    else:
                        break
                max_burst = max(max_burst, count)
            
            if max_burst >= threshold_outdegree:
                suspicious_nodes.append({'node': node, 'burst_size': max_burst, 'risk': 'Fan-Out'})
        
        return suspicious_nodes

    def detect_cycles(self):
        """
        Detects elementary cycles in the graph.
        Note: This can be computationally expensive on very large graphs.
        """
        # Convert to simple DiGraph for cycle detection (ignore multi-edges for connectivity)
        G_simple = nx.DiGraph(self.G)

        # Safeguards: limit number of cycles returned and their length to avoid long runs
        max_cycles = 200
        max_cycle_length = 8

        suspicious_cycles = []

        # Work per strongly-connected-component to reduce search space
        for scc in nx.strongly_connected_components(G_simple):
            if len(scc) <= 2:
                continue

            # If the SCC is large, report the nodes as suspicious instead of enumerating all cycles
            if len(scc) > max_cycle_length:
                # create a representative cycle-like report using the first few nodes
                nodes = list(scc)
                suspicious_cycles.append(nodes[:max_cycle_length])
                if len(suspicious_cycles) >= max_cycles:
                    break
                continue

            subG = G_simple.subgraph(scc).copy()
            try:
                for cycle in nx.simple_cycles(subG):
                    if len(cycle) > 2 and len(cycle) <= max_cycle_length:
                        suspicious_cycles.append(cycle)
                    if len(suspicious_cycles) >= max_cycles:
                        break
                if len(suspicious_cycles) >= max_cycles:
                    break
            except Exception:
                # If cycle search fails or is interrupted, stop and return what we have so far
                break

        return suspicious_cycles

    def calculate_centrality(self):
        """Calculates Betweenness Centrality to find bridge nodes."""
        G_simple = nx.DiGraph(self.G)
        centrality = nx.betweenness_centrality(G_simple)
        # Return top 10 central nodes
        return sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]

    def get_node_stats(self, node):
        in_edges = self.G.in_edges(node, data=True)
        out_edges = self.G.out_edges(node, data=True)
        
        total_in = sum([d['amount'] for _, _, d in in_edges])
        total_out = sum([d['amount'] for _, _, d in out_edges])
        
        return {
            "total_in": total_in,
            "total_out": total_out,
            "balance_change": total_in - total_out
        }
