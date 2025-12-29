import pandas as pd
from src.simulator import TransactionSimulator
from src.graph_analyzer import AMLGraphAnalyzer
from src.csr_cycle_detector import build_csr, detect_cycles_csr
from src.visualizer import plot_graph

def main():
    print("=== AML Graph Analysis Engine ===")
    
    # 1. Simulation Phase
    print("\n[1] Initializing Simulation...")
    sim = TransactionSimulator(num_accounts=50)
    
    # Generate background noise
    sim.generate_organic_traffic(num_transactions=300)
    
    # Inject Specific Typologies
    # Fan-In (Structuring)
    fan_in_hub = "ACC_0010"
    sim.inject_fan_in(fan_in_hub, num_spokes=8, avg_amount=9500)
    
    # Fan-Out (Integration)
    fan_out_hub = "ACC_0025"
    sim.inject_fan_out(fan_out_hub, num_beneficiaries=6, total_amount=50000)
    
    # Cycle (Layering)
    sim.inject_cycle(length=5, amount=20000)
    
    df = sim.get_dataframe()
    print(f"\nGenerated {len(df)} transactions.")
    df.to_csv("transactions.csv", index=False)
    print("Saved transactions to 'transactions.csv'.")

    # 2. Analysis Phase
    print("\n[2] Analyzing Graph Topology...")
    analyzer = AMLGraphAnalyzer(df)
    
    # Detect Patterns
    print("\n--- Detecting Fan-In (Structuring) ---")
    fan_ins = analyzer.detect_fan_in(threshold_indegree=5)
    suspicious_set = set()
    if fan_ins:
        for item in fan_ins:
            print(f"ALERT: Node {item['node']} received {item['burst_size']} txs in short window.")
            suspicious_set.add(item['node'])
    else:
        print("No Fan-In patterns detected.")

    print("\n--- Detecting Fan-Out (Dissipation) ---")
    fan_outs = analyzer.detect_fan_out(threshold_outdegree=5)
    if fan_outs:
        for item in fan_outs:
            print(f"ALERT: Node {item['node']} sent {item['burst_size']} txs in short window.")
            suspicious_set.add(item['node'])
    else:
        print("No Fan-Out patterns detected.")

    print("--- Detecting Cycles (Round Tripping) ---")
    # Use CSR-based bounded-k cycle detector (prototype A)
    node_map, node_list, indptr, indices = build_csr(df)
    cycles = detect_cycles_csr(indptr, indices, node_list, max_k=6, max_cycles=500)
    if cycles:
        for cycle in cycles:
            print(f"ALERT: Cycle detected: {' -> '.join(cycle)} -> {cycle[0]}")
            for node in cycle:
                suspicious_set.add(node)
    else:
        print("No Cyclic patterns detected.")

    print("\n--- High Betweenness Centrality (Bridge Nodes) ---")
    bridges = analyzer.calculate_centrality()
    for node, score in bridges[:3]:
        print(f"Node {node}: Score {score:.4f}")

    # 3. Visualization Phase
    print("\n[3] Generating Visualization...")
    if suspicious_set:
        print(f"Highlighting {len(suspicious_set)} suspicious nodes.")
    plot_graph(analyzer.G, suspicious_nodes=list(suspicious_set), filename="aml_network_graph.png")

    print("\nDone.")

if __name__ == "__main__":
    main()
