import pandas as pd
from src.simulator import TransactionSimulator
from src.graph_analyzer import AMLGraphAnalyzer
from src.csr_cycle_detector import build_csr, detect_cycles_csr
from src.visualizer import plot_graph
from src.behavioral_detector import BehavioralDetector

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
    # --- Cyber-behavioral analysis demo (synthetic) ---
    print("\n[2.1] Cyber-behavioral heuristics demo...")
    detector = BehavioralDetector()

    # Create simple synthetic login events for demo purposes
    login_events = [
        # Pre-compromise: many failures from same subnet
        {'user_id': 'ACC_0010', 'timestamp': sim.start_time + 10, 'success': False, 'ip': '1.2.3.1', 'subnet': '1.2.3.0/24', 'user_agent': 'UA-1', 'time_to_login': 0.5},
        {'user_id': 'ACC_0010', 'timestamp': sim.start_time + 15, 'success': False, 'ip': '1.2.3.2', 'subnet': '1.2.3.0/24', 'user_agent': 'UA-1', 'time_to_login': 0.6},
        {'user_id': 'ACC_0010', 'timestamp': sim.start_time + 20, 'success': False, 'ip': '1.2.3.3', 'subnet': '1.2.3.0/24', 'user_agent': 'UA-1', 'time_to_login': 0.4},
        {'user_id': 'ACC_0010', 'timestamp': sim.start_time + 25, 'success': True,  'ip': '5.6.7.8',   'subnet': '5.6.7.0/24', 'user_agent': 'UA-1', 'time_to_login': 0.3, 'new_device': True},
        # Impossible travel example (far apart)
        {'user_id': 'ACC_0025', 'timestamp': sim.start_time + 100, 'success': True, 'ip': '8.8.8.8', 'lat': 51.5074, 'lon': -0.1278, 'user_agent': 'UA-2', 'time_to_login': 2.0},
        {'user_id': 'ACC_0025', 'timestamp': sim.start_time + 1000, 'success': True,'ip': '9.9.9.9', 'lat': 35.6895, 'lon': 139.6917, 'user_agent': 'UA-3', 'time_to_login': 2.5}
    ]

    logins_df = pd.DataFrame(login_events)

    cs_flags = detector.detect_credential_stuffing(logins_df, window_seconds=60, fail_ratio_threshold=0.6, min_attempts=3)
    bf_flags = detector.detect_bruteforce_and_new_device(logins_df)
    it_flags = detector.detect_impossible_travel(logins_df, velocity_kmph_threshold=1000.0)

    if cs_flags:
        for f in cs_flags:
            print(f"CYBER ALERT: {f}")
    if bf_flags:
        for f in bf_flags:
            print(f"CYBER ALERT: {f}")
    if it_flags:
        for f in it_flags:
            print(f"CYBER ALERT: {f}")
    if not (cs_flags or bf_flags or it_flags):
        print("No immediate cyber behavioral alerts from demo data.")
    
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
