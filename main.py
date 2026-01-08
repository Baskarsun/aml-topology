import pandas as pd
import numpy as np
from src.simulator import TransactionSimulator
from src.graph_analyzer import AMLGraphAnalyzer
from src.csr_cycle_detector import build_csr, detect_cycles_csr
from src.visualizer import plot_graph
from src.behavioral_detector import BehavioralDetector
from src.temporal_predictor import TemporalPredictor, SequenceAnalyzer
from src.embedding_builder import build_time_series_node_embeddings, build_pair_sequences_for_pairs
from src.lstm_link_predictor import LSTMLinkPredictor, train_model, predict_proba

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

    # 3. TEMPORAL PREDICTION PHASE (Predictive Subsystem)
    print("\n[3] Temporal & Predictive Analysis...")
    temporal_pred = TemporalPredictor(lookback_days=30, forecast_days=7)
    
    print("\n[3.1] Establishing temporal baselines for all accounts...")
    baselines = temporal_pred.establish_baselines(df)
    print(f"Baseline established for {len(baselines)} accounts.")
    
    print("\n[3.2] Detecting transaction volume acceleration patterns...")
    volume_alerts = temporal_pred.detect_volume_acceleration(df, threshold_sigma=2.5)
    if volume_alerts:
        for alert in volume_alerts:
            print(f"PREDICTIVE ALERT: Account {alert['account']} showing {alert['acceleration_rate']:.1%} volume acceleration")
            suspicious_set.add(alert['account'])
    else:
        print("No volume acceleration patterns detected.")
    
    print("\n[3.3] Detecting behavioral shifts...")
    behavior_alerts = temporal_pred.detect_behavioral_shift(df, deviation_threshold=2.0)
    if behavior_alerts:
        for alert in behavior_alerts:
            print(f"PREDICTIVE ALERT: Account {alert['account']} shows significant behavioral shift")
            suspicious_set.add(alert['account'])
    else:
        print("No behavioral shifts detected.")
    
    print("\n[3.4] Forecasting risk escalation (multi-signal analysis)...")
    risk_predictions = temporal_pred.forecast_risk_escalation(df, early_warning_threshold=0.6)
    if risk_predictions:
        for pred in risk_predictions:
            print(f"PREDICTIVE ALERT: Account {pred['account']} has {pred['predicted_risk_probability']:.1%} predicted risk of escalation")
            print(f"  Risk signals: {', '.join(pred['risk_signals'])}")
            suspicious_set.add(pred['account'])
    else:
        print("No risk escalation predictions.")
    
    print("\n[3.5] Detecting temporal concentration of transactions...")
    temporal_bursts = temporal_pred.detect_temporal_concentration(df, min_transactions=4, time_window_hours=24)
    if temporal_bursts:
        for burst in temporal_bursts:
            print(f"PREDICTIVE ALERT: Account {burst['account']} shows {burst['concentration_pct']:.0f}% temporal concentration")
            suspicious_set.add(burst['account'])
    else:
        print("No temporal concentration patterns detected.")
    
    print("\n[3.6] Predicting cycle emergence...")
    cycle_predictions = temporal_pred.predict_cycle_emergence(df, min_chain_length=3)
    if cycle_predictions:
        for pred in cycle_predictions:
            print(f"PREDICTIVE ALERT: Account {pred['account']} likely to form cycles ({pred['predicted_cycle_probability']:.1%} probability)")
            suspicious_set.add(pred['account'])
    else:
        print("No cycle emergence predictions.")
    
    print("\n[3.7] Analyzing transaction sequences for structuring...")
    sequence_analyzer = SequenceAnalyzer()
    structuring_seqs = sequence_analyzer.detect_structuring_sequence(df, threshold_amount=10000, just_below_threshold=9000)
    if structuring_seqs:
        for seq in structuring_seqs:
            print(f"PREDICTIVE ALERT: Account {seq['account']} shows structuring sequence ({seq['transaction_count']} txs)")
            suspicious_set.add(seq['account'])
    else:
        print("No structuring sequences detected.")
    
    print("\n[3.8] Generating temporal forecast summary...")
    forecast_report = temporal_pred.forecast_account_summary(df, include_all_detections=False)
    print(f"\nTemporal Forecast Summary:")
    print(f"  Total flagged accounts: {forecast_report['detection_summary']['total_flagged_accounts']}")
    print(f"  Risk escalation predictions: {forecast_report['detection_summary']['risk_escalation_predictions']}")
    print(f"  Temporal concentration alerts: {forecast_report['detection_summary']['temporal_concentration_alerts']}")
    if forecast_report['highest_risk_accounts']:
        print(f"\n  Top 5 highest temporal risk accounts:")
        for acc in forecast_report['highest_risk_accounts'][:5]:
            print(f"    - {acc['account']}: Risk Score {acc['temporal_risk_score']:.1f} ({len(acc['signals'])} signals)")

    # 4. Visualization Phase
    print("\n[4] Generating Visualization...")
    if suspicious_set:
        print(f"Highlighting {len(suspicious_set)} suspicious nodes (spatial + temporal).")
    plot_graph(analyzer.G, suspicious_nodes=list(suspicious_set), filename="aml_network_graph.png")

    # 5. LSTM LINK PREDICTION PHASE (Predictive ML)
    print("\n[5] LSTM Link Prediction for Emerging Links...")
    try:
        print("\n[5.1] Building time-series node embeddings...")
        fraud_scores = {node: (1.0 if node in suspicious_set else 0.0) for node in analyzer.G.nodes()}
        emb_map, feature_names = build_time_series_node_embeddings(df, freq='12H', fraud_scores=fraud_scores)
        print(f"Built embeddings for {len(emb_map)} nodes; feature dimension={len(feature_names)}")
        
        # Generate candidate pairs from suspicious nodes (link prediction focus)
        if suspicious_set and len(suspicious_set) > 1:
            print(f"\n[5.2] Generating candidate pairs for suspicious nodes...")
            pair_candidates = []
            susp_list = list(suspicious_set)
            # Sample edges: suspicious-to-suspicious and suspicious-to-others
            for i, node_u in enumerate(susp_list):
                for node_v in susp_list[i+1:]:
                    pair_candidates.append((node_u, node_v))
                # Also sample some suspicious-to-other nodes
                others = [n for n in analyzer.G.nodes() if n not in suspicious_set]
                sampled_others = np.random.choice(others, min(5, len(others)), replace=False) if others else []
                for node_v in sampled_others:
                    pair_candidates.append((node_u, node_v))
            
            if pair_candidates:
                print(f"Generated {len(pair_candidates)} candidate pairs for link prediction.")
                
                print("\n[5.3] Building pair sequences (with zero-padding for young nodes)...")
                sequences, valid_pairs = build_pair_sequences_for_pairs(
                    emb_map, pair_candidates, seq_len=3, allow_padding=True
                )
                print(f"Built {len(valid_pairs)} valid sequences (padded).")
                
                if len(valid_pairs) > 0:
                    # Create synthetic labels: 1 if link already exists in graph
                    labels = np.array([
                        float(analyzer.G.has_edge(u, v) or analyzer.G.has_edge(v, u))
                        for u, v in valid_pairs
                    ], dtype=np.float32)
                    
                    # Ensure some class balance for training
                    if len(np.unique(labels)) < 2:
                        n = len(labels)
                        k = max(1, int(0.2 * n))
                        inds = np.random.choice(n, k, replace=False)
                        labels[inds] = 1 - labels[inds]
                    
                    print(f"Labels: {int((labels==1).sum())} positive, {int((labels==0).sum())} negative")
                    
                    print("\n[5.4] Training LSTM link predictor...")
                    input_size = sequences.shape[2]
                    model = LSTMLinkPredictor(input_size=input_size, hidden_size=64, num_layers=1, dropout=0.1)
                    model, history = train_model(
                        model, sequences, labels, epochs=15, batch_size=32, lr=1e-3, use_class_weight=True
                    )
                    
                    print("\n[5.5] Predicting emerging links...")
                    predictions = predict_proba(model, sequences)
                    emerging_links = [
                        (valid_pairs[i], predictions[i])
                        for i in np.argsort(predictions)[::-1][:10]  # Top 10
                    ]
                    
                    print("\nTop 10 Predicted Emerging Suspicious Links:")
                    for (u, v), score in emerging_links:
                        print(f"  {u} -> {v}: {score:.3f} (link formation probability)")
                        suspicious_set.add(u)
                        suspicious_set.add(v)
                else:
                    print("Insufficient sequences for LSTM training.")
            else:
                print("Not enough suspicious nodes for pair generation.")
        else:
            print("Insufficient suspicious nodes detected for link prediction phase.")
    except Exception as e:
        print(f"LSTM phase skipped (optional): {e}")

    print("\n[4] Generating Visualization...")
    if suspicious_set:
        print(f"Highlighting {len(suspicious_set)} suspicious nodes (spatial + temporal + LSTM).")
    plot_graph(analyzer.G, suspicious_nodes=list(suspicious_set), filename="aml_network_graph.png")

    print("\nDone.")

if __name__ == "__main__":
    main()
