import pandas as pd
import numpy as np
import json
import os
from src.simulator import TransactionSimulator
from src.graph_analyzer import AMLGraphAnalyzer
from src.csr_cycle_detector import build_csr, detect_cycles_csr
from src.visualizer import plot_graph
from src.behavioral_detector import BehavioralDetector
from src.temporal_predictor import TemporalPredictor, SequenceAnalyzer
from src.embedding_builder import build_time_series_node_embeddings, build_pair_sequences_for_pairs
from src.lstm_link_predictor import LSTMLinkPredictor, train_model, predict_proba, save_model
from src.risk_consolidator import RiskConsolidator

# Additional model trainers with persistence
try:
    from src.gnn_trainer import train_demo as train_gnn_demo
    _HAS_GNN = True
except ImportError:
    _HAS_GNN = False
    print("Warning: GNN trainer not available")

try:
    from src.sequence_detector import demo_run as sequence_detector_demo
    _HAS_SEQUENCE = True
except ImportError:
    _HAS_SEQUENCE = False
    print("Warning: Sequence detector not available")

try:
    from src.gbdt_detector import demo_run as gbdt_detector_demo
    _HAS_GBDT = True
except ImportError:
    _HAS_GBDT = False
    print("Warning: GBDT detector not available")

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
                    
                    # Save trained LSTM model
                    print("\n[5.4.1] Saving LSTM model weights...")
                    os.makedirs('models', exist_ok=True)
                    lstm_model_path = 'models/lstm_link_predictor.pt'
                    save_model(model, lstm_model_path)
                    print(f"LSTM model saved to '{lstm_model_path}'")
                    
                    # Save LSTM training metadata
                    lstm_metadata = {
                        'input_size': input_size,
                        'hidden_size': 64,
                        'num_layers': 1,
                        'dropout': 0.1,
                        'num_sequences': len(sequences),
                        'num_epochs_trained': len(history['train_loss']),
                        'final_val_auc': float(history['val_auc'][-1]) if history['val_auc'] else 0.0,
                        'final_train_loss': float(history['train_loss'][-1]) if history['train_loss'] else 0.0,
                        'feature_names': feature_names
                    }
                    lstm_metadata_path = 'models/lstm_metadata.json'
                    with open(lstm_metadata_path, 'w') as f:
                        json.dump(lstm_metadata, f, indent=2)
                    print(f"LSTM metadata saved to '{lstm_metadata_path}'")
                    
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

    # 5.5 OPTIONAL: Train and Persist Additional Models (GNN, Sequence Detector, GBDT)
    print("\n[5.5] Training and Persisting Additional ML Models (with model persistence)...")
    
    # GNN Phase (Optional)
    if _HAS_GNN:
        try:
            print("\n[5.5.1] Training GNN for node classification...")
            train_gnn_demo(num_nodes=50, epochs=20, save_model_flag=True)
            print("GNN model trained and persisted to models/gnn_model.pt")
        except Exception as e:
            print(f"GNN training skipped: {e}")
    
    # Sequence Detector Phase (Optional)
    if _HAS_SEQUENCE:
        try:
            print("\n[5.5.2] Training Sequence Detector for event anomaly detection...")
            sequence_detector_demo(num_sequences=1000, seq_len=15, epochs=5, model_type='lstm', save_model_flag=True)
            print("Sequence detector model trained and persisted to models/sequence_detector_model.pt")
        except Exception as e:
            print(f"Sequence detector training skipped: {e}")
    
    # GBDT Phase (Optional)
    if _HAS_GBDT:
        try:
            print("\n[5.5.3] Training GBDT classifier for transaction-level anomalies...")
            gbdt_detector_demo(n=5000, save_model_flag=True)
            print("GBDT model trained and persisted to models/")
        except Exception as e:
            print(f"GBDT training skipped: {e}")

    print("\n[4] Generating Visualization...")
    if suspicious_set:
        print(f"Highlighting {len(suspicious_set)} suspicious nodes (spatial + temporal + LSTM).")
    plot_graph(analyzer.G, suspicious_nodes=list(suspicious_set), filename="aml_network_graph.png")

    # 6. RISK CONSOLIDATION & FINAL RANKING (Phase 6)
    print("\n[6] Consolidating Multi-Phase Risk Scores...")
    
    # Organize signals from all phases into consolidated structure
    spatial_signals = {
        'cycles': cycles,
        'fan_ins': fan_ins,
        'fan_outs': fan_outs,
        'centrality': bridges
    }
    
    behavioral_signals = {
        'cyber_alerts': cs_flags + bf_flags + it_flags
    }
    
    temporal_signals = {
        'risk_predictions': risk_predictions,
        'concentration': temporal_bursts,
        'cycle_pred': cycle_predictions
    }
    
    lstm_signals = {
        'emerging_links': emerging_links if 'emerging_links' in locals() else []
    }
    
    # Initialize consolidator with custom weights (parameterized)
    risk_weights = {
        'spatial': 0.20,      # Graph topology (cycles, fan-in/out, centrality)
        'behavioral': 0.10,   # Cyber behavioral alerts
        'temporal': 0.35,     # Temporal patterns (strongest predictor)
        'lstm': 0.25,         # ML-based link prediction
        'cyber': 0.10         # Additional cyber signals
    }
    
    consolidator = RiskConsolidator(weights=risk_weights)
    consolidated_risks = consolidator.consolidate_risks(
        spatial_signals=spatial_signals,
        behavioral_signals=behavioral_signals,
        temporal_signals=temporal_signals,
        lstm_signals=lstm_signals,
        all_nodes=list(analyzer.G.nodes())
    )
    
    # Save risk consolidation configuration
    print("\n[6.0.1] Saving risk consolidation weights and parameters...")
    os.makedirs('models', exist_ok=True)
    consolidation_config = {
        'weights': consolidator.weights,
        'signal_thresholds': consolidator.signal_thresholds,
        'normalize_output': consolidator.normalize_output,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    consolidation_config_path = 'models/consolidation_config.json'
    with open(consolidation_config_path, 'w') as f:
        json.dump(consolidation_config, f, indent=2)
    print(f"Consolidation weights saved to '{consolidation_config_path}'")
    print(f"  Phase Weights: {consolidator.weights}")
    
    # Generate summary report
    print("\n[6.1] Risk Score Summary")
    summary = consolidator.get_summary_report(consolidated_risks)
    print(f"  Total accounts analyzed: {summary['total_accounts']}")
    print(f"  High-risk (>0.7): {summary['high_risk_count']}")
    print(f"  Medium-risk (0.4-0.7): {summary['medium_risk_count']}")
    print(f"  Low-risk (0.0-0.4): {summary['low_risk_count']}")
    print(f"  Clean (0.0): {summary['clean_count']}")
    print(f"  Average risk score: {summary['avg_risk_score']:.3f}")
    print(f"  Highest risk score: {summary['max_risk_score']:.3f}")
    
    # Top 15 highest-risk accounts
    print("\n[6.2] TOP 15 HIGHEST-RISK ACCOUNTS (CONSOLIDATED)")
    print("-" * 95)
    print(f"{'Rank':<6}{'Account':<15}{'Final Score':<15}{'Spatial':<12}{'Behavioral':<14}{'Temporal':<12}{'LSTM':<12}{'Signals':<20}")
    print("-" * 95)
    
    top_accounts = consolidator.get_top_accounts(consolidated_risks, top_n=15)
    for account, risk_data in top_accounts:
        rank = risk_data['risk_rank']
        final_score = risk_data['final_risk_score']
        spatial_score = risk_data['spatial_score']
        behavioral_score = risk_data['behavioral_score']
        temporal_score = risk_data['temporal_score']
        lstm_score = risk_data['lstm_score']
        signal_count = len(risk_data['signals'])
        signals_str = ','.join(risk_data['signals'][:3]) + ('...' if signal_count > 3 else '')
        
        print(f"{rank:<6}{account:<15}{final_score:<15.4f}{spatial_score:<12.3f}{behavioral_score:<14.3f}{temporal_score:<12.3f}{lstm_score:<12.3f}{signals_str:<20}")
    
    print("-" * 95)
    
    # Save consolidated results to CSV
    print("\n[6.3] Saving consolidated risk scores to CSV...")
    consolidated_df = pd.DataFrame([
        {
            'account': account,
            'final_risk_score': data['final_risk_score'],
            'spatial_score': data['spatial_score'],
            'behavioral_score': data['behavioral_score'],
            'temporal_score': data['temporal_score'],
            'lstm_score': data['lstm_score'],
            'risk_rank': data['risk_rank'],
            'signal_count': len(data['signals']),
            'signals': '|'.join(data['signals'])
        }
        for account, data in consolidated_risks.items()
    ])
    
    consolidated_df = consolidated_df.sort_values('final_risk_score', ascending=False)
    consolidated_df.to_csv('consolidated_risk_scores.csv', index=False)
    print("Consolidated risk scores saved to 'consolidated_risk_scores.csv'")
    
    print("\n[6.4] Phase-Specific Insights")
    print(f"  Spatial: {len([s for s in consolidated_risks.values() if s['spatial_score'] > 0.0])} accounts flagged")
    print(f"  Behavioral: {len([s for s in consolidated_risks.values() if s['behavioral_score'] > 0.0])} accounts flagged")
    print(f"  Temporal: {len([s for s in consolidated_risks.values() if s['temporal_score'] > 0.0])} accounts flagged")
    print(f"  LSTM: {len([s for s in consolidated_risks.values() if s['lstm_score'] > 0.0])} accounts flagged")

    print("\nDone.")

if __name__ == "__main__":
    main()
