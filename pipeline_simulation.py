
import pandas as pd
import numpy as np
import json
import os
import time
from datetime import datetime
import sys
from src.simulator import TransactionSimulator
from src.graph_analyzer import AMLGraphAnalyzer
from src.csr_cycle_detector import build_csr, detect_cycles_csr
from src.visualizer import plot_graph
from src.behavioral_detector import BehavioralDetector
from src.temporal_predictor import TemporalPredictor, SequenceAnalyzer
from src.embedding_builder import build_time_series_node_embeddings, build_pair_sequences_for_pairs
try:
    from src.lstm_link_predictor import LSTMLinkPredictor, load_model as load_lstm_model, predict_proba
    _HAS_LSTM = True
except ImportError:
    _HAS_LSTM = False
    print("Warning: LSTM Link Predictor not available (missing torch)")
from src.risk_consolidator import RiskConsolidator
from src.metrics_logger import get_metrics_logger

def run_pipeline():
    print("="*60)
    print("🚀 AML PIPELINE SIMULATION")
    print("="*60)
    print("Simulating full batch inference pipeline (Graph + Temporal + ML)...\n")
    
    # ---------------------------------------------------------
    # 1. SIMULATION (DATA INGESTION)
    # ---------------------------------------------------------
    print("[Phase 1] Data Ingestion & Simulation")
    try:
        sim = TransactionSimulator(num_accounts=50)
        sim.generate_organic_traffic(num_transactions=400)
        
        # Inject Typologies to ensure we have signals to detect
        sim.inject_fan_in("ACC_0010", num_spokes=8, avg_amount=9500)
        sim.inject_fan_out("ACC_0025", num_beneficiaries=6, total_amount=50000)
        sim.inject_cycle(length=5, amount=25000)
        
        df = sim.get_dataframe()
        print(f"   ✅ Generated {len(df)} transactions")
        print(f"   ✅ Ingested {len(sim.accounts)} accounts")
    except Exception as e:
        print(f"   ❌ Simulation failed: {e}")
        return

    # ---------------------------------------------------------
    # 2. GRAPH TOPOLOGY ANALYSIS
    # ---------------------------------------------------------
    print("\n[Phase 2] Graph Topology Analysis")
    suspicious_set = set()
    spatial_signals = {'cycles': [], 'fan_ins': [], 'fan_outs': [], 'centrality': []}
    
    try:
        analyzer = AMLGraphAnalyzer(df)
        
        # Fan-In
        fan_ins = analyzer.detect_fan_in(threshold_indegree=5)
        if fan_ins:
            print(f"   ⚠️  Detected {len(fan_ins)} Fan-In patterns")
        spatial_signals['fan_ins'] = fan_ins
        
        # Fan-Out
        fan_outs = analyzer.detect_fan_out(threshold_outdegree=5)
        if fan_outs:
            print(f"   ⚠️  Detected {len(fan_outs)} Fan-Out patterns")
        spatial_signals['fan_outs'] = fan_outs
        
        # Cycles
        node_map, node_list, indptr, indices = build_csr(df)
        cycles = detect_cycles_csr(indptr, indices, node_list, max_k=6, max_cycles=500)
        if cycles:
            print(f"   ⚠️  Detected {len(cycles)} Cyclic patterns")
        spatial_signals['cycles'] = cycles
        
        # Centrality
        bridges = analyzer.calculate_centrality()
        spatial_signals['centrality'] = bridges
        print("   ✅ Topology analysis complete")
        
        # Aggregate suspicious nodes
        for item in fan_ins + fan_outs: suspicious_set.add(item['node'])
        for cycle in cycles: 
            for node in cycle: suspicious_set.add(node)
            
    except Exception as e:
        print(f"   ❌ Topology analysis failed: {e}")

    # ---------------------------------------------------------
    # 3. BEHAVIORAL ANALYSIS (Synthetic)
    # ---------------------------------------------------------
    print("\n[Phase 3] Cyber-Behavioral Analysis")
    behavioral_signals = {'cyber_alerts': []}
    
    try:
        detector = BehavioralDetector()
        
        # Generate synthetic login events matching our bad actors
        login_events = []
        if 'ACC_0010' in suspicious_set:
             # Brute force pattern for Fan-In hub
             for i in range(5):
                 login_events.append({'user_id': 'ACC_0010', 'timestamp': sim.start_time + i*5, 'success': False, 'ip': '1.2.3.4'})
             login_events.append({'user_id': 'ACC_0010', 'timestamp': sim.start_time + 30, 'success': True, 'ip': '1.2.3.4'})
        
        if 'ACC_0025' in suspicious_set:
             # Impossible travel
             login_events.append({'user_id': 'ACC_0025', 'timestamp': sim.start_time, 'success': True, 'ip': '8.8.8.8', 'lat': 51.5, 'lon': -0.1})
             login_events.append({'user_id': 'ACC_0025', 'timestamp': sim.start_time + 600, 'success': True, 'ip': '9.9.9.9', 'lat': 40.7, 'lon': -74.0})

        logins_df = pd.DataFrame(login_events)
        if not logins_df.empty:
            cs_flags = detector.detect_credential_stuffing(logins_df)
            bf_flags = detector.detect_bruteforce_and_new_device(logins_df)
            it_flags = detector.detect_impossible_travel(logins_df)
            
            alerts = cs_flags + bf_flags + it_flags
            behavioral_signals['cyber_alerts'] = alerts
            if alerts:
                print(f"   ⚠️  Detected {len(alerts)} Cyber-behavioral alerts")
        
        print("   ✅ Behavioral analysis complete")
    except Exception as e:
         print(f"   ❌ Behavioral analysis failed: {e}")

    # ---------------------------------------------------------
    # 4. TEMPORAL PREDICTION
    # ---------------------------------------------------------
    print("\n[Phase 4] Temporal Predictive Analysis")
    temporal_signals = {'risk_predictions': [], 'concentration': [], 'cycle_pred': []}
    
    try:
        temporal_pred = TemporalPredictor(lookback_days=30, forecast_days=7)
        temporal_pred.establish_baselines(df)
        
        # Volume Acceleration
        vol_alerts = temporal_pred.detect_volume_acceleration(df)
        
        # Risk Escalation
        risk_preds = temporal_pred.forecast_risk_escalation(df)
        if risk_preds:
             print(f"   🔮 Predicted risk escalation for {len(risk_preds)} accounts")
        temporal_signals['risk_predictions'] = risk_preds
        
        # Temporal Concentration
        conc_alerts = temporal_pred.detect_temporal_concentration(df)
        if conc_alerts:
             print(f"   ⚠️  Detected {len(conc_alerts)} Temporal Concentration bursts")
        temporal_signals['concentration'] = conc_alerts
        
        print("   ✅ Temporal analysis complete")
    except Exception as e:
        print(f"   ❌ Temporal analysis failed: {e}")

    # ---------------------------------------------------------
    # 5. ML INFERENCE (LSTM Link Prediction)
    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # 5. ML INFERENCE (LSTM Link Prediction)
    # ---------------------------------------------------------
    print("\n[Phase 5] LSTM Link Prediction Inference")
    lstm_signals = {'emerging_links': []}
    
    if not _HAS_LSTM:
         print("   ⚠️  LSTM module not loaded (missing torch). Skipping Phase 5.")
    else:
        try:
            # Load Model
            lstm_path = 'models/lstm_link_predictor.pt'
            meta_path = 'models/lstm_metadata.json'
            
            if os.path.exists(lstm_path) and os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                
                # Reconstruct model architecture from metadata
                input_size = meta.get('input_size', 10) # Fallback if missing
                model = LSTMLinkPredictor(input_size=input_size, hidden_size=64, num_layers=1)
                model = load_lstm_model(model, lstm_path)
                
                # Build Embeddings for Inference
                # We don't have ground truth 'fraud_scores' for new data in reality, 
                # so we use heuristic suspicious set as a proxy or just raw embeddings.
                # Here we follow main.py logic:
                heuristic_labels = {node: (1.0 if node in suspicious_set else 0.0) for node in analyzer.G.nodes()}
                emb_map, _ = build_time_series_node_embeddings(df, freq='12H', fraud_scores=heuristic_labels)
                
                # Generate Candidate Pairs (All pairs or heuristic filter)
                # For efficiency in simulation, use similar logic to main.py
                candidates = []
                nodes = list(analyzer.G.nodes())
                if len(nodes) > 1:
                    # Sample random pairs + suspicious pairs
                    for _ in range(50):
                        u, v = np.random.choice(nodes, 2, replace=False)
                        candidates.append((u,v))
                    
                    sequences, valid_pairs = build_pair_sequences_for_pairs(emb_map, candidates, seq_len=3, allow_padding=True)
                    
                    if len(valid_pairs) > 0:
                        probs = predict_proba(model, sequences)
                        
                        # Filter for high probability links
                        high_prob_links = []
                        for i, p in enumerate(probs):
                            if p > 0.5:
                                high_prob_links.append((valid_pairs[i], float(p)))
                        
                        lstm_signals['emerging_links'] = high_prob_links
                        if high_prob_links:
                            print(f"   🔮 Predicted {len(high_prob_links)} emerging suspicious links")
                
                print("   ✅ LSTM inference complete")
            else:
                print("   ⚠️  LSTM model not found. Skipping Phase 5.")
                
        except Exception as e:
            print(f"   ❌ LSTM inference failed: {e}")

    # ---------------------------------------------------------
    # 6. RISK CONSOLIDATION
    # ---------------------------------------------------------
    print("\n[Phase 6] Risk Consolidation")
    
    try:
        # Load weights/config if available, else usage default
        config_path = 'models/consolidation_config.json'
        weights = None
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                weights = config.get('weights')
        
        consolidator = RiskConsolidator(weights=weights)
        
        consolidated_risks = consolidator.consolidate_risks(
            spatial_signals=spatial_signals,
            behavioral_signals=behavioral_signals,
            temporal_signals=temporal_signals,
            lstm_signals=lstm_signals,
            all_nodes=list(analyzer.G.nodes())
        )
        
        summary = consolidator.get_summary_report(consolidated_risks)
        print("\n   📊 PIPELINE RESULTS SUMMARY")
        print(f"   Total Accounts: {summary['total_accounts']}")
        print(f"   🔴 High Risk:   {summary['high_risk_count']}")
        print(f"   🟡 Medium Risk: {summary['medium_risk_count']}")
        print(f"   🟢 Low Risk:    {summary['low_risk_count']}")
        print(f"   ⚪ Clean:       {summary['clean_count']}")
        
        # Helper for risk level
        def get_level(s):
            if s >= 0.7: return 'HIGH'
            elif s >= 0.4: return 'MEDIUM'
            elif s > 0.0: return 'LOW'
            else: return 'CLEAN'
        
        # Log to Dashboard DB
        print("\n   📡 Sending results to Dashboard (metrics.db)...")
        logger = get_metrics_logger()
        for node, data in consolidated_risks.items():
            final_score = data['final_risk_score']
            risk_level = get_level(final_score)
            
            logger.log_inference({
                'timestamp': datetime.utcnow().isoformat(),
                'account_id': node,
                'endpoint': '/simulation/pipeline',
                'engine': 'PipelineShim',
                'latency_ms': 0.0,
                'risk_score': final_score,
                'risk_level': risk_level,
                'component_scores': {
                    'signals': data['signals'],
                    'spatial': data.get('spatial_score', 0),
                    'temporal': data.get('temporal_score', 0),
                    'behavioral': data.get('behavioral_score', 0)
                },
                'status': 'success'
            })
            
        # Save output
        out_file = 'simulation_pipeline_results.csv'
        pd.DataFrame([
            {
                'account': k, 
                'score': v['final_risk_score'], 
                'risk_level': get_level(v['final_risk_score']),
                'signals': '|'.join(v['signals'])
            }
            for k, v in consolidated_risks.items()
        ]).sort_values('score', ascending=False).to_csv(out_file, index=False)
        print(f"\n   ✅ Results saved to '{out_file}'")
        
    except Exception as e:
        print(f"   ❌ Consolidation failed: {e}")

    print("\n" + "="*60)
    print("FINISHED")
    print("="*60)

if __name__ == "__main__":
    run_pipeline()
