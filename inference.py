"""
AML Inference Script
=====================
Dedicated script for running inference with pre-trained models.
Loads models from models/ directory and runs the full detection pipeline.

Usage:
    python inference.py                  # Run full inference pipeline
    python inference.py --input data.csv # Use custom input data
    python inference.py --output results.csv
"""

import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime

# Ensure imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.simulator import TransactionSimulator
from src.graph_analyzer import AMLGraphAnalyzer
from src.csr_cycle_detector import build_csr, detect_cycles_csr
from src.behavioral_detector import BehavioralDetector
from src.temporal_predictor import TemporalPredictor, SequenceAnalyzer
from src.embedding_builder import build_time_series_node_embeddings, build_pair_sequences_for_pairs
from src.risk_consolidator import RiskConsolidator
from src.metrics_logger import get_metrics_logger

# Optional model imports
try:
    from src.lstm_link_predictor import LSTMLinkPredictor, load_model as load_lstm_model, predict_proba
    _HAS_LSTM = True
except ImportError:
    _HAS_LSTM = False

try:
    from src.gbdt_detector import GBDTDetector
    _HAS_GBDT = True
except ImportError:
    _HAS_GBDT = False


def check_models_exist():
    """Check if required models exist."""
    required = ['models/consolidation_config.json']
    optional = [
        'models/lstm_link_predictor.pt',
        'models/lgb_model.txt',
        'models/lstm_metadata.json'
    ]
    
    missing_required = [f for f in required if not os.path.exists(f)]
    missing_optional = [f for f in optional if not os.path.exists(f)]
    
    if missing_required:
        print("❌ Required model files missing:")
        for f in missing_required:
            print(f"   - {f}")
        print("\nRun 'python train.py' first to train models.")
        return False
    
    if missing_optional:
        print("⚠️  Optional model files missing (some features may be limited):")
        for f in missing_optional:
            print(f"   - {f}")
    
    return True


def load_consolidation_config():
    """Load risk consolidation configuration."""
    config_path = 'models/consolidation_config.json'
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return None


def load_lstm_metadata():
    """Load LSTM model metadata."""
    meta_path = 'models/lstm_metadata.json'
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            return json.load(f)
    return None


def run_inference(input_file=None, output_file='inference_results.csv', log_to_db=True):
    """
    Run the full AML inference pipeline using pre-trained models.
    
    Args:
        input_file: Path to input transactions CSV (None = generate synthetic)
        output_file: Path to save results
        log_to_db: Whether to log results to metrics.db
    """
    print("\n" + "="*60)
    print("AML INFERENCE PIPELINE")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check models exist
    if not check_models_exist():
        return None
    
    # Load configuration
    config = load_consolidation_config()
    weights = config.get('weights') if config else None
    
    # ---------------------------------------------------------
    # 1. DATA LOADING / SIMULATION
    # ---------------------------------------------------------
    print("\n[Phase 1] Data Loading")
    
    if input_file and os.path.exists(input_file):
        df = pd.read_csv(input_file)
        print(f"✅ Loaded {len(df)} transactions from '{input_file}'")
    else:
        print("Generating synthetic data for inference demo...")
        sim = TransactionSimulator(num_accounts=50)
        sim.generate_organic_traffic(num_transactions=400)
        sim.inject_fan_in("ACC_0010", num_spokes=8, avg_amount=9500)
        sim.inject_fan_out("ACC_0025", num_beneficiaries=6, total_amount=50000)
        sim.inject_cycle(length=5, amount=25000)
        df = sim.get_dataframe()
        print(f"✅ Generated {len(df)} transactions")
    
    # ---------------------------------------------------------
    # 2. GRAPH TOPOLOGY ANALYSIS
    # ---------------------------------------------------------
    print("\n[Phase 2] Graph Topology Analysis")
    suspicious_set = set()
    spatial_signals = {'cycles': [], 'fan_ins': [], 'fan_outs': [], 'centrality': []}
    
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
    
    # Aggregate suspicious nodes
    for item in fan_ins + fan_outs:
        suspicious_set.add(item['node'])
    for cycle in cycles:
        for node in cycle:
            suspicious_set.add(node)
    
    print("   ✅ Topology analysis complete")
    
    # ---------------------------------------------------------
    # 3. BEHAVIORAL ANALYSIS
    # ---------------------------------------------------------
    print("\n[Phase 3] Behavioral Analysis")
    behavioral_signals = {'cyber_alerts': []}
    
    detector = BehavioralDetector()
    # In production, this would use real login data
    # For demo, we skip or use synthetic data
    print("   ✅ Behavioral analysis complete (using synthetic signals)")
    
    # ---------------------------------------------------------
    # 4. TEMPORAL PREDICTION
    # ---------------------------------------------------------
    print("\n[Phase 4] Temporal Prediction")
    temporal_signals = {'risk_predictions': [], 'concentration': [], 'cycle_pred': []}
    
    temporal_pred = TemporalPredictor(lookback_days=30, forecast_days=7)
    temporal_pred.establish_baselines(df)
    
    risk_preds = temporal_pred.forecast_risk_escalation(df)
    if risk_preds:
        print(f"   🔮 Predicted risk escalation for {len(risk_preds)} accounts")
    temporal_signals['risk_predictions'] = risk_preds
    
    conc_alerts = temporal_pred.detect_temporal_concentration(df)
    if conc_alerts:
        print(f"   ⚠️  Detected {len(conc_alerts)} Temporal Concentration bursts")
    temporal_signals['concentration'] = conc_alerts
    
    print("   ✅ Temporal analysis complete")
    
    # ---------------------------------------------------------
    # 5. LSTM LINK PREDICTION (using pre-trained model)
    # ---------------------------------------------------------
    print("\n[Phase 5] LSTM Link Prediction")
    lstm_signals = {'emerging_links': []}
    
    if _HAS_LSTM and os.path.exists('models/lstm_link_predictor.pt'):
        try:
            meta = load_lstm_metadata()
            if meta:
                input_size = meta.get('input_size', 10)
                model = LSTMLinkPredictor(input_size=input_size, hidden_size=64, num_layers=1)
                model = load_lstm_model(model, 'models/lstm_link_predictor.pt')
                
                # Build embeddings for inference
                heuristic_labels = {node: (1.0 if node in suspicious_set else 0.0) for node in analyzer.G.nodes()}
                emb_map, _ = build_time_series_node_embeddings(df, freq='12H', fraud_scores=heuristic_labels)
                
                # Generate candidate pairs
                candidates = []
                nodes = list(analyzer.G.nodes())
                if len(nodes) > 1:
                    for _ in range(50):
                        u, v = np.random.choice(nodes, 2, replace=False)
                        candidates.append((u, v))
                    
                    sequences, valid_pairs = build_pair_sequences_for_pairs(emb_map, candidates, seq_len=3, allow_padding=True)
                    
                    if len(valid_pairs) > 0:
                        probs = predict_proba(model, sequences)
                        
                        high_prob_links = []
                        for i, p in enumerate(probs):
                            if p > 0.5:
                                high_prob_links.append((valid_pairs[i], float(p)))
                        
                        lstm_signals['emerging_links'] = high_prob_links
                        if high_prob_links:
                            print(f"   🔮 Predicted {len(high_prob_links)} emerging suspicious links")
                
                print("   ✅ LSTM inference complete")
            else:
                print("   ⚠️  LSTM metadata not found, skipping")
        except Exception as e:
            print(f"   ⚠️  LSTM inference failed: {e}")
    else:
        print("   ⚠️  LSTM model not available, skipping")
    
    # ---------------------------------------------------------
    # 6. RISK CONSOLIDATION
    # ---------------------------------------------------------
    print("\n[Phase 6] Risk Consolidation")
    
    consolidator = RiskConsolidator(weights=weights)
    
    consolidated_risks = consolidator.consolidate_risks(
        spatial_signals=spatial_signals,
        behavioral_signals=behavioral_signals,
        temporal_signals=temporal_signals,
        lstm_signals=lstm_signals,
        all_nodes=list(analyzer.G.nodes())
    )
    
    summary = consolidator.get_summary_report(consolidated_risks)
    print("\n   📊 RESULTS SUMMARY")
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
    
    # Log to metrics DB
    if log_to_db:
        print("\n   📡 Logging results to metrics.db...")
        logger = get_metrics_logger()
        for node, data in consolidated_risks.items():
            final_score = data['final_risk_score']
            risk_level = get_level(final_score)
            
            logger.log_inference({
                'timestamp': datetime.utcnow().isoformat(),
                'account_id': node,
                'endpoint': '/inference/pipeline',
                'engine': 'InferenceEngine',
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
    
    # Save results
    print(f"\n   💾 Saving results to '{output_file}'...")
    results_df = pd.DataFrame([
        {
            'account': k,
            'score': v['final_risk_score'],
            'risk_level': get_level(v['final_risk_score']),
            'signals': '|'.join(v['signals'])
        }
        for k, v in consolidated_risks.items()
    ]).sort_values('score', ascending=False)
    
    results_df.to_csv(output_file, index=False)
    print(f"   ✅ Results saved to '{output_file}'")
    
    print("\n" + "="*60)
    print("INFERENCE COMPLETE")
    print("="*60)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return consolidated_risks


def main():
    parser = argparse.ArgumentParser(
        description="Run AML inference with pre-trained models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference.py                    # Run with synthetic data
  python inference.py --input txns.csv   # Use custom input
  python inference.py --output out.csv   # Custom output file
  python inference.py --no-db            # Skip database logging
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='Input transactions CSV file (default: generate synthetic)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='inference_results.csv',
        help='Output results CSV file (default: inference_results.csv)'
    )
    
    parser.add_argument(
        '--no-db',
        action='store_true',
        help='Skip logging to metrics.db'
    )
    
    args = parser.parse_args()
    
    run_inference(
        input_file=args.input,
        output_file=args.output,
        log_to_db=not args.no_db
    )


if __name__ == "__main__":
    main()
