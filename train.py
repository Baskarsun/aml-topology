"""
AML Model Training Script
==========================
Dedicated script for training all ML models used in the AML detection pipeline.
Separates training from inference for cleaner architecture.

Usage:
    python train.py                     # Train all models
    python train.py --model lstm        # Train specific model
    python train.py --model gbdt --samples 10000
    python train.py --epochs 20         # Custom epochs for neural models
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
from src.temporal_predictor import TemporalPredictor
from src.embedding_builder import build_time_series_node_embeddings, build_pair_sequences_for_pairs
from src.risk_consolidator import RiskConsolidator

# Optional model imports with availability flags
try:
    from src.lstm_link_predictor import LSTMLinkPredictor, train_model, save_model
    _HAS_LSTM = True
except ImportError:
    _HAS_LSTM = False
    print("Warning: LSTM Link Predictor not available (missing torch)")

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

try:
    from src.adversarial_agent import AdversarialPatternAgent
    from src.adversarial_env import create_adversarial_env
    _HAS_ADVERSARIAL = True
except ImportError:
    _HAS_ADVERSARIAL = False
    print("Warning: Adversarial agent not available")


def ensure_models_dir():
    """Create models directory if it doesn't exist."""
    os.makedirs('models', exist_ok=True)


def generate_training_data(num_accounts=50, num_transactions=300):
    """
    Generate synthetic training data with known fraud patterns.
    
    Returns:
        df: Transaction DataFrame
        sim: TransactionSimulator instance
        suspicious_set: Set of known suspicious nodes
    """
    print("\n" + "="*60)
    print("GENERATING TRAINING DATA")
    print("="*60)
    
    sim = TransactionSimulator(num_accounts=num_accounts)
    sim.generate_organic_traffic(num_transactions=num_transactions)
    
    # Inject known fraud typologies for supervised learning
    fan_in_hub = "ACC_0010"
    sim.inject_fan_in(fan_in_hub, num_spokes=8, avg_amount=9500)
    
    fan_out_hub = "ACC_0025"
    sim.inject_fan_out(fan_out_hub, num_beneficiaries=6, total_amount=50000)
    
    sim.inject_cycle(length=5, amount=20000)
    
    df = sim.get_dataframe()
    print(f"✅ Generated {len(df)} transactions with {num_accounts} accounts")
    
    # Analyze to find suspicious nodes
    analyzer = AMLGraphAnalyzer(df)
    suspicious_set = set()
    
    # Fan-In detection
    fan_ins = analyzer.detect_fan_in(threshold_indegree=5)
    for item in fan_ins:
        suspicious_set.add(item['node'])
    
    # Fan-Out detection
    fan_outs = analyzer.detect_fan_out(threshold_outdegree=5)
    for item in fan_outs:
        suspicious_set.add(item['node'])
    
    # Cycle detection
    node_map, node_list, indptr, indices = build_csr(df)
    cycles = detect_cycles_csr(indptr, indices, node_list, max_k=6, max_cycles=500)
    for cycle in cycles:
        for node in cycle:
            suspicious_set.add(node)
    
    print(f"✅ Identified {len(suspicious_set)} suspicious nodes for labeling")
    
    return df, sim, analyzer, suspicious_set


def train_lstm_model(df, analyzer, suspicious_set, epochs=15, save_path='models/lstm_link_predictor.pt'):
    """
    Train LSTM Link Predictor model.
    
    Args:
        df: Transaction DataFrame
        analyzer: AMLGraphAnalyzer with built graph
        suspicious_set: Set of known suspicious nodes
        epochs: Training epochs
        save_path: Path to save model
    """
    if not _HAS_LSTM:
        print("❌ LSTM training skipped (torch not available)")
        return False
    
    print("\n" + "-"*60)
    print("TRAINING LSTM LINK PREDICTOR")
    print("-"*60)
    
    try:
        # Build time-series embeddings
        print("[1/4] Building time-series node embeddings...")
        fraud_scores = {node: (1.0 if node in suspicious_set else 0.0) for node in analyzer.G.nodes()}
        emb_map, feature_names = build_time_series_node_embeddings(df, freq='12H', fraud_scores=fraud_scores)
        print(f"  Built embeddings for {len(emb_map)} nodes, {len(feature_names)} features")
        
        # Generate candidate pairs
        print("[2/4] Generating candidate pairs...")
        pair_candidates = []
        susp_list = list(suspicious_set)
        
        if len(susp_list) < 2:
            print("❌ Insufficient suspicious nodes for LSTM training")
            return False
        
        # Suspicious-to-suspicious pairs
        for i, node_u in enumerate(susp_list):
            for node_v in susp_list[i+1:]:
                pair_candidates.append((node_u, node_v))
            
            # Suspicious-to-other pairs
            others = [n for n in analyzer.G.nodes() if n not in suspicious_set]
            sampled_others = np.random.choice(others, min(5, len(others)), replace=False) if others else []
            for node_v in sampled_others:
                pair_candidates.append((node_u, node_v))
        
        print(f"  Generated {len(pair_candidates)} candidate pairs")
        
        # Build sequences
        print("[3/4] Building pair sequences...")
        sequences, valid_pairs = build_pair_sequences_for_pairs(
            emb_map, pair_candidates, seq_len=3, allow_padding=True
        )
        print(f"  Built {len(valid_pairs)} valid sequences")
        
        if len(valid_pairs) == 0:
            print("❌ No valid sequences generated")
            return False
        
        # Create labels
        labels = np.array([
            float(analyzer.G.has_edge(u, v) or analyzer.G.has_edge(v, u))
            for u, v in valid_pairs
        ], dtype=np.float32)
        
        # Ensure class balance
        if len(np.unique(labels)) < 2:
            n = len(labels)
            k = max(1, int(0.2 * n))
            inds = np.random.choice(n, k, replace=False)
            labels[inds] = 1 - labels[inds]
        
        print(f"  Labels: {int((labels==1).sum())} positive, {int((labels==0).sum())} negative")
        
        # Train model
        print(f"[4/4] Training LSTM model ({epochs} epochs)...")
        input_size = sequences.shape[2]
        model = LSTMLinkPredictor(input_size=input_size, hidden_size=64, num_layers=1, dropout=0.1)
        model, history = train_model(
            model, sequences, labels, epochs=epochs, batch_size=32, lr=1e-3, use_class_weight=True
        )
        
        # Save model
        ensure_models_dir()
        save_model(model, save_path)
        print(f"✅ LSTM model saved to '{save_path}'")
        
        # Save metadata
        metadata = {
            'input_size': input_size,
            'hidden_size': 64,
            'num_layers': 1,
            'dropout': 0.1,
            'num_sequences': len(sequences),
            'num_epochs_trained': len(history['train_loss']),
            'final_val_auc': float(history['val_auc'][-1]) if history['val_auc'] else 0.0,
            'final_train_loss': float(history['train_loss'][-1]) if history['train_loss'] else 0.0,
            'feature_names': feature_names,
            'trained_at': datetime.now().isoformat()
        }
        metadata_path = save_path.replace('.pt', '_metadata.json').replace('lstm_link_predictor_metadata', 'lstm_metadata')
        if 'lstm_link_predictor' in save_path:
            metadata_path = 'models/lstm_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✅ LSTM metadata saved to '{metadata_path}'")
        
        return True
        
    except Exception as e:
        print(f"❌ LSTM training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def train_gbdt_model(n_samples=5000, save_path='models/lgb_model.txt'):
    """
    Train GBDT (LightGBM) classifier for transaction-level anomalies.
    
    Args:
        n_samples: Number of training samples to generate
        save_path: Path to save model (handled internally by gbdt_detector)
    """
    if not _HAS_GBDT:
        print("❌ GBDT training skipped (lightgbm not available)")
        return False
    
    print("\n" + "-"*60)
    print("TRAINING GBDT CLASSIFIER")
    print("-"*60)
    
    try:
        ensure_models_dir()
        gbdt_detector_demo(n=n_samples, save_model_flag=True)
        print(f"✅ GBDT model trained and saved to 'models/'")
        return True
    except Exception as e:
        print(f"❌ GBDT training failed: {e}")
        return False


def train_gnn_model(num_nodes=50, epochs=20, save_path='models/gnn_model.pt'):
    """
    Train GNN for node classification.
    
    Args:
        num_nodes: Number of nodes in synthetic graph
        epochs: Training epochs
        save_path: Path to save model (handled internally)
    """
    if not _HAS_GNN:
        print("❌ GNN training skipped (torch_geometric not available)")
        return False
    
    print("\n" + "-"*60)
    print("TRAINING GNN MODEL")
    print("-"*60)
    
    try:
        ensure_models_dir()
        train_gnn_demo(num_nodes=num_nodes, epochs=epochs, save_model_flag=True)
        print(f"✅ GNN model trained and saved to '{save_path}'")
        return True
    except Exception as e:
        print(f"❌ GNN training failed: {e}")
        return False


def train_sequence_model(num_sequences=1000, seq_len=15, epochs=5, save_path='models/sequence_detector_model.pt'):
    """
    Train Sequence Detector for event anomaly detection.
    
    Args:
        num_sequences: Number of training sequences
        seq_len: Sequence length
        epochs: Training epochs
        save_path: Path to save model (handled internally)
    """
    if not _HAS_SEQUENCE:
        print("❌ Sequence detector training skipped (dependencies not available)")
        return False
    
    print("\n" + "-"*60)
    print("TRAINING SEQUENCE DETECTOR")
    print("-"*60)
    
    try:
        ensure_models_dir()
        sequence_detector_demo(num_sequences=num_sequences, seq_len=seq_len, epochs=epochs, 
                               model_type='lstm', save_model_flag=True)
        print(f"✅ Sequence detector trained and saved to '{save_path}'")
        return True
    except Exception as e:
        print(f"❌ Sequence detector training failed: {e}")
        return False


def train_adversarial_agent(
    num_episodes: int = 100,
    steps_per_episode: int = 10,
    num_base_accounts: int = 300,
    save_path: str = 'models/adversarial_agent.pt',
    device: str = 'cpu'
):
    """
    Train Adversarial Pattern Agent (PPO) to generate evasive AML patterns.
    
    This trains a reinforcement learning agent that learns to inject patterns
    (peel chains, wash trading, etc.) that evade GNN-based detection.
    
    Args:
        num_episodes: Number of training episodes
        steps_per_episode: Pattern injections per episode
        num_base_accounts: Size of base transaction graph
        save_path: Path to save trained agent
        device: 'cpu' or 'cuda'
    """
    if not _HAS_ADVERSARIAL:
        print("❌ Adversarial training skipped (dependencies not available)")
        return False
    
    print("\n" + "-"*60)
    print("TRAINING ADVERSARIAL PATTERN AGENT (PPO)")
    print("-"*60)
    print(f"Episodes: {num_episodes}")
    print(f"Steps/Episode: {steps_per_episode}")
    print(f"Base accounts: {num_base_accounts}")
    print(f"Device: {device}")
    
    try:
        import torch
        from collections import deque
        import time
        
        # Check CUDA
        if device == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU")
            device = 'cpu'
        
        # Create environment
        print("\n[1/4] Creating adversarial environment...")
        env = create_adversarial_env(
            num_base_accounts=num_base_accounts,
            seed=42,
            device=device
        )
        
        # Create agent
        print("[2/4] Initializing PPO agent...")
        agent = AdversarialPatternAgent(
            state_dim=49,
            hidden_dim=128,
            lr=3e-4,
            device=device
        )
        
        # Training metrics
        episode_rewards = deque(maxlen=100)
        episode_detections = deque(maxlen=100)
        
        print(f"[3/4] Training for {num_episodes} episodes...")
        start_time = time.time()
        
        for episode in range(1, num_episodes + 1):
            state = env.reset()
            episode_reward = 0
            detection_rates = []
            
            for step in range(steps_per_episode):
                action_dict, raw_output = agent.select_action(state)
                next_state, reward, done, info = env.step(action_dict)
                agent.store_experience(state, raw_output, reward, done)
                
                episode_reward += reward
                detection_rates.append(info['mean_detection'])
                state = next_state
                
                if done:
                    break
            
            # Update detection history
            mean_detection = np.mean(detection_rates) if detection_rates else 0.5
            agent.update_detection_history(mean_detection)
            
            episode_rewards.append(episode_reward)
            episode_detections.append(mean_detection)
            
            # PPO update every 10 episodes or when buffer is reasonably full
            if len(agent.buffer) >= 128 or episode == num_episodes:
                agent.update(epochs=4, batch_size=64)
            
            # Progress logging
            if episode % 10 == 0 or episode == num_episodes:
                avg_reward = np.mean(episode_rewards)
                avg_detection = np.mean(episode_detections)
                elapsed = time.time() - start_time
                print(f"  Episode {episode:4d}/{num_episodes} | "
                      f"Reward: {avg_reward:7.3f} | "
                      f"Detection: {avg_detection:.3f} | "
                      f"Time: {elapsed:.1f}s")
        
        # Save model
        print(f"[4/4] Saving agent to '{save_path}'...")
        ensure_models_dir()
        agent.save(save_path)
        
        # Save metadata
        metadata = {
            'num_episodes': num_episodes,
            'steps_per_episode': steps_per_episode,
            'num_base_accounts': num_base_accounts,
            'final_avg_reward': float(np.mean(episode_rewards)),
            'final_avg_detection': float(np.mean(episode_detections)),
            'trained_at': datetime.now().isoformat()
        }
        metadata_path = save_path.replace('.pt', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✅ Adversarial agent trained and saved!")
        print(f"   Final avg reward: {np.mean(episode_rewards):.3f}")
        print(f"   Final avg detection: {np.mean(episode_detections):.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Adversarial training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def save_consolidation_config(weights=None):
    """
    Save risk consolidation configuration.
    
    Args:
        weights: Optional custom weights dict
    """
    print("\n" + "-"*60)
    print("SAVING CONSOLIDATION CONFIG")
    print("-"*60)
    
    if weights is None:
        weights = {
            'spatial': 0.20,
            'behavioral': 0.10,
            'temporal': 0.35,
            'lstm': 0.25,
            'cyber': 0.10
        }
    
    consolidator = RiskConsolidator(weights=weights)
    
    ensure_models_dir()
    config = {
        'weights': consolidator.weights,
        'signal_thresholds': consolidator.signal_thresholds,
        'normalize_output': consolidator.normalize_output,
        'trained_at': datetime.now().isoformat()
    }
    
    config_path = 'models/consolidation_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Consolidation config saved to '{config_path}'")
    print(f"   Weights: {weights}")
    return True


def train_all(epochs=15, gbdt_samples=5000, adversarial_episodes=0):
    """
    Train all models in the AML pipeline.
    
    Args:
        epochs: Epochs for neural models (LSTM, GNN, Sequence)
        gbdt_samples: Number of samples for GBDT training
        adversarial_episodes: Episodes for adversarial agent (0 to skip)
    """
    print("\n" + "="*60)
    print("AML MODEL TRAINING PIPELINE")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Generate training data
    df, sim, analyzer, suspicious_set = generate_training_data()
    
    # Track results
    results = {}
    
    # Train LSTM
    results['lstm'] = train_lstm_model(df, analyzer, suspicious_set, epochs=epochs)
    
    # Train GBDT
    results['gbdt'] = train_gbdt_model(n_samples=gbdt_samples)
    
    # Train GNN (optional)
    results['gnn'] = train_gnn_model(epochs=epochs)
    
    # Train Sequence Detector (optional)
    results['sequence'] = train_sequence_model(epochs=max(5, epochs//3))
    
    # Train Adversarial Agent (optional - for red-teaming)
    if adversarial_episodes > 0:
        results['adversarial'] = train_adversarial_agent(
            num_episodes=adversarial_episodes,
            steps_per_episode=10,
            num_base_accounts=300
        )
    else:
        results['adversarial'] = None  # Skipped
    
    # Save consolidation config
    results['consolidation'] = save_consolidation_config()
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    
    for model, success in results.items():
        status = "✅ Success" if success else "❌ Skipped/Failed"
        print(f"  {model.upper():15} {status}")
    
    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nModel files saved to 'models/' directory.")
    print("You can now run inference with: python pipeline_simulation.py")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train AML detection models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py                        # Train all models
  python train.py --model lstm           # Train only LSTM
  python train.py --model gbdt           # Train only GBDT
  python train.py --model adversarial    # Train adversarial agent
  python train.py --epochs 20            # Custom epochs
  python train.py --samples 10000        # Custom GBDT samples
  python train.py --adversarial 100      # Include adversarial (100 episodes)
        """
    )
    
    parser.add_argument(
        '--model', 
        choices=['all', 'lstm', 'gbdt', 'gnn', 'sequence', 'config', 'adversarial'],
        default='all',
        help='Which model to train (default: all)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=15,
        help='Training epochs for neural models (default: 15)'
    )
    
    parser.add_argument(
        '--samples',
        type=int,
        default=5000,
        help='Number of samples for GBDT training (default: 5000)'
    )
    
    parser.add_argument(
        '--adversarial',
        type=int,
        default=0,
        help='Adversarial agent episodes (0 to skip, default: 0)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device for adversarial training (default: cpu)'
    )
    
    args = parser.parse_args()
    
    if args.model == 'all':
        train_all(epochs=args.epochs, gbdt_samples=args.samples, 
                  adversarial_episodes=args.adversarial)
    
    elif args.model == 'lstm':
        df, sim, analyzer, suspicious_set = generate_training_data()
        train_lstm_model(df, analyzer, suspicious_set, epochs=args.epochs)
    
    elif args.model == 'gbdt':
        train_gbdt_model(n_samples=args.samples)
    
    elif args.model == 'gnn':
        train_gnn_model(epochs=args.epochs)
    
    elif args.model == 'sequence':
        train_sequence_model(epochs=max(5, args.epochs//3))
    
    elif args.model == 'config':
        save_consolidation_config()
    
    elif args.model == 'adversarial':
        # Default to 100 episodes if not specified
        episodes = args.adversarial if args.adversarial > 0 else 100
        train_adversarial_agent(
            num_episodes=episodes,
            steps_per_episode=10,
            num_base_accounts=300,
            device=args.device
        )


if __name__ == "__main__":
    main()
