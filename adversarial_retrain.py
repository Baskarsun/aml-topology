"""
Adversarial GNN Retraining Loop with Experience Replay
========================================================
Uses a trained adversarial agent to generate challenging patterns,
then retrains the GNN discriminator on MIXED data (original + adversarial)
to prevent catastrophic forgetting.

Key Features:
- Experience Replay: Maintains buffer of original training patterns
- Mixed Training: 50% original + 50% adversarial data each epoch
- Separate Evaluation: Tracks performance on both pattern types

Usage:
    python adversarial_retrain.py                    # One round of retraining
    python adversarial_retrain.py --rounds 5         # Multiple adversarial rounds
    python adversarial_retrain.py --device cuda      # Use GPU
"""

import os
import sys
import argparse
import json
import time
import random
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

# Add project root to path
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.adversarial_agent import AdversarialPatternAgent, action_to_pattern_name
from src.adversarial_env import AdversarialAMLEnv
from src.gnn_trainer import (
    GraphSage, 
    build_node_features, 
    generate_synthetic_graph,
    adjacency_sparse_from_nx,
    graph_to_tx_df
)
from src.pattern_library import PatternLibrary


# =============================================================================
# EXPERIENCE REPLAY BUFFER
# =============================================================================

class ExperienceReplayBuffer:
    """
    Buffer for storing training graphs to prevent catastrophic forgetting.
    
    Stores (graph, suspicious_nodes, data_type) tuples where data_type
    is 'organic', 'original_fraud', or 'adversarial'.
    
    Three-way classification:
    - organic: Clean graphs with no fraud patterns (mostly label=0)
    - original_fraud: Traditional Fan-In patterns (mule hubs, smurfing)
    - adversarial: Agent-generated Fan-Out patterns (peel chains)
    """
    
    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)
        self.counts = {'organic': 0, 'original_fraud': 0, 'adversarial': 0}
    
    def add(self, graph, suspicious_nodes: List[str], data_type: str):
        """Add a graph to the buffer."""
        self.buffer.append({
            'graph': graph,
            'suspicious_nodes': set(suspicious_nodes),
            'data_type': data_type,
        })
        if data_type in self.counts:
            self.counts[data_type] = min(self.counts[data_type] + 1, self.capacity // 3)
    
    def sample(self, n: int, organic_ratio: float = 0.6, fraud_ratio: float = 0.2) -> List[Dict]:
        """
        Sample n graphs with 3-way mix ratio.
        
        Args:
            n: Number of graphs to sample
            organic_ratio: Fraction of clean organic samples (0.6 = 60%)
            fraud_ratio: Fraction of original fraud samples (0.2 = 20%)
            # Remaining is adversarial (0.2 = 20%)
            
        Returns:
            List of graph dicts
        """
        organic = [x for x in self.buffer if x['data_type'] == 'organic']
        original_fraud = [x for x in self.buffer if x['data_type'] == 'original_fraud']
        adversarial = [x for x in self.buffer if x['data_type'] == 'adversarial']
        
        n_organic = int(n * organic_ratio)
        n_original = int(n * fraud_ratio)
        n_adversarial = n - n_organic - n_original
        
        samples = []
        
        if organic:
            samples.extend(random.choices(organic, k=min(n_organic, len(organic) * 3)))
        if original_fraud:
            samples.extend(random.choices(original_fraud, k=min(n_original, len(original_fraud) * 3)))
        if adversarial:
            samples.extend(random.choices(adversarial, k=min(n_adversarial, len(adversarial) * 3)))
        
        random.shuffle(samples)
        return samples[:n]
    
    def get_all(self) -> List[Dict]:
        """Get all items in buffer."""
        return list(self.buffer)
    
    def __len__(self):
        return len(self.buffer)
    
    def stats(self) -> Dict:
        """Get buffer statistics."""
        organic = sum(1 for x in self.buffer if x['data_type'] == 'organic')
        original_fraud = sum(1 for x in self.buffer if x['data_type'] == 'original_fraud')
        adversarial = sum(1 for x in self.buffer if x['data_type'] == 'adversarial')
        return {
            'total': len(self.buffer),
            'organic': organic,
            'original_fraud': original_fraud,
            'adversarial': adversarial,
            'capacity': self.capacity,
        }


# =============================================================================
# ORGANIC DATA GENERATION (Clean graphs - no fraud)
# =============================================================================

def generate_organic_data(
    base_graph,
    num_graphs: int = 30,
) -> Tuple[List, List[str], Dict]:
    """
    Generate CLEAN organic graphs with NO fraud patterns.
    
    These represent normal, legitimate transaction behavior.
    All nodes have label=0 (not suspicious).
    
    This is CRITICAL to prevent the GNN from being "too paranoid"
    and flagging everything as suspicious.
    """
    print(f"\n[Generating ORGANIC (clean) data: {num_graphs} graphs]")
    print("  No fraud patterns - all nodes are legitimate (label=0)")
    
    graphs = []
    all_nodes = []
    
    base_nodes = list(base_graph.nodes())
    
    for i in range(num_graphs):
        # Copy base graph with no modifications
        G = base_graph.copy()
        for node in base_graph.nodes():
            G.nodes[node].update(dict(base_graph.nodes[node]))
            # Ensure all nodes are labeled as NOT suspicious
            G.nodes[node]['label'] = 0
        
        # Add some normal transaction edges (not suspicious patterns)
        num_normal_edges = random.randint(20, 50)
        for _ in range(num_normal_edges):
            src = random.choice(base_nodes)
            tgt = random.choice([n for n in base_nodes if n != src])
            # Normal transactions: small to medium amounts, random timing
            G.add_edge(src, tgt, 
                      amount=random.uniform(100, 2000),  # Normal amounts
                      timestamp=int(time.time()) - random.randint(0, 86400 * 90),
                      type='organic')
        
        graphs.append(G)
        all_nodes.extend(base_nodes)
        
        if (i + 1) % 10 == 0:
            print(f"  Graph {i+1}/{num_graphs}: {len(base_nodes)} clean nodes")
    
    stats = {
        'num_graphs': len(graphs),
        'total_nodes': len(all_nodes),
        'suspicious_nodes': 0,  # None - all clean
    }
    
    return graphs, [], stats  # Empty suspicious list


# =============================================================================
# ORIGINAL FRAUD PATTERN GENERATION (Fan-In: mule hubs, smurfing)
# =============================================================================

def generate_baseline_data(
    base_graph,
    pattern_lib: PatternLibrary,
    num_graphs: int = 20,
) -> Tuple[List, List[str], Dict]:
    """
    Generate baseline training data with BALANCED pattern types.
    
    CRITICAL: Must include both Fan-In (mule hubs) AND Fan-Out (peel chains)
    to prevent the GNN from learning directional bias.
    
    Pattern Balance Strategy:
    - 50% of graphs: Fan-In heavy (mule hubs, smurfing destinations)
    - 50% of graphs: Fan-Out heavy (peel chains, layering)
    - All graphs: Include both directions for diversity
    """
    print(f"\n[Generating BALANCED baseline data: {num_graphs} graphs]")
    print("  Ensuring both Fan-In (mule hubs) AND Fan-Out (peel chains)")
    
    graphs = []
    all_suspicious_nodes = []
    pattern_counts = {
        'fan_in_hub': 0,      # Many -> One (CRITICAL for mule detection)
        'smurfing': 0,         # Structured deposits (Fan-In variant)
        'peel_chain': 0,       # One -> Many linear
        'forked_peel': 0,      # One -> Many branching
        'funnel': 0,           # Fan-In + Fan-Out
        'mule': 0,             # Pass-through
        'cycle': 0,
    }
    
    base_nodes = list(base_graph.nodes())
    
    for i in range(num_graphs):
        # Copy base graph
        G = base_graph.copy()
        for node in base_graph.nodes():
            G.nodes[node].update(dict(base_graph.nodes[node]))
        
        suspicious = []
        
        # GUARANTEED: Every graph gets Fan-In pattern (mule hub detection)
        # This prevents GNN from learning "fraud = high out-degree only"
        try:
            # Create explicit Fan-In mule hub (Many -> One)
            hub_id = f"MuleHub_{i}_{random.randint(1000, 9999)}"
            G.add_node(hub_id, ntype='account', label=1, pattern='fan_in_hub')
            
            # 8-15 sources sending TO this hub
            num_sources = random.randint(8, 15)
            sources = random.sample(base_nodes, k=min(num_sources, len(base_nodes)))
            
            for src in sources:
                amt = random.uniform(5000, 15000)  # Substantial amounts
                G.add_edge(src, hub_id, 
                          amount=amt, 
                          timestamp=int(time.time()) - random.randint(0, 86400*30),
                          type='fan_in_deposit')
            
            suspicious.append(hub_id)
            pattern_counts['fan_in_hub'] += 1
            
            # Mark the hub node explicitly
            G.nodes[hub_id]['in_degree_suspicious'] = True
            
        except Exception as e:
            pass
        
        # GUARANTEED: Add smurfing (structured Fan-In)
        try:
            target = random.choice(base_nodes)
            smurfs = pattern_lib.inject_smurfing(G, target, num_smurfs=random.randint(6, 12))
            suspicious.extend(smurfs)
            suspicious.append(target)  # Target is also suspicious
            G.nodes[target]['label'] = 1
            pattern_counts['smurfing'] += 1
        except Exception:
            pass
        
        # Alternate between Fan-Out heavy and Fan-In heavy graphs
        if i % 2 == 0:
            # Even graphs: Add more Fan-Out patterns (adversarial-like)
            try:
                result = pattern_lib.inject_simple_peel_chain(G, base_nodes, length=random.randint(8, 12))
                if isinstance(result, list):
                    suspicious.extend(result)
                else:
                    suspicious.extend(result.get('chain', []))
                pattern_counts['peel_chain'] += 1
            except Exception:
                pass
        else:
            # Odd graphs: Add more Fan-In patterns (traditional fraud)
            try:
                # Second mule hub
                hub2 = f"MuleHub2_{i}_{random.randint(1000, 9999)}"
                G.add_node(hub2, ntype='account', label=1, pattern='fan_in_hub')
                sources2 = random.sample(base_nodes, k=min(10, len(base_nodes)))
                for src in sources2:
                    G.add_edge(src, hub2, amount=random.uniform(3000, 12000), 
                              timestamp=int(time.time()) - random.randint(0, 86400*30))
                suspicious.append(hub2)
                pattern_counts['fan_in_hub'] += 1
            except Exception:
                pass
        
        # Always add funnel (balanced: Fan-In + Fan-Out through hub)
        try:
            funnel_hub = f"Funnel_{i}_{random.randint(1000, 9999)}"
            G.add_node(funnel_hub, ntype='account', label=1, pattern='funnel')
            fan_in = random.sample(base_nodes, k=min(6, len(base_nodes)))
            fan_out = random.sample(base_nodes, k=min(4, len(base_nodes)))
            pattern_lib.inject_funnel_pattern(G, funnel_hub, fan_in, fan_out, total_amount=50000)
            suspicious.append(funnel_hub)
            pattern_counts['funnel'] += 1
        except Exception:
            pass
        
        # Mule account (pass-through = Fan-In + Fan-Out)
        try:
            mule = f"Mule_{i}_{random.randint(1000, 9999)}"
            G.add_node(mule, ntype='account', label=1, pattern='mule')
            sources = random.sample(base_nodes, k=min(5, len(base_nodes)))
            targets = random.sample(base_nodes, k=min(3, len(base_nodes)))
            pattern_lib.inject_mule_account(G, mule, sources, targets, pass_through_amount=30000)
            suspicious.append(mule)
            pattern_counts['mule'] += 1
        except Exception:
            pass
        
        graphs.append(G)
        all_suspicious_nodes.extend(suspicious)
        
        if (i + 1) % 10 == 0:
            print(f"  Graph {i+1}/{num_graphs}: {len(suspicious)} suspicious nodes")
    
    # Final balance check
    total_fan_in = pattern_counts['fan_in_hub'] + pattern_counts['smurfing'] + pattern_counts['funnel']
    total_fan_out = pattern_counts['peel_chain'] + pattern_counts['forked_peel']
    
    print(f"  Pattern Balance: Fan-In heavy={total_fan_in}, Fan-Out heavy={total_fan_out}")
    
    stats = {
        'num_graphs': len(graphs),
        'total_suspicious_nodes': len(all_suspicious_nodes),
        'unique_suspicious_nodes': len(set(all_suspicious_nodes)),
        'pattern_counts': pattern_counts,
        'fan_in_patterns': total_fan_in,
        'fan_out_patterns': total_fan_out,
    }
    
    return graphs, list(set(all_suspicious_nodes)), stats


# =============================================================================
# ADVERSARIAL DATA GENERATION
# =============================================================================

def generate_adversarial_data(
    agent: AdversarialPatternAgent,
    base_graph,
    pattern_lib: PatternLibrary,
    num_episodes: int = 50,
    steps_per_episode: int = 10,
    device: str = 'cpu'
) -> Tuple[List, List[str], Dict]:
    """
    Use trained adversarial agent to generate challenging patterns.
    """
    print(f"\n[Generating adversarial data: {num_episodes} episodes x {steps_per_episode} patterns]")
    
    graphs = []
    all_suspicious_nodes = []
    pattern_counts = {i: 0 for i in range(5)}
    
    base_nodes = list(base_graph.nodes())
    
    for ep in range(num_episodes):
        G = base_graph.copy()
        for node in base_graph.nodes():
            G.nodes[node].update(dict(base_graph.nodes[node]))
        
        episode_suspicious = []
        state = np.zeros(49, dtype=np.float32)
        
        for step in range(steps_per_episode):
            action_dict, _ = agent.select_action(state)
            
            try:
                result = pattern_lib.inject_pattern_from_action(G, base_nodes, action_dict)
                injected = result.get('injected_nodes', [])
                episode_suspicious.extend(injected)
                pattern_counts[action_dict['pattern_type']] += 1
            except Exception:
                pass
        
        graphs.append(G)
        all_suspicious_nodes.extend(episode_suspicious)
        
        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}/{num_episodes}: {len(episode_suspicious)} suspicious nodes")
    
    stats = {
        'num_episodes': num_episodes,
        'total_graphs': len(graphs),
        'total_suspicious_nodes': len(all_suspicious_nodes),
        'unique_suspicious_nodes': len(set(all_suspicious_nodes)),
        'pattern_distribution': pattern_counts,
    }
    
    return graphs, list(set(all_suspicious_nodes)), stats


# =============================================================================
# MIXED TRAINING
# =============================================================================

def retrain_gnn_mixed(
    gnn_model: GraphSage,
    replay_buffer: ExperienceReplayBuffer,
    epochs: int = 30,
    batch_size: int = 10,
    lr: float = 1e-3,
    mix_ratio: float = 0.5,
    device: str = 'cpu'
) -> Tuple[GraphSage, Dict]:
    """
    Retrain GNN with mixed original + adversarial data (Experience Replay).
    
    Args:
        gnn_model: Current GNN model
        replay_buffer: Buffer containing both original and adversarial graphs
        epochs: Training epochs
        batch_size: Graphs per batch
        lr: Learning rate
        mix_ratio: Fraction of adversarial samples (0.5 = balanced)
        device: torch device
        
    Returns:
        retrained_model: Updated GNN
        training_history: Training metrics
    """
    buffer_stats = replay_buffer.stats()
    print(f"\n[3-Way Mixed Training with Experience Replay: {epochs} epochs]")
    print(f"  Buffer: {buffer_stats['organic']} organic + {buffer_stats['original_fraud']} fraud + {buffer_stats['adversarial']} adversarial")
    print(f"  Target mix: 60% Organic / 20% Original Fraud / 20% Adversarial")
    
    # Pre-process all graphs in buffer
    processed_data = []
    
    for item in replay_buffer.get_all():
        try:
            G = item['graph']
            suspicious_set = item['suspicious_nodes']
            data_type = item['data_type']
            
            df_tx = graph_to_tx_df(G)
            features, node_index, raw_feats = build_node_features(G, df_tx)
            adj = adjacency_sparse_from_nx(G, node_index)
            
            labels = np.zeros(len(node_index), dtype=np.float32)
            for node, idx in node_index.items():
                if node in suspicious_set:
                    labels[idx] = 1.0
            
            processed_data.append({
                'features': torch.FloatTensor(features),
                'labels': torch.FloatTensor(labels),
                'adj': adj,
                'data_type': data_type,
            })
        except Exception:
            continue
    
    if len(processed_data) == 0:
        print("  Warning: No valid training data")
        return gnn_model, {'error': 'No valid data'}
    
    organic_data = [d for d in processed_data if d['data_type'] == 'organic']
    original_fraud_data = [d for d in processed_data if d['data_type'] == 'original_fraud']
    adversarial_data = [d for d in processed_data if d['data_type'] == 'adversarial']
    
    print(f"  Processed: {len(organic_data)} organic + {len(original_fraud_data)} fraud + {len(adversarial_data)} adversarial graphs")
    
    # Training loop
    gnn_model = gnn_model.to(device)
    gnn_model.train()
    
    optimizer = optim.Adam(gnn_model.parameters(), lr=lr)
    
    history = {
        'loss': [], 
        'accuracy': [],
        'organic_acc': [],
        'fraud_acc': [],
        'adversarial_acc': [],
    }
    
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        organic_correct, organic_total = 0, 0
        fraud_correct, fraud_total = 0, 0
        adv_correct, adv_total = 0, 0
        
        # Sample 3-way mixed batch for this epoch (60/20/20)
        n_organic = int(batch_size * 0.6)  # 60%
        n_fraud = int(batch_size * 0.2)     # 20%
        n_adv = batch_size - n_organic - n_fraud  # 20%
        
        batch = []
        if organic_data:
            batch.extend(random.choices(organic_data, k=min(n_organic, len(organic_data) * 2)))
        if original_fraud_data:
            batch.extend(random.choices(original_fraud_data, k=min(n_fraud, len(original_fraud_data) * 2)))
        if adversarial_data:
            batch.extend(random.choices(adversarial_data, k=min(n_adv, len(adversarial_data) * 2)))
        
        random.shuffle(batch)
        
        for item in batch:
            features = item['features'].to(device)
            labels = item['labels'].to(device)
            adj = item['adj'].to(device)
            data_type = item['data_type']
            
            optimizer.zero_grad()
            
            probs = gnn_model(features, adj)
            
            # Weighted BCE
            pos_weight = (labels == 0).sum() / max(1, (labels == 1).sum())
            pos_weight = min(pos_weight, 10.0)
            
            loss = nn.functional.binary_cross_entropy(
                probs, labels,
                weight=labels * pos_weight + (1 - labels)
            )
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Metrics
            preds = (probs > 0.5).float()
            correct = (preds == labels).sum().item()
            total = len(labels)
            
            epoch_correct += correct
            epoch_total += total
            
            if data_type == 'organic':
                organic_correct += correct
                organic_total += total
            elif data_type == 'original_fraud':
                fraud_correct += correct
                fraud_total += total
            else:  # adversarial
                adv_correct += correct
                adv_total += total
        
        if len(batch) > 0:
            avg_loss = epoch_loss / len(batch)
            accuracy = epoch_correct / max(1, epoch_total)
            org_acc = organic_correct / max(1, organic_total)
            fraud_acc = fraud_correct / max(1, fraud_total)
            adv_acc = adv_correct / max(1, adv_total)
            
            history['loss'].append(avg_loss)
            history['accuracy'].append(accuracy)
            history['organic_acc'].append(org_acc)
            history['fraud_acc'].append(fraud_acc)
            history['adversarial_acc'].append(adv_acc)
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, "
                      f"Acc={accuracy:.3f} (Org={org_acc:.3f}, Fraud={fraud_acc:.3f}, Adv={adv_acc:.3f})")
    
    gnn_model.eval()
    
    return gnn_model, history


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_gnn_comprehensive(
    gnn_model: GraphSage,
    agent: AdversarialPatternAgent,
    base_graph,
    pattern_lib: PatternLibrary,
    num_episodes: int = 10,
    device: str = 'cpu'
) -> Dict:
    """
    Evaluate GNN on BOTH adversarial AND original patterns.
    
    Returns separate detection rates for each to track catastrophic forgetting.
    """
    print(f"\n[Comprehensive GNN Evaluation]")
    
    gnn_model.eval()
    gnn_model.to(device)
    
    base_nodes = list(base_graph.nodes())
    
    # Test on adversarial patterns
    adv_detection_rates = []
    for ep in range(num_episodes):
        G = base_graph.copy()
        for node in base_graph.nodes():
            G.nodes[node].update(dict(base_graph.nodes[node]))
        
        state = np.zeros(49, dtype=np.float32)
        injected = []
        
        for step in range(5):
            action_dict, _ = agent.select_action(state)
            try:
                result = pattern_lib.inject_pattern_from_action(G, base_nodes, action_dict)
                injected.extend(result.get('injected_nodes', []))
            except:
                pass
        
        if injected:
            try:
                df_tx = graph_to_tx_df(G)
                features, node_index, _ = build_node_features(G, df_tx)
                adj = adjacency_sparse_from_nx(G, node_index)
                
                with torch.no_grad():
                    features_t = torch.FloatTensor(features).to(device)
                    probs = gnn_model(features_t, adj.to(device)).cpu().numpy()
                
                indices = [node_index[n] for n in injected if n in node_index]
                if indices:
                    adv_detection_rates.append(probs[indices].mean())
            except:
                pass
    
    # Test on original patterns (simple patterns)
    orig_detection_rates = []
    for ep in range(num_episodes):
        G = base_graph.copy()
        for node in base_graph.nodes():
            G.nodes[node].update(dict(base_graph.nodes[node]))
        
        injected = []
        
        # Inject simple patterns
        try:
            # Fan-in pattern
            hub = random.choice(base_nodes)
            spokes = random.sample([n for n in base_nodes if n != hub], k=min(8, len(base_nodes)-1))
            for spoke in spokes:
                G.add_edge(spoke, hub, amount=random.uniform(5000, 15000), timestamp=int(time.time()))
                G.nodes[hub]['label'] = 1
            injected.append(hub)
            
            # Simple chain
            chain_result = pattern_lib.inject_simple_peel_chain(G, base_nodes, length=10)
            if isinstance(chain_result, list):
                injected.extend(chain_result)
            else:
                injected.extend(chain_result.get('chain', []))
        except:
            pass
        
        if injected:
            try:
                df_tx = graph_to_tx_df(G)
                features, node_index, _ = build_node_features(G, df_tx)
                adj = adjacency_sparse_from_nx(G, node_index)
                
                with torch.no_grad():
                    features_t = torch.FloatTensor(features).to(device)
                    probs = gnn_model(features_t, adj.to(device)).cpu().numpy()
                
                indices = [node_index[n] for n in injected if n in node_index]
                if indices:
                    orig_detection_rates.append(probs[indices].mean())
            except:
                pass
    
    mean_adv = float(np.mean(adv_detection_rates)) if adv_detection_rates else 0.5
    mean_orig = float(np.mean(orig_detection_rates)) if orig_detection_rates else 0.5
    
    print(f"  Adversarial detection: {mean_adv:.3f}")
    print(f"  Original detection: {mean_orig:.3f}")
    
    return {
        'adversarial_detection': mean_adv,
        'original_detection': mean_orig,
        'mean_detection': (mean_adv + mean_orig) / 2,
        'forgetting_gap': mean_orig - mean_adv,  # Positive = good, Negative = forgetting
    }


# =============================================================================
# MAIN ADVERSARIAL LOOP WITH EXPERIENCE REPLAY
# =============================================================================

def adversarial_retrain_loop(
    num_rounds: int = 3,
    episodes_per_round: int = 50,
    gnn_epochs: int = 30,
    mix_ratio: float = 0.5,
    replay_capacity: int = 100,
    agent_path: str = 'models/adversarial_agent.pt',
    gnn_save_path: str = 'models/gnn_adversarial.pt',
    device: str = 'cpu',
    seed: int = 42
) -> Dict:
    """
    Full adversarial retraining loop with Experience Replay.
    
    Prevents catastrophic forgetting by mixing original + adversarial data.
    """
    print("=" * 60)
    print("ADVERSARIAL GNN RETRAINING WITH 3-WAY EXPERIENCE REPLAY")
    print("=" * 60)
    print(f"Rounds: {num_rounds}")
    print(f"Episodes per round: {episodes_per_round}")
    print(f"GNN epochs per round: {gnn_epochs}")
    print(f"Mix ratio: 60% Organic / 20% Original Fraud / 20% Adversarial")
    print(f"Replay buffer capacity: {replay_capacity}")
    print(f"Agent: {agent_path}")
    print(f"Device: {device}")
    print()
    
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    # Initialize experience replay buffer
    replay_buffer = ExperienceReplayBuffer(capacity=replay_capacity)
    
    # Load adversarial agent
    print("[Loading adversarial agent...]")
    agent = AdversarialPatternAgent(state_dim=49, hidden_dim=128, device=device)
    if os.path.exists(agent_path):
        agent.load(agent_path)
        print(f"  Loaded from {agent_path}")
    else:
        print(f"  Warning: No saved agent found, using random agent")
    agent.eval()
    
    # Generate base graph
    print("\n[Generating base graph...]")
    base_graph, labels, profiles = generate_synthetic_graph(
        num_nodes=500,
        mule_fraction=0.02,
        seed=seed
    )
    print(f"  Created graph with {base_graph.number_of_nodes()} nodes")
    
    # Initialize GNN
    print("\n[Initializing GNN...]")
    gnn_model = GraphSage(in_feats=12, hidden=64, out_feats=32)
    gnn_model.to(device)
    
    # Pattern library
    pattern_lib = PatternLibrary(seed=seed)
    
    # STEP 1: Generate ORGANIC data (clean - no fraud)
    print("\n[1/3 Populating buffer: ORGANIC data (60% of training)]")
    organic_graphs, _, organic_stats = generate_organic_data(
        base_graph, num_graphs=30
    )
    for G in organic_graphs:
        replay_buffer.add(G, [], 'organic')  # No suspicious nodes
    print(f"  Added {len(organic_graphs)} organic (clean) graphs")
    
    # STEP 2: Generate ORIGINAL FRAUD data (Fan-In patterns - legacy)
    print("\n[2/3 Populating buffer: ORIGINAL FRAUD data (20% of training)]")
    baseline_graphs, baseline_suspicious, baseline_stats = generate_baseline_data(
        base_graph, pattern_lib, num_graphs=15
    )
    for G in baseline_graphs:
        replay_buffer.add(G, baseline_suspicious, 'original_fraud')
    print(f"  Added {len(baseline_graphs)} original_fraud (mule hub) graphs")
    
    # Track metrics across rounds
    round_metrics = []
    
    for round_num in range(1, num_rounds + 1):
        print(f"\n{'='*60}")
        print(f"ROUND {round_num}/{num_rounds}")
        print(f"{'='*60}")
        
        round_start = time.time()
        
        # Step 1: Generate adversarial data
        graphs, suspicious_nodes, gen_stats = generate_adversarial_data(
            agent=agent,
            base_graph=base_graph,
            pattern_lib=pattern_lib,
            num_episodes=episodes_per_round,
            steps_per_episode=10,
            device=device
        )
        
        # Add to replay buffer
        for G in graphs:
            replay_buffer.add(G, suspicious_nodes, 'adversarial')
        
        print(f"  Generated {gen_stats['unique_suspicious_nodes']} unique suspicious nodes")
        print(f"  Pattern distribution: {gen_stats['pattern_distribution']}")
        print(f"  Buffer: {replay_buffer.stats()}")
        
        # Step 2: Mixed training with experience replay
        gnn_model, train_history = retrain_gnn_mixed(
            gnn_model=gnn_model,
            replay_buffer=replay_buffer,
            epochs=gnn_epochs,
            batch_size=15,
            lr=1e-3,
            mix_ratio=mix_ratio,
            device=device
        )
        
        # Step 3: Comprehensive evaluation (both pattern types)
        eval_results = evaluate_gnn_comprehensive(
            gnn_model=gnn_model,
            agent=agent,
            base_graph=base_graph,
            pattern_lib=pattern_lib,
            num_episodes=10,
            device=device
        )
        
        round_time = time.time() - round_start
        
        round_metrics.append({
            'round': round_num,
            'generation_stats': gen_stats,
            'buffer_stats': replay_buffer.stats(),
            'final_train_loss': train_history['loss'][-1] if train_history.get('loss') else None,
            'final_train_acc': train_history['accuracy'][-1] if train_history.get('accuracy') else None,
            'original_acc': train_history['original_acc'][-1] if train_history.get('original_acc') else None,
            'adversarial_acc': train_history['adversarial_acc'][-1] if train_history.get('adversarial_acc') else None,
            'eval_adversarial': eval_results['adversarial_detection'],
            'eval_original': eval_results['original_detection'],
            'forgetting_gap': eval_results['forgetting_gap'],
            'time_seconds': round_time,
        })
        
        print(f"\n  Round {round_num} Summary:")
        print(f"    Train Loss: {round_metrics[-1]['final_train_loss']:.4f}")
        print(f"    Train Acc: {round_metrics[-1]['final_train_acc']:.3f}")
        print(f"    Eval (Adversarial): {eval_results['adversarial_detection']:.3f}")
        print(f"    Eval (Original): {eval_results['original_detection']:.3f}")
        print(f"    Forgetting Gap: {eval_results['forgetting_gap']:+.3f}")
        print(f"    Time: {round_time:.1f}s")
    
    # Save retrained GNN
    print(f"\n[Saving retrained GNN to {gnn_save_path}...]")
    os.makedirs(os.path.dirname(gnn_save_path) or '.', exist_ok=True)
    torch.save(gnn_model.state_dict(), gnn_save_path)
    
    # Save metadata
    metadata = {
        'num_rounds': num_rounds,
        'episodes_per_round': episodes_per_round,
        'gnn_epochs': gnn_epochs,
        'mix_ratio': mix_ratio,
        'replay_capacity': replay_capacity,
        'agent_path': agent_path,
        'round_metrics': round_metrics,
        'final_adversarial_detection': round_metrics[-1]['eval_adversarial'],
        'final_original_detection': round_metrics[-1]['eval_original'],
        'final_forgetting_gap': round_metrics[-1]['forgetting_gap'],
        'trained_at': datetime.now().isoformat(),
    }
    
    metadata_path = gnn_save_path.replace('.pt', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Summary
    print("\n" + "=" * 60)
    print("ADVERSARIAL RETRAINING COMPLETE (WITH EXPERIENCE REPLAY)")
    print("=" * 60)
    
    print("\nDetection Rate Evolution:")
    print(f"{'Round':<6} {'Adversarial':<12} {'Original':<12} {'Forgetting Gap':<15}")
    print("-" * 45)
    for m in round_metrics:
        print(f"{m['round']:<6} {m['eval_adversarial']:<12.3f} {m['eval_original']:<12.3f} {m['forgetting_gap']:+.3f}")
    
    initial_orig = round_metrics[0]['eval_original'] if round_metrics else 0.5
    final_orig = round_metrics[-1]['eval_original'] if round_metrics else 0.5
    forgetting = initial_orig - final_orig
    
    print(f"\nOriginal Pattern Performance: {initial_orig:.3f} → {final_orig:.3f} (Δ={forgetting:+.3f})")
    if forgetting > 0.05:
        print("  ⚠️  Some forgetting detected on original patterns")
    else:
        print("  ✅ Minimal forgetting - experience replay working!")
    
    print(f"\nSaved: {gnn_save_path}")
    print(f"Metadata: {metadata_path}")
    
    return {
        'round_metrics': round_metrics,
        'final_adversarial_detection': metadata['final_adversarial_detection'],
        'final_original_detection': metadata['final_original_detection'],
        'forgetting_gap': metadata['final_forgetting_gap'],
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Adversarial GNN Retraining with Experience Replay',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python adversarial_retrain.py                    # Default retraining
  python adversarial_retrain.py --rounds 5         # 5 rounds
  python adversarial_retrain.py --mix-ratio 0.6    # 60% adversarial data
  python adversarial_retrain.py --device cuda      # Use GPU
        """
    )
    
    parser.add_argument('--rounds', type=int, default=3,
                        help='Number of adversarial training rounds')
    parser.add_argument('--episodes', type=int, default=50,
                        help='Episodes per round for data generation')
    parser.add_argument('--gnn-epochs', type=int, default=30,
                        help='GNN training epochs per round')
    parser.add_argument('--mix-ratio', type=float, default=0.5,
                        help='Fraction of adversarial data in training (default: 0.5)')
    parser.add_argument('--replay-capacity', type=int, default=100,
                        help='Experience replay buffer capacity')
    parser.add_argument('--agent', type=str, default='models/adversarial_agent.pt',
                        help='Path to trained adversarial agent')
    parser.add_argument('--output', type=str, default='models/gnn_adversarial.pt',
                        help='Output path for retrained GNN')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device for training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    adversarial_retrain_loop(
        num_rounds=args.rounds,
        episodes_per_round=args.episodes,
        gnn_epochs=args.gnn_epochs,
        mix_ratio=args.mix_ratio,
        replay_capacity=args.replay_capacity,
        agent_path=args.agent,
        gnn_save_path=args.output,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
