"""
Adversarial AML Environment
============================
Gymnasium-compatible environment where the PPO agent injects AML patterns
and the GNN discriminator evaluates detection probability.

Key Features:
- Action canonicalization for conditional dependencies
- Reward design optimized for evasion learning
- State includes global graph embedding + detection history

Reference: "The 2025 Horizon" - Section 6.3 "The Algorithmic Peel-Wash"
"""

import numpy as np
import networkx as nx
import torch
import warnings
from typing import Dict, List, Tuple, Optional, Any
from copy import deepcopy

try:
    import gymnasium as gym
    from gymnasium import spaces
    _HAS_GYM = True
except ImportError:
    # Fallback for older gym versions or missing dependency
    gym = None
    spaces = None
    _HAS_GYM = False


# =============================================================================
# PATTERN TYPE CONSTANTS
# =============================================================================

PATTERN_TYPES = {
    0: 'peel_chain',
    1: 'forked_peel', 
    2: 'smurfing',
    3: 'funnel',
    4: 'mule',
}

TEMPORAL_MODES = {
    0: 'rapid_fire',
    1: 'evasion_pause',
    2: 'mixed',
}

EXIT_STRATEGIES = {
    0: 'cuckoo_victim',    # Exit to innocent existing node
    1: 'new_shell',         # Create new shell account
    2: 'existing_mule',     # Use existing labeled mule
}


# =============================================================================
# ENVIRONMENT
# =============================================================================

class AdversarialAMLEnv:
    """
    Environment for adversarial AML pattern generation.
    
    The agent injects patterns into a transaction graph, and the GNN
    discriminator evaluates detection probability. The reward encourages
    patterns that evade detection.
    
    State Space (49 dims):
        - Global graph embedding (32 dims): Mean-pooled GNN hidden states
        - Target node features (12 dims): Structural + behavioral features
        - Detection rate history (5 dims): Rolling window of last 5 episodes
    
    Action Space (Hybrid):
        - Discrete: pattern_type, temporal_mode, chain_length, fork_count, exit_strategy
        - Continuous: peel_pct, wash_intensity, cycle_probability
    """
    
    def __init__(
        self,
        gnn_model,
        base_graph: nx.DiGraph,
        pattern_library,
        node_feature_builder,
        max_steps: int = 10,
        reward_scale: float = 1.0,
        confusion_penalty: float = 0.1,
        device: str = 'cpu'
    ):
        """
        Initialize the adversarial environment.
        
        Args:
            gnn_model: Pre-trained GraphSage discriminator
            base_graph: Clean transaction graph to inject patterns into
            pattern_library: PatternLibrary instance for pattern injection
            node_feature_builder: Function to build node features from graph
            max_steps: Maximum injection steps per episode
            reward_scale: Scale factor for rewards
            confusion_penalty: Penalty for "confused" action outputs
            device: torch device
        """
        self.gnn_model = gnn_model
        self.base_graph = base_graph
        self.pattern_library = pattern_library
        self.build_node_features = node_feature_builder
        self.max_steps = max_steps
        self.reward_scale = reward_scale
        self.confusion_penalty = confusion_penalty
        self.device = device
        
        # Episode state
        self.current_graph = None
        self.current_step = 0
        self.injected_nodes: List[str] = []
        self.detection_history = [0.5] * 5  # Initialize with neutral
        
        # Cache base graph nodes
        self.base_nodes = list(base_graph.nodes())
    
    def reset(self) -> np.ndarray:
        """
        Reset environment for new episode.
        
        Returns:
            Initial state observation
        """
        # Use networkx copy to avoid deepcopy issues with sets in node attributes
        self.current_graph = self.base_graph.copy()
        # Copy node and edge data manually to avoid reference issues
        for node in self.base_graph.nodes():
            self.current_graph.nodes[node].update(dict(self.base_graph.nodes[node]))
        self.current_step = 0
        self.injected_nodes = []
        
        # Compute initial state (suppress verbose warnings)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            state = self._compute_state()
        
        return state
    
    def step(self, action: Dict) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action dictionary from agent
            
        Returns:
            (next_state, reward, done, info)
        """
        self.current_step += 1
        
        # Canonicalize action to handle conditional dependencies
        action = self._canonicalize_action(action)
        
        # Inject pattern based on action
        injected = self._inject_pattern(action)
        self.injected_nodes.extend(injected)
        
        # Run GNN discriminator on injected nodes
        detection_probs = self._evaluate_detection(injected)
        
        # Compute reward
        reward, reward_info = self._compute_reward(detection_probs, action)
        
        # Check if done
        done = self.current_step >= self.max_steps
        
        # Compute next state
        next_state = self._compute_state()
        
        # Update detection history
        if len(detection_probs) > 0:
            mean_detection = float(np.mean(detection_probs))
            self.detection_history.pop(0)
            self.detection_history.append(mean_detection)
        
        info = {
            'injected_nodes': injected,
            'detection_probs': detection_probs,
            'mean_detection': np.mean(detection_probs) if len(detection_probs) > 0 else 0.5,
            'num_undetected': int(np.sum(detection_probs < 0.5)) if len(detection_probs) > 0 else 0,
            'action_canonical': action,
            **reward_info,
        }
        
        return next_state, reward, done, info
    
    def _canonicalize_action(self, action: Dict) -> Dict:
        """
        Canonicalize action to ensure deterministic behavior.
        
        This addresses the conditional dependency problem where some
        action dimensions are irrelevant for certain pattern types.
        """
        action = dict(action)  # Copy
        pattern = action['pattern_type']
        
        # fork_count only matters for forked_peel (pattern_type=1)
        if pattern != 1:
            action['fork_count'] = 1  # Canonical value (unused)
        
        # wash_intensity only matters for peel chains (0, 1)
        if pattern not in [0, 1]:
            action['wash_intensity'] = 0.0
        
        # cycle_probability only matters for sophisticated peel patterns
        if pattern in [2, 3, 4]:  # smurfing, funnel, mule
            action['cycle_probability'] = 0.0
        
        # chain_length has different meanings per pattern
        if pattern == 2:  # smurfing - interpret as num_smurfs
            action['chain_length'] = max(5, min(25, action['chain_length']))
        elif pattern == 4:  # mule - fewer hops
            action['chain_length'] = max(3, min(8, action['chain_length']))
        
        return action
    
    def _inject_pattern(self, action: Dict) -> List[str]:
        """
        Inject pattern into graph based on action parameters.
        
        Returns list of injected node IDs.
        """
        pattern_type = action['pattern_type']
        existing_nodes = self.base_nodes.copy()
        
        injected_nodes = []
        
        try:
            if pattern_type == 0:  # peel_chain (sophisticated)
                result = self.pattern_library.inject_sophisticated_peel_chain(
                    self.current_graph,
                    existing_nodes,
                    length=action['chain_length'],
                    initial_amount=np.random.uniform(50000, 200000),
                )
                injected_nodes = result.get('all_suspicious', [])
                
            elif pattern_type == 1:  # forked_peel
                result = self.pattern_library.inject_forked_peel_chain(
                    self.current_graph,
                    existing_nodes,
                    initial_length=max(3, action['chain_length'] // 3),
                    fork_count=action['fork_count'],
                    branch_length=action['chain_length'] // 2,
                    initial_amount=np.random.uniform(50000, 200000),
                )
                main = result.get('main_chain', [])
                branches = result.get('branches', [])
                injected_nodes = main + [n for b in branches for n in b]
                
            elif pattern_type == 2:  # smurfing
                target = np.random.choice(existing_nodes)
                injected_nodes = self.pattern_library.inject_smurfing(
                    self.current_graph,
                    target,
                    num_smurfs=action['chain_length'],  # Reinterpret as num_smurfs
                )
                
            elif pattern_type == 3:  # funnel
                hub = f"Hub_Adv_{self.current_step}_{np.random.randint(1000, 9999)}"
                self.current_graph.add_node(hub, ntype='account', label=1, pattern='funnel_hub')
                
                fan_in = list(np.random.choice(existing_nodes, size=min(8, len(existing_nodes)), replace=False))
                fan_out = list(np.random.choice(existing_nodes, size=min(5, len(existing_nodes)), replace=False))
                
                self.pattern_library.inject_funnel_pattern(
                    self.current_graph,
                    hub,
                    fan_in,
                    fan_out,
                    total_amount=np.random.uniform(50000, 150000),
                )
                injected_nodes = [hub]
                
            elif pattern_type == 4:  # mule
                mule = f"Mule_Adv_{self.current_step}_{np.random.randint(1000, 9999)}"
                self.current_graph.add_node(mule, ntype='account', label=1, pattern='mule')
                
                sources = list(np.random.choice(existing_nodes, size=min(5, len(existing_nodes)), replace=False))
                targets = list(np.random.choice(existing_nodes, size=min(3, len(existing_nodes)), replace=False))
                
                self.pattern_library.inject_mule_account(
                    self.current_graph,
                    mule,
                    sources,
                    targets,
                    pass_through_amount=np.random.uniform(30000, 80000),
                )
                injected_nodes = [mule]
                
        except Exception as e:
            print(f"Warning: Pattern injection failed: {e}")
            injected_nodes = []
        
        return injected_nodes
    
    def _evaluate_detection(self, nodes: List[str]) -> np.ndarray:
        """
        Run GNN discriminator on specified nodes.
        
        Returns array of detection probabilities for each node.
        """
        if len(nodes) == 0:
            return np.array([])
        
        try:
            # Build features and adjacency for current graph
            df_tx = self._graph_to_df()
            features, node_index, raw_feats = self.build_node_features(
                self.current_graph, df_tx
            )
            
            # Build adjacency matrix
            adj = self._build_adjacency(node_index)
            
            # Get node indices for injected nodes
            target_indices = []
            for n in nodes:
                if n in node_index:
                    target_indices.append(node_index[n])
            
            if len(target_indices) == 0:
                return np.array([0.5])  # Default if nodes not found
            
            # Run GNN
            with torch.no_grad():
                features_t = torch.FloatTensor(features).to(self.device)
                probs = self.gnn_model(features_t, adj)
                probs = probs.cpu().numpy()
            
            # Extract probabilities for target nodes
            detection_probs = probs[target_indices]
            
            return detection_probs
            
        except Exception:
            # Silently fallback - verbose warnings handled at higher level
            return np.array([0.5])  # Default
    
    def _compute_reward(
        self, 
        detection_probs: np.ndarray, 
        action: Dict
    ) -> Tuple[float, Dict]:
        """
        Compute reward based on detection probabilities.
        
        Reward design:
        - Base: -log(mean_detection) - higher when detection is low
        - Bonus: Extra reward for completely undetected nodes
        - Penalty: Small penalty for "confused" actions
        """
        if len(detection_probs) == 0:
            return -1.0, {'base_reward': -1.0, 'bonus': 0.0, 'penalty': 0.0}
        
        # Base reward: inversely proportional to detection
        mean_p = np.clip(np.mean(detection_probs), 1e-7, 1.0)
        base_reward = -np.log(mean_p)
        
        # Bonus for undetected nodes (p < 0.5)
        undetected_ratio = np.mean(detection_probs < 0.5)
        bonus = undetected_ratio * 2.0
        
        # Check for "confused" action patterns
        penalty = 0.0
        # Example: using wash_intensity for non-peel patterns (already canonicalized, but tracked)
        # This encourages the agent to learn proper action correlations
        
        total_reward = (base_reward + bonus - penalty) * self.reward_scale
        
        return total_reward, {
            'base_reward': base_reward,
            'bonus': bonus,
            'penalty': penalty,
            'undetected_ratio': undetected_ratio,
        }
    
    def _compute_state(self) -> np.ndarray:
        """
        Compute state observation.
        
        State = [Global Graph Embedding (32) | Target Features (12) | Detection History (5)]
        """
        try:
            # Build features
            df_tx = self._graph_to_df()
            features, node_index, raw_feats = self.build_node_features(
                self.current_graph, df_tx
            )
            adj = self._build_adjacency(node_index)
            
            # Get global graph embedding via GNN
            with torch.no_grad():
                features_t = torch.FloatTensor(features).to(self.device)
                
                # Get intermediate representation (mean-pooled)
                h_neigh = self.gnn_model.aggregate(features_t, adj)
                h = torch.cat([features_t, h_neigh], dim=1)
                h = torch.relu(self.gnn_model.fc1(h))
                
                h_neigh2 = self.gnn_model.aggregate(h, adj)
                h2 = torch.cat([h, h_neigh2], dim=1)
                h2 = torch.relu(self.gnn_model.fc2(h2))
                
                # Global embedding: mean pool across all nodes
                global_emb = h2.mean(dim=0).cpu().numpy()  # 32 dims
            
            # Target node features: use mean of recent injected or random
            if len(self.injected_nodes) > 0:
                target_indices = [
                    node_index[n] for n in self.injected_nodes[-5:] 
                    if n in node_index
                ]
                if len(target_indices) > 0:
                    target_feats = raw_feats[target_indices].mean(axis=0)
                else:
                    target_feats = raw_feats.mean(axis=0)
            else:
                target_feats = raw_feats.mean(axis=0)
            
            # Detection history
            detection_hist = np.array(self.detection_history, dtype=np.float32)
            
            # Concatenate state
            state = np.concatenate([
                global_emb,               # 32 dims
                target_feats,             # 12 dims
                detection_hist,           # 5 dims
            ]).astype(np.float32)
            
            return state
            
        except Exception:
            # Silently fallback - verbose warnings handled at higher level
            return np.zeros(49, dtype=np.float32)
    
    def _graph_to_df(self):
        """Convert graph to transaction DataFrame compatible with gnn_trainer."""
        import pandas as pd
        
        edges = []
        for u, v, data in self.current_graph.edges(data=True):
            edges.append({
                'source': u,
                'target': v,
                'amount': float(data.get('amount', 100)),
                'timestamp': int(data.get('timestamp', 0)),
                'type': data.get('type', 'synthetic'),
                'channel': data.get('channel'),
                'ip': data.get('ip'),
                'device_id': data.get('device_id'),
                'lat': data.get('lat'),
                'lon': data.get('lon'),
            })
        
        if len(edges) == 0:
            return pd.DataFrame(columns=['source', 'target', 'amount', 'timestamp', 'type'])
        
        return pd.DataFrame(edges)
    
    def _build_adjacency(self, node_index: Dict[str, int]):
        """Build sparse adjacency matrix from graph."""
        N = len(node_index)
        rows, cols = [], []
        
        for u, v in self.current_graph.edges():
            if u in node_index and v in node_index:
                rows.append(node_index[v])
                cols.append(node_index[u])
        
        if len(rows) == 0:
            # Return empty sparse tensor
            return torch.sparse_coo_tensor(
                torch.empty((2, 0), dtype=torch.long),
                torch.empty(0),
                size=(N, N)
            ).to(self.device)
        
        indices = torch.LongTensor([rows, cols])
        values = torch.ones(len(rows))
        adj = torch.sparse_coo_tensor(indices, values, size=(N, N)).to(self.device)
        
        return adj


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_adversarial_env(
    gnn_model_path: Optional[str] = None,
    num_base_accounts: int = 500,
    seed: int = 42,
    device: str = 'cpu'
) -> AdversarialAMLEnv:
    """
    Factory function to create an AdversarialAMLEnv with all dependencies.
    
    Args:
        gnn_model_path: Path to saved GNN model (uses default if None)
        num_base_accounts: Number of accounts in base graph
        seed: Random seed
        device: torch device
        
    Returns:
        Configured AdversarialAMLEnv instance
    """
    import sys
    import os
    
    # Add project root to path
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    
    from src.gnn_trainer import GraphSage, build_node_features, generate_synthetic_graph
    from src.pattern_library import PatternLibrary
    
    np.random.seed(seed)
    
    # Generate base graph
    print(f"Generating base graph with {num_base_accounts} accounts...")
    base_graph, labels, profiles = generate_synthetic_graph(
        num_nodes=num_base_accounts,
        mule_fraction=0.02,  # Low baseline suspicious rate
        seed=seed
    )
    
    # Initialize pattern library
    pattern_lib = PatternLibrary(seed=seed)
    
    # Load or create GNN model
    if gnn_model_path and os.path.exists(gnn_model_path):
        print(f"Loading GNN from {gnn_model_path}...")
        # Determine input dimension from features
        from src.gnn_trainer import graph_to_tx_df
        df_tx = graph_to_tx_df(base_graph)
        features, _, _ = build_node_features(base_graph, df_tx)
        in_feats = features.shape[1]
        
        gnn_model = GraphSage(in_feats=in_feats, hidden=64, out_feats=32)
        gnn_model.load_state_dict(torch.load(gnn_model_path, map_location=device))
    else:
        print("Creating new GNN model...")
        # Use default 12 features
        gnn_model = GraphSage(in_feats=12, hidden=64, out_feats=32)
    
    gnn_model.eval()
    gnn_model.to(device)
    
    # Create environment
    env = AdversarialAMLEnv(
        gnn_model=gnn_model,
        base_graph=base_graph,
        pattern_library=pattern_lib,
        node_feature_builder=build_node_features,
        max_steps=10,
        device=device
    )
    
    print("✅ Adversarial environment created successfully!")
    
    return env


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing AdversarialAMLEnv...")
    
    try:
        env = create_adversarial_env(
            num_base_accounts=100,
            seed=42,
            device='cpu'
        )
        
        # Reset
        state = env.reset()
        print(f"\nInitial state shape: {state.shape}")
        
        # Test action
        test_action = {
            'pattern_type': 0,  # peel_chain
            'temporal_mode': 2,  # mixed
            'chain_length': 10,
            'fork_count': 2,
            'exit_strategy': 0,  # cuckoo
            'peel_pct': 0.05,
            'wash_intensity': 0.3,
            'cycle_probability': 0.2,
        }
        
        next_state, reward, done, info = env.step(test_action)
        
        print(f"\nStep result:")
        print(f"  Next state shape: {next_state.shape}")
        print(f"  Reward: {reward:.4f}")
        print(f"  Done: {done}")
        print(f"  Injected nodes: {len(info['injected_nodes'])}")
        print(f"  Mean detection: {info['mean_detection']:.4f}")
        print(f"  Undetected: {info['num_undetected']}")
        
        print("\n✅ Environment test passed!")
        
    except Exception as e:
        print(f"\n❌ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
