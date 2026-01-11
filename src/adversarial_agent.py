"""
Adversarial Pattern Agent with PPO
===================================
A PPO-based reinforcement learning agent that learns to generate AML patterns
(Peel Chains, Nested Cycles, etc.) that evade GNN-based detection.

Key Features:
- Hybrid action space: Multi-Discrete + Beta-Continuous
- Beta distributions for bounded continuous actions (no gradient death)
- Canonicalization-ready action output for conditional dependencies

Reference: "The 2025 Horizon" - Section 6.3 "The Algorithmic Peel-Wash"
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Beta
from typing import Dict, List, Tuple, Optional, NamedTuple
from collections import deque


# =============================================================================
# ACTION SPACE CONFIGURATION
# =============================================================================

class ActionSpaceConfig:
    """Configuration for the hybrid action space."""
    
    # Discrete action dimensions
    DISCRETE_ACTIONS = {
        'pattern_type': 5,      # [peel_chain, forked_peel, smurfing, funnel, mule]
        'temporal_mode': 3,     # [rapid_fire, evasion_pause, mixed]
        'chain_length': 16,     # [5..20] mapped to indices [0..15]
        'fork_count': 4,        # [1..4] mapped to indices [0..3]
        'exit_strategy': 3,     # [cuckoo_victim, new_shell, existing_mule]
    }
    
    # Continuous action ranges (using Beta distribution)
    CONTINUOUS_ACTIONS = {
        'peel_pct': (0.02, 0.15),           # Percentage peeled per hop
        'wash_intensity': (0.0, 1.0),        # Wash trading volume ratio
        'cycle_probability': (0.0, 0.5),     # Nested cycle injection rate
    }
    
    # Pattern type names for reference
    PATTERN_NAMES = ['peel_chain', 'forked_peel', 'smurfing', 'funnel', 'mule']
    TEMPORAL_MODES = ['rapid_fire', 'evasion_pause', 'mixed']
    EXIT_STRATEGIES = ['cuckoo_victim', 'new_shell', 'existing_mule']
    
    @classmethod
    def get_discrete_dims(cls) -> List[int]:
        """Return list of discrete action dimensions."""
        return list(cls.DISCRETE_ACTIONS.values())
    
    @classmethod
    def get_continuous_ranges(cls) -> List[Tuple[float, float]]:
        """Return list of (min, max) for continuous actions."""
        return list(cls.CONTINUOUS_ACTIONS.values())


# =============================================================================
# EXPERIENCE BUFFER
# =============================================================================

class Experience(NamedTuple):
    """Single experience tuple for PPO training."""
    state: np.ndarray
    discrete_actions: np.ndarray       # Shape: (num_discrete,)
    continuous_actions: np.ndarray     # Shape: (num_continuous,)
    discrete_log_probs: np.ndarray     # Shape: (num_discrete,)
    continuous_log_probs: np.ndarray   # Shape: (num_continuous,)
    reward: float
    done: bool
    value: float


class RolloutBuffer:
    """Buffer for storing PPO rollout experiences."""
    
    def __init__(self, capacity: int = 2048):
        self.capacity = capacity
        self.buffer: List[Experience] = []
    
    def add(self, exp: Experience):
        self.buffer.append(exp)
    
    def clear(self):
        self.buffer = []
    
    def __len__(self):
        return len(self.buffer)
    
    def is_full(self):
        return len(self.buffer) >= self.capacity
    
    def get_batch(self) -> Dict[str, torch.Tensor]:
        """Convert buffer to tensors for training."""
        states = torch.FloatTensor(np.array([e.state for e in self.buffer]))
        discrete_actions = torch.LongTensor(np.array([e.discrete_actions for e in self.buffer]))
        continuous_actions = torch.FloatTensor(np.array([e.continuous_actions for e in self.buffer]))
        discrete_log_probs = torch.FloatTensor(np.array([e.discrete_log_probs for e in self.buffer]))
        continuous_log_probs = torch.FloatTensor(np.array([e.continuous_log_probs for e in self.buffer]))
        rewards = torch.FloatTensor([e.reward for e in self.buffer])
        dones = torch.FloatTensor([e.done for e in self.buffer])
        values = torch.FloatTensor([e.value for e in self.buffer])
        
        return {
            'states': states,
            'discrete_actions': discrete_actions,
            'continuous_actions': continuous_actions,
            'discrete_log_probs': discrete_log_probs,
            'continuous_log_probs': continuous_log_probs,
            'rewards': rewards,
            'dones': dones,
            'values': values,
        }


# =============================================================================
# NEURAL NETWORK COMPONENTS
# =============================================================================

class StateEncoder(nn.Module):
    """Encodes the state vector (graph embedding + node features + history)."""
    
    def __init__(self, state_dim: int = 49, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class DiscreteActionHead(nn.Module):
    """Multi-head output for discrete actions."""
    
    def __init__(self, hidden_dim: int, action_dims: List[int]):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, dim) for dim in action_dims
        ])
    
    def forward(self, h: torch.Tensor) -> List[Categorical]:
        """Returns list of Categorical distributions for each discrete action."""
        distributions = []
        for head in self.heads:
            logits = head(h)
            distributions.append(Categorical(logits=logits))
        return distributions
    
    def get_actions_and_log_probs(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample actions and compute log probabilities."""
        distributions = self.forward(h)
        actions = []
        log_probs = []
        
        for dist in distributions:
            action = dist.sample()
            actions.append(action)
            log_probs.append(dist.log_prob(action))
        
        return torch.stack(actions, dim=-1), torch.stack(log_probs, dim=-1)
    
    def evaluate_actions(self, h: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log probs and entropy for given actions."""
        distributions = self.forward(h)
        log_probs = []
        entropies = []
        
        for i, dist in enumerate(distributions):
            log_probs.append(dist.log_prob(actions[:, i]))
            entropies.append(dist.entropy())
        
        return torch.stack(log_probs, dim=-1), torch.stack(entropies, dim=-1)


class BetaActionHead(nn.Module):
    """
    Beta distribution head for bounded continuous actions.
    
    Key insight: Beta distribution naturally exists on [0,1], avoiding
    gradient death from clipping Gaussian outputs.
    """
    
    def __init__(self, hidden_dim: int, min_val: float, max_val: float):
        super().__init__()
        self.alpha_head = nn.Linear(hidden_dim, 1)
        self.beta_head = nn.Linear(hidden_dim, 1)
        self.min_val = min_val
        self.max_val = max_val
        self.range = max_val - min_val
    
    def forward(self, h: torch.Tensor) -> Beta:
        """Returns Beta distribution."""
        # Softplus ensures α, β > 0; add 1 for numerical stability
        alpha = F.softplus(self.alpha_head(h)) + 1.0
        beta = F.softplus(self.beta_head(h)) + 1.0
        return Beta(alpha.squeeze(-1), beta.squeeze(-1))
    
    def sample_and_scale(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample from Beta and scale to [min_val, max_val]."""
        dist = self.forward(h)
        x = dist.rsample()  # Reparameterized sample for gradients
        action = self.min_val + x * self.range
        log_prob = dist.log_prob(x)
        return action, log_prob
    
    def evaluate_action(self, h: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log prob and entropy for given scaled action."""
        dist = self.forward(h)
        # Rescale action back to [0,1] for Beta
        x = (action - self.min_val) / self.range
        x = torch.clamp(x, 1e-6, 1 - 1e-6)  # Avoid boundary issues
        log_prob = dist.log_prob(x)
        entropy = dist.entropy()
        return log_prob, entropy


class ContinuousActionHeads(nn.Module):
    """Collection of Beta heads for all continuous actions."""
    
    def __init__(self, hidden_dim: int, action_ranges: List[Tuple[float, float]]):
        super().__init__()
        self.heads = nn.ModuleList([
            BetaActionHead(hidden_dim, min_val, max_val)
            for min_val, max_val in action_ranges
        ])
    
    def get_actions_and_log_probs(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample all continuous actions."""
        actions = []
        log_probs = []
        
        for head in self.heads:
            action, log_prob = head.sample_and_scale(h)
            actions.append(action)
            log_probs.append(log_prob)
        
        return torch.stack(actions, dim=-1), torch.stack(log_probs, dim=-1)
    
    def evaluate_actions(self, h: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log probs and entropy for given actions."""
        log_probs = []
        entropies = []
        
        for i, head in enumerate(self.heads):
            log_prob, entropy = head.evaluate_action(h, actions[:, i])
            log_probs.append(log_prob)
            entropies.append(entropy)
        
        return torch.stack(log_probs, dim=-1), torch.stack(entropies, dim=-1)


class Critic(nn.Module):
    """Value function estimator for advantage computation."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h).squeeze(-1)


# =============================================================================
# PPO AGENT
# =============================================================================

class AdversarialPatternAgent(nn.Module):
    """
    PPO-based agent for generating adversarial AML patterns.
    
    The agent learns to select pattern parameters that evade GNN detection,
    using a hybrid Multi-Discrete + Beta-Continuous action space.
    """
    
    def __init__(
        self,
        state_dim: int = 49,
        hidden_dim: int = 128,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = 'cpu'
    ):
        super().__init__()
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        
        # Network components
        self.encoder = StateEncoder(state_dim, hidden_dim)
        self.discrete_head = DiscreteActionHead(
            hidden_dim, 
            ActionSpaceConfig.get_discrete_dims()
        )
        self.continuous_head = ContinuousActionHeads(
            hidden_dim,
            ActionSpaceConfig.get_continuous_ranges()
        )
        self.critic = Critic(hidden_dim)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        # Experience buffer
        self.buffer = RolloutBuffer()
        
        # Detection rate history for state augmentation
        self.detection_history = deque(maxlen=5)
        for _ in range(5):
            self.detection_history.append(0.5)  # Initialize with neutral rate
        
        self.to(device)
    
    def _encode_state(self, state: np.ndarray) -> torch.Tensor:
        """Convert numpy state to encoded tensor."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        return self.encoder(state_t)
    
    def select_action(self, state: np.ndarray) -> Tuple[Dict, Dict]:
        """
        Select action given current state.
        
        Returns:
            action_dict: Human-readable action dictionary
            raw_output: Raw tensors for PPO update
        """
        with torch.no_grad():
            h = self._encode_state(state)
            
            # Sample discrete actions
            discrete_actions, discrete_log_probs = self.discrete_head.get_actions_and_log_probs(h)
            
            # Sample continuous actions
            continuous_actions, continuous_log_probs = self.continuous_head.get_actions_and_log_probs(h)
            
            # Get value estimate
            value = self.critic(h)
        
        # Convert to numpy
        discrete_np = discrete_actions.squeeze(0).cpu().numpy()
        continuous_np = continuous_actions.squeeze(0).cpu().numpy()
        
        # Build human-readable action dict
        action_dict = {
            'pattern_type': int(discrete_np[0]),
            'temporal_mode': int(discrete_np[1]),
            'chain_length': int(discrete_np[2]) + 5,  # Map [0..15] to [5..20]
            'fork_count': int(discrete_np[3]) + 1,     # Map [0..3] to [1..4]
            'exit_strategy': int(discrete_np[4]),
            'peel_pct': float(continuous_np[0]),
            'wash_intensity': float(continuous_np[1]),
            'cycle_probability': float(continuous_np[2]),
        }
        
        # Raw output for buffer
        raw_output = {
            'discrete_actions': discrete_np,
            'continuous_actions': continuous_np,
            'discrete_log_probs': discrete_log_probs.squeeze(0).cpu().numpy(),
            'continuous_log_probs': continuous_log_probs.squeeze(0).cpu().numpy(),
            'value': value.item(),
        }
        
        return action_dict, raw_output
    
    def store_experience(
        self,
        state: np.ndarray,
        raw_output: Dict,
        reward: float,
        done: bool
    ):
        """Store experience in buffer."""
        exp = Experience(
            state=state,
            discrete_actions=raw_output['discrete_actions'],
            continuous_actions=raw_output['continuous_actions'],
            discrete_log_probs=raw_output['discrete_log_probs'],
            continuous_log_probs=raw_output['continuous_log_probs'],
            reward=reward,
            done=done,
            value=raw_output['value'],
        )
        self.buffer.add(exp)
    
    def update_detection_history(self, detection_rate: float):
        """Update rolling detection rate history."""
        self.detection_history.append(detection_rate)
    
    def get_detection_history(self) -> np.ndarray:
        """Get detection history as numpy array."""
        return np.array(list(self.detection_history), dtype=np.float32)
    
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation."""
        batch_size = len(rewards)
        advantages = torch.zeros(batch_size)
        returns = torch.zeros(batch_size)
        
        gae = 0
        next_val = next_value
        
        for t in reversed(range(batch_size)):
            if dones[t]:
                next_val = 0
                gae = 0
            
            delta = rewards[t] + self.gamma * next_val - values[t]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
            next_val = values[t]
        
        return advantages, returns
    
    def update(self, epochs: int = 4, batch_size: int = 64) -> Dict[str, float]:
        """
        PPO update with clipped objective.
        
        Returns dict of training metrics.
        """
        if len(self.buffer) == 0:
            return {}
        
        batch = self.buffer.get_batch()
        
        # Compute advantages
        with torch.no_grad():
            # Get value of last state for GAE
            last_state = batch['states'][-1].unsqueeze(0).to(self.device)
            h = self.encoder(last_state)
            next_value = self.critic(h).item()
        
        advantages, returns = self.compute_gae(
            batch['rewards'],
            batch['values'],
            batch['dones'],
            next_value
        )
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Move to device
        states = batch['states'].to(self.device)
        discrete_actions = batch['discrete_actions'].to(self.device)
        continuous_actions = batch['continuous_actions'].to(self.device)
        old_discrete_log_probs = batch['discrete_log_probs'].to(self.device)
        old_continuous_log_probs = batch['continuous_log_probs'].to(self.device)
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)
        
        # Training metrics
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0
        
        # Multiple epochs of PPO updates
        dataset_size = len(states)
        indices = np.arange(dataset_size)
        
        for epoch in range(epochs):
            np.random.shuffle(indices)
            
            for start in range(0, dataset_size, batch_size):
                end = min(start + batch_size, dataset_size)
                batch_indices = indices[start:end]
                
                # Get batch
                b_states = states[batch_indices]
                b_discrete_actions = discrete_actions[batch_indices]
                b_continuous_actions = continuous_actions[batch_indices]
                b_old_discrete_lp = old_discrete_log_probs[batch_indices]
                b_old_continuous_lp = old_continuous_log_probs[batch_indices]
                b_advantages = advantages[batch_indices]
                b_returns = returns[batch_indices]
                
                # Forward pass
                h = self.encoder(b_states)
                
                # Evaluate discrete actions
                new_discrete_lp, discrete_entropy = self.discrete_head.evaluate_actions(
                    h, b_discrete_actions
                )
                
                # Evaluate continuous actions
                new_continuous_lp, continuous_entropy = self.continuous_head.evaluate_actions(
                    h, b_continuous_actions
                )
                
                # Combined log probs
                old_log_prob = b_old_discrete_lp.sum(dim=-1) + b_old_continuous_lp.sum(dim=-1)
                new_log_prob = new_discrete_lp.sum(dim=-1) + new_continuous_lp.sum(dim=-1)
                
                # Ratio for PPO
                ratio = torch.exp(new_log_prob - old_log_prob)
                
                # Clipped surrogate objective
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                values = self.critic(h)
                value_loss = F.mse_loss(values, b_returns)
                
                # Entropy bonus
                entropy = (discrete_entropy.mean() + continuous_entropy.mean()) / 2
                
                # Total loss
                loss = (
                    policy_loss 
                    + self.value_coef * value_loss 
                    - self.entropy_coef * entropy
                )
                
                # Backprop
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                num_updates += 1
        
        # Clear buffer
        self.buffer.clear()
        
        return {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates,
        }
    
    def save(self, path: str):
        """Save model weights."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'detection_history': list(self.detection_history),
        }, path)
    
    def load(self, path: str):
        """Load model weights."""
        # Use weights_only=False for checkpoints containing optimizer state and numpy arrays
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.detection_history = deque(
            checkpoint.get('detection_history', [0.5] * 5),
            maxlen=5
        )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def action_to_pattern_name(action_dict: Dict) -> str:
    """Convert action dict to human-readable pattern description."""
    pattern = ActionSpaceConfig.PATTERN_NAMES[action_dict['pattern_type']]
    temporal = ActionSpaceConfig.TEMPORAL_MODES[action_dict['temporal_mode']]
    exit_strat = ActionSpaceConfig.EXIT_STRATEGIES[action_dict['exit_strategy']]
    
    return (
        f"{pattern} | "
        f"temporal={temporal} | "
        f"length={action_dict['chain_length']} | "
        f"peel={action_dict['peel_pct']:.2%} | "
        f"wash={action_dict['wash_intensity']:.2f} | "
        f"exit={exit_strat}"
    )


if __name__ == "__main__":
    # Quick test
    print("Testing AdversarialPatternAgent...")
    
    agent = AdversarialPatternAgent(state_dim=49, hidden_dim=128)
    
    # Random state
    state = np.random.randn(49).astype(np.float32)
    
    # Select action
    action_dict, raw_output = agent.select_action(state)
    
    print(f"\nAction selected:")
    print(f"  Pattern: {action_to_pattern_name(action_dict)}")
    print(f"\nRaw output shapes:")
    print(f"  discrete_actions: {raw_output['discrete_actions'].shape}")
    print(f"  continuous_actions: {raw_output['continuous_actions'].shape}")
    
    # Simulate storing experience
    agent.store_experience(state, raw_output, reward=0.5, done=False)
    
    print(f"\nBuffer size: {len(agent.buffer)}")
    print("\n✅ Agent test passed!")
