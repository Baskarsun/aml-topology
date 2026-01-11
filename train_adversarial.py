"""
Adversarial RL Training Script
==============================
Trains a PPO-based AdversarialPatternAgent to generate AML patterns
that evade GNN-based detection.

Usage:
    python train_adversarial.py                    # Full training
    python train_adversarial.py --episodes 100    # Quick test
    python train_adversarial.py --debug           # Debug mode

Reference: "The 2025 Horizon" - Section 6.3 "The Algorithmic Peel-Wash"
"""

import os
import sys
import argparse
import json
import time
import numpy as np
from datetime import datetime
from collections import deque

import torch

# Add project root to path
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.adversarial_agent import AdversarialPatternAgent, action_to_pattern_name
from src.adversarial_env import create_adversarial_env, AdversarialAMLEnv


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_adversarial(
    num_episodes: int = 1000,
    steps_per_episode: int = 10,
    ppo_epochs: int = 4,
    batch_size: int = 64,
    rollout_size: int = 2048,
    discriminator_update_freq: int = 100,
    save_freq: int = 100,
    log_freq: int = 10,
    num_base_accounts: int = 500,
    seed: int = 42,
    device: str = 'cpu',
    debug: bool = False,
    save_dir: str = 'models',
) -> dict:
    """
    Train the adversarial pattern agent.
    
    Args:
        num_episodes: Total training episodes
        steps_per_episode: Pattern injections per episode
        ppo_epochs: PPO update epochs per rollout
        batch_size: Mini-batch size for PPO updates
        rollout_size: Experience buffer size before update
        discriminator_update_freq: Episodes between GNN retraining
        save_freq: Episodes between checkpoints
        log_freq: Episodes between logging
        num_base_accounts: Base graph size
        seed: Random seed
        device: torch device
        debug: Enable debug output
        save_dir: Directory for saved models
        
    Returns:
        Training metrics dict
    """
    print("=" * 60)
    print("ADVERSARIAL RL TRAINING")
    print("=" * 60)
    print(f"Episodes: {num_episodes}")
    print(f"Steps/Episode: {steps_per_episode}")
    print(f"Device: {device}")
    print(f"Seed: {seed}")
    print()
    
    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create environment
    print("Creating environment...")
    env = create_adversarial_env(
        num_base_accounts=num_base_accounts,
        seed=seed,
        device=device
    )
    
    # Create agent
    print("Creating PPO agent...")
    agent = AdversarialPatternAgent(
        state_dim=49,
        hidden_dim=128,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        device=device
    )
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Training metrics
    episode_rewards = deque(maxlen=100)
    episode_detections = deque(maxlen=100)
    episode_undetected = deque(maxlen=100)
    
    all_metrics = {
        'episode': [],
        'reward': [],
        'mean_detection': [],
        'undetected_ratio': [],
        'policy_loss': [],
        'value_loss': [],
        'entropy': [],
    }
    
    total_steps = 0
    best_reward = float('-inf')
    
    start_time = time.time()
    
    print("\nStarting training loop...")
    print("-" * 60)
    
    for episode in range(1, num_episodes + 1):
        # Reset environment
        state = env.reset()
        episode_reward = 0
        episode_info = []
        
        for step in range(steps_per_episode):
            # Select action
            action_dict, raw_output = agent.select_action(state)
            
            # Execute action
            next_state, reward, done, info = env.step(action_dict)
            
            # Store experience
            agent.store_experience(state, raw_output, reward, done)
            
            episode_reward += reward
            episode_info.append(info)
            total_steps += 1
            
            if debug:
                print(f"  Step {step+1}: {action_to_pattern_name(action_dict)}")
                print(f"    Reward: {reward:.4f}, Detection: {info['mean_detection']:.4f}")
            
            state = next_state
            
            if done:
                break
        
        # Update detection history
        mean_detection = np.mean([i['mean_detection'] for i in episode_info])
        agent.update_detection_history(mean_detection)
        
        # Track metrics
        undetected = np.mean([i.get('undetected_ratio', 0) for i in episode_info])
        
        episode_rewards.append(episode_reward)
        episode_detections.append(mean_detection)
        episode_undetected.append(undetected)
        
        # PPO update when buffer is full
        update_metrics = {}
        if len(agent.buffer) >= rollout_size or episode == num_episodes:
            update_metrics = agent.update(epochs=ppo_epochs, batch_size=batch_size)
        
        # Logging
        if episode % log_freq == 0:
            avg_reward = np.mean(episode_rewards)
            avg_detection = np.mean(episode_detections)
            avg_undetected = np.mean(episode_undetected)
            
            elapsed = time.time() - start_time
            eps_per_sec = episode / elapsed if elapsed > 0 else 0
            
            print(f"Episode {episode:5d} | "
                  f"Reward: {avg_reward:7.3f} | "
                  f"Detection: {avg_detection:.3f} | "
                  f"Undetected: {avg_undetected:.3f} | "
                  f"Speed: {eps_per_sec:.1f} ep/s")
            
            if update_metrics:
                print(f"             | "
                      f"Policy Loss: {update_metrics.get('policy_loss', 0):.4f} | "
                      f"Value Loss: {update_metrics.get('value_loss', 0):.4f} | "
                      f"Entropy: {update_metrics.get('entropy', 0):.4f}")
            
            # Store metrics
            all_metrics['episode'].append(episode)
            all_metrics['reward'].append(avg_reward)
            all_metrics['mean_detection'].append(avg_detection)
            all_metrics['undetected_ratio'].append(avg_undetected)
            all_metrics['policy_loss'].append(update_metrics.get('policy_loss', 0))
            all_metrics['value_loss'].append(update_metrics.get('value_loss', 0))
            all_metrics['entropy'].append(update_metrics.get('entropy', 0))
        
        # Save checkpoint
        if episode % save_freq == 0:
            checkpoint_path = os.path.join(save_dir, f'adversarial_agent_ep{episode}.pt')
            agent.save(checkpoint_path)
            print(f"  → Checkpoint saved: {checkpoint_path}")
            
            # Save best model
            avg_reward = np.mean(episode_rewards)
            if avg_reward > best_reward:
                best_reward = avg_reward
                best_path = os.path.join(save_dir, 'adversarial_agent_best.pt')
                agent.save(best_path)
                print(f"  → New best model saved: {best_path}")
        
        # Discriminator update (placeholder for future implementation)
        if episode % discriminator_update_freq == 0 and episode > 0:
            print(f"\n  [!] Discriminator update scheduled (Episode {episode})")
            # TODO: Retrain GNN on new patterns
            print()
    
    # Final save
    final_path = os.path.join(save_dir, 'adversarial_agent_final.pt')
    agent.save(final_path)
    
    # Save metrics
    metrics_path = os.path.join(save_dir, 'adversarial_training_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    # Training summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Total episodes: {num_episodes}")
    print(f"Total steps: {total_steps}")
    print(f"Training time: {elapsed/60:.1f} minutes")
    print(f"Final avg reward: {np.mean(episode_rewards):.3f}")
    print(f"Final avg detection: {np.mean(episode_detections):.3f}")
    print(f"Final undetected ratio: {np.mean(episode_undetected):.3f}")
    print(f"Best reward: {best_reward:.3f}")
    print(f"\nSaved models:")
    print(f"  - {final_path}")
    print(f"  - {metrics_path}")
    
    return all_metrics


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_agent(
    agent_path: str,
    num_episodes: int = 10,
    num_base_accounts: int = 500,
    seed: int = 123,
    device: str = 'cpu',
):
    """
    Evaluate a trained agent.
    
    Args:
        agent_path: Path to saved agent
        num_episodes: Number of evaluation episodes
        num_base_accounts: Base graph size
        seed: Random seed
        device: torch device
    """
    print("=" * 60)
    print("AGENT EVALUATION")
    print("=" * 60)
    
    # Create environment
    env = create_adversarial_env(
        num_base_accounts=num_base_accounts,
        seed=seed,
        device=device
    )
    
    # Load agent
    agent = AdversarialPatternAgent(state_dim=49, hidden_dim=128, device=device)
    agent.load(agent_path)
    agent.eval()
    
    print(f"Loaded agent from: {agent_path}")
    print(f"Running {num_episodes} evaluation episodes...\n")
    
    all_rewards = []
    all_detections = []
    pattern_counts = {i: 0 for i in range(5)}
    
    for ep in range(1, num_episodes + 1):
        state = env.reset()
        ep_reward = 0
        ep_detections = []
        
        for step in range(10):
            action_dict, _ = agent.select_action(state)
            next_state, reward, done, info = env.step(action_dict)
            
            ep_reward += reward
            ep_detections.append(info['mean_detection'])
            pattern_counts[action_dict['pattern_type']] += 1
            
            print(f"  {action_to_pattern_name(action_dict)}")
            print(f"    → Detection: {info['mean_detection']:.3f}, "
                  f"Undetected: {info['num_undetected']}/{len(info['injected_nodes'])}")
            
            state = next_state
            if done:
                break
        
        mean_det = np.mean(ep_detections)
        all_rewards.append(ep_reward)
        all_detections.append(mean_det)
        
        print(f"\nEpisode {ep}: Reward={ep_reward:.3f}, Mean Detection={mean_det:.3f}")
        print("-" * 40)
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Mean Reward: {np.mean(all_rewards):.3f} ± {np.std(all_rewards):.3f}")
    print(f"Mean Detection: {np.mean(all_detections):.3f} ± {np.std(all_detections):.3f}")
    print(f"\nPattern Distribution:")
    pattern_names = ['peel_chain', 'forked_peel', 'smurfing', 'funnel', 'mule']
    for i, name in enumerate(pattern_names):
        print(f"  {name}: {pattern_counts[i]}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train Adversarial Pattern Agent',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_adversarial.py                    # Full training
  python train_adversarial.py --episodes 100    # Quick test
  python train_adversarial.py --debug           # Debug mode
  python train_adversarial.py --eval models/adversarial_agent_best.pt
        """
    )
    
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of training episodes')
    parser.add_argument('--steps', type=int, default=10,
                        help='Steps per episode')
    parser.add_argument('--accounts', type=int, default=500,
                        help='Base graph account count')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device for training')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug output')
    parser.add_argument('--save-dir', type=str, default='models',
                        help='Directory for saved models')
    parser.add_argument('--eval', type=str, default=None,
                        help='Path to agent for evaluation (skips training)')
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    if args.eval:
        # Evaluation mode
        evaluate_agent(
            agent_path=args.eval,
            num_episodes=10,
            num_base_accounts=args.accounts,
            seed=args.seed + 1000,  # Different seed for eval
            device=args.device,
        )
    else:
        # Training mode
        train_adversarial(
            num_episodes=args.episodes,
            steps_per_episode=args.steps,
            num_base_accounts=args.accounts,
            seed=args.seed,
            device=args.device,
            debug=args.debug,
            save_dir=args.save_dir,
        )


if __name__ == "__main__":
    main()
