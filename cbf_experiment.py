#!/usr/bin/env python3
import os
import torch
import numpy as np
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm

from gcbfplus.env import DoubleIntegratorEnv
from gcbfplus.policy import BPTTPolicy, create_policy_from_config

def run_cbf_experiment(config_path, log_dir, seed=42, device="cpu"):
    """
    Run a CBF alpha experiment with the given configuration.
    
    Args:
        config_path: Path to the configuration file
        log_dir: Directory to save logs and models
        seed: Random seed for reproducibility
        device: Device to run on ("cpu" or "cuda")
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create log directory
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir = log_dir / "models" / "final"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the configuration to the log directory
    with open(log_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    # Extract necessary configurations
    run_name = config.get('run_name', 'GCBF_Experiment')
    env_config = config.get('env', {})
    network_config = config.get('networks', {})
    policy_network_config = network_config.get('policy', {})
    training_config = config.get('training', {})
    
    # Extract CBF alpha value
    cbf_alpha = env_config.get('cbf_alpha', 1.0)
    
    # Override environment configuration to make it more challenging
    # More agents in a constrained space will better demonstrate CBF behavior
    env_config['num_agents'] = 12  # Increase number of agents
    env_config['area_size'] = 1.0  # Keep area size the same for density
    
    # Create environment
    env = DoubleIntegratorEnv(env_config)
    env = env.to(device)
    
    # Create policy network
    hidden_dim = policy_network_config.get('hidden_dim', 64)
    n_layers = policy_network_config.get('n_layers', 2)
    
    policy_config = {
        'type': 'bptt',
        'perception': {
            'input_dim': env.observation_shape[-1],
            'hidden_dim': hidden_dim,
            'num_layers': n_layers,
            'activation': 'relu',
            'use_batch_norm': False
        },
        'memory': {
            'hidden_dim': hidden_dim,
            'num_layers': 1
        },
        'policy_head': {
            'output_dim': env.action_shape[-1],
            'hidden_dims': [hidden_dim],
            'action_scaling': True,
            'action_bound': 1.0
        }
    }
    
    policy = create_policy_from_config(policy_config)
    policy = policy.to(device)
    
    # Create optimizer
    learning_rate = training_config.get('learning_rate', 0.001)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    
    # Training parameters
    num_steps = training_config.get('training_steps', 1000)
    horizon = training_config.get('horizon_length', 50)
    goal_weight = training_config.get('goal_weight', 1.0)
    safety_weight = training_config.get('safety_weight', 10.0)
    control_weight = training_config.get('control_weight', 0.1)
    
    print(f"Starting training for {run_name}:")
    print(f"  CBF Alpha: {cbf_alpha}")
    print(f"  Num Agents: {env_config['num_agents']}")
    print(f"  Training steps: {num_steps}")
    print(f"  Horizon length: {horizon}")
    print(f"  Device: {device}")
    
    # Run training
    for step in tqdm(range(num_steps)):
        # Reset environment
        policy.train()
        state = env.reset(batch_size=1, randomize=True)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Run rollout for horizon steps
        total_reward = 0
        total_control_cost = 0
        total_safety_cost = 0
        
        current_state = state
        for t in range(horizon):
            # Get observation
            obs = env.get_observation(current_state)
            
            # Get action
            action = policy(obs, reset_memory=(t==0))
            
            # Step environment
            step_result = env.step(current_state, action)
            next_state = step_result.next_state
            reward = step_result.reward
            cost = step_result.cost
            
            # Accumulate metrics
            total_reward += reward
            total_control_cost += torch.mean(action**2)
            total_safety_cost += cost
            
            # Update state
            current_state = next_state
        
        # Compute losses
        goal_loss = -torch.mean(total_reward)
        control_loss = torch.mean(total_control_cost) / horizon
        safety_loss = torch.mean(total_safety_cost)
        
        total_loss = (
            goal_weight * goal_loss + 
            control_weight * control_loss +
            safety_weight * safety_loss
        )
        
        # Backpropagation
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        
        # Print metrics every 100 steps
        if (step + 1) % 100 == 0:
            print(f"\nStep {step+1}/{num_steps}")
            print(f"  Goal Loss: {goal_loss.item():.4f}")
            print(f"  Control Loss: {control_loss.item():.4f}")
            print(f"  Safety Loss: {safety_loss.item():.4f}")
            print(f"  Total Loss: {total_loss.item():.4f}")
    
    # Save the trained policy
    policy_path = model_dir / "policy.pt"
    torch.save(policy.state_dict(), policy_path)
    print(f"\nTraining completed. Policy saved to {policy_path}")
    
    # Test the policy
    print("\nTesting the trained policy...")
    policy.eval()
    state = env.reset(batch_size=1, randomize=True)
    
    # Initial distances
    distances = env.get_goal_distance(state)
    print(f"Initial goal distances: avg={distances.mean().item():.4f}, min={distances.min().item():.4f}, max={distances.max().item():.4f}")
    
    # Run for 50 steps
    collision_count = 0
    goal_reached_count = 0
    
    for i in range(50):
        # Get observation
        obs = env.get_observation(state)
        
        # Get action
        with torch.no_grad():
            action = policy(obs, reset_memory=(i==0))
        
        # Step environment
        step_result = env.step(state, action)
        state = step_result.next_state
        
        # Check for collisions
        if torch.any(step_result.cost > 0):
            collision_count += 1
        
        # Check if reached goals
        distances = env.get_goal_distance(state)
        if torch.all(distances < 0.1):
            goal_reached_count += 1
            print(f"All agents reached their goals at step {i+1}!")
            break
            
        if (i + 1) % 10 == 0:
            print(f"Step {i+1}: avg_dist={distances.mean().item():.4f}")
    
    # Final distances
    distances = env.get_goal_distance(state)
    print(f"Final goal distances: avg={distances.mean().item():.4f}, min={distances.min().item():.4f}, max={distances.max().item():.4f}")
    print(f"Collision count: {collision_count}")
    print(f"Goal reached: {'Yes' if goal_reached_count > 0 else 'No'}")
    print(f"Experiment with CBF alpha={cbf_alpha} completed!")
    
    return {
        'cbf_alpha': cbf_alpha,
        'final_avg_distance': distances.mean().item(),
        'collision_count': collision_count,
        'goal_reached': goal_reached_count > 0,
    }

def main():
    parser = argparse.ArgumentParser(description="Run CBF Alpha Experiments")
    parser.add_argument("--config", type=str, default="config/alpha_medium.yaml", help="Path to configuration file")
    parser.add_argument("--log_dir", type=str, default="logs/cbf_experiment", help="Directory to save logs and models")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    args = parser.parse_args()
    
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    
    # Run the experiment
    results = run_cbf_experiment(args.config, args.log_dir, args.seed, device)
    
    # Print results
    print("\nExperiment Results:")
    print(f"  CBF Alpha: {results['cbf_alpha']}")
    print(f"  Final Average Distance: {results['final_avg_distance']:.4f}")
    print(f"  Collision Count: {results['collision_count']}")
    print(f"  Goal Reached: {results['goal_reached']}")

if __name__ == "__main__":
    main() 