#!/usr/bin/env python3
import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from gcbfplus.env import DoubleIntegratorEnv
from gcbfplus.policy import BPTTPolicy, create_policy_from_config

def main():
    print("Running simple training for verification...")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set up device
    device = torch.device("cpu")
    
    # Create log directory
    log_dir = Path("logs/simple_training")
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir = log_dir / "models" / "final"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Create environment
    env_config = {
        'num_agents': 3,
        'area_size': 1.0,
        'dt': 0.03,
        'mass': 0.1,
        'car_radius': 0.05,
        'comm_radius': 0.5,
        'max_steps': 100
    }
    
    env = DoubleIntegratorEnv(env_config)
    print(f"Created environment: {env.__class__.__name__}")
    
    # Create policy
    policy_config = {
        'type': 'bptt',
        'perception': {
            'input_dim': env.observation_shape[-1],
            'hidden_dim': 32,
            'num_layers': 2
        },
        'memory': {
            'hidden_dim': 32,
            'num_layers': 1
        },
        'policy_head': {
            'output_dim': env.action_shape[-1],
            'hidden_dims': [32],
            'action_scaling': True,
            'action_bound': 1.0
        }
    }
    
    policy = create_policy_from_config(policy_config)
    print(f"Created policy: {policy.__class__.__name__}")
    
    # Create optimizer
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
    
    # Training parameters
    num_steps = 10
    horizon = 20
    goal_weight = 1.0
    control_weight = 0.1
    
    # Run training
    print(f"Starting simple training for {num_steps} steps...")
    
    for step in tqdm(range(num_steps)):
        # Reset environment
        policy.train()
        state = env.reset(batch_size=1, randomize=True)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Run rollout for horizon steps
        total_reward = 0
        total_control_cost = 0
        
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
            
            # Accumulate metrics
            total_reward += reward
            total_control_cost += torch.mean(action**2)
            
            # Update state
            current_state = next_state
        
        # Compute loss
        goal_loss = -torch.mean(total_reward)
        control_loss = torch.mean(total_control_cost) / horizon
        total_loss = goal_weight * goal_loss + control_weight * control_loss
        
        # Backpropagation
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        
        # Print metrics
        if (step + 1) % 5 == 0:
            print(f"\nStep {step+1}/{num_steps}")
            print(f"  Goal Loss: {goal_loss.item():.4f}")
            print(f"  Control Loss: {control_loss.item():.4f}")
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
    for i in range(50):
        # Get observation
        obs = env.get_observation(state)
        
        # Get action
        with torch.no_grad():
            action = policy(obs, reset_memory=(i==0))
        
        # Step environment
        step_result = env.step(state, action)
        state = step_result.next_state
        
        # Check if reached goals
        distances = env.get_goal_distance(state)
        if torch.all(distances < 0.1):
            print(f"All agents reached their goals at step {i+1}!")
            break
            
        if (i + 1) % 10 == 0:
            print(f"Step {i+1}: avg_dist={distances.mean().item():.4f}")
    
    # Final distances
    distances = env.get_goal_distance(state)
    print(f"Final goal distances: avg={distances.mean().item():.4f}, min={distances.min().item():.4f}, max={distances.max().item():.4f}")
    print("Simple training completed!")

if __name__ == "__main__":
    main() 