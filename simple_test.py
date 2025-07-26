#!/usr/bin/env python3
import torch
from gcbfplus.env import DoubleIntegratorEnv
from gcbfplus.policy import BPTTPolicy, create_policy_from_config

def main():
    print("Running simple verification test...")
    
    # Test environment
    env_config = {
        'num_agents': 2,
        'area_size': 1.0,
        'dt': 0.03,
        'mass': 0.1,
        'car_radius': 0.05,
        'comm_radius': 0.5,
        'max_steps': 100
    }
    
    env = DoubleIntegratorEnv(env_config)
    print(f"Created environment: {env.__class__.__name__}")
    
    # Reset environment
    state = env.reset(batch_size=1, randomize=True)
    print(f"Reset environment, state: positions.shape={state.positions.shape}, velocities.shape={state.velocities.shape}")
    
    # Get observation
    obs = env.get_observation(state)
    print(f"Got observation with shape: {obs.shape}")
    
    # Create policy network
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
    
    # Get action from policy
    actions = policy(obs, reset_memory=True)
    print(f"Got actions with shape: {actions.shape}")
    
    # Step the environment
    step_result = env.step(state, actions)
    print(f"Stepped environment, next_state: positions.shape={step_result.next_state.positions.shape}")
    print(f"Reward shape: {step_result.reward.shape}, Cost shape: {step_result.cost.shape}")
    
    # Test 5 steps of rollout
    print("\nTesting 5-step rollout...")
    current_state = state
    for i in range(5):
        # Get observation
        obs = env.get_observation(current_state)
        
        # Get action
        action = policy(obs)
        
        # Step environment
        step_result = env.step(current_state, action)
        next_state = step_result.next_state
        
        # Print distances to goals
        distances = env.get_goal_distance(next_state)
        print(f"Step {i+1}: Average goal distance = {distances.mean().item():.4f}")
        
        # Update state
        current_state = next_state
    
    print("Simple test completed successfully!")

if __name__ == "__main__":
    main() 