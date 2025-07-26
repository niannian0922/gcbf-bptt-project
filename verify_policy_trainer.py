#!/usr/bin/env python3
import torch
import numpy as np
from gcbfplus.env import DoubleIntegratorEnv
from gcbfplus.policy import BPTTPolicy, create_policy_from_config
from gcbfplus.trainer.bptt_trainer import BPTTTrainer

def main():
    print("Verifying Policy Network and Trainer refactoring...")
    
    # Create a simple environment
    env_config = {
        'num_agents': 2,
        'area_size': 1.0,
        'dt': 0.03,
        'mass': 0.1,
        'car_radius': 0.05,
        'comm_radius': 0.5
    }
    env = DoubleIntegratorEnv(env_config)
    
    # Create a policy network
    policy_config = {
        'perception': {
            'input_dim': env.observation_shape[-1],
            'hidden_dim': 32,
            'num_layers': 2
        },
        'memory': {
            'hidden_dim': 32
        },
        'policy_head': {
            'output_dim': env.action_shape[-1],
            'hidden_dims': [32],
            'action_scaling': True
        }
    }
    policy = create_policy_from_config(policy_config)
    
    # Create a trainer
    trainer_config = {
        'run_name': 'verification_test',
        'log_dir': 'logs/verification',
        'num_agents': env_config['num_agents'],
        'training_steps': 2,
        'horizon_length': 5,
        'eval_horizon': 10,
        'eval_interval': 1,
        'save_interval': 1,
        'goal_weight': 1.0,
        'safety_weight': 10.0,
        'control_weight': 0.1
    }
    
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
    trainer = BPTTTrainer(
        env=env,
        policy_network=policy,
        optimizer=optimizer,
        config=trainer_config
    )
    
    # Test initialization and one training step
    print("Testing initialization and one training step...")
    
    # Initialize a scenario
    state = trainer.initialize_scenario()
    print(f"Initialized state with {state.positions.shape[1]} agents")
    
    # Get observation
    obs = env.get_observation(state)
    print(f"Observation shape: {obs.shape}")
    
    # Forward pass through policy
    action = policy(obs)
    print(f"Action shape: {action.shape}")
    
    # Run a very short training loop
    print("Running a very short training loop...")
    metrics = trainer.train()
    
    print("Verification completed successfully!")
    
if __name__ == "__main__":
    main() 