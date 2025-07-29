#!/usr/bin/env python3
import os
import torch
import yaml
import argparse
import numpy as np
from pathlib import Path

from gcbfplus.env import DoubleIntegratorEnv, CrazyFlieEnv
from gcbfplus.env.gcbf_safety_layer import GCBFSafetyLayer
from gcbfplus.policy import BPTTPolicy, create_policy_from_config
from gcbfplus.trainer.bptt_trainer import BPTTTrainer


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a multi-agent safe control system using BPTT")
    parser.add_argument("--config", type=str, default="config/bptt_config.yaml", help="Path to configuration file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--log_dir", type=str, default="logs/bptt", help="Directory to save logs and models")
    parser.add_argument("--env_type", type=str, default="double_integrator", help="Environment type (double_integrator or crazyflie)")
    parser.add_argument("--load_checkpoint", type=str, help="Load from checkpoint")
    args = parser.parse_args()
    
    # Set up the device - Force GPU usage if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Legacy support for cuda flag
    use_cuda = torch.cuda.is_available()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)
        
    # Enable anomaly detection to help debug gradient issues
    torch.autograd.set_detect_anomaly(True)
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create log directory
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the configuration to the log directory
    with open(log_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    # Extract configuration sections with default values for missing sections
    run_name = config.get('run_name', 'GCBF+_BTN_BPTT')
    env_config = config.get('env', {})
    network_config = config.get('networks', {})
    training_config = config.get('training', {})
    
    # Extract policy and CBF network configurations
    policy_network_config = network_config.get('policy', {})
    cbf_network_config = network_config.get('cbf')  # Can be None
    
    # Environment type from command-line arguments
    env_type = args.env_type.lower()
    
    # 1. Create environment based on config
    if env_type == "double_integrator":
        env = DoubleIntegratorEnv(env_config)
    elif env_type == "crazyflie":
        env = CrazyFlieEnv(env_config)
    else:
        raise ValueError(f"Unsupported environment type: {env_type}. Choose 'double_integrator' or 'crazyflie'")
    
    # Move environment to device
    env = env.to(device)
    
    # 2. Create policy network
    # Use the policy configuration from YAML file
    if policy_network_config:
        # Use configuration from YAML file (supports vision and other advanced features)
        policy_config = policy_network_config.copy()
        policy_config['type'] = 'bptt'
        
        # Ensure policy_head has correct output dimension
        if 'policy_head' not in policy_config:
            policy_config['policy_head'] = {}
        policy_config['policy_head']['output_dim'] = env.action_shape[-1]
        
        # Add default values for missing perception config if needed
        if 'perception' not in policy_config:
            hidden_dim = policy_network_config.get('hidden_dim', 64)
            n_layers = policy_network_config.get('n_layers', 2)
            obs_dim = env.observation_shape[-1]
            print(f"Environment observation dimension: {obs_dim}")
            
            policy_config['perception'] = {
                'input_dim': obs_dim,
                'hidden_dim': hidden_dim,
                'num_layers': n_layers,
                'activation': 'relu',
                'use_batch_norm': False
            }
        
        # Add default memory config if needed
        if 'memory' not in policy_config:
            hidden_dim = policy_network_config.get('hidden_dim', 64)
            policy_config['memory'] = {
                'hidden_dim': hidden_dim,
                'num_layers': 1
            }
    else:
        # Fallback: create default configuration if no policy config in YAML
        hidden_dim = 64
        n_layers = 2
        obs_dim = env.observation_shape[-1]
        print(f"Environment observation dimension: {obs_dim}")
        
        policy_config = {
            'type': 'bptt',
            'perception': {
                'input_dim': obs_dim,
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
    
    # Create the policy network
    policy_network = create_policy_from_config(policy_config)
    policy_network = policy_network.to(device)
    
    # 3. Create CBF network (optional)
    cbf_network = None
    
    # Extract CBF alpha parameter from configuration (needed for trainer config)
    cbf_alpha = env_config.get('cbf_alpha', training_config.get('cbf_alpha', 1.0))
    
    if cbf_network_config is not None:
        # Extract CBF configuration values with defaults
        cbf_hidden_dim = cbf_network_config.get('hidden_dim', 64)
        safety_margin = env_config.get('safety_margin', env_config.get('car_radius', 0.05) * 1.1)
        
        # Prepare CBF safety layer configuration
        safety_layer_config = {
            'alpha': cbf_alpha,
            'eps': 0.02,
            'safety_margin': safety_margin,
            'use_qp': True
        }
        
        # Create CBF safety layer
        safety_layer = GCBFSafetyLayer(safety_layer_config)
        safety_layer = safety_layer.to(device)
        
        # For now, we'll use a simple MLP as CBF network
        cbf_input_dim = env.observation_shape[-1]
        cbf_network = torch.nn.Sequential(
            torch.nn.Linear(cbf_input_dim, cbf_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(cbf_hidden_dim, cbf_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(cbf_hidden_dim, 1)
        ).to(device)
    
    # 4. Create optimizer
    optimizer_params = []
    optimizer_params.extend(policy_network.parameters())
    if cbf_network is not None:
        optimizer_params.extend(cbf_network.parameters())
    
    # Get optimizer parameters with defaults
    learning_rate = training_config.get('learning_rate', 0.001)
    
    optimizer = torch.optim.Adam(
        optimizer_params,
        lr=learning_rate
    )
    
    # 5. Extract training parameters with defaults
    training_steps = training_config.get('training_steps', 10000)
    horizon_length = training_config.get('horizon_length', 50)
    eval_horizon = training_config.get('eval_horizon', 100)
    eval_interval = training_config.get('eval_interval', 100)
    save_interval = training_config.get('save_interval', 1000)
    max_grad_norm = training_config.get('max_grad_norm', 1.0)
    goal_weight = training_config.get('goal_weight', 1.0)
    safety_weight = training_config.get('safety_weight', 10.0)
    control_weight = training_config.get('control_weight', 0.1)
    use_lr_scheduler = training_config.get('use_lr_scheduler', False)
    
    # Get environment parameters with defaults
    num_agents = env_config.get('num_agents', 8)
    area_size = env_config.get('area_size', 1.0)
    
    # Create trainer configuration
    trainer_config = {
        'run_name': run_name,
        'log_dir': str(log_dir),
        'num_agents': num_agents,
        'area_size': area_size,
        'training_steps': training_steps,
        'horizon_length': horizon_length,
        'eval_horizon': eval_horizon,
        'eval_interval': eval_interval,
        'save_interval': save_interval,
        'max_grad_norm': max_grad_norm,
        'goal_weight': goal_weight,
        'safety_weight': safety_weight,
        'control_weight': control_weight,
        'cbf_alpha': cbf_alpha,
        'use_lr_scheduler': use_lr_scheduler
    }
    
    # 6. Create the trainer
    trainer = BPTTTrainer(
        env=env,
        policy_network=policy_network,
        cbf_network=cbf_network,
        optimizer=optimizer,
        config=trainer_config
    )
    
    # 7. Load checkpoint if provided
    if args.load_checkpoint:
        checkpoint_path = Path(args.load_checkpoint)
        if checkpoint_path.exists():
            checkpoint_step = int(checkpoint_path.name)
            trainer.load_models(checkpoint_step)
            print(f"Loaded checkpoint from step {checkpoint_step}")
        else:
            print(f"Warning: Checkpoint {checkpoint_path} not found")
    
    # 8. Run training
    print(f"Starting training with configuration:")
    print(f"  Run name: {run_name}")
    print(f"  Environment: {env_type}")
    print(f"  Num agents: {num_agents}")
    print(f"  CBF alpha: {cbf_alpha}")
    print(f"  Training steps: {training_steps}")
    print(f"  BPTT horizon length: {horizon_length}")
    print(f"  Device: {device}")
    
    trainer.train()
    print("Training completed successfully!")


if __name__ == "__main__":
    main() 