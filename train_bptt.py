#!/usr/bin/env python3
import os
import torch
import yaml
import argparse
import numpy as np
from pathlib import Path

from gcbfplus.nn.torch_gnn import PolicyGNN, CBFGNN
from gcbfplus.trainer.bptt_trainer import BPTTTrainer


def flatten_dict(d, parent_key='', sep='_'):
    """
    Flatten a nested dictionary into a single-level dictionary.
    
    Args:
        d (dict): The dictionary to flatten
        parent_key (str): The string to prepend to dictionary keys
        sep (str): The separator between flattened keys
        
    Returns:
        dict: A flattened version of the input dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a multi-agent safe control system using BPTT")
    parser.add_argument("--config", type=str, default="config/bptt_config.yaml", help="Path to configuration file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--log_dir", type=str, default="logs/bptt", help="Directory to save logs and models")
    args = parser.parse_args()
    
    # Set up the device
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create log directory
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the configuration to the log directory
    with open(log_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    # Flatten the configuration for the trainer
    flat_config = flatten_dict(config)
    
    # Create the neural networks
    policy_network = PolicyGNN(
        node_dim=config['networks']['policy']['node_dim'],
        edge_dim=config['networks']['policy']['edge_dim'],
        action_dim=2,  # Fixed for the double integrator system
        hidden_dim=config['networks']['policy']['hidden_dim'],
        n_layers=config['networks']['policy']['n_layers']
    )
    
    cbf_network = CBFGNN(
        node_dim=config['networks']['cbf']['node_dim'],
        edge_dim=config['networks']['cbf']['edge_dim'],
        hidden_dim=config['networks']['cbf']['hidden_dim'],
        n_layers=config['networks']['cbf']['n_layers']
    )
    
    # Create the trainer
    trainer = BPTTTrainer(
        policy_network=policy_network,
        cbf_network=cbf_network,
        num_agents=config['env']['num_agents'],
        area_size=config['env']['area_size'],
        log_dir=args.log_dir,
        device=device,
        params={
            'run_name': config['run_name'],
            'dt': config['env']['dt'],
            'mass': config['env']['mass'],
            'car_radius': config['env']['car_radius'],
            'comm_radius': config['env']['comm_radius'],
            'training_steps': config['training']['training_steps'],
            'horizon_length': config['training']['horizon_length'],
            'eval_horizon': config['training']['eval_horizon'],
            'eval_interval': config['training']['eval_interval'],
            'save_interval': config['training']['save_interval'],
            'learning_rate': config['training']['learning_rate'],
            'max_grad_norm': config['training']['max_grad_norm'],
            'use_lr_scheduler': config['training']['use_lr_scheduler'],
            'lr_step_size': config['training']['lr_step_size'],
            'lr_gamma': config['training']['lr_gamma'],
            'goal_weight': config['training']['goal_weight'],
            'safety_weight': config['training']['safety_weight'],
            'control_weight': config['training']['control_weight'],
            'cbf_alpha': config['training']['cbf_alpha'],
        }
    )
    
    # Run training
    print(f"Starting training with configuration:")
    print(f"  Run name: {config['run_name']}")
    print(f"  Num agents: {config['env']['num_agents']}")
    print(f"  Training steps: {config['training']['training_steps']}")
    print(f"  BPTT horizon length: {config['training']['horizon_length']}")
    print(f"  Device: {device}")
    
    trainer.train()
    print("Training completed successfully!")


if __name__ == "__main__":
    main() 