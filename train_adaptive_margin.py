#!/usr/bin/env python3
"""
Adaptive Margin Training Script - Dynamic Safety Margin Prediction

Train our second core innovation: Let the policy network learn to output 
dynamic safety margins based on environment context for optimal efficiency-safety balance.
"""

import os
import sys
import yaml
import torch
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from gcbfplus.trainer.bptt_trainer import BPTTTrainer
from gcbfplus.env.double_integrator import DoubleIntegratorEnv
from gcbfplus.policy.bptt_policy import BPTTPolicy
from gcbfplus.env.gcbf_safety_layer import GCBFSafetyLayer


def create_policy_network(config):
    """Create BPTT policy network from config."""
    policy_config = config['networks']['policy']
    return BPTTPolicy(policy_config)


def create_cbf_network(config):
    """Create CBF safety layer from config."""
    cbf_config = config['networks']['cbf']
    return GCBFSafetyLayer(cbf_config)


def create_environment(config):
    """Create environment from config."""
    env_config = config['env']
    return DoubleIntegratorEnv(env_config)


def main():
    """Main adaptive margin training function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train innovation: Adaptive Safety Margin")
    parser.add_argument('--config', type=str, default='config/innovation_adaptive_margin.yaml',
                       help='Path to configuration file')
    parser.add_argument('--pretrained', type=str, default='logs/innovation_safety_gated/models/2000',
                       help='Path to pretrained model (Safety-Gated champion)')
    args = parser.parse_args()
    
    print("🚀 Starting Innovation Training: Adaptive Safety Margin")
    print("=" * 80)
    print("🎯 CORE INNOVATION #2: Dynamic safety margin prediction by policy network")
    print("   - Policy learns to output dynamic safety radius based on context")
    print("   - Optimal efficiency-safety balance in different scenarios")
    print("   - Smart margin regularization with safety-weighted penalties")
    print("=" * 80)
    
    # Load configuration
    config_path = args.config
    print(f"📄 Loading config: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Create environment
    print("🌍 Creating environment...")
    env = create_environment(config)
    print(f"✅ Environment created: {env.num_agents} agents")
    
    # Create networks
    print("🏗️ Creating policy network...")
    policy_network = create_policy_network(config).to(device)
    print("✅ Policy network created")
    
    print("🛡️ Creating CBF safety layer...")
    cbf_network = create_cbf_network(config).to(device)
    print("✅ CBF safety layer created")
    
    # Load pretrained weights from Safety-Gated champion
    pretrained_path = args.pretrained
    if pretrained_path and os.path.exists(f"{pretrained_path}/policy.pt"):
        try:
            print(f"📦 Loading Safety-Gated champion weights from: {pretrained_path}/policy.pt")
            checkpoint = torch.load(f"{pretrained_path}/policy.pt", map_location=device)
            # Use strict=False to allow new margin network parameters
            policy_network.load_state_dict(checkpoint, strict=False)
            print("✅ Champion weights loaded successfully (with new margin network)")
        except Exception as e:
            print(f"⚠️ Could not load champion weights: {e}")
            print("   Proceeding with random initialization...")
    else:
        print("ℹ️ No champion model found, using random initialization")
    
    # Display innovation configuration
    print("\n🚀 INNOVATION Configuration:")
    print(f"   Adaptive Safety Margin: {config.get('use_adaptive_margin', False)}")
    print(f"   Min Safety Margin: {config.get('min_safety_margin', 0.15):.3f}")
    print(f"   Max Safety Margin: {config.get('max_safety_margin', 0.4):.3f}")
    print(f"   Margin Regularization Weight: {config.get('margin_reg_weight', 0.05):.3f}")
    
    # Display inherited innovations
    print("\n🔗 Inherited Innovations:")
    print(f"   Safety-Gated Alpha Reg: {config.get('use_safety_gated_alpha_reg', False)}")
    if config.get('use_safety_gated_alpha_reg', False):
        print(f"   Safety Loss Threshold: {config.get('safety_loss_threshold', 0.01):.3f}")
    
    # Display loss weights
    print("\n📊 Loss Weight Configuration:")
    loss_weights = config['loss_weights']
    print(f"   Goal Weight: {loss_weights['goal_weight']:.3f}")
    print(f"   Safety Weight: {loss_weights['safety_weight']:.1f}")
    print(f"   Alpha Reg Weight: {loss_weights['alpha_reg_weight']:.3f} (GATED)")
    print(f"   Margin Reg Weight: {loss_weights['margin_reg_weight']:.3f} (NEW)")
    print(f"   Acceleration Loss Weight: {loss_weights['acceleration_loss_weight']:.3f}")
    print(f"   Jerk Loss Weight: {loss_weights['jerk_loss_weight']:.3f}")
    
    # Create trainer
    print("\n🎓 Initializing BPTT trainer with dual innovations...")
    trainer = BPTTTrainer(
        env=env,
        policy_network=policy_network,
        cbf_network=cbf_network,
        config=config
    )
    print("✅ Dual innovation trainer initialized")
    
    # Start training
    print(f"\n🏃 Starting adaptive margin training for {config['training']['training_steps']} steps...")
    print("💡 Focus: Learning dynamic safety margins for optimal efficiency-safety balance")
    
    try:
        trainer.train()
        print("\n🎉 Adaptive margin training completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Adaptive margin training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
