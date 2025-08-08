#!/usr/bin/env python3
"""
Smooth Control Training Script

Train a BPTT policy with control regularization to produce smooth, stable trajectories.
Focus on acceleration and jerk penalties to reduce erratic control behavior.
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
    """Main training function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train BPTT policy with control regularization")
    parser.add_argument('--config', type=str, default='config/smooth_control_training.yaml',
                       help='Path to configuration file')
    args = parser.parse_args()
    
    print("🚀 Starting Smooth Control Training")
    print("=" * 60)
    print("🎯 Hypothesis: Adding acceleration and jerk penalties will")
    print("   produce smoother, more stable trajectories with better")
    print("   obstacle avoidance and reduced oscillations.")
    print("=" * 60)
    
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
    env = env.to(device)
    print(f"✅ Environment created: {config['env']['num_agents']} agents")
    
    # Create networks
    print("🏗️ Creating policy network...")
    policy_network = create_policy_network(config)
    policy_network = policy_network.to(device)
    print("✅ Policy network created")
    
    print("🛡️ Creating CBF safety layer...")
    cbf_network = create_cbf_network(config)
    cbf_network = cbf_network.to(device)
    print("✅ CBF safety layer created")
    
    # Load pretrained model if available
    pretrained_path = "logs/fresh_gpu_safety_gated/models/10000/policy.pt"
    if os.path.exists(pretrained_path):
        print(f"📦 Loading pretrained weights from: {pretrained_path}")
        try:
            pretrained_state = torch.load(pretrained_path, map_location=device, weights_only=True)
            missing_keys, unexpected_keys = policy_network.load_state_dict(pretrained_state, strict=False)
            
            if missing_keys:
                print(f"⚠️ Missing keys: {len(missing_keys)} (expected for architecture differences)")
            if unexpected_keys:
                print(f"⚠️ Unexpected keys: {len(unexpected_keys)} (expected for architecture differences)")
            
            print("✅ Pretrained weights loaded successfully")
        except Exception as e:
            print(f"⚠️ Could not load pretrained weights: {e}")
            print("   Proceeding with random initialization...")
    else:
        print("ℹ️ No pretrained model found, using random initialization")
    
    # Display loss weights
    print("\n📊 Loss Weight Configuration:")
    loss_weights = config['loss_weights']
    print(f"   Goal Weight: {loss_weights['goal_weight']:.3f} (reduced for smooth learning)")
    print(f"   Safety Weight: {loss_weights['safety_weight']:.1f} (kept high)")
    print(f"   Acceleration Loss Weight: {loss_weights['acceleration_loss_weight']:.3f} (NEW)")
    print(f"   Jerk Loss Weight: {loss_weights['jerk_loss_weight']:.3f} (enhanced)")
    print(f"   Control Weight: {loss_weights['control_weight']:.3f}")
    print(f"   Alpha Reg Weight: {loss_weights['alpha_reg_weight']:.3f}")
    
    # Create trainer
    print("\n🎓 Initializing BPTT trainer...")
    trainer = BPTTTrainer(
        env=env,
        policy_network=policy_network,
        cbf_network=cbf_network,
        config=config
    )
    print("✅ Trainer initialized")
    
    # Start training
    print(f"\n🏃 Starting training for {config['training']['training_steps']} steps...")
    print("💡 Focus: Learning smooth control with regularization penalties")
    
    try:
        trainer.train()
        print("\n🎉 Training completed successfully!")
        
        # Save final model
        final_step = config['training']['training_steps']
        trainer.save_models(final_step)
        print(f"💾 Final model saved at step {final_step}")
        
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        # Save current model
        current_step = getattr(trainer, '_current_step', 0)
        if current_step > 0:
            trainer.save_models(current_step)
            print(f"💾 Model saved at step {current_step}")
    
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Provide next steps
    print(f"\n💡 Next Steps:")
    print(f"   1. Evaluate the trained model:")
    print(f"      python evaluate_with_logging.py \\")
    print(f"         --model-dir {config['log_dir']}/models/{config['training']['training_steps']} \\")
    print(f"         --config config/baseline_evaluation.yaml \\")
    print(f"         --episodes 3 --auto-plot")
    print(f"   2. Compare trajectories with baseline to assess smoothness")
    print(f"   3. Adjust loss weights if needed and retrain")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
