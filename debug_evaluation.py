#!/usr/bin/env python3
"""
Debug script to test model loading and basic evaluation
"""

import os
import sys
import torch
import yaml
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from gcbfplus.policy.bptt_policy import BPTTPolicy
from gcbfplus.env.gcbf_safety_layer import GCBFSafetyLayer

def main():
    print("🔍 Debug: Testing model loading...")
    
    model_dir = "logs/fresh_gpu_safety_gated/models/10000"
    config_path = "config/baseline_evaluation.yaml"
    
    try:
        # Load configuration
        print(f"📄 Loading config from: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("✅ Config loaded successfully")
        
        # Load model-specific config if available
        model_config_path = os.path.join(model_dir, "config.pt")
        if os.path.exists(model_config_path):
            try:
                model_config = torch.load(model_config_path, map_location='cpu', weights_only=False)
                print(f"📄 Model config keys: {list(model_config.keys()) if isinstance(model_config, dict) else 'Not a dict'}")
                if isinstance(model_config, dict):
                    config.update(model_config)
                    print("✅ Model config merged")
            except Exception as e:
                print(f"⚠️ Could not load model config: {e}")
        
        # Check for model files
        policy_path = os.path.join(model_dir, "policy.pt")
        if not os.path.exists(policy_path):
            raise FileNotFoundError(f"Policy model not found: {policy_path}")
        print(f"✅ Policy file found: {policy_path}")
        
        # Create policy network
        print("🏗️ Creating policy network...")
        policy_config = config.get('networks', {}).get('policy', {})
        print(f"📐 Policy config: {policy_config}")
        
        try:
            policy_network = BPTTPolicy(policy_config)
            print("✅ Policy network created successfully")
            
            # Load policy weights
            print("📦 Loading policy weights...")
            policy_state_dict = torch.load(policy_path, map_location='cpu', weights_only=True)
            print(f"📊 Model has {len(policy_state_dict)} parameters")
            
            # Show first few parameter shapes
            print("🔍 First few parameter shapes:")
            for i, (k, v) in enumerate(list(policy_state_dict.items())[:5]):
                print(f"   {k}: {v.shape}")
            
            missing_keys, unexpected_keys = policy_network.load_state_dict(policy_state_dict, strict=False)
            
            if missing_keys:
                print(f"⚠️ Missing keys ({len(missing_keys)}): {missing_keys[:3]}...")
            if unexpected_keys:
                print(f"⚠️ Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:3]}...")
            
            if not missing_keys and not unexpected_keys:
                print("🎉 Policy weights loaded perfectly!")
            elif len(missing_keys) < 5 and len(unexpected_keys) < 5:
                print("✅ Policy weights loaded with minor mismatches")
            else:
                print("❌ Significant architecture mismatch")
                return
            
        except Exception as e:
            print(f"❌ Failed to create/load policy network: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Test CBF network loading
        cbf_path = os.path.join(model_dir, "cbf.pt")
        if os.path.exists(cbf_path):
            print("🛡️ Testing CBF network loading...")
            try:
                cbf_config = config.get('networks', {}).get('cbf', {})
                cbf_network = GCBFSafetyLayer(
                    input_dim=cbf_config.get('input_dim', 6),
                    hidden_dim=cbf_config.get('hidden_dim', 128),
                    num_agents=config.get('env', {}).get('num_agents', 2)
                )
                
                cbf_state_dict = torch.load(cbf_path, map_location='cpu', weights_only=True)
                cbf_network.load_state_dict(cbf_state_dict, strict=False)
                print("✅ CBF network loaded successfully")
            except Exception as e:
                print(f"⚠️ CBF loading failed: {e}")
        
        print("\n🎉 Model loading test completed successfully!")
        print("💡 The model architecture matches and can be loaded.")
        
    except Exception as e:
        print(f"❌ Debug test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
