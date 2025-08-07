#!/usr/bin/env python3
"""
簡單的訓練測試
"""

import sys
import traceback

def test_imports():
    """測試所有必要的imports"""
    print("🔍 測試導入...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        import yaml
        print("✅ YAML")
        
        from gcbfplus.env import DoubleIntegratorEnv
        print("✅ DoubleIntegratorEnv")
        
        from gcbfplus.policy import create_policy_from_config
        print("✅ Policy")
        
        from gcbfplus.trainer.bptt_trainer import BPTTTrainer
        print("✅ BPTTTrainer")
        
        return True
        
    except Exception as e:
        print(f"❌ 導入失敗: {e}")
        traceback.print_exc()
        return False

def test_config_and_env():
    """測試配置和環境"""
    print("\n🏗️ 測試配置和環境...")
    
    try:
        import yaml
        
        # 加載配置
        with open('config/simple_collaboration_pretrain.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("✅ 配置加載成功")
        
        # 創建環境
        from gcbfplus.env import DoubleIntegratorEnv
        env_config = config['env']
        env = DoubleIntegratorEnv(env_config)
        print(f"✅ 環境創建成功: {env.observation_shape}")
        
        # 移動到CPU
        env = env.to('cpu')
        print("✅ 環境移動到CPU成功")
        
        return True, config, env
        
    except Exception as e:
        print(f"❌ 配置或環境失敗: {e}")
        traceback.print_exc()
        return False, None, None

def test_policy_creation(config, env):
    """測試策略創建"""
    print("\n🧠 測試策略創建...")
    
    try:
        from gcbfplus.policy import create_policy_from_config
        
        # 獲取配置
        policy_config = config.get('networks', {}).get('policy', {})
        obs_shape = env.observation_shape
        action_shape = env.action_shape
        
        print(f"📊 觀測形狀: {obs_shape}")
        print(f"📊 動作形狀: {action_shape}")
        print(f"📊 策略配置: {policy_config}")
        
        # 如果沒有策略配置，創建默認的
        if not policy_config:
            print("⚠️ 沒有策略配置，創建默認配置")
            policy_config = {
                'perception': {
                    'use_vision': False,
                    'input_dim': obs_shape[-1],
                    'output_dim': 128,
                    'hidden_dims': [256, 256],
                    'activation': 'relu'
                },
                'memory': {
                    'hidden_dim': 128,
                    'num_layers': 1
                },
                'policy_head': {
                    'output_dim': action_shape[-1],
                    'hidden_dims': [128],
                    'activation': 'relu',
                    'predict_alpha': True
                }
            }
        
        # 創建策略
        policy_network = create_policy_from_config(policy_config)
        print("✅ 策略網絡創建成功")
        
        return True, policy_network
        
    except Exception as e:
        print(f"❌ 策略創建失敗: {e}")
        traceback.print_exc()
        return False, None

def main():
    """主測試流程"""
    print("🚀 簡單訓練測試")
    print("=" * 50)
    
    # 1. 測試導入
    if not test_imports():
        print("❌ 導入測試失敗，退出")
        return False
    
    # 2. 測試配置和環境
    success, config, env = test_config_and_env()
    if not success:
        print("❌ 配置/環境測試失敗，退出")
        return False
    
    # 3. 測試策略創建
    success, policy = test_policy_creation(config, env)
    if not success:
        print("❌ 策略創建測試失敗，退出")
        return False
    
    print("\n🎉 所有基礎測試通過！")
    print("✅ 可以進行訓練")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n🎯 準備運行完整訓練...")
        else:
            print("\n❌ 測試失敗")
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 意外錯誤: {e}")
        traceback.print_exc()
        sys.exit(1)
 
"""
簡單的訓練測試
"""

import sys
import traceback

def test_imports():
    """測試所有必要的imports"""
    print("🔍 測試導入...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        import yaml
        print("✅ YAML")
        
        from gcbfplus.env import DoubleIntegratorEnv
        print("✅ DoubleIntegratorEnv")
        
        from gcbfplus.policy import create_policy_from_config
        print("✅ Policy")
        
        from gcbfplus.trainer.bptt_trainer import BPTTTrainer
        print("✅ BPTTTrainer")
        
        return True
        
    except Exception as e:
        print(f"❌ 導入失敗: {e}")
        traceback.print_exc()
        return False

def test_config_and_env():
    """測試配置和環境"""
    print("\n🏗️ 測試配置和環境...")
    
    try:
        import yaml
        
        # 加載配置
        with open('config/simple_collaboration_pretrain.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("✅ 配置加載成功")
        
        # 創建環境
        from gcbfplus.env import DoubleIntegratorEnv
        env_config = config['env']
        env = DoubleIntegratorEnv(env_config)
        print(f"✅ 環境創建成功: {env.observation_shape}")
        
        # 移動到CPU
        env = env.to('cpu')
        print("✅ 環境移動到CPU成功")
        
        return True, config, env
        
    except Exception as e:
        print(f"❌ 配置或環境失敗: {e}")
        traceback.print_exc()
        return False, None, None

def test_policy_creation(config, env):
    """測試策略創建"""
    print("\n🧠 測試策略創建...")
    
    try:
        from gcbfplus.policy import create_policy_from_config
        
        # 獲取配置
        policy_config = config.get('networks', {}).get('policy', {})
        obs_shape = env.observation_shape
        action_shape = env.action_shape
        
        print(f"📊 觀測形狀: {obs_shape}")
        print(f"📊 動作形狀: {action_shape}")
        print(f"📊 策略配置: {policy_config}")
        
        # 如果沒有策略配置，創建默認的
        if not policy_config:
            print("⚠️ 沒有策略配置，創建默認配置")
            policy_config = {
                'perception': {
                    'use_vision': False,
                    'input_dim': obs_shape[-1],
                    'output_dim': 128,
                    'hidden_dims': [256, 256],
                    'activation': 'relu'
                },
                'memory': {
                    'hidden_dim': 128,
                    'num_layers': 1
                },
                'policy_head': {
                    'output_dim': action_shape[-1],
                    'hidden_dims': [128],
                    'activation': 'relu',
                    'predict_alpha': True
                }
            }
        
        # 創建策略
        policy_network = create_policy_from_config(policy_config)
        print("✅ 策略網絡創建成功")
        
        return True, policy_network
        
    except Exception as e:
        print(f"❌ 策略創建失敗: {e}")
        traceback.print_exc()
        return False, None

def main():
    """主測試流程"""
    print("🚀 簡單訓練測試")
    print("=" * 50)
    
    # 1. 測試導入
    if not test_imports():
        print("❌ 導入測試失敗，退出")
        return False
    
    # 2. 測試配置和環境
    success, config, env = test_config_and_env()
    if not success:
        print("❌ 配置/環境測試失敗，退出")
        return False
    
    # 3. 測試策略創建
    success, policy = test_policy_creation(config, env)
    if not success:
        print("❌ 策略創建測試失敗，退出")
        return False
    
    print("\n🎉 所有基礎測試通過！")
    print("✅ 可以進行訓練")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n🎯 準備運行完整訓練...")
        else:
            print("\n❌ 測試失敗")
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 意外錯誤: {e}")
        traceback.print_exc()
        sys.exit(1)
 
 
 
 