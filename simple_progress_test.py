#!/usr/bin/env python3
"""
簡單測試進度獎勵功能
"""

import torch
import yaml
import numpy as np
from gcbfplus.env import DoubleIntegratorEnv
from gcbfplus.policy import BPTTPolicy, create_policy_from_config
from gcbfplus.trainer.bptt_trainer import BPTTTrainer

def test_progress_reward():
    """測試進度獎勵功能"""
    print("🧪 測試進度獎勵功能")
    
    # 簡單配置
    config = {
        'env': {
            'num_agents': 4,
            'area_size': 2.0,
            'car_radius': 0.1,
            'comm_radius': 0.5,
            'dt': 0.05,
            'obstacles': {'enabled': False}
        },
        'loss_weights': {
            'goal_weight': 1.0,
            'safety_weight': 2.0,
            'control_weight': 0.1,
            'progress_weight': 0.2  # 測試進度權重
        },
        'training': {
            'training_steps': 10,  # 很短的測試
            'horizon_length': 5,
            'learning_rate': 0.01
        },
        'networks': {
            'policy': {}
        }
    }
    
    # 創建環境
    env = DoubleIntegratorEnv(config['env'])
    print(f"✅ 環境創建成功: {env.observation_shape}")
    
    # 創建簡單策略網絡
    obs_shape = env.observation_shape
    action_shape = env.action_shape
    
    policy_config = {
        'perception': {
            'use_vision': False,
            'input_dim': obs_shape[-1],
            'output_dim': 64,
            'hidden_dims': [64, 64],
            'activation': 'relu'
        },
        'memory': {
            'hidden_dim': 64,
            'num_layers': 1
        },
        'policy_head': {
            'output_dim': action_shape[-1],
            'hidden_dims': [64],
            'activation': 'relu',
            'predict_alpha': True
        }
    }
    
    policy_network = create_policy_from_config(policy_config)
    print(f"✅ 策略網絡創建成功")
    
    # 創建訓練器
    trainer_config = {
        'horizon_length': config['training']['horizon_length'],
        'learning_rate': config['training']['learning_rate'],
        'training_steps': config['training']['training_steps'],
        'goal_weight': config['loss_weights']['goal_weight'],
        'safety_weight': config['loss_weights']['safety_weight'],
        'control_weight': config['loss_weights']['control_weight'],
        'progress_weight': config['loss_weights']['progress_weight'],  # 關鍵：進度權重
        'jerk_weight': 0.02,
        'alpha_reg_weight': 0.01,
        'cbf_alpha': 1.0,
        'device': 'cpu',
        'log_interval': 5,
        'save_interval': 10
    }
    
    trainer = BPTTTrainer(
        env=env,
        policy_network=policy_network,
        cbf_network=None,
        config=trainer_config
    )
    
    print(f"✅ 訓練器創建成功")
    print(f"📊 進度權重: {trainer.progress_weight}")
    
    # 測試進度損失計算
    print("\n🔍 測試進度損失計算...")
    
    # 創建模擬軌迹
    state1 = env.reset()
    state2 = env.reset()
    
    # 手動修改位置以模擬進度
    state2.positions = state1.positions + torch.tensor([[[0.1, 0.1]]] * config['env']['num_agents'])
    
    trajectory_states = [state1, state2]
    
    try:
        progress_loss = trainer._calculate_progress_loss(trajectory_states)
        print(f"✅ 進度損失計算成功: {progress_loss.item():.6f}")
        
        if progress_loss.requires_grad:
            print("✅ 進度損失支持梯度計算")
        else:
            print("⚠️ 進度損失不支持梯度計算")
            
    except Exception as e:
        print(f"❌ 進度損失計算失敗: {e}")
        return False
    
    # 運行一個超短的訓練步驟
    print("\n🏃 運行短訓練測試...")
    
    try:
        # 設置保存目錄
        import os
        save_dir = "logs/progress_test"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        trainer.model_dir = os.path.join(save_dir, "models")
        if not os.path.exists(trainer.model_dir):
            os.makedirs(trainer.model_dir, exist_ok=True)
        
        # 只運行幾步訓練來測試
        original_steps = trainer.training_steps
        trainer.training_steps = 3
        
        trainer.train()
        
        print("✅ 短訓練測試成功")
        return True
        
    except Exception as e:
        print(f"❌ 短訓練測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函數"""
    print("🎯 戰略性改進 - 進度獎勵測試")
    print("=" * 50)
    
    success = test_progress_reward()
    
    if success:
        print("\n🎉 進度獎勵功能測試成功！")
        print("✅ 基於潛力的獎勵塑形已實施")
        print("✅ 系統準備好進行課程學習")
        print("\n🚀 下一步：運行完整的課程學習實驗")
        print("   python run_curriculum_experiments.bat")
    else:
        print("\n❌ 進度獎勵功能測試失敗")
        print("請檢查實施並修復問題")
    
    return success

if __name__ == "__main__":
    main()
 
"""
簡單測試進度獎勵功能
"""

import torch
import yaml
import numpy as np
from gcbfplus.env import DoubleIntegratorEnv
from gcbfplus.policy import BPTTPolicy, create_policy_from_config
from gcbfplus.trainer.bptt_trainer import BPTTTrainer

def test_progress_reward():
    """測試進度獎勵功能"""
    print("🧪 測試進度獎勵功能")
    
    # 簡單配置
    config = {
        'env': {
            'num_agents': 4,
            'area_size': 2.0,
            'car_radius': 0.1,
            'comm_radius': 0.5,
            'dt': 0.05,
            'obstacles': {'enabled': False}
        },
        'loss_weights': {
            'goal_weight': 1.0,
            'safety_weight': 2.0,
            'control_weight': 0.1,
            'progress_weight': 0.2  # 測試進度權重
        },
        'training': {
            'training_steps': 10,  # 很短的測試
            'horizon_length': 5,
            'learning_rate': 0.01
        },
        'networks': {
            'policy': {}
        }
    }
    
    # 創建環境
    env = DoubleIntegratorEnv(config['env'])
    print(f"✅ 環境創建成功: {env.observation_shape}")
    
    # 創建簡單策略網絡
    obs_shape = env.observation_shape
    action_shape = env.action_shape
    
    policy_config = {
        'perception': {
            'use_vision': False,
            'input_dim': obs_shape[-1],
            'output_dim': 64,
            'hidden_dims': [64, 64],
            'activation': 'relu'
        },
        'memory': {
            'hidden_dim': 64,
            'num_layers': 1
        },
        'policy_head': {
            'output_dim': action_shape[-1],
            'hidden_dims': [64],
            'activation': 'relu',
            'predict_alpha': True
        }
    }
    
    policy_network = create_policy_from_config(policy_config)
    print(f"✅ 策略網絡創建成功")
    
    # 創建訓練器
    trainer_config = {
        'horizon_length': config['training']['horizon_length'],
        'learning_rate': config['training']['learning_rate'],
        'training_steps': config['training']['training_steps'],
        'goal_weight': config['loss_weights']['goal_weight'],
        'safety_weight': config['loss_weights']['safety_weight'],
        'control_weight': config['loss_weights']['control_weight'],
        'progress_weight': config['loss_weights']['progress_weight'],  # 關鍵：進度權重
        'jerk_weight': 0.02,
        'alpha_reg_weight': 0.01,
        'cbf_alpha': 1.0,
        'device': 'cpu',
        'log_interval': 5,
        'save_interval': 10
    }
    
    trainer = BPTTTrainer(
        env=env,
        policy_network=policy_network,
        cbf_network=None,
        config=trainer_config
    )
    
    print(f"✅ 訓練器創建成功")
    print(f"📊 進度權重: {trainer.progress_weight}")
    
    # 測試進度損失計算
    print("\n🔍 測試進度損失計算...")
    
    # 創建模擬軌迹
    state1 = env.reset()
    state2 = env.reset()
    
    # 手動修改位置以模擬進度
    state2.positions = state1.positions + torch.tensor([[[0.1, 0.1]]] * config['env']['num_agents'])
    
    trajectory_states = [state1, state2]
    
    try:
        progress_loss = trainer._calculate_progress_loss(trajectory_states)
        print(f"✅ 進度損失計算成功: {progress_loss.item():.6f}")
        
        if progress_loss.requires_grad:
            print("✅ 進度損失支持梯度計算")
        else:
            print("⚠️ 進度損失不支持梯度計算")
            
    except Exception as e:
        print(f"❌ 進度損失計算失敗: {e}")
        return False
    
    # 運行一個超短的訓練步驟
    print("\n🏃 運行短訓練測試...")
    
    try:
        # 設置保存目錄
        import os
        save_dir = "logs/progress_test"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        trainer.model_dir = os.path.join(save_dir, "models")
        if not os.path.exists(trainer.model_dir):
            os.makedirs(trainer.model_dir, exist_ok=True)
        
        # 只運行幾步訓練來測試
        original_steps = trainer.training_steps
        trainer.training_steps = 3
        
        trainer.train()
        
        print("✅ 短訓練測試成功")
        return True
        
    except Exception as e:
        print(f"❌ 短訓練測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函數"""
    print("🎯 戰略性改進 - 進度獎勵測試")
    print("=" * 50)
    
    success = test_progress_reward()
    
    if success:
        print("\n🎉 進度獎勵功能測試成功！")
        print("✅ 基於潛力的獎勵塑形已實施")
        print("✅ 系統準備好進行課程學習")
        print("\n🚀 下一步：運行完整的課程學習實驗")
        print("   python run_curriculum_experiments.bat")
    else:
        print("\n❌ 進度獎勵功能測試失敗")
        print("請檢查實施並修復問題")
    
    return success

if __name__ == "__main__":
    main()
 
 
 
 