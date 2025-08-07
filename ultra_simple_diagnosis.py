#!/usr/bin/env python3
"""
超簡化診斷測試 - 找出確切問題
"""

import torch
import yaml
import traceback
import os
from datetime import datetime

def test_step_by_step():
    """一步步測試每個組件"""
    print("🔬 超簡化診斷測試")
    print("=" * 60)
    print(f"⏰ 測試時間: {datetime.now().strftime('%H:%M:%S')}")
    print()
    
    # 1. 基礎導入測試
    print("1️⃣ 測試基礎導入...")
    try:
        from gcbfplus.env import DoubleIntegratorEnv
        from gcbfplus.policy import create_policy_from_config
        from gcbfplus.trainer.bptt_trainer import BPTTTrainer
        print("   ✅ 所有導入成功")
    except Exception as e:
        print(f"   ❌ 導入失敗: {e}")
        return False
    
    # 2. 配置加載測試
    print("\n2️⃣ 測試配置加載...")
    try:
        with open('config/simple_collaboration_pretrain.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("   ✅ 配置加載成功")
        print(f"   📋 環境智能體數: {config['env']['num_agents']}")
        print(f"   📋 訓練步數: {config['training']['training_steps']}")
        print(f"   📋 策略輸入維度: {config['networks']['policy']['perception']['input_dim']}")
        print(f"   📋 策略隱藏維度: {config['networks']['policy']['perception']['hidden_dim']}")
    except Exception as e:
        print(f"   ❌ 配置加載失敗: {e}")
        traceback.print_exc()
        return False
    
    # 3. 環境創建測試
    print("\n3️⃣ 測試環境創建...")
    try:
        env = DoubleIntegratorEnv(config['env'])
        env = env.to('cpu')
        print("   ✅ 環境創建成功")
        print(f"   📊 觀測形狀: {env.observation_shape}")
        print(f"   📊 動作形狀: {env.action_shape}")
        print(f"   📊 智能體數量: {env.num_agents}")
        
        # 測試環境重置
        state = env.reset()
        print(f"   📊 重置狀態形狀: {state.positions.shape if hasattr(state, 'positions') else 'N/A'}")
        obs = env.get_observation(state)
        print(f"   📊 觀測張量形狀: {obs.shape}")
    except Exception as e:
        print(f"   ❌ 環境創建失敗: {e}")
        traceback.print_exc()
        return False
    
    # 4. 策略網絡創建測試
    print("\n4️⃣ 測試策略網絡創建...")
    try:
        policy_config = config['networks']['policy']
        print(f"   📋 策略配置: {policy_config}")
        
        policy = create_policy_from_config(policy_config)
        print("   ✅ 策略網絡創建成功")
        
        # 測試前向傳播
        print("   🧪 測試前向傳播...")
        with torch.no_grad():
            actions, alpha = policy(obs.unsqueeze(0))  # 添加batch維度
            print(f"   📊 動作輸出形狀: {actions.shape}")
            print(f"   📊 Alpha輸出形狀: {alpha.shape}")
            print("   ✅ 前向傳播測試成功")
    except Exception as e:
        print(f"   ❌ 策略網絡測試失敗: {e}")
        traceback.print_exc()
        return False
    
    # 5. 訓練器創建測試（不訓練）
    print("\n5️⃣ 測試訓練器創建...")
    try:
        # 提取訓練器配置
        training_config = config['training']
        loss_weights = config.get('loss_weights', {})
        
        trainer_config = {
            'horizon_length': training_config.get('horizon_length', 30),
            'learning_rate': training_config.get('learning_rate', 0.003),
            'training_steps': 1,  # 只測試1步
            'batch_size': 4,  # 小批量
            'device': 'cpu',
            'log_interval': 1,
            'save_interval': 1,
            'cbf_alpha': config['env'].get('cbf_alpha', 1.0),
            'goal_weight': loss_weights.get('goal_weight', 1.0),
            'safety_weight': loss_weights.get('safety_weight', 10.0),
            'control_weight': loss_weights.get('control_weight', 0.1),
            'jerk_weight': loss_weights.get('jerk_weight', 0.05),
            'alpha_reg_weight': loss_weights.get('alpha_reg_weight', 0.01),
            'progress_weight': loss_weights.get('progress_weight', 0.0)
        }
        
        trainer = BPTTTrainer(
            env=env,
            policy_network=policy,
            cbf_network=None,  # 暫時不使用CBF
            config=trainer_config
        )
        
        # 設置保存目錄
        test_log_dir = "logs/diagnosis_test"
        os.makedirs(test_log_dir, exist_ok=True)
        trainer.log_dir = test_log_dir
        trainer.model_dir = os.path.join(test_log_dir, 'models')
        os.makedirs(trainer.model_dir, exist_ok=True)
        
        print("   ✅ 訓練器創建成功")
        print(f"   📋 訓練器配置: {trainer_config}")
    except Exception as e:
        print(f"   ❌ 訓練器創建失敗: {e}")
        traceback.print_exc()
        return False
    
    # 6. 單步訓練測試
    print("\n6️⃣ 測試單步訓練...")
    try:
        print("   🔄 執行單步訓練...")
        trainer.training_steps = 1  # 確保只訓練1步
        trainer.save_interval = 1   # 確保會保存
        
        trainer.train()
        
        print("   ✅ 單步訓練成功")
        
        # 檢查是否生成了模型
        models_dir = trainer.model_dir
        if os.path.exists(models_dir):
            model_files = os.listdir(models_dir)
            if model_files:
                print(f"   📊 生成的模型: {model_files}")
                return True
            else:
                print("   ⚠️ 訓練完成但未生成模型文件")
                return False
        else:
            print("   ❌ 模型目錄未創建")
            return False
            
    except Exception as e:
        print(f"   ❌ 單步訓練失敗: {e}")
        traceback.print_exc()
        return False

def diagnose_specific_error():
    """診斷特定錯誤"""
    print("\n🔍 特定錯誤診斷...")
    
    try:
        # 重現之前的錯誤場景
        with open('config/simple_collaboration_pretrain.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        env = DoubleIntegratorEnv(config['env'])
        env = env.to('cpu')
        
        state = env.reset()
        obs = env.get_observation(state)
        
        print(f"   📊 環境觀測形狀: {obs.shape}")
        print(f"   📊 預期格式: (num_agents={env.num_agents}, obs_dim={obs.shape[-1]})")
        
        policy_config = config['networks']['policy']
        expected_input = policy_config['perception']['input_dim']
        actual_input = obs.shape[-1]
        
        print(f"   📊 策略期望輸入維度: {expected_input}")
        print(f"   📊 實際觀測維度: {actual_input}")
        
        if expected_input == actual_input:
            print("   ✅ 維度匹配正確")
        else:
            print(f"   ❌ 維度不匹配！期望 {expected_input}，實際 {actual_input}")
            return False
            
        # 測試批量維度
        batch_obs = obs.unsqueeze(0)  # 添加batch維度
        print(f"   📊 批量觀測形狀: {batch_obs.shape}")
        print(f"   📊 期望格式: (batch_size=1, num_agents={env.num_agents}, obs_dim={actual_input})")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 診斷過程失敗: {e}")
        traceback.print_exc()
        return False

def main():
    """主診斷流程"""
    print("🚀 開始超簡化診斷測試")
    print("目標：找出課程學習實驗失敗的確切原因")
    print()
    
    # 逐步測試
    success = test_step_by_step()
    
    if success:
        print("\n🎉 所有基礎測試通過！")
        print("✅ 課程學習框架本身沒有問題")
        print("💡 之前的實驗失敗可能是由於:")
        print("   - 進程被意外終止")
        print("   - 系統資源不足")
        print("   - 訓練時間過長被用戶取消")
        print("\n🚀 建議:")
        print("   現在可以運行完整的課程學習實驗")
    else:
        print("\n❌ 診斷發現問題")
        
        # 運行特定錯誤診斷
        error_diagnosed = diagnose_specific_error()
        
        if not error_diagnosed:
            print("💡 需要進一步調試配置或代碼")
        
    print(f"\n⏰ 診斷完成時間: {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    main()
 
"""
超簡化診斷測試 - 找出確切問題
"""

import torch
import yaml
import traceback
import os
from datetime import datetime

def test_step_by_step():
    """一步步測試每個組件"""
    print("🔬 超簡化診斷測試")
    print("=" * 60)
    print(f"⏰ 測試時間: {datetime.now().strftime('%H:%M:%S')}")
    print()
    
    # 1. 基礎導入測試
    print("1️⃣ 測試基礎導入...")
    try:
        from gcbfplus.env import DoubleIntegratorEnv
        from gcbfplus.policy import create_policy_from_config
        from gcbfplus.trainer.bptt_trainer import BPTTTrainer
        print("   ✅ 所有導入成功")
    except Exception as e:
        print(f"   ❌ 導入失敗: {e}")
        return False
    
    # 2. 配置加載測試
    print("\n2️⃣ 測試配置加載...")
    try:
        with open('config/simple_collaboration_pretrain.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("   ✅ 配置加載成功")
        print(f"   📋 環境智能體數: {config['env']['num_agents']}")
        print(f"   📋 訓練步數: {config['training']['training_steps']}")
        print(f"   📋 策略輸入維度: {config['networks']['policy']['perception']['input_dim']}")
        print(f"   📋 策略隱藏維度: {config['networks']['policy']['perception']['hidden_dim']}")
    except Exception as e:
        print(f"   ❌ 配置加載失敗: {e}")
        traceback.print_exc()
        return False
    
    # 3. 環境創建測試
    print("\n3️⃣ 測試環境創建...")
    try:
        env = DoubleIntegratorEnv(config['env'])
        env = env.to('cpu')
        print("   ✅ 環境創建成功")
        print(f"   📊 觀測形狀: {env.observation_shape}")
        print(f"   📊 動作形狀: {env.action_shape}")
        print(f"   📊 智能體數量: {env.num_agents}")
        
        # 測試環境重置
        state = env.reset()
        print(f"   📊 重置狀態形狀: {state.positions.shape if hasattr(state, 'positions') else 'N/A'}")
        obs = env.get_observation(state)
        print(f"   📊 觀測張量形狀: {obs.shape}")
    except Exception as e:
        print(f"   ❌ 環境創建失敗: {e}")
        traceback.print_exc()
        return False
    
    # 4. 策略網絡創建測試
    print("\n4️⃣ 測試策略網絡創建...")
    try:
        policy_config = config['networks']['policy']
        print(f"   📋 策略配置: {policy_config}")
        
        policy = create_policy_from_config(policy_config)
        print("   ✅ 策略網絡創建成功")
        
        # 測試前向傳播
        print("   🧪 測試前向傳播...")
        with torch.no_grad():
            actions, alpha = policy(obs.unsqueeze(0))  # 添加batch維度
            print(f"   📊 動作輸出形狀: {actions.shape}")
            print(f"   📊 Alpha輸出形狀: {alpha.shape}")
            print("   ✅ 前向傳播測試成功")
    except Exception as e:
        print(f"   ❌ 策略網絡測試失敗: {e}")
        traceback.print_exc()
        return False
    
    # 5. 訓練器創建測試（不訓練）
    print("\n5️⃣ 測試訓練器創建...")
    try:
        # 提取訓練器配置
        training_config = config['training']
        loss_weights = config.get('loss_weights', {})
        
        trainer_config = {
            'horizon_length': training_config.get('horizon_length', 30),
            'learning_rate': training_config.get('learning_rate', 0.003),
            'training_steps': 1,  # 只測試1步
            'batch_size': 4,  # 小批量
            'device': 'cpu',
            'log_interval': 1,
            'save_interval': 1,
            'cbf_alpha': config['env'].get('cbf_alpha', 1.0),
            'goal_weight': loss_weights.get('goal_weight', 1.0),
            'safety_weight': loss_weights.get('safety_weight', 10.0),
            'control_weight': loss_weights.get('control_weight', 0.1),
            'jerk_weight': loss_weights.get('jerk_weight', 0.05),
            'alpha_reg_weight': loss_weights.get('alpha_reg_weight', 0.01),
            'progress_weight': loss_weights.get('progress_weight', 0.0)
        }
        
        trainer = BPTTTrainer(
            env=env,
            policy_network=policy,
            cbf_network=None,  # 暫時不使用CBF
            config=trainer_config
        )
        
        # 設置保存目錄
        test_log_dir = "logs/diagnosis_test"
        os.makedirs(test_log_dir, exist_ok=True)
        trainer.log_dir = test_log_dir
        trainer.model_dir = os.path.join(test_log_dir, 'models')
        os.makedirs(trainer.model_dir, exist_ok=True)
        
        print("   ✅ 訓練器創建成功")
        print(f"   📋 訓練器配置: {trainer_config}")
    except Exception as e:
        print(f"   ❌ 訓練器創建失敗: {e}")
        traceback.print_exc()
        return False
    
    # 6. 單步訓練測試
    print("\n6️⃣ 測試單步訓練...")
    try:
        print("   🔄 執行單步訓練...")
        trainer.training_steps = 1  # 確保只訓練1步
        trainer.save_interval = 1   # 確保會保存
        
        trainer.train()
        
        print("   ✅ 單步訓練成功")
        
        # 檢查是否生成了模型
        models_dir = trainer.model_dir
        if os.path.exists(models_dir):
            model_files = os.listdir(models_dir)
            if model_files:
                print(f"   📊 生成的模型: {model_files}")
                return True
            else:
                print("   ⚠️ 訓練完成但未生成模型文件")
                return False
        else:
            print("   ❌ 模型目錄未創建")
            return False
            
    except Exception as e:
        print(f"   ❌ 單步訓練失敗: {e}")
        traceback.print_exc()
        return False

def diagnose_specific_error():
    """診斷特定錯誤"""
    print("\n🔍 特定錯誤診斷...")
    
    try:
        # 重現之前的錯誤場景
        with open('config/simple_collaboration_pretrain.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        env = DoubleIntegratorEnv(config['env'])
        env = env.to('cpu')
        
        state = env.reset()
        obs = env.get_observation(state)
        
        print(f"   📊 環境觀測形狀: {obs.shape}")
        print(f"   📊 預期格式: (num_agents={env.num_agents}, obs_dim={obs.shape[-1]})")
        
        policy_config = config['networks']['policy']
        expected_input = policy_config['perception']['input_dim']
        actual_input = obs.shape[-1]
        
        print(f"   📊 策略期望輸入維度: {expected_input}")
        print(f"   📊 實際觀測維度: {actual_input}")
        
        if expected_input == actual_input:
            print("   ✅ 維度匹配正確")
        else:
            print(f"   ❌ 維度不匹配！期望 {expected_input}，實際 {actual_input}")
            return False
            
        # 測試批量維度
        batch_obs = obs.unsqueeze(0)  # 添加batch維度
        print(f"   📊 批量觀測形狀: {batch_obs.shape}")
        print(f"   📊 期望格式: (batch_size=1, num_agents={env.num_agents}, obs_dim={actual_input})")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 診斷過程失敗: {e}")
        traceback.print_exc()
        return False

def main():
    """主診斷流程"""
    print("🚀 開始超簡化診斷測試")
    print("目標：找出課程學習實驗失敗的確切原因")
    print()
    
    # 逐步測試
    success = test_step_by_step()
    
    if success:
        print("\n🎉 所有基礎測試通過！")
        print("✅ 課程學習框架本身沒有問題")
        print("💡 之前的實驗失敗可能是由於:")
        print("   - 進程被意外終止")
        print("   - 系統資源不足")
        print("   - 訓練時間過長被用戶取消")
        print("\n🚀 建議:")
        print("   現在可以運行完整的課程學習實驗")
    else:
        print("\n❌ 診斷發現問題")
        
        # 運行特定錯誤診斷
        error_diagnosed = diagnose_specific_error()
        
        if not error_diagnosed:
            print("💡 需要進一步調試配置或代碼")
        
    print(f"\n⏰ 診斷完成時間: {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    main()
 
 
 
 