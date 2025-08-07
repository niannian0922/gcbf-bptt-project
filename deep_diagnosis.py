#!/usr/bin/env python3
"""
深入診斷測試 - 重現訓練問題
"""

import torch
import yaml
import os
import traceback

def test_training_pipeline():
    """測試完整的訓練流程"""
    print("🔬 深入診斷 - 測試訓練流程")
    print("=" * 50)
    
    try:
        # 1. 加載配置
        print("1️⃣ 加載配置...")
        with open('config/simple_collaboration_pretrain.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("   ✅ 配置加載成功")
        
        # 2. 創建環境
        print("\n2️⃣ 創建環境...")
        from gcbfplus.env import DoubleIntegratorEnv
        env = DoubleIntegratorEnv(config['env'])
        env = env.to('cpu')
        print(f"   ✅ 環境創建成功: {env.observation_shape}")
        
        # 3. 測試環境重置和觀測
        print("\n3️⃣ 測試環境重置...")
        state = env.reset()
        obs = env.get_observation(state)
        print(f"   📊 觀測形狀: {obs.shape}")
        print(f"   📊 觀測內容示例: {obs[0, :3]}")  # 第一個智能體的前3維
        
        # 4. 創建策略網絡
        print("\n4️⃣ 創建策略網絡...")
        from gcbfplus.policy import create_policy_from_config
        policy_config = config['networks']['policy']
        
        print(f"   📋 策略配置檢查:")
        print(f"     - 輸入維度: {policy_config['perception']['input_dim']}")
        print(f"     - 隱藏維度: {policy_config['perception']['hidden_dim']}")
        print(f"     - 輸出維度: {policy_config['policy_head']['output_dim']}")
        
        policy = create_policy_from_config(policy_config)
        print("   ✅ 策略網絡創建成功")
        
        # 5. 測試前向傳播
        print("\n5️⃣ 測試前向傳播...")
        batch_obs = obs.unsqueeze(0)  # [1, 6, 9]
        print(f"   📊 批量觀測形狀: {batch_obs.shape}")
        
        with torch.no_grad():
            actions, alpha = policy(batch_obs)
            print(f"   📊 動作輸出: {actions.shape}")
            print(f"   📊 Alpha輸出: {alpha.shape}")
            print(f"   📊 動作值示例: {actions[0, 0]}")  # 第一個智能體的動作
        
        print("   ✅ 前向傳播測試成功")
        
        # 6. 創建訓練器（簡化配置）
        print("\n6️⃣ 創建訓練器...")
        from gcbfplus.trainer.bptt_trainer import BPTTTrainer
        
        # 最小化訓練配置
        training_config = {
            'horizon_length': 5,  # 很短的時間範圍
            'learning_rate': 0.01,
            'training_steps': 2,  # 只訓練2步
            'batch_size': 2,      # 小批量
            'device': 'cpu',
            'log_interval': 1,
            'save_interval': 1,
            'cbf_alpha': 1.0,
            'goal_weight': 1.0,
            'safety_weight': 1.0,
            'control_weight': 0.1,
            'jerk_weight': 0.01,
            'alpha_reg_weight': 0.01,
            'progress_weight': 0.1
        }
        
        print(f"   📋 訓練配置: {training_config}")
        
        trainer = BPTTTrainer(
            env=env,
            policy_network=policy,
            cbf_network=None,  # 不使用CBF以簡化
            config=training_config
        )
        
        # 設置保存目錄
        log_dir = "logs/deep_diagnosis"
        os.makedirs(log_dir, exist_ok=True)
        trainer.log_dir = log_dir
        trainer.model_dir = os.path.join(log_dir, 'models')
        os.makedirs(trainer.model_dir, exist_ok=True)
        
        print("   ✅ 訓練器創建成功")
        
        # 7. 嘗試單步訓練
        print("\n7️⃣ 執行微型訓練...")
        print("   🔄 開始2步訓練...")
        
        try:
            trainer.train()
            print("   ✅ 訓練完成！")
            
            # 檢查生成的文件
            if os.path.exists(trainer.model_dir):
                files = os.listdir(trainer.model_dir)
                if files:
                    print(f"   📁 生成的文件: {files}")
                    return True, "訓練成功"
                else:
                    return False, "訓練完成但未生成文件"
            else:
                return False, "模型目錄未創建"
                
        except Exception as train_error:
            print(f"   ❌ 訓練失敗: {train_error}")
            traceback.print_exc()
            return False, f"訓練錯誤: {train_error}"
            
    except Exception as e:
        print(f"❌ 流程測試失敗: {e}")
        traceback.print_exc()
        return False, f"流程錯誤: {e}"

def main():
    """主診斷函數"""
    print("🚀 深入診斷開始")
    print("目標: 找出訓練無法生成模型的確切原因")
    print()
    
    success, message = test_training_pipeline()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 診斷結果: 成功！")
        print("✅ 訓練流程正常工作")
        print("💡 之前的實驗失敗可能是因為:")
        print("   1. 訓練步數設置過多，時間太長")
        print("   2. 進程被用戶中斷")
        print("   3. 系統資源問題")
        print("\n🚀 建議: 可以運行完整實驗，確保有足夠時間")
    else:
        print("❌ 診斷結果: 發現問題！")
        print(f"🔍 問題詳情: {message}")
        print("💡 需要修復此問題才能繼續")
    
    print(f"\n📊 診斷詳情: {message}")

if __name__ == "__main__":
    main()
 
"""
深入診斷測試 - 重現訓練問題
"""

import torch
import yaml
import os
import traceback

def test_training_pipeline():
    """測試完整的訓練流程"""
    print("🔬 深入診斷 - 測試訓練流程")
    print("=" * 50)
    
    try:
        # 1. 加載配置
        print("1️⃣ 加載配置...")
        with open('config/simple_collaboration_pretrain.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("   ✅ 配置加載成功")
        
        # 2. 創建環境
        print("\n2️⃣ 創建環境...")
        from gcbfplus.env import DoubleIntegratorEnv
        env = DoubleIntegratorEnv(config['env'])
        env = env.to('cpu')
        print(f"   ✅ 環境創建成功: {env.observation_shape}")
        
        # 3. 測試環境重置和觀測
        print("\n3️⃣ 測試環境重置...")
        state = env.reset()
        obs = env.get_observation(state)
        print(f"   📊 觀測形狀: {obs.shape}")
        print(f"   📊 觀測內容示例: {obs[0, :3]}")  # 第一個智能體的前3維
        
        # 4. 創建策略網絡
        print("\n4️⃣ 創建策略網絡...")
        from gcbfplus.policy import create_policy_from_config
        policy_config = config['networks']['policy']
        
        print(f"   📋 策略配置檢查:")
        print(f"     - 輸入維度: {policy_config['perception']['input_dim']}")
        print(f"     - 隱藏維度: {policy_config['perception']['hidden_dim']}")
        print(f"     - 輸出維度: {policy_config['policy_head']['output_dim']}")
        
        policy = create_policy_from_config(policy_config)
        print("   ✅ 策略網絡創建成功")
        
        # 5. 測試前向傳播
        print("\n5️⃣ 測試前向傳播...")
        batch_obs = obs.unsqueeze(0)  # [1, 6, 9]
        print(f"   📊 批量觀測形狀: {batch_obs.shape}")
        
        with torch.no_grad():
            actions, alpha = policy(batch_obs)
            print(f"   📊 動作輸出: {actions.shape}")
            print(f"   📊 Alpha輸出: {alpha.shape}")
            print(f"   📊 動作值示例: {actions[0, 0]}")  # 第一個智能體的動作
        
        print("   ✅ 前向傳播測試成功")
        
        # 6. 創建訓練器（簡化配置）
        print("\n6️⃣ 創建訓練器...")
        from gcbfplus.trainer.bptt_trainer import BPTTTrainer
        
        # 最小化訓練配置
        training_config = {
            'horizon_length': 5,  # 很短的時間範圍
            'learning_rate': 0.01,
            'training_steps': 2,  # 只訓練2步
            'batch_size': 2,      # 小批量
            'device': 'cpu',
            'log_interval': 1,
            'save_interval': 1,
            'cbf_alpha': 1.0,
            'goal_weight': 1.0,
            'safety_weight': 1.0,
            'control_weight': 0.1,
            'jerk_weight': 0.01,
            'alpha_reg_weight': 0.01,
            'progress_weight': 0.1
        }
        
        print(f"   📋 訓練配置: {training_config}")
        
        trainer = BPTTTrainer(
            env=env,
            policy_network=policy,
            cbf_network=None,  # 不使用CBF以簡化
            config=training_config
        )
        
        # 設置保存目錄
        log_dir = "logs/deep_diagnosis"
        os.makedirs(log_dir, exist_ok=True)
        trainer.log_dir = log_dir
        trainer.model_dir = os.path.join(log_dir, 'models')
        os.makedirs(trainer.model_dir, exist_ok=True)
        
        print("   ✅ 訓練器創建成功")
        
        # 7. 嘗試單步訓練
        print("\n7️⃣ 執行微型訓練...")
        print("   🔄 開始2步訓練...")
        
        try:
            trainer.train()
            print("   ✅ 訓練完成！")
            
            # 檢查生成的文件
            if os.path.exists(trainer.model_dir):
                files = os.listdir(trainer.model_dir)
                if files:
                    print(f"   📁 生成的文件: {files}")
                    return True, "訓練成功"
                else:
                    return False, "訓練完成但未生成文件"
            else:
                return False, "模型目錄未創建"
                
        except Exception as train_error:
            print(f"   ❌ 訓練失敗: {train_error}")
            traceback.print_exc()
            return False, f"訓練錯誤: {train_error}"
            
    except Exception as e:
        print(f"❌ 流程測試失敗: {e}")
        traceback.print_exc()
        return False, f"流程錯誤: {e}"

def main():
    """主診斷函數"""
    print("🚀 深入診斷開始")
    print("目標: 找出訓練無法生成模型的確切原因")
    print()
    
    success, message = test_training_pipeline()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 診斷結果: 成功！")
        print("✅ 訓練流程正常工作")
        print("💡 之前的實驗失敗可能是因為:")
        print("   1. 訓練步數設置過多，時間太長")
        print("   2. 進程被用戶中斷")
        print("   3. 系統資源問題")
        print("\n🚀 建議: 可以運行完整實驗，確保有足夠時間")
    else:
        print("❌ 診斷結果: 發現問題！")
        print(f"🔍 問題詳情: {message}")
        print("💡 需要修復此問題才能繼續")
    
    print(f"\n📊 診斷詳情: {message}")

if __name__ == "__main__":
    main()
 
 
 
 