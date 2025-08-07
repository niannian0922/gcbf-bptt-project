#!/usr/bin/env python3
"""
診斷訓練問題
"""

import yaml
import torch
import traceback
from gcbfplus.env import DoubleIntegratorEnv

def test_config_loading():
    """測試配置文件加載"""
    print("🔍 測試配置文件加載...")
    
    try:
        # 測試預訓練配置
        with open('config/simple_collaboration_pretrain.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print("✅ 預訓練配置加載成功")
        print(f"📊 環境配置: {config.get('env', {})}")
        print(f"📊 網絡配置: {config.get('networks', {})}")
        print(f"📊 損失權重: {config.get('loss_weights', {})}")
        
        return config
        
    except Exception as e:
        print(f"❌ 配置加載失敗: {e}")
        traceback.print_exc()
        return None

def test_environment_creation(config):
    """測試環境創建"""
    print("\n🏗️ 測試環境創建...")
    
    try:
        env_config = config['env']
        env = DoubleIntegratorEnv(env_config)
        
        print(f"✅ 環境創建成功")
        print(f"📊 觀測形狀: {env.observation_shape}")
        print(f"📊 動作形狀: {env.action_shape}")
        print(f"📊 智能體數量: {env.num_agents}")
        print(f"📊 障礙物: {env_config.get('obstacles', {})}")
        
        return env
        
    except Exception as e:
        print(f"❌ 環境創建失敗: {e}")
        traceback.print_exc()
        return None

def test_simple_training_command():
    """測試簡單的訓練命令"""
    print("\n🏃 測試簡單訓練命令...")
    
    import subprocess
    
    # 創建一個非常簡短的測試命令
    cmd = "python train_bptt.py --config config/simple_collaboration_pretrain.yaml --device cpu --log_dir logs/quick_test --seed 42"
    
    print(f"📝 測試命令: {cmd}")
    
    try:
        # 只運行30秒看看是否有錯誤
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ 訓練腳本啟動成功（30秒測試）")
        else:
            print("❌ 訓練腳本失敗")
            print(f"錯誤輸出:\n{result.stderr}")
            
        if result.stdout:
            print(f"標準輸出:\n{result.stdout}")
            
    except subprocess.TimeoutExpired:
        print("⏰ 30秒測試完成 - 腳本正常啟動")
    except Exception as e:
        print(f"❌ 測試失敗: {e}")

def main():
    """主診斷流程"""
    print("🔧 課程學習訓練問題診斷")
    print("=" * 60)
    
    # 1. 測試配置加載
    config = test_config_loading()
    if not config:
        print("❌ 配置加載失敗，停止診斷")
        return
    
    # 2. 測試環境創建
    env = test_environment_creation(config)
    if not env:
        print("❌ 環境創建失敗，停止診斷")
        return
    
    # 3. 測試訓練命令
    test_simple_training_command()
    
    print("\n" + "=" * 60)
    print("🔍 診斷完成")
    
    # 提供解決方案
    print("\n🚀 建議解決方案：")
    print("1. 確保所有依賴包已安裝")
    print("2. 檢查配置文件格式")
    print("3. 運行簡化版本的訓練")
    print("4. 檢查設備兼容性（CPU vs GPU）")

if __name__ == "__main__":
    main()
 
"""
診斷訓練問題
"""

import yaml
import torch
import traceback
from gcbfplus.env import DoubleIntegratorEnv

def test_config_loading():
    """測試配置文件加載"""
    print("🔍 測試配置文件加載...")
    
    try:
        # 測試預訓練配置
        with open('config/simple_collaboration_pretrain.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print("✅ 預訓練配置加載成功")
        print(f"📊 環境配置: {config.get('env', {})}")
        print(f"📊 網絡配置: {config.get('networks', {})}")
        print(f"📊 損失權重: {config.get('loss_weights', {})}")
        
        return config
        
    except Exception as e:
        print(f"❌ 配置加載失敗: {e}")
        traceback.print_exc()
        return None

def test_environment_creation(config):
    """測試環境創建"""
    print("\n🏗️ 測試環境創建...")
    
    try:
        env_config = config['env']
        env = DoubleIntegratorEnv(env_config)
        
        print(f"✅ 環境創建成功")
        print(f"📊 觀測形狀: {env.observation_shape}")
        print(f"📊 動作形狀: {env.action_shape}")
        print(f"📊 智能體數量: {env.num_agents}")
        print(f"📊 障礙物: {env_config.get('obstacles', {})}")
        
        return env
        
    except Exception as e:
        print(f"❌ 環境創建失敗: {e}")
        traceback.print_exc()
        return None

def test_simple_training_command():
    """測試簡單的訓練命令"""
    print("\n🏃 測試簡單訓練命令...")
    
    import subprocess
    
    # 創建一個非常簡短的測試命令
    cmd = "python train_bptt.py --config config/simple_collaboration_pretrain.yaml --device cpu --log_dir logs/quick_test --seed 42"
    
    print(f"📝 測試命令: {cmd}")
    
    try:
        # 只運行30秒看看是否有錯誤
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ 訓練腳本啟動成功（30秒測試）")
        else:
            print("❌ 訓練腳本失敗")
            print(f"錯誤輸出:\n{result.stderr}")
            
        if result.stdout:
            print(f"標準輸出:\n{result.stdout}")
            
    except subprocess.TimeoutExpired:
        print("⏰ 30秒測試完成 - 腳本正常啟動")
    except Exception as e:
        print(f"❌ 測試失敗: {e}")

def main():
    """主診斷流程"""
    print("🔧 課程學習訓練問題診斷")
    print("=" * 60)
    
    # 1. 測試配置加載
    config = test_config_loading()
    if not config:
        print("❌ 配置加載失敗，停止診斷")
        return
    
    # 2. 測試環境創建
    env = test_environment_creation(config)
    if not env:
        print("❌ 環境創建失敗，停止診斷")
        return
    
    # 3. 測試訓練命令
    test_simple_training_command()
    
    print("\n" + "=" * 60)
    print("🔍 診斷完成")
    
    # 提供解決方案
    print("\n🚀 建議解決方案：")
    print("1. 確保所有依賴包已安裝")
    print("2. 檢查配置文件格式")
    print("3. 運行簡化版本的訓練")
    print("4. 檢查設備兼容性（CPU vs GPU）")

if __name__ == "__main__":
    main()
 
 
 
 