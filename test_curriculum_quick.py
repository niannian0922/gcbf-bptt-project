#!/usr/bin/env python3
"""
快速測試課程學習框架
運行小規模實驗以驗證系統工作正常
"""

import subprocess
import sys
import os
import time

def run_command(cmd, description):
    """運行命令並處理錯誤"""
    print(f"\n🔄 {description}")
    print(f"📝 執行命令: {cmd}")
    
    start_time = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    duration = time.time() - start_time
    
    if result.returncode == 0:
        print(f"✅ {description} 成功 ({duration:.1f}s)")
        return True
    else:
        print(f"❌ {description} 失敗")
        print(f"錯誤輸出: {result.stderr}")
        return False

def main():
    """主測試函數"""
    print("🧪 課程學習框架快速測試")
    print("=" * 50)
    
    # 測試配置
    device = "cpu"
    seed = 42
    
    # 創建必要目錄
    os.makedirs("logs/test_curriculum", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # ===== 測試1: 預訓練階段 =====
    print("\n📚 測試階段1: 預訓練（簡化版）")
    
    # 修改預訓練配置為超快速版本
    quick_pretrain_config = """
env:
  num_agents: 4
  area_size: 1.5
  car_radius: 0.05
  comm_radius: 0.5
  mass: 0.1
  dt: 0.05
  obstacles:
    enabled: false

training:
  training_steps: 100  # 超短訓練
  horizon_length: 10
  learning_rate: 0.01
  eval_interval: 50
  save_interval: 50
  
loss_weights:
  goal_weight: 1.0
  safety_weight: 2.0
  control_weight: 0.1
  jerk_weight: 0.02
  alpha_reg_weight: 0.01
  progress_weight: 0.2
"""
    
    # 保存快速配置
    with open("config/quick_pretrain_test.yaml", "w") as f:
        f.write(quick_pretrain_config)
    
    # 運行預訓練
    cmd1 = f"python train_bptt.py --config config/quick_pretrain_test.yaml --device {device} --log_dir logs/test_curriculum/pretrain --seed {seed}"
    
    if not run_command(cmd1, "快速預訓練測試"):
        print("❌ 預訓練測試失敗，停止測試")
        return False
    
    # ===== 測試2: Fine-tuning階段 =====
    print("\n🎯 測試階段2: Fine-tuning（添加障礙物）")
    
    # 修改fine-tuning配置
    quick_finetune_config = """
env:
  num_agents: 4
  area_size: 1.5
  car_radius: 0.05
  comm_radius: 0.5
  mass: 0.1
  dt: 0.05
  obstacles:
    enabled: true
    positions: [[0.0, 0.0]]
    radii: [0.2]

training:
  training_steps: 100  # 超短fine-tuning
  horizon_length: 10
  learning_rate: 0.005
  eval_interval: 50
  save_interval: 50
  
loss_weights:
  goal_weight: 1.0
  safety_weight: 5.0
  control_weight: 0.1
  jerk_weight: 0.02
  alpha_reg_weight: 0.01
  progress_weight: 0.15
"""
    
    # 保存fine-tuning配置
    with open("config/quick_finetune_test.yaml", "w") as f:
        f.write(quick_finetune_config)
    
    # 運行fine-tuning
    cmd2 = f"python train_bptt.py --config config/quick_finetune_test.yaml --device {device} --log_dir logs/test_curriculum/finetune --load_pretrained_model_from logs/test_curriculum/pretrain --seed {seed}"
    
    if not run_command(cmd2, "快速Fine-tuning測試"):
        print("❌ Fine-tuning測試失敗")
        return False
    
    # ===== 測試3: 可視化生成 =====
    print("\n🎨 測試階段3: 可視化生成")
    
    cmd3 = f"python visualize_bptt.py --model_dir logs/test_curriculum/finetune --output results/test_curriculum_result.gif --device {device}"
    
    if not run_command(cmd3, "可視化生成測試"):
        print("⚠️ 可視化測試失敗，但核心功能可能正常")
    
    # ===== 測試結果 =====
    print("\n📊 測試結果總結")
    print("=" * 50)
    
    # 檢查生成的文件
    pretrain_exists = os.path.exists("logs/test_curriculum/pretrain/models")
    finetune_exists = os.path.exists("logs/test_curriculum/finetune/models")
    viz_exists = os.path.exists("results/test_curriculum_result.gif")
    
    print(f"✅ 預訓練模型: {'存在' if pretrain_exists else '不存在'}")
    print(f"✅ Fine-tuning模型: {'存在' if finetune_exists else '不存在'}")
    print(f"✅ 可視化文件: {'存在' if viz_exists else '不存在'}")
    
    if pretrain_exists and finetune_exists:
        print("\n🎉 課程學習框架測試成功！")
        print("🚀 現在可以運行完整的實驗管道：")
        print("   Windows: run_curriculum_experiments.bat")
        print("   Linux/Mac: bash run_curriculum_experiments.sh")
        return True
    else:
        print("\n❌ 課程學習框架測試失敗")
        print("請檢查錯誤日誌並修復問題")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
 
"""
快速測試課程學習框架
運行小規模實驗以驗證系統工作正常
"""

import subprocess
import sys
import os
import time

def run_command(cmd, description):
    """運行命令並處理錯誤"""
    print(f"\n🔄 {description}")
    print(f"📝 執行命令: {cmd}")
    
    start_time = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    duration = time.time() - start_time
    
    if result.returncode == 0:
        print(f"✅ {description} 成功 ({duration:.1f}s)")
        return True
    else:
        print(f"❌ {description} 失敗")
        print(f"錯誤輸出: {result.stderr}")
        return False

def main():
    """主測試函數"""
    print("🧪 課程學習框架快速測試")
    print("=" * 50)
    
    # 測試配置
    device = "cpu"
    seed = 42
    
    # 創建必要目錄
    os.makedirs("logs/test_curriculum", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # ===== 測試1: 預訓練階段 =====
    print("\n📚 測試階段1: 預訓練（簡化版）")
    
    # 修改預訓練配置為超快速版本
    quick_pretrain_config = """
env:
  num_agents: 4
  area_size: 1.5
  car_radius: 0.05
  comm_radius: 0.5
  mass: 0.1
  dt: 0.05
  obstacles:
    enabled: false

training:
  training_steps: 100  # 超短訓練
  horizon_length: 10
  learning_rate: 0.01
  eval_interval: 50
  save_interval: 50
  
loss_weights:
  goal_weight: 1.0
  safety_weight: 2.0
  control_weight: 0.1
  jerk_weight: 0.02
  alpha_reg_weight: 0.01
  progress_weight: 0.2
"""
    
    # 保存快速配置
    with open("config/quick_pretrain_test.yaml", "w") as f:
        f.write(quick_pretrain_config)
    
    # 運行預訓練
    cmd1 = f"python train_bptt.py --config config/quick_pretrain_test.yaml --device {device} --log_dir logs/test_curriculum/pretrain --seed {seed}"
    
    if not run_command(cmd1, "快速預訓練測試"):
        print("❌ 預訓練測試失敗，停止測試")
        return False
    
    # ===== 測試2: Fine-tuning階段 =====
    print("\n🎯 測試階段2: Fine-tuning（添加障礙物）")
    
    # 修改fine-tuning配置
    quick_finetune_config = """
env:
  num_agents: 4
  area_size: 1.5
  car_radius: 0.05
  comm_radius: 0.5
  mass: 0.1
  dt: 0.05
  obstacles:
    enabled: true
    positions: [[0.0, 0.0]]
    radii: [0.2]

training:
  training_steps: 100  # 超短fine-tuning
  horizon_length: 10
  learning_rate: 0.005
  eval_interval: 50
  save_interval: 50
  
loss_weights:
  goal_weight: 1.0
  safety_weight: 5.0
  control_weight: 0.1
  jerk_weight: 0.02
  alpha_reg_weight: 0.01
  progress_weight: 0.15
"""
    
    # 保存fine-tuning配置
    with open("config/quick_finetune_test.yaml", "w") as f:
        f.write(quick_finetune_config)
    
    # 運行fine-tuning
    cmd2 = f"python train_bptt.py --config config/quick_finetune_test.yaml --device {device} --log_dir logs/test_curriculum/finetune --load_pretrained_model_from logs/test_curriculum/pretrain --seed {seed}"
    
    if not run_command(cmd2, "快速Fine-tuning測試"):
        print("❌ Fine-tuning測試失敗")
        return False
    
    # ===== 測試3: 可視化生成 =====
    print("\n🎨 測試階段3: 可視化生成")
    
    cmd3 = f"python visualize_bptt.py --model_dir logs/test_curriculum/finetune --output results/test_curriculum_result.gif --device {device}"
    
    if not run_command(cmd3, "可視化生成測試"):
        print("⚠️ 可視化測試失敗，但核心功能可能正常")
    
    # ===== 測試結果 =====
    print("\n📊 測試結果總結")
    print("=" * 50)
    
    # 檢查生成的文件
    pretrain_exists = os.path.exists("logs/test_curriculum/pretrain/models")
    finetune_exists = os.path.exists("logs/test_curriculum/finetune/models")
    viz_exists = os.path.exists("results/test_curriculum_result.gif")
    
    print(f"✅ 預訓練模型: {'存在' if pretrain_exists else '不存在'}")
    print(f"✅ Fine-tuning模型: {'存在' if finetune_exists else '不存在'}")
    print(f"✅ 可視化文件: {'存在' if viz_exists else '不存在'}")
    
    if pretrain_exists and finetune_exists:
        print("\n🎉 課程學習框架測試成功！")
        print("🚀 現在可以運行完整的實驗管道：")
        print("   Windows: run_curriculum_experiments.bat")
        print("   Linux/Mac: bash run_curriculum_experiments.sh")
        return True
    else:
        print("\n❌ 課程學習框架測試失敗")
        print("請檢查錯誤日誌並修復問題")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
 
 
 
 