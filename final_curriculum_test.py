#!/usr/bin/env python3
"""
最終課程學習實驗 - 完整的兩階段訓練流程
"""

import os
import subprocess
import time
from datetime import datetime

def run_command(cmd, description):
    """運行命令並處理結果"""
    print(f"\n🔄 {description}")
    print(f"📝 執行命令: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)  # 5分鐘超時
        if result.returncode == 0:
            print(f"✅ {description} - 成功")
            if result.stdout.strip():
                print(f"📊 輸出: {result.stdout.strip()[-500:]}")  # 只顯示最後500字符
            return True
        else:
            print(f"❌ {description} - 失敗")
            print(f"錯誤: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} - 超時")
        return False
    except Exception as e:
        print(f"❌ {description} - 異常: {e}")
        return False

def main():
    """主要實驗流程"""
    print("🎯 課程學習完整實驗")
    print("=" * 60)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_log_dir = f"logs/curriculum_experiment_{timestamp}"
    
    # Phase 1: 預訓練（無障礙物）
    print("\n📚 階段1: 預訓練（無障礙物環境）")
    print("-" * 40)
    
    pretrain_log_dir = os.path.join(base_log_dir, "pretrain")
    
    cmd1 = f"python train_bptt.py --config config/simple_collaboration_pretrain.yaml --device cpu --log_dir {pretrain_log_dir} --seed 42"
    
    if not run_command(cmd1, "預訓練階段"):
        print("❌ 預訓練失敗，停止實驗")
        return False
    
    print("✅ 預訓練階段完成")
    
    # 等待一下確保文件系統同步
    time.sleep(2)
    
    # Phase 2: Fine-tuning（有障礙物）
    print("\n🎓 階段2: Fine-tuning（有障礙物環境）")
    print("-" * 40)
    
    finetune_log_dir = os.path.join(base_log_dir, "finetune")
    
    cmd2 = f"python train_bptt.py --config config/simple_collaboration.yaml --device cpu --log_dir {finetune_log_dir} --load_pretrained_model_from {pretrain_log_dir} --seed 42"
    
    if not run_command(cmd2, "Fine-tuning階段"):
        print("❌ Fine-tuning失敗")
        return False
    
    print("✅ Fine-tuning階段完成")
    
    # 生成可視化
    print("\n🎬 生成協作可視化")
    print("-" * 40)
    
    viz_cmd = f"python unified_visualize_bptt.py {finetune_log_dir}"
    
    if run_command(viz_cmd, "生成可視化"):
        print("✅ 可視化生成成功")
    else:
        print("⚠️ 可視化生成失敗，但訓練成功")
    
    # 實驗結果總結
    print("\n🎉 課程學習實驗完成！")
    print("=" * 60)
    print(f"📁 實驗結果保存在: {base_log_dir}")
    print(f"📁 預訓練模型: {pretrain_log_dir}")
    print(f"📁 最終模型: {finetune_log_dir}")
    print("\n🚀 下一步：")
    print("  1. 檢查生成的可視化文件")
    print("  2. 分析訓練日誌")
    print("  3. 與原始模型比較性能")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎯 實驗成功完成")
    else:
        print("\n❌ 實驗失敗")
 
"""
最終課程學習實驗 - 完整的兩階段訓練流程
"""

import os
import subprocess
import time
from datetime import datetime

def run_command(cmd, description):
    """運行命令並處理結果"""
    print(f"\n🔄 {description}")
    print(f"📝 執行命令: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)  # 5分鐘超時
        if result.returncode == 0:
            print(f"✅ {description} - 成功")
            if result.stdout.strip():
                print(f"📊 輸出: {result.stdout.strip()[-500:]}")  # 只顯示最後500字符
            return True
        else:
            print(f"❌ {description} - 失敗")
            print(f"錯誤: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} - 超時")
        return False
    except Exception as e:
        print(f"❌ {description} - 異常: {e}")
        return False

def main():
    """主要實驗流程"""
    print("🎯 課程學習完整實驗")
    print("=" * 60)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_log_dir = f"logs/curriculum_experiment_{timestamp}"
    
    # Phase 1: 預訓練（無障礙物）
    print("\n📚 階段1: 預訓練（無障礙物環境）")
    print("-" * 40)
    
    pretrain_log_dir = os.path.join(base_log_dir, "pretrain")
    
    cmd1 = f"python train_bptt.py --config config/simple_collaboration_pretrain.yaml --device cpu --log_dir {pretrain_log_dir} --seed 42"
    
    if not run_command(cmd1, "預訓練階段"):
        print("❌ 預訓練失敗，停止實驗")
        return False
    
    print("✅ 預訓練階段完成")
    
    # 等待一下確保文件系統同步
    time.sleep(2)
    
    # Phase 2: Fine-tuning（有障礙物）
    print("\n🎓 階段2: Fine-tuning（有障礙物環境）")
    print("-" * 40)
    
    finetune_log_dir = os.path.join(base_log_dir, "finetune")
    
    cmd2 = f"python train_bptt.py --config config/simple_collaboration.yaml --device cpu --log_dir {finetune_log_dir} --load_pretrained_model_from {pretrain_log_dir} --seed 42"
    
    if not run_command(cmd2, "Fine-tuning階段"):
        print("❌ Fine-tuning失敗")
        return False
    
    print("✅ Fine-tuning階段完成")
    
    # 生成可視化
    print("\n🎬 生成協作可視化")
    print("-" * 40)
    
    viz_cmd = f"python unified_visualize_bptt.py {finetune_log_dir}"
    
    if run_command(viz_cmd, "生成可視化"):
        print("✅ 可視化生成成功")
    else:
        print("⚠️ 可視化生成失敗，但訓練成功")
    
    # 實驗結果總結
    print("\n🎉 課程學習實驗完成！")
    print("=" * 60)
    print(f"📁 實驗結果保存在: {base_log_dir}")
    print(f"📁 預訓練模型: {pretrain_log_dir}")
    print(f"📁 最終模型: {finetune_log_dir}")
    print("\n🚀 下一步：")
    print("  1. 檢查生成的可視化文件")
    print("  2. 分析訓練日誌")
    print("  3. 與原始模型比較性能")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎯 實驗成功完成")
    else:
        print("\n❌ 實驗失敗")
 
 
 
 