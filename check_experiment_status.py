#!/usr/bin/env python3
"""
檢查課程學習實驗狀態
"""

import os
import glob
from datetime import datetime

def check_experiment_status():
    """檢查實驗狀態"""
    print("🔍 課程學習實驗狀態檢查")
    print("=" * 50)
    
    # 查找最新的課程實驗目錄
    curriculum_dirs = glob.glob("logs/curriculum_experiment_*")
    
    if not curriculum_dirs:
        print("❌ 沒有找到課程學習實驗目錄")
        return
    
    # 按時間排序，獲取最新的
    curriculum_dirs.sort()
    latest_dir = curriculum_dirs[-1]
    
    print(f"📁 最新實驗目錄: {latest_dir}")
    
    # 檢查預訓練階段
    pretrain_dir = os.path.join(latest_dir, "pretrain")
    if os.path.exists(pretrain_dir):
        print(f"✅ 預訓練目錄存在: {pretrain_dir}")
        
        # 檢查模型文件
        models_dir = os.path.join(pretrain_dir, "models")
        if os.path.exists(models_dir):
            model_steps = [d for d in os.listdir(models_dir) if d.isdigit()]
            if model_steps:
                model_steps.sort(key=int)
                print(f"📊 預訓練模型步數: {model_steps}")
                print(f"🏆 最新預訓練模型: {model_steps[-1]}")
            else:
                print("⚠️ 預訓練模型目錄為空")
        else:
            print("❌ 預訓練模型目錄不存在")
    else:
        print("❌ 預訓練目錄不存在")
    
    # 檢查Fine-tuning階段
    finetune_dir = os.path.join(latest_dir, "finetune")
    if os.path.exists(finetune_dir):
        print(f"✅ Fine-tuning目錄存在: {finetune_dir}")
        
        # 檢查模型文件
        models_dir = os.path.join(finetune_dir, "models")
        if os.path.exists(models_dir):
            model_steps = [d for d in os.listdir(models_dir) if d.isdigit()]
            if model_steps:
                model_steps.sort(key=int)
                print(f"📊 Fine-tuning模型步數: {model_steps}")
                print(f"🏆 最新Fine-tuning模型: {model_steps[-1]}")
            else:
                print("⚠️ Fine-tuning模型目錄為空")
        else:
            print("❌ Fine-tuning模型目錄不存在")
    else:
        print("❌ Fine-tuning目錄不存在")
    
    # 檢查日誌文件
    log_files = glob.glob(os.path.join(latest_dir, "*.log"))
    if log_files:
        print(f"\n📄 日誌文件:")
        for log_file in log_files:
            size = os.path.getsize(log_file)
            print(f"  - {os.path.basename(log_file)}: {size} bytes")
    
    # 檢查可視化文件
    viz_files = glob.glob(os.path.join(latest_dir, "**/*.gif"), recursive=True)
    viz_files.extend(glob.glob(os.path.join(latest_dir, "**/*.mp4"), recursive=True))
    
    if viz_files:
        print(f"\n🎬 可視化文件:")
        for viz_file in viz_files:
            print(f"  - {viz_file}")
    else:
        print("\n⚠️ 沒有找到可視化文件")
    
    print("\n" + "=" * 50)
    
    # 實驗狀態總結
    if os.path.exists(pretrain_dir) and os.path.exists(finetune_dir):
        print("🎉 課程學習實驗完整完成！")
        print("📈 兩個階段都已執行")
    elif os.path.exists(pretrain_dir):
        print("🔄 課程學習實驗進行中...")
        print("✅ 預訓練階段完成")
        print("⏳ Fine-tuning階段待完成")
    else:
        print("🚀 課程學習實驗剛開始")
        print("⏳ 預訓練階段進行中")

if __name__ == "__main__":
    check_experiment_status()
 
"""
檢查課程學習實驗狀態
"""

import os
import glob
from datetime import datetime

def check_experiment_status():
    """檢查實驗狀態"""
    print("🔍 課程學習實驗狀態檢查")
    print("=" * 50)
    
    # 查找最新的課程實驗目錄
    curriculum_dirs = glob.glob("logs/curriculum_experiment_*")
    
    if not curriculum_dirs:
        print("❌ 沒有找到課程學習實驗目錄")
        return
    
    # 按時間排序，獲取最新的
    curriculum_dirs.sort()
    latest_dir = curriculum_dirs[-1]
    
    print(f"📁 最新實驗目錄: {latest_dir}")
    
    # 檢查預訓練階段
    pretrain_dir = os.path.join(latest_dir, "pretrain")
    if os.path.exists(pretrain_dir):
        print(f"✅ 預訓練目錄存在: {pretrain_dir}")
        
        # 檢查模型文件
        models_dir = os.path.join(pretrain_dir, "models")
        if os.path.exists(models_dir):
            model_steps = [d for d in os.listdir(models_dir) if d.isdigit()]
            if model_steps:
                model_steps.sort(key=int)
                print(f"📊 預訓練模型步數: {model_steps}")
                print(f"🏆 最新預訓練模型: {model_steps[-1]}")
            else:
                print("⚠️ 預訓練模型目錄為空")
        else:
            print("❌ 預訓練模型目錄不存在")
    else:
        print("❌ 預訓練目錄不存在")
    
    # 檢查Fine-tuning階段
    finetune_dir = os.path.join(latest_dir, "finetune")
    if os.path.exists(finetune_dir):
        print(f"✅ Fine-tuning目錄存在: {finetune_dir}")
        
        # 檢查模型文件
        models_dir = os.path.join(finetune_dir, "models")
        if os.path.exists(models_dir):
            model_steps = [d for d in os.listdir(models_dir) if d.isdigit()]
            if model_steps:
                model_steps.sort(key=int)
                print(f"📊 Fine-tuning模型步數: {model_steps}")
                print(f"🏆 最新Fine-tuning模型: {model_steps[-1]}")
            else:
                print("⚠️ Fine-tuning模型目錄為空")
        else:
            print("❌ Fine-tuning模型目錄不存在")
    else:
        print("❌ Fine-tuning目錄不存在")
    
    # 檢查日誌文件
    log_files = glob.glob(os.path.join(latest_dir, "*.log"))
    if log_files:
        print(f"\n📄 日誌文件:")
        for log_file in log_files:
            size = os.path.getsize(log_file)
            print(f"  - {os.path.basename(log_file)}: {size} bytes")
    
    # 檢查可視化文件
    viz_files = glob.glob(os.path.join(latest_dir, "**/*.gif"), recursive=True)
    viz_files.extend(glob.glob(os.path.join(latest_dir, "**/*.mp4"), recursive=True))
    
    if viz_files:
        print(f"\n🎬 可視化文件:")
        for viz_file in viz_files:
            print(f"  - {viz_file}")
    else:
        print("\n⚠️ 沒有找到可視化文件")
    
    print("\n" + "=" * 50)
    
    # 實驗狀態總結
    if os.path.exists(pretrain_dir) and os.path.exists(finetune_dir):
        print("🎉 課程學習實驗完整完成！")
        print("📈 兩個階段都已執行")
    elif os.path.exists(pretrain_dir):
        print("🔄 課程學習實驗進行中...")
        print("✅ 預訓練階段完成")
        print("⏳ Fine-tuning階段待完成")
    else:
        print("🚀 課程學習實驗剛開始")
        print("⏳ 預訓練階段進行中")

if __name__ == "__main__":
    check_experiment_status()
 
 
 
 