#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
實時監控課程學習實驗進度
持續顯示當前狀態，每30秒更新一次
"""

import os
import time
import glob
from datetime import datetime

def get_latest_experiment():
    """獲取最新的實驗目錄"""
    pattern = "logs/fixed_curriculum_*"
    dirs = glob.glob(pattern)
    if not dirs:
        return None
    return max(dirs, key=os.path.getctime)

def count_files_in_dir(directory):
    """計算目錄中的文件數量"""
    if not os.path.exists(directory):
        return 0
    return len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])

def get_experiment_status():
    """獲取實驗狀態"""
    experiment_dir = get_latest_experiment()
    if not experiment_dir:
        return None, "❌ 沒有找到實驗目錄"
    
    # 檢查各階段狀態
    pretrain_dir = os.path.join(experiment_dir, "pretrain")
    finetune_dir = os.path.join(experiment_dir, "finetune")
    
    # 計算檢查點
    pretrain_checkpoints = 0
    finetune_checkpoints = 0
    visualizations = 0
    
    if os.path.exists(pretrain_dir):
        pretrain_checkpoints = len(glob.glob(os.path.join(pretrain_dir, "*.pt")))
    
    if os.path.exists(finetune_dir):
        finetune_checkpoints = len(glob.glob(os.path.join(finetune_dir, "*.pt")))
    
    # 檢查可視化文件
    viz_files = glob.glob(os.path.join(experiment_dir, "*.gif")) + glob.glob(os.path.join(experiment_dir, "*.mp4"))
    visualizations = len(viz_files)
    
    # 判斷當前階段
    if pretrain_checkpoints == 0:
        phase = "📚 Phase 1: 預訓練準備中..."
        progress = "⏳ 等待第一個檢查點"
    elif finetune_checkpoints == 0:
        phase = "📚 Phase 1: 預訓練進行中"
        progress = f"✅ 預訓練檢查點: {pretrain_checkpoints}"
    elif visualizations == 0:
        phase = "🎓 Phase 2: Fine-tuning進行中"
        progress = f"✅ Fine-tuning檢查點: {finetune_checkpoints}"
    else:
        phase = "🎬 實驗完成！"
        progress = f"🎉 可視化文件已生成: {visualizations}"
    
    return experiment_dir, {
        'phase': phase,
        'progress': progress,
        'pretrain_checkpoints': pretrain_checkpoints,
        'finetune_checkpoints': finetune_checkpoints,
        'visualizations': visualizations
    }

def main():
    """主監控循環"""
    print("🚀 實時監控課程學習實驗")
    print("=" * 60)
    print("💡 每30秒自動更新，按 Ctrl+C 停止監控")
    print()
    
    try:
        while True:
            # 清屏（Windows）
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print("🚀 實時監控課程學習實驗")
            print("=" * 60)
            
            experiment_dir, status = get_experiment_status()
            current_time = datetime.now().strftime("%H:%M:%S")
            
            if experiment_dir:
                print(f"📁 監控實驗: {experiment_dir}")
                print(f"⏰ 當前時間: {current_time}")
                print()
                
                if isinstance(status, dict):
                    print(f"{status['phase']}")
                    print(f"{status['progress']}")
                    print()
                    
                    # 詳細進度
                    print("📊 詳細進度:")
                    print(f"   📚 預訓練檢查點: {status['pretrain_checkpoints']}")
                    print(f"   🎓 Fine-tuning檢查點: {status['finetune_checkpoints']}")
                    print(f"   🎬 可視化文件: {status['visualizations']}")
                    print()
                    
                    # 預計完成時間
                    if status['pretrain_checkpoints'] == 0:
                        print("⏰ 預計: 預訓練約需4-6分鐘")
                    elif status['finetune_checkpoints'] == 0:
                        print("⏰ 預計: Fine-tuning約需4-6分鐘")
                    elif status['visualizations'] == 0:
                        print("⏰ 預計: 可視化生成約需1-2分鐘")
                    else:
                        print("🎉 實驗完成！可以查看生成的可視化文件")
                        break
                else:
                    print(status)
            else:
                print("❌ 沒有找到進行中的實驗")
            
            print()
            print("💡 每30秒自動更新，按 Ctrl+C 停止監控")
            print(f"下次更新: {datetime.now().strftime('%H:%M:%S')}")
            
            # 等待30秒
            for i in range(30, 0, -1):
                print(f"\r⏱️  下次更新倒計時: {i:2d}秒", end="", flush=True)
                time.sleep(1)
            print()
            
    except KeyboardInterrupt:
        print("\n\n👋 監控已停止")
        # 顯示最終狀態
        experiment_dir, status = get_experiment_status()
        if experiment_dir and isinstance(status, dict):
            print("\n📊 停止時的狀態:")
            print(f"   預訓練檢查點: {status['pretrain_checkpoints']}")
            print(f"   Fine-tuning檢查點: {status['finetune_checkpoints']}")
            print(f"   可視化文件: {status['visualizations']}")
        
        print("\n🔄 要重新開始監控，請運行: python realtime_monitor.py")

if __name__ == "__main__":
    main()