#!/usr/bin/env python3
"""
追蹤課程學習實驗進度
"""

import os
import glob
import time
from datetime import datetime

def track_current_experiment():
    """追蹤當前實驗進度"""
    print("📡 追蹤課程學習實驗進度")
    print("=" * 60)
    
    # 找到最新的fixed_curriculum實驗
    experiment_dirs = glob.glob("logs/fixed_curriculum_*")
    
    if not experiment_dirs:
        print("❌ 沒有找到fixed_curriculum實驗")
        return
    
    experiment_dirs.sort()
    latest_exp = experiment_dirs[-1]
    
    print(f"📁 監控實驗: {latest_exp}")
    print(f"⏰ 當前時間: {datetime.now().strftime('%H:%M:%S')}")
    print()
    
    # 檢查預訓練階段
    pretrain_dir = os.path.join(latest_exp, "pretrain")
    
    print("📚 Phase 1: 預訓練階段")
    print("-" * 40)
    
    if os.path.exists(pretrain_dir):
        print("✅ 預訓練目錄已創建")
        
        # 檢查配置文件
        config_file = os.path.join(pretrain_dir, "config.yaml")
        if os.path.exists(config_file):
            print("✅ 配置文件已生成")
        
        # 檢查模型目錄
        models_dir = os.path.join(pretrain_dir, "models")
        if os.path.exists(models_dir):
            model_steps = [d for d in os.listdir(models_dir) if d.isdigit()]
            if model_steps:
                model_steps.sort(key=int)
                print(f"🔄 訓練進行中: {len(model_steps)} 個檢查點")
                print(f"📊 當前步數: {model_steps[-1]}")
                print(f"📈 進度: {int(model_steps[-1])/2500*100:.1f}% (目標: 2500步)")
            else:
                print("🔄 模型目錄已創建，等待第一個檢查點...")
        else:
            print("⏳ 等待模型目錄創建...")
    else:
        print("⏳ 等待預訓練開始...")
    
    # 檢查Fine-tuning階段
    finetune_dir = os.path.join(latest_exp, "finetune")
    
    print("\n🎓 Phase 2: Fine-tuning階段")
    print("-" * 40)
    
    if os.path.exists(finetune_dir):
        print("✅ Fine-tuning目錄已創建")
        
        models_dir = os.path.join(finetune_dir, "models")
        if os.path.exists(models_dir):
            model_steps = [d for d in os.listdir(models_dir) if d.isdigit()]
            if model_steps:
                model_steps.sort(key=int)
                print(f"🔄 Fine-tuning進行中: {len(model_steps)} 個檢查點")
                print(f"📊 當前步數: {model_steps[-1]}")
            else:
                print("🔄 Fine-tuning已開始，等待檢查點...")
        else:
            print("⏳ 等待Fine-tuning開始...")
    else:
        print("⏳ 等待預訓練完成...")
    
    # 檢查可視化
    viz_files = glob.glob(os.path.join(latest_exp, "**/*.gif"), recursive=True)
    viz_files.extend(glob.glob(os.path.join(latest_exp, "**/*.mp4"), recursive=True))
    
    print("\n🎬 可視化文件")
    print("-" * 40)
    
    if viz_files:
        print(f"✅ 已生成 {len(viz_files)} 個可視化文件:")
        for viz in viz_files:
            print(f"   📹 {os.path.basename(viz)}")
    else:
        print("⏳ 等待可視化生成...")
    
    # 檢查日誌文件
    log_files = glob.glob(os.path.join(latest_exp, "**/*.log"), recursive=True)
    
    if log_files:
        print(f"\n📄 日誌文件: {len(log_files)} 個")
        for log in log_files:
            size = os.path.getsize(log)
            print(f"   📝 {os.path.basename(log)}: {size} bytes")
    
    # 實驗狀態評估
    print("\n" + "=" * 60)
    
    pretrain_models = []
    finetune_models = []
    
    pretrain_models_dir = os.path.join(pretrain_dir, "models")
    if os.path.exists(pretrain_models_dir):
        pretrain_models = [d for d in os.listdir(pretrain_models_dir) if d.isdigit()]
    
    finetune_models_dir = os.path.join(finetune_dir, "models")
    if os.path.exists(finetune_models_dir):
        finetune_models = [d for d in os.listdir(finetune_models_dir) if d.isdigit()]
    
    if pretrain_models and finetune_models:
        print("🎉 實驗狀態: 兩階段都在進行！")
        print("✅ 課程學習成功運行")
    elif pretrain_models:
        print("🔄 實驗狀態: 預訓練階段進行中")
        if len(pretrain_models) >= 5:  # 如果有5個或更多檢查點，說明進展良好
            print("✅ 預訓練進展順利")
        else:
            print("🔄 預訓練剛開始")
    else:
        print("🚀 實驗狀態: 剛開始，等待第一個檢查點")
    
    total_models = len(pretrain_models) + len(finetune_models)
    print(f"📊 總檢查點: {total_models}")
    
    if total_models > 0:
        print("💪 實驗正在健康運行！")
    else:
        print("⏳ 實驗剛開始，請稍候...")

def main():
    """主函數"""
    print("🚀 開始追蹤課程學習實驗")
    print("這個腳本會顯示當前實驗的實時狀態")
    print()
    
    track_current_experiment()
    
    print(f"\n💡 提示:")
    print("  - 實驗大約需要8-12分鐘完成")
    print("  - 可以重複運行此腳本查看進度")
    print("  - 實驗完成後會自動生成可視化")
    
    print(f"\n⏰ 檢查完成時間: {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    main()
 
 
 
 
 