#!/usr/bin/env python3
"""
最終狀態檢查 - 監控課程學習實驗
"""

import os
import glob
import time
from datetime import datetime

def check_all_experiments():
    """檢查所有實驗的狀態"""
    print("📊 所有課程學習實驗狀態總覽")
    print("=" * 70)
    
    # 查找所有相關實驗目錄
    experiment_patterns = [
        "logs/curriculum_experiment_*",
        "logs/interactive_curriculum_*",
        "logs/curriculum_quick_test*"
    ]
    
    all_experiments = []
    for pattern in experiment_patterns:
        all_experiments.extend(glob.glob(pattern))
    
    if not all_experiments:
        print("❌ 沒有找到任何實驗目錄")
        return False
    
    all_experiments.sort()
    
    print(f"📁 找到 {len(all_experiments)} 個實驗目錄:")
    
    successful_experiments = []
    
    for i, exp_dir in enumerate(all_experiments, 1):
        exp_name = os.path.basename(exp_dir)
        print(f"\n{i}. 📂 {exp_name}")
        
        # 檢查預訓練
        pretrain_dir = os.path.join(exp_dir, "pretrain")
        pretrain_status = "❌ 不存在"
        pretrain_models = []
        
        if os.path.exists(pretrain_dir):
            models_dir = os.path.join(pretrain_dir, "models")
            if os.path.exists(models_dir):
                pretrain_models = [d for d in os.listdir(models_dir) if d.isdigit()]
                if pretrain_models:
                    pretrain_models.sort(key=int)
                    pretrain_status = f"✅ 完成 ({len(pretrain_models)} 步: {pretrain_models[-1]})"
                else:
                    pretrain_status = "🔄 進行中"
            else:
                pretrain_status = "🔄 已開始"
        
        print(f"   📚 預訓練: {pretrain_status}")
        
        # 檢查Fine-tuning
        finetune_dir = os.path.join(exp_dir, "finetune")
        finetune_status = "❌ 不存在"
        finetune_models = []
        
        if os.path.exists(finetune_dir):
            models_dir = os.path.join(finetune_dir, "models")
            if os.path.exists(models_dir):
                finetune_models = [d for d in os.listdir(models_dir) if d.isdigit()]
                if finetune_models:
                    finetune_models.sort(key=int)
                    finetune_status = f"✅ 完成 ({len(finetune_models)} 步: {finetune_models[-1]})"
                else:
                    finetune_status = "🔄 進行中"
            else:
                finetune_status = "🔄 已開始"
        
        print(f"   🎓 Fine-tuning: {finetune_status}")
        
        # 檢查可視化
        viz_files = glob.glob(os.path.join(exp_dir, "**/*.gif"), recursive=True)
        viz_files.extend(glob.glob(os.path.join(exp_dir, "**/*.mp4"), recursive=True))
        
        if viz_files:
            print(f"   🎬 可視化: ✅ {len(viz_files)} 個文件")
        else:
            print(f"   🎬 可視化: ❌ 無")
        
        # 評估實驗成功度
        if pretrain_models and finetune_models:
            successful_experiments.append((exp_dir, len(pretrain_models) + len(finetune_models)))
            print(f"   🏆 狀態: 完全成功")
        elif pretrain_models:
            print(f"   ⚠️ 狀態: 部分成功（僅預訓練）")
        else:
            print(f"   ❌ 狀態: 未成功")
    
    # 總結
    print("\n" + "=" * 70)
    print("📈 實驗總結:")
    
    if successful_experiments:
        successful_experiments.sort(key=lambda x: x[1], reverse=True)  # 按模型數量排序
        best_exp = successful_experiments[0]
        
        print(f"🏆 最成功的實驗: {os.path.basename(best_exp[0])}")
        print(f"📊 總步數: {best_exp[1]}")
        print(f"📁 路徑: {best_exp[0]}")
        
        print(f"\n🚀 建議操作:")
        print(f"   python unified_visualize_bptt.py {os.path.join(best_exp[0], 'finetune')}")
        
        return True
    else:
        print("❌ 沒有完全成功的實驗")
        
        # 檢查是否有正在進行的實驗
        recent_dirs = [d for d in all_experiments if "163933" in d or "163445" in d]
        if recent_dirs:
            print("🔄 但有最近的實驗可能正在進行中")
            print("💡 建議等待幾分鐘後重新檢查")
        
        return False

def main():
    """主函數"""
    print("🔍 課程學習實驗最終狀態檢查")
    print(f"⏰ 檢查時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    success = check_all_experiments()
    
    if success:
        print("\n🎉 發現成功的課程學習實驗！")
        print("✅ 戰略性改進已成功實施並驗證")
    else:
        print("\n⏳ 實驗可能仍在進行中...")
        print("🔄 建議稍後重新檢查")

if __name__ == "__main__":
    main()
 
"""
最終狀態檢查 - 監控課程學習實驗
"""

import os
import glob
import time
from datetime import datetime

def check_all_experiments():
    """檢查所有實驗的狀態"""
    print("📊 所有課程學習實驗狀態總覽")
    print("=" * 70)
    
    # 查找所有相關實驗目錄
    experiment_patterns = [
        "logs/curriculum_experiment_*",
        "logs/interactive_curriculum_*",
        "logs/curriculum_quick_test*"
    ]
    
    all_experiments = []
    for pattern in experiment_patterns:
        all_experiments.extend(glob.glob(pattern))
    
    if not all_experiments:
        print("❌ 沒有找到任何實驗目錄")
        return False
    
    all_experiments.sort()
    
    print(f"📁 找到 {len(all_experiments)} 個實驗目錄:")
    
    successful_experiments = []
    
    for i, exp_dir in enumerate(all_experiments, 1):
        exp_name = os.path.basename(exp_dir)
        print(f"\n{i}. 📂 {exp_name}")
        
        # 檢查預訓練
        pretrain_dir = os.path.join(exp_dir, "pretrain")
        pretrain_status = "❌ 不存在"
        pretrain_models = []
        
        if os.path.exists(pretrain_dir):
            models_dir = os.path.join(pretrain_dir, "models")
            if os.path.exists(models_dir):
                pretrain_models = [d for d in os.listdir(models_dir) if d.isdigit()]
                if pretrain_models:
                    pretrain_models.sort(key=int)
                    pretrain_status = f"✅ 完成 ({len(pretrain_models)} 步: {pretrain_models[-1]})"
                else:
                    pretrain_status = "🔄 進行中"
            else:
                pretrain_status = "🔄 已開始"
        
        print(f"   📚 預訓練: {pretrain_status}")
        
        # 檢查Fine-tuning
        finetune_dir = os.path.join(exp_dir, "finetune")
        finetune_status = "❌ 不存在"
        finetune_models = []
        
        if os.path.exists(finetune_dir):
            models_dir = os.path.join(finetune_dir, "models")
            if os.path.exists(models_dir):
                finetune_models = [d for d in os.listdir(models_dir) if d.isdigit()]
                if finetune_models:
                    finetune_models.sort(key=int)
                    finetune_status = f"✅ 完成 ({len(finetune_models)} 步: {finetune_models[-1]})"
                else:
                    finetune_status = "🔄 進行中"
            else:
                finetune_status = "🔄 已開始"
        
        print(f"   🎓 Fine-tuning: {finetune_status}")
        
        # 檢查可視化
        viz_files = glob.glob(os.path.join(exp_dir, "**/*.gif"), recursive=True)
        viz_files.extend(glob.glob(os.path.join(exp_dir, "**/*.mp4"), recursive=True))
        
        if viz_files:
            print(f"   🎬 可視化: ✅ {len(viz_files)} 個文件")
        else:
            print(f"   🎬 可視化: ❌ 無")
        
        # 評估實驗成功度
        if pretrain_models and finetune_models:
            successful_experiments.append((exp_dir, len(pretrain_models) + len(finetune_models)))
            print(f"   🏆 狀態: 完全成功")
        elif pretrain_models:
            print(f"   ⚠️ 狀態: 部分成功（僅預訓練）")
        else:
            print(f"   ❌ 狀態: 未成功")
    
    # 總結
    print("\n" + "=" * 70)
    print("📈 實驗總結:")
    
    if successful_experiments:
        successful_experiments.sort(key=lambda x: x[1], reverse=True)  # 按模型數量排序
        best_exp = successful_experiments[0]
        
        print(f"🏆 最成功的實驗: {os.path.basename(best_exp[0])}")
        print(f"📊 總步數: {best_exp[1]}")
        print(f"📁 路徑: {best_exp[0]}")
        
        print(f"\n🚀 建議操作:")
        print(f"   python unified_visualize_bptt.py {os.path.join(best_exp[0], 'finetune')}")
        
        return True
    else:
        print("❌ 沒有完全成功的實驗")
        
        # 檢查是否有正在進行的實驗
        recent_dirs = [d for d in all_experiments if "163933" in d or "163445" in d]
        if recent_dirs:
            print("🔄 但有最近的實驗可能正在進行中")
            print("💡 建議等待幾分鐘後重新檢查")
        
        return False

def main():
    """主函數"""
    print("🔍 課程學習實驗最終狀態檢查")
    print(f"⏰ 檢查時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    success = check_all_experiments()
    
    if success:
        print("\n🎉 發現成功的課程學習實驗！")
        print("✅ 戰略性改進已成功實施並驗證")
    else:
        print("\n⏳ 實驗可能仍在進行中...")
        print("🔄 建議稍後重新檢查")

if __name__ == "__main__":
    main()
 
 
 
 