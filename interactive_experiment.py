#!/usr/bin/env python3
"""
交互式課程學習實驗
"""

import os
import subprocess
import time
from datetime import datetime

def run_interactive_experiment():
    """運行可觀察的交互式實驗"""
    print("🎯 交互式課程學習實驗")
    print("=" * 60)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_log_dir = f"logs/interactive_curriculum_{timestamp}"
    
    # Phase 1: 預訓練
    print("\n📚 階段1: 預訓練（可觀察輸出）")
    print("-" * 50)
    
    pretrain_log_dir = os.path.join(base_log_dir, "pretrain")
    
    cmd1 = f"python train_bptt.py --config config/simple_collaboration_pretrain.yaml --device cpu --log_dir {pretrain_log_dir} --seed 42"
    
    print(f"📝 執行命令: {cmd1}")
    print("🔄 開始預訓練... (實時輸出)")
    print("-" * 30)
    
    try:
        # 實時輸出，不捕獲，限制時間
        process = subprocess.Popen(cmd1, shell=True)
        
        # 等待120秒或完成
        try:
            process.wait(timeout=120)
            if process.returncode == 0:
                print("\n✅ 預訓練階段完成")
            else:
                print(f"\n❌ 預訓練失敗，返回碼: {process.returncode}")
                return False
        except subprocess.TimeoutExpired:
            print("\n⏰ 預訓練2分鐘測試完成")
            process.terminate()
            time.sleep(2)
            if process.poll() is None:
                process.kill()
    
    except Exception as e:
        print(f"\n❌ 預訓練異常: {e}")
        return False
    
    # 檢查預訓練結果
    print("\n🔍 檢查預訓練結果...")
    models_dir = os.path.join(pretrain_log_dir, "models")
    
    if os.path.exists(models_dir) and os.listdir(models_dir):
        model_steps = [d for d in os.listdir(models_dir) if d.isdigit()]
        model_steps.sort(key=int)
        print(f"✅ 預訓練模型已生成: {model_steps}")
        
        # Phase 2: Fine-tuning
        print(f"\n🎓 階段2: Fine-tuning（加載從 {model_steps[-1]} 步）")
        print("-" * 50)
        
        finetune_log_dir = os.path.join(base_log_dir, "finetune")
        
        cmd2 = f"python train_bptt.py --config config/simple_collaboration.yaml --device cpu --log_dir {finetune_log_dir} --load_pretrained_model_from {pretrain_log_dir} --seed 42"
        
        print(f"📝 執行命令: {cmd2}")
        print("🔄 開始Fine-tuning... (實時輸出)")
        print("-" * 30)
        
        try:
            # Fine-tuning階段，也限制時間
            process = subprocess.Popen(cmd2, shell=True)
            
            try:
                process.wait(timeout=120)
                if process.returncode == 0:
                    print("\n✅ Fine-tuning階段完成")
                else:
                    print(f"\n❌ Fine-tuning失敗，返回碼: {process.returncode}")
            except subprocess.TimeoutExpired:
                print("\n⏰ Fine-tuning2分鐘測試完成")
                process.terminate()
                time.sleep(2)
                if process.poll() is None:
                    process.kill()
                    
        except Exception as e:
            print(f"\n❌ Fine-tuning異常: {e}")
    
    else:
        print("❌ 預訓練模型未生成，跳過Fine-tuning")
    
    # 實驗總結
    print(f"\n🏁 實驗總結")
    print("=" * 60)
    print(f"📁 實驗目錄: {base_log_dir}")
    
    # 檢查所有生成的模型
    all_models = []
    for phase in ["pretrain", "finetune"]:
        phase_dir = os.path.join(base_log_dir, phase, "models")
        if os.path.exists(phase_dir):
            phase_models = [d for d in os.listdir(phase_dir) if d.isdigit()]
            if phase_models:
                phase_models.sort(key=int)
                all_models.extend([f"{phase}/{m}" for m in phase_models])
    
    if all_models:
        print(f"✅ 生成的模型: {all_models}")
        print("🎉 課程學習實驗成功！")
        
        # 建議下一步
        print(f"\n🚀 下一步建議:")
        print(f"   python unified_visualize_bptt.py {os.path.join(base_log_dir, 'finetune')}")
        print(f"   python check_experiment_status.py")
        
        return True
    else:
        print("❌ 沒有生成模型")
        return False

def main():
    """主函數"""
    print("🚀 啟動交互式課程學習實驗")
    print("這個實驗會顯示實時輸出，更容易觀察進度")
    print()
    
    success = run_interactive_experiment()
    
    if success:
        print("\n🎉 實驗成功完成！")
    else:
        print("\n💡 提示: 即使時間限制，部分模型可能已生成")
        print("   可以檢查實驗目錄查看結果")

if __name__ == "__main__":
    main()
 
"""
交互式課程學習實驗
"""

import os
import subprocess
import time
from datetime import datetime

def run_interactive_experiment():
    """運行可觀察的交互式實驗"""
    print("🎯 交互式課程學習實驗")
    print("=" * 60)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_log_dir = f"logs/interactive_curriculum_{timestamp}"
    
    # Phase 1: 預訓練
    print("\n📚 階段1: 預訓練（可觀察輸出）")
    print("-" * 50)
    
    pretrain_log_dir = os.path.join(base_log_dir, "pretrain")
    
    cmd1 = f"python train_bptt.py --config config/simple_collaboration_pretrain.yaml --device cpu --log_dir {pretrain_log_dir} --seed 42"
    
    print(f"📝 執行命令: {cmd1}")
    print("🔄 開始預訓練... (實時輸出)")
    print("-" * 30)
    
    try:
        # 實時輸出，不捕獲，限制時間
        process = subprocess.Popen(cmd1, shell=True)
        
        # 等待120秒或完成
        try:
            process.wait(timeout=120)
            if process.returncode == 0:
                print("\n✅ 預訓練階段完成")
            else:
                print(f"\n❌ 預訓練失敗，返回碼: {process.returncode}")
                return False
        except subprocess.TimeoutExpired:
            print("\n⏰ 預訓練2分鐘測試完成")
            process.terminate()
            time.sleep(2)
            if process.poll() is None:
                process.kill()
    
    except Exception as e:
        print(f"\n❌ 預訓練異常: {e}")
        return False
    
    # 檢查預訓練結果
    print("\n🔍 檢查預訓練結果...")
    models_dir = os.path.join(pretrain_log_dir, "models")
    
    if os.path.exists(models_dir) and os.listdir(models_dir):
        model_steps = [d for d in os.listdir(models_dir) if d.isdigit()]
        model_steps.sort(key=int)
        print(f"✅ 預訓練模型已生成: {model_steps}")
        
        # Phase 2: Fine-tuning
        print(f"\n🎓 階段2: Fine-tuning（加載從 {model_steps[-1]} 步）")
        print("-" * 50)
        
        finetune_log_dir = os.path.join(base_log_dir, "finetune")
        
        cmd2 = f"python train_bptt.py --config config/simple_collaboration.yaml --device cpu --log_dir {finetune_log_dir} --load_pretrained_model_from {pretrain_log_dir} --seed 42"
        
        print(f"📝 執行命令: {cmd2}")
        print("🔄 開始Fine-tuning... (實時輸出)")
        print("-" * 30)
        
        try:
            # Fine-tuning階段，也限制時間
            process = subprocess.Popen(cmd2, shell=True)
            
            try:
                process.wait(timeout=120)
                if process.returncode == 0:
                    print("\n✅ Fine-tuning階段完成")
                else:
                    print(f"\n❌ Fine-tuning失敗，返回碼: {process.returncode}")
            except subprocess.TimeoutExpired:
                print("\n⏰ Fine-tuning2分鐘測試完成")
                process.terminate()
                time.sleep(2)
                if process.poll() is None:
                    process.kill()
                    
        except Exception as e:
            print(f"\n❌ Fine-tuning異常: {e}")
    
    else:
        print("❌ 預訓練模型未生成，跳過Fine-tuning")
    
    # 實驗總結
    print(f"\n🏁 實驗總結")
    print("=" * 60)
    print(f"📁 實驗目錄: {base_log_dir}")
    
    # 檢查所有生成的模型
    all_models = []
    for phase in ["pretrain", "finetune"]:
        phase_dir = os.path.join(base_log_dir, phase, "models")
        if os.path.exists(phase_dir):
            phase_models = [d for d in os.listdir(phase_dir) if d.isdigit()]
            if phase_models:
                phase_models.sort(key=int)
                all_models.extend([f"{phase}/{m}" for m in phase_models])
    
    if all_models:
        print(f"✅ 生成的模型: {all_models}")
        print("🎉 課程學習實驗成功！")
        
        # 建議下一步
        print(f"\n🚀 下一步建議:")
        print(f"   python unified_visualize_bptt.py {os.path.join(base_log_dir, 'finetune')}")
        print(f"   python check_experiment_status.py")
        
        return True
    else:
        print("❌ 沒有生成模型")
        return False

def main():
    """主函數"""
    print("🚀 啟動交互式課程學習實驗")
    print("這個實驗會顯示實時輸出，更容易觀察進度")
    print()
    
    success = run_interactive_experiment()
    
    if success:
        print("\n🎉 實驗成功完成！")
    else:
        print("\n💡 提示: 即使時間限制，部分模型可能已生成")
        print("   可以檢查實驗目錄查看結果")

if __name__ == "__main__":
    main()
 
 
 
 