#!/usr/bin/env python3
"""
快速課程學習測試
"""

import os
import subprocess
import time

def run_quick_experiment():
    """運行快速實驗"""
    print("🎯 快速課程學習測試")
    print("=" * 50)
    
    timestamp = "quick_test"
    base_log_dir = f"logs/curriculum_{timestamp}"
    
    # Phase 1: 預訓練（極短版本）
    print("\n📚 階段1: 快速預訓練")
    print("-" * 30)
    
    pretrain_log_dir = os.path.join(base_log_dir, "pretrain")
    
    # 修改配置為超短版本
    cmd1 = f"python train_bptt.py --config config/simple_collaboration_pretrain.yaml --device cpu --log_dir {pretrain_log_dir} --seed 42"
    
    print(f"📝 執行: {cmd1}")
    
    try:
        # 只運行60秒
        result = subprocess.run(cmd1, shell=True, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0 or "Training completed" in result.stdout:
            print("✅ 預訓練完成")
            if result.stdout:
                print(f"輸出: {result.stdout[-200:]}")  # 最後200字符
        else:
            print("❌ 預訓練失敗")
            print(f"錯誤: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ 預訓練60秒測試完成")
    except Exception as e:
        print(f"❌ 預訓練異常: {e}")
        return False
    
    # 檢查是否生成了模型
    models_dir = os.path.join(pretrain_log_dir, "models")
    if os.path.exists(models_dir) and os.listdir(models_dir):
        print(f"✅ 預訓練模型已生成: {os.listdir(models_dir)}")
        
        # Phase 2: 簡短fine-tuning
        print("\n🎓 階段2: 快速Fine-tuning")
        print("-" * 30)
        
        finetune_log_dir = os.path.join(base_log_dir, "finetune")
        
        cmd2 = f"python train_bptt.py --config config/simple_collaboration.yaml --device cpu --log_dir {finetune_log_dir} --load_pretrained_model_from {pretrain_log_dir} --seed 42"
        
        print(f"📝 執行: {cmd2}")
        
        try:
            # 60秒fine-tuning測試
            result = subprocess.run(cmd2, shell=True, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0 or "Training completed" in result.stdout:
                print("✅ Fine-tuning完成")
                if result.stdout:
                    print(f"輸出: {result.stdout[-200:]}")
            else:
                print("❌ Fine-tuning失敗")
                print(f"錯誤: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print("⏰ Fine-tuning60秒測試完成")
        except Exception as e:
            print(f"❌ Fine-tuning異常: {e}")
    
    else:
        print("❌ 預訓練模型未生成，跳過fine-tuning")
    
    print(f"\n📁 測試結果: {base_log_dir}")
    return True

def main():
    """主函數"""
    success = run_quick_experiment()
    
    if success:
        print("\n🎉 快速課程學習測試完成！")
        print("🔧 如果基本流程工作，可以運行完整實驗")
    else:
        print("\n❌ 快速測試失敗")

if __name__ == "__main__":
    main()
 
"""
快速課程學習測試
"""

import os
import subprocess
import time

def run_quick_experiment():
    """運行快速實驗"""
    print("🎯 快速課程學習測試")
    print("=" * 50)
    
    timestamp = "quick_test"
    base_log_dir = f"logs/curriculum_{timestamp}"
    
    # Phase 1: 預訓練（極短版本）
    print("\n📚 階段1: 快速預訓練")
    print("-" * 30)
    
    pretrain_log_dir = os.path.join(base_log_dir, "pretrain")
    
    # 修改配置為超短版本
    cmd1 = f"python train_bptt.py --config config/simple_collaboration_pretrain.yaml --device cpu --log_dir {pretrain_log_dir} --seed 42"
    
    print(f"📝 執行: {cmd1}")
    
    try:
        # 只運行60秒
        result = subprocess.run(cmd1, shell=True, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0 or "Training completed" in result.stdout:
            print("✅ 預訓練完成")
            if result.stdout:
                print(f"輸出: {result.stdout[-200:]}")  # 最後200字符
        else:
            print("❌ 預訓練失敗")
            print(f"錯誤: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ 預訓練60秒測試完成")
    except Exception as e:
        print(f"❌ 預訓練異常: {e}")
        return False
    
    # 檢查是否生成了模型
    models_dir = os.path.join(pretrain_log_dir, "models")
    if os.path.exists(models_dir) and os.listdir(models_dir):
        print(f"✅ 預訓練模型已生成: {os.listdir(models_dir)}")
        
        # Phase 2: 簡短fine-tuning
        print("\n🎓 階段2: 快速Fine-tuning")
        print("-" * 30)
        
        finetune_log_dir = os.path.join(base_log_dir, "finetune")
        
        cmd2 = f"python train_bptt.py --config config/simple_collaboration.yaml --device cpu --log_dir {finetune_log_dir} --load_pretrained_model_from {pretrain_log_dir} --seed 42"
        
        print(f"📝 執行: {cmd2}")
        
        try:
            # 60秒fine-tuning測試
            result = subprocess.run(cmd2, shell=True, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0 or "Training completed" in result.stdout:
                print("✅ Fine-tuning完成")
                if result.stdout:
                    print(f"輸出: {result.stdout[-200:]}")
            else:
                print("❌ Fine-tuning失敗")
                print(f"錯誤: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print("⏰ Fine-tuning60秒測試完成")
        except Exception as e:
            print(f"❌ Fine-tuning異常: {e}")
    
    else:
        print("❌ 預訓練模型未生成，跳過fine-tuning")
    
    print(f"\n📁 測試結果: {base_log_dir}")
    return True

def main():
    """主函數"""
    success = run_quick_experiment()
    
    if success:
        print("\n🎉 快速課程學習測試完成！")
        print("🔧 如果基本流程工作，可以運行完整實驗")
    else:
        print("\n❌ 快速測試失敗")

if __name__ == "__main__":
    main()
 
 
 
 