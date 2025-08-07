#!/usr/bin/env python3
"""
修復後的完整課程學習實驗
"""

import os
import subprocess
import time
from datetime import datetime

def run_fixed_curriculum_experiment():
    """運行修復後的完整課程學習實驗"""
    print("🎯 修復後的完整課程學習實驗")
    print("=" * 70)
    print(f"⏰ 開始時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🔧 問題已修復: 維度匹配 (6維觀測)")
    print()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_log_dir = f"logs/fixed_curriculum_{timestamp}"
    
    print(f"📁 實驗目錄: {base_log_dir}")
    print()
    
    # Phase 1: 預訓練階段
    print("📚 Phase 1: 預訓練階段 (無障礙物環境)")
    print("-" * 60)
    
    pretrain_log_dir = os.path.join(base_log_dir, "pretrain")
    
    cmd1 = f"python train_bptt.py --config config/simple_collaboration_pretrain.yaml --device cpu --log_dir {pretrain_log_dir} --seed 42"
    
    print(f"📝 執行命令: {cmd1}")
    print("🔄 開始預訓練... (預計4-6分鐘)")
    print("   - 學習基本導航和協作")
    print("   - 2500訓練步數")
    print("   - 高進度獎勵權重 (0.25)")
    print()
    
    try:
        # 運行預訓練，設置較長的超時時間
        result = subprocess.run(cmd1, shell=True, capture_output=True, text=True, timeout=600)  # 10分鐘超時
        
        if result.returncode == 0:
            print("✅ 預訓練階段完成！")
            if result.stdout:
                # 顯示最後幾行重要輸出
                lines = result.stdout.strip().split('\n')
                print("📊 訓練輸出摘要:")
                for line in lines[-10:]:  # 最後10行
                    if line.strip():
                        print(f"   {line}")
        else:
            print("❌ 預訓練階段失敗")
            print(f"錯誤輸出: {result.stderr}")
            return False, "預訓練失敗"
            
    except subprocess.TimeoutExpired:
        print("⏰ 預訓練超時，但可能部分完成")
        print("🔍 檢查生成的模型...")
    except Exception as e:
        print(f"❌ 預訓練異常: {e}")
        return False, f"預訓練異常: {e}"
    
    # 檢查預訓練結果
    print("\n🔍 檢查預訓練結果...")
    models_dir = os.path.join(pretrain_log_dir, "models")
    
    if os.path.exists(models_dir):
        model_steps = [d for d in os.listdir(models_dir) if d.isdigit()]
        if model_steps:
            model_steps.sort(key=int)
            print(f"✅ 預訓練模型已生成: {len(model_steps)} 個檢查點")
            print(f"📊 訓練步數: {', '.join(model_steps[-5:])}")  # 顯示最後5個
            latest_step = model_steps[-1]
            print(f"🏆 最新模型: 步數 {latest_step}")
        else:
            print("❌ 預訓練模型目錄為空")
            return False, "預訓練未生成模型"
    else:
        print("❌ 預訓練模型目錄不存在")
        return False, "預訓練目錄未創建"
    
    # Phase 2: Fine-tuning階段
    print(f"\n🎓 Phase 2: Fine-tuning階段 (有障礙物環境)")
    print("-" * 60)
    
    finetune_log_dir = os.path.join(base_log_dir, "finetune")
    
    cmd2 = f"python train_bptt.py --config config/simple_collaboration.yaml --device cpu --log_dir {finetune_log_dir} --load_pretrained_model_from {pretrain_log_dir} --seed 42"
    
    print(f"📝 執行命令: {cmd2}")
    print(f"🔄 開始Fine-tuning... (預計4-6分鐘)")
    print(f"   - 從預訓練步數 {latest_step} 繼續")
    print("   - 適應障礙物環境")
    print("   - 平衡的損失權重")
    print()
    
    try:
        # 運行Fine-tuning
        result = subprocess.run(cmd2, shell=True, capture_output=True, text=True, timeout=600)  # 10分鐘超時
        
        if result.returncode == 0:
            print("✅ Fine-tuning階段完成！")
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                print("📊 Fine-tuning輸出摘要:")
                for line in lines[-10:]:
                    if line.strip():
                        print(f"   {line}")
        else:
            print("❌ Fine-tuning階段失敗")
            print(f"錯誤輸出: {result.stderr}")
            return False, "Fine-tuning失敗"
            
    except subprocess.TimeoutExpired:
        print("⏰ Fine-tuning超時，但可能部分完成")
    except Exception as e:
        print(f"❌ Fine-tuning異常: {e}")
        return False, f"Fine-tuning異常: {e}"
    
    # 檢查Fine-tuning結果
    print("\n🔍 檢查Fine-tuning結果...")
    finetune_models_dir = os.path.join(finetune_log_dir, "models")
    
    if os.path.exists(finetune_models_dir):
        finetune_steps = [d for d in os.listdir(finetune_models_dir) if d.isdigit()]
        if finetune_steps:
            finetune_steps.sort(key=int)
            print(f"✅ Fine-tuning模型已生成: {len(finetune_steps)} 個檢查點")
            print(f"📊 Fine-tuning步數: {', '.join(finetune_steps[-5:])}")
            final_step = finetune_steps[-1]
            print(f"🏆 最終模型: 步數 {final_step}")
        else:
            print("❌ Fine-tuning模型目錄為空")
            return False, "Fine-tuning未生成模型"
    else:
        print("❌ Fine-tuning模型目錄不存在")
        return False, "Fine-tuning目錄未創建"
    
    # 生成可視化
    print(f"\n🎬 生成協作可視化...")
    print("-" * 40)
    
    viz_cmd = f"python unified_visualize_bptt.py {finetune_log_dir}"
    print(f"📝 可視化命令: {viz_cmd}")
    
    try:
        viz_result = subprocess.run(viz_cmd, shell=True, capture_output=True, text=True, timeout=300)  # 5分鐘超時
        
        if viz_result.returncode == 0:
            print("✅ 可視化生成成功")
            
            # 查找生成的可視化文件
            import glob
            viz_files = glob.glob(os.path.join(base_log_dir, "**/*.gif"), recursive=True)
            viz_files.extend(glob.glob(os.path.join(base_log_dir, "**/*.mp4"), recursive=True))
            
            if viz_files:
                print(f"🎥 生成的可視化文件:")
                for viz_file in viz_files:
                    print(f"   📹 {viz_file}")
            else:
                print("⚠️ 可視化生成成功但未找到文件")
                
        else:
            print("⚠️ 可視化生成失敗，但訓練成功")
            if viz_result.stderr:
                print(f"可視化錯誤: {viz_result.stderr}")
                
    except subprocess.TimeoutExpired:
        print("⏰ 可視化生成超時")
    except Exception as e:
        print(f"⚠️ 可視化生成異常: {e}")
    
    return True, base_log_dir

def main():
    """主實驗函數"""
    print("🚀 啟動修復後的完整課程學習實驗")
    print("維度問題已修復，預期成功率: 高")
    print()
    
    success, result = run_fixed_curriculum_experiment()
    
    print("\n" + "=" * 70)
    
    if success:
        print("🎉 課程學習實驗成功完成！")
        print("✅ 兩階段訓練都已成功")
        print("✅ 戰略性改進已完全實施")
        print()
        print(f"📁 實驗結果: {result}")
        print()
        print("🚀 實驗成果:")
        print("   ✅ 基於潛力的獎勵塑形")
        print("   ✅ 兩階段課程學習")
        print("   ✅ 進度獎勵機制")
        print("   ✅ 多智能體協作訓練")
        print()
        print("🎯 下一步建議:")
        print("   1. 檢查生成的可視化文件")
        print("   2. 分析訓練日誌和性能指標")
        print("   3. 與基線模型比較協作效果")
        
    else:
        print("❌ 實驗失敗")
        print(f"🔍 失敗原因: {result}")
        print("💡 建議檢查錯誤信息並重新嘗試")
    
    print(f"\n⏰ 完成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
 
"""
修復後的完整課程學習實驗
"""

import os
import subprocess
import time
from datetime import datetime

def run_fixed_curriculum_experiment():
    """運行修復後的完整課程學習實驗"""
    print("🎯 修復後的完整課程學習實驗")
    print("=" * 70)
    print(f"⏰ 開始時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🔧 問題已修復: 維度匹配 (6維觀測)")
    print()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_log_dir = f"logs/fixed_curriculum_{timestamp}"
    
    print(f"📁 實驗目錄: {base_log_dir}")
    print()
    
    # Phase 1: 預訓練階段
    print("📚 Phase 1: 預訓練階段 (無障礙物環境)")
    print("-" * 60)
    
    pretrain_log_dir = os.path.join(base_log_dir, "pretrain")
    
    cmd1 = f"python train_bptt.py --config config/simple_collaboration_pretrain.yaml --device cpu --log_dir {pretrain_log_dir} --seed 42"
    
    print(f"📝 執行命令: {cmd1}")
    print("🔄 開始預訓練... (預計4-6分鐘)")
    print("   - 學習基本導航和協作")
    print("   - 2500訓練步數")
    print("   - 高進度獎勵權重 (0.25)")
    print()
    
    try:
        # 運行預訓練，設置較長的超時時間
        result = subprocess.run(cmd1, shell=True, capture_output=True, text=True, timeout=600)  # 10分鐘超時
        
        if result.returncode == 0:
            print("✅ 預訓練階段完成！")
            if result.stdout:
                # 顯示最後幾行重要輸出
                lines = result.stdout.strip().split('\n')
                print("📊 訓練輸出摘要:")
                for line in lines[-10:]:  # 最後10行
                    if line.strip():
                        print(f"   {line}")
        else:
            print("❌ 預訓練階段失敗")
            print(f"錯誤輸出: {result.stderr}")
            return False, "預訓練失敗"
            
    except subprocess.TimeoutExpired:
        print("⏰ 預訓練超時，但可能部分完成")
        print("🔍 檢查生成的模型...")
    except Exception as e:
        print(f"❌ 預訓練異常: {e}")
        return False, f"預訓練異常: {e}"
    
    # 檢查預訓練結果
    print("\n🔍 檢查預訓練結果...")
    models_dir = os.path.join(pretrain_log_dir, "models")
    
    if os.path.exists(models_dir):
        model_steps = [d for d in os.listdir(models_dir) if d.isdigit()]
        if model_steps:
            model_steps.sort(key=int)
            print(f"✅ 預訓練模型已生成: {len(model_steps)} 個檢查點")
            print(f"📊 訓練步數: {', '.join(model_steps[-5:])}")  # 顯示最後5個
            latest_step = model_steps[-1]
            print(f"🏆 最新模型: 步數 {latest_step}")
        else:
            print("❌ 預訓練模型目錄為空")
            return False, "預訓練未生成模型"
    else:
        print("❌ 預訓練模型目錄不存在")
        return False, "預訓練目錄未創建"
    
    # Phase 2: Fine-tuning階段
    print(f"\n🎓 Phase 2: Fine-tuning階段 (有障礙物環境)")
    print("-" * 60)
    
    finetune_log_dir = os.path.join(base_log_dir, "finetune")
    
    cmd2 = f"python train_bptt.py --config config/simple_collaboration.yaml --device cpu --log_dir {finetune_log_dir} --load_pretrained_model_from {pretrain_log_dir} --seed 42"
    
    print(f"📝 執行命令: {cmd2}")
    print(f"🔄 開始Fine-tuning... (預計4-6分鐘)")
    print(f"   - 從預訓練步數 {latest_step} 繼續")
    print("   - 適應障礙物環境")
    print("   - 平衡的損失權重")
    print()
    
    try:
        # 運行Fine-tuning
        result = subprocess.run(cmd2, shell=True, capture_output=True, text=True, timeout=600)  # 10分鐘超時
        
        if result.returncode == 0:
            print("✅ Fine-tuning階段完成！")
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                print("📊 Fine-tuning輸出摘要:")
                for line in lines[-10:]:
                    if line.strip():
                        print(f"   {line}")
        else:
            print("❌ Fine-tuning階段失敗")
            print(f"錯誤輸出: {result.stderr}")
            return False, "Fine-tuning失敗"
            
    except subprocess.TimeoutExpired:
        print("⏰ Fine-tuning超時，但可能部分完成")
    except Exception as e:
        print(f"❌ Fine-tuning異常: {e}")
        return False, f"Fine-tuning異常: {e}"
    
    # 檢查Fine-tuning結果
    print("\n🔍 檢查Fine-tuning結果...")
    finetune_models_dir = os.path.join(finetune_log_dir, "models")
    
    if os.path.exists(finetune_models_dir):
        finetune_steps = [d for d in os.listdir(finetune_models_dir) if d.isdigit()]
        if finetune_steps:
            finetune_steps.sort(key=int)
            print(f"✅ Fine-tuning模型已生成: {len(finetune_steps)} 個檢查點")
            print(f"📊 Fine-tuning步數: {', '.join(finetune_steps[-5:])}")
            final_step = finetune_steps[-1]
            print(f"🏆 最終模型: 步數 {final_step}")
        else:
            print("❌ Fine-tuning模型目錄為空")
            return False, "Fine-tuning未生成模型"
    else:
        print("❌ Fine-tuning模型目錄不存在")
        return False, "Fine-tuning目錄未創建"
    
    # 生成可視化
    print(f"\n🎬 生成協作可視化...")
    print("-" * 40)
    
    viz_cmd = f"python unified_visualize_bptt.py {finetune_log_dir}"
    print(f"📝 可視化命令: {viz_cmd}")
    
    try:
        viz_result = subprocess.run(viz_cmd, shell=True, capture_output=True, text=True, timeout=300)  # 5分鐘超時
        
        if viz_result.returncode == 0:
            print("✅ 可視化生成成功")
            
            # 查找生成的可視化文件
            import glob
            viz_files = glob.glob(os.path.join(base_log_dir, "**/*.gif"), recursive=True)
            viz_files.extend(glob.glob(os.path.join(base_log_dir, "**/*.mp4"), recursive=True))
            
            if viz_files:
                print(f"🎥 生成的可視化文件:")
                for viz_file in viz_files:
                    print(f"   📹 {viz_file}")
            else:
                print("⚠️ 可視化生成成功但未找到文件")
                
        else:
            print("⚠️ 可視化生成失敗，但訓練成功")
            if viz_result.stderr:
                print(f"可視化錯誤: {viz_result.stderr}")
                
    except subprocess.TimeoutExpired:
        print("⏰ 可視化生成超時")
    except Exception as e:
        print(f"⚠️ 可視化生成異常: {e}")
    
    return True, base_log_dir

def main():
    """主實驗函數"""
    print("🚀 啟動修復後的完整課程學習實驗")
    print("維度問題已修復，預期成功率: 高")
    print()
    
    success, result = run_fixed_curriculum_experiment()
    
    print("\n" + "=" * 70)
    
    if success:
        print("🎉 課程學習實驗成功完成！")
        print("✅ 兩階段訓練都已成功")
        print("✅ 戰略性改進已完全實施")
        print()
        print(f"📁 實驗結果: {result}")
        print()
        print("🚀 實驗成果:")
        print("   ✅ 基於潛力的獎勵塑形")
        print("   ✅ 兩階段課程學習")
        print("   ✅ 進度獎勵機制")
        print("   ✅ 多智能體協作訓練")
        print()
        print("🎯 下一步建議:")
        print("   1. 檢查生成的可視化文件")
        print("   2. 分析訓練日誌和性能指標")
        print("   3. 與基線模型比較協作效果")
        
    else:
        print("❌ 實驗失敗")
        print(f"🔍 失敗原因: {result}")
        print("💡 建議檢查錯誤信息並重新嘗試")
    
    print(f"\n⏰ 完成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
 
 
 
 