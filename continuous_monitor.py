#!/usr/bin/env python3
"""
持續監控課程學習實驗進度
"""

import os
import glob
import time
from datetime import datetime

def clear_screen():
    """清除屏幕"""
    os.system('cls' if os.name == 'nt' else 'clear')

def get_experiment_status():
    """獲取實驗狀態"""
    # 找到最新的fixed_curriculum實驗
    experiment_dirs = glob.glob("logs/fixed_curriculum_*")
    
    if not experiment_dirs:
        return None, "沒有找到實驗"
    
    experiment_dirs.sort()
    latest_exp = experiment_dirs[-1]
    
    status = {
        'exp_dir': latest_exp,
        'exp_name': os.path.basename(latest_exp),
        'pretrain_models': [],
        'finetune_models': [],
        'viz_files': [],
        'log_files': []
    }
    
    # 檢查預訓練
    pretrain_dir = os.path.join(latest_exp, "pretrain")
    if os.path.exists(pretrain_dir):
        models_dir = os.path.join(pretrain_dir, "models")
        if os.path.exists(models_dir):
            status['pretrain_models'] = [d for d in os.listdir(models_dir) if d.isdigit()]
            status['pretrain_models'].sort(key=int)
    
    # 檢查Fine-tuning
    finetune_dir = os.path.join(latest_exp, "finetune")
    if os.path.exists(finetune_dir):
        models_dir = os.path.join(finetune_dir, "models")
        if os.path.exists(models_dir):
            status['finetune_models'] = [d for d in os.listdir(models_dir) if d.isdigit()]
            status['finetune_models'].sort(key=int)
    
    # 檢查可視化文件
    status['viz_files'] = glob.glob(os.path.join(latest_exp, "**/*.gif"), recursive=True)
    status['viz_files'].extend(glob.glob(os.path.join(latest_exp, "**/*.mp4"), recursive=True))
    
    # 檢查日誌文件
    status['log_files'] = glob.glob(os.path.join(latest_exp, "**/*.log"), recursive=True)
    
    return status, None

def display_status(status, iteration):
    """顯示狀態"""
    clear_screen()
    
    print("📡 課程學習實驗持續監控")
    print("=" * 80)
    print(f"⏰ 監控時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (第 {iteration} 次檢查)")
    print(f"📁 實驗目錄: {status['exp_name']}")
    print()
    
    # Phase 1 狀態
    print("📚 Phase 1: 預訓練階段 (無障礙物)")
    print("-" * 50)
    
    pretrain_count = len(status['pretrain_models'])
    if pretrain_count > 0:
        latest_pretrain = status['pretrain_models'][-1]
        progress = int(latest_pretrain) / 2500 * 100
        print(f"🔄 進行中: {pretrain_count} 個檢查點")
        print(f"📊 當前步數: {latest_pretrain}")
        print(f"📈 預訓練進度: {progress:.1f}%")
        
        # 顯示進度條
        bar_length = 30
        filled_length = int(bar_length * progress / 100)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        print(f"📊 進度條: [{bar}] {progress:.1f}%")
        
        if progress >= 100:
            print("✅ 預訓練完成！")
        elif progress >= 50:
            print("🔥 預訓練進展良好")
        else:
            print("🚀 預訓練穩步進行")
    else:
        print("⏳ 等待第一個檢查點...")
        print("💭 這通常需要2-3分鐘")
    
    # Phase 2 狀態
    print(f"\n🎓 Phase 2: Fine-tuning階段 (有障礙物)")
    print("-" * 50)
    
    finetune_count = len(status['finetune_models'])
    if finetune_count > 0:
        latest_finetune = status['finetune_models'][-1]
        print(f"🔄 進行中: {finetune_count} 個檢查點")
        print(f"📊 當前步數: {latest_finetune}")
        print("✅ Fine-tuning已開始")
    elif pretrain_count > 0 and int(status['pretrain_models'][-1]) >= 2500:
        print("🔄 預訓練完成，準備開始Fine-tuning...")
    else:
        print("⏳ 等待預訓練完成...")
    
    # 可視化狀態
    print(f"\n🎬 可視化與結果")
    print("-" * 50)
    
    if status['viz_files']:
        print(f"✅ 已生成 {len(status['viz_files'])} 個可視化文件:")
        for viz in status['viz_files']:
            file_size = os.path.getsize(viz) / 1024  # KB
            print(f"   📹 {os.path.basename(viz)} ({file_size:.1f} KB)")
    else:
        print("⏳ 等待可視化生成...")
        if finetune_count > 0:
            print("💭 可視化將在Fine-tuning完成後生成")
    
    # 總體進度
    print(f"\n📊 總體實驗狀態")
    print("-" * 50)
    
    total_checkpoints = pretrain_count + finetune_count
    
    if total_checkpoints == 0:
        status_text = "🚀 實驗剛開始"
        stage = "初始化"
    elif pretrain_count > 0 and finetune_count == 0:
        status_text = "📚 預訓練階段"
        stage = "Phase 1"
    elif finetune_count > 0 and len(status['viz_files']) == 0:
        status_text = "🎓 Fine-tuning階段"
        stage = "Phase 2"
    elif len(status['viz_files']) > 0:
        status_text = "🎉 實驗完成"
        stage = "完成"
    else:
        status_text = "🔄 進行中"
        stage = "運行中"
    
    print(f"狀態: {status_text}")
    print(f"階段: {stage}")
    print(f"總檢查點: {total_checkpoints}")
    
    # 時間估計
    if pretrain_count > 0:
        pretrain_progress = int(status['pretrain_models'][-1]) / 2500
        if pretrain_progress < 1.0:
            # 估算剩餘時間（假設每500步需要1分鐘）
            remaining_steps = 2500 - int(status['pretrain_models'][-1])
            estimated_minutes = remaining_steps / 500
            print(f"⏱️ 預訓練預計剩餘: {estimated_minutes:.1f} 分鐘")
        elif finetune_count > 0:
            print("⏱️ Fine-tuning進行中，預計2-4分鐘")
        else:
            print("⏱️ 等待Fine-tuning開始...")
    
    # 健康檢查
    print(f"\n💚 健康狀態")
    print("-" * 50)
    
    if total_checkpoints > 0:
        print("✅ 實驗正常運行")
        print("✅ 模型保存正常")
        if len(status['log_files']) > 0:
            print("✅ 日誌記錄正常")
    else:
        elapsed_time = datetime.now().hour * 60 + datetime.now().minute - (16 * 60 + 59)  # 從開始時間計算
        if elapsed_time > 5:
            print("⚠️ 超過5分鐘未見檢查點，可能有問題")
        else:
            print("✅ 實驗正常啟動中")
    
    print(f"\n🔄 自動刷新 (每30秒) | 按 Ctrl+C 停止監控")
    print("💡 實驗大約需要8-12分鐘完成")

def continuous_monitor():
    """持續監控主函數"""
    iteration = 1
    
    try:
        while True:
            status, error = get_experiment_status()
            
            if error:
                clear_screen()
                print("❌ 監控錯誤:", error)
                time.sleep(30)
                continue
            
            display_status(status, iteration)
            
            # 檢查是否完成
            if len(status['viz_files']) > 0:
                print("\n🎉 實驗已完成！")
                print("✅ 課程學習成功")
                print("✅ 可視化已生成")
                
                print(f"\n🚀 實驗成果:")
                print(f"   📁 實驗目錄: {status['exp_dir']}")
                print(f"   📊 預訓練檢查點: {len(status['pretrain_models'])}")
                print(f"   📊 Fine-tuning檢查點: {len(status['finetune_models'])}")
                print(f"   🎬 可視化文件: {len(status['viz_files'])}")
                
                print(f"\n💡 下一步:")
                print("   1. 查看生成的可視化文件")
                print("   2. 分析訓練日誌")
                print("   3. 評估協作效果")
                
                break
            
            # 等待30秒
            time.sleep(30)
            iteration += 1
            
    except KeyboardInterrupt:
        print(f"\n\n👋 監控已停止")
        
        # 顯示最終狀態
        if status:
            print(f"\n📊 停止時的狀態:")
            print(f"   預訓練檢查點: {len(status['pretrain_models'])}")
            print(f"   Fine-tuning檢查點: {len(status['finetune_models'])}")
            print(f"   可視化文件: {len(status['viz_files'])}")
            
            if len(status['pretrain_models']) > 0:
                print(f"   最新預訓練步數: {status['pretrain_models'][-1]}")
            if len(status['finetune_models']) > 0:
                print(f"   最新Fine-tuning步數: {status['finetune_models'][-1]}")
        
        print(f"\n🔄 要重新開始監控，請運行: python continuous_monitor.py")

def main():
    """主函數"""
    print("🚀 啟動課程學習實驗持續監控系統")
    print("這個系統會每30秒自動更新實驗狀態")
    print("按 Ctrl+C 可以隨時停止監控")
    print()
    print("⏰ 3秒後開始監控...")
    time.sleep(3)
    
    continuous_monitor()

if __name__ == "__main__":
    main()
 
 
 
 
 