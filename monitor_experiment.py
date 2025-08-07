#!/usr/bin/env python3
"""
實時監控課程學習實驗進度
"""

import os
import glob
import time
from datetime import datetime

def monitor_experiment():
    """實時監控實驗"""
    print("📡 課程學習實驗實時監控")
    print("=" * 60)
    print("⏰ 監控開始時間:", datetime.now().strftime("%H:%M:%S"))
    print("🔄 每10秒更新一次...\n")
    
    last_status = ""
    
    while True:
        try:
            # 清除屏幕內容，保持標題
            os.system('cls' if os.name == 'nt' else 'clear')
            print("📡 課程學習實驗實時監控")
            print("=" * 60)
            print("⏰ 當前時間:", datetime.now().strftime("%H:%M:%S"))
            print()
            
            # 查找最新實驗目錄
            curriculum_dirs = glob.glob("logs/curriculum_*")
            
            if not curriculum_dirs:
                print("❌ 沒有找到實驗目錄")
                time.sleep(10)
                continue
            
            curriculum_dirs.sort()
            latest_dir = curriculum_dirs[-1]
            
            print(f"📁 監控目錄: {latest_dir}")
            print()
            
            # 檢查階段1: 預訓練
            pretrain_dir = os.path.join(latest_dir, "pretrain")
            pretrain_status = "❌ 未開始"
            pretrain_models = []
            
            if os.path.exists(pretrain_dir):
                models_dir = os.path.join(pretrain_dir, "models")
                if os.path.exists(models_dir):
                    pretrain_models = [d for d in os.listdir(models_dir) if d.isdigit()]
                    if pretrain_models:
                        pretrain_models.sort(key=int)
                        pretrain_status = f"🔄 進行中 (步數: {pretrain_models[-1]})"
                    else:
                        pretrain_status = "🔄 已創建，等待模型"
                else:
                    pretrain_status = "🔄 目錄已創建"
            
            print(f"📚 階段1 - 預訓練: {pretrain_status}")
            if pretrain_models:
                print(f"   📊 已完成步數: {', '.join(pretrain_models[-5:])}")  # 顯示最近5個
            
            # 檢查階段2: Fine-tuning
            finetune_dir = os.path.join(latest_dir, "finetune")
            finetune_status = "⏳ 等待預訓練完成"
            finetune_models = []
            
            if os.path.exists(finetune_dir):
                models_dir = os.path.join(finetune_dir, "models")
                if os.path.exists(models_dir):
                    finetune_models = [d for d in os.listdir(models_dir) if d.isdigit()]
                    if finetune_models:
                        finetune_models.sort(key=int)
                        finetune_status = f"🔄 進行中 (步數: {finetune_models[-1]})"
                    else:
                        finetune_status = "🔄 已創建，等待模型"
                else:
                    finetune_status = "🔄 目錄已創建"
            
            print(f"🎓 階段2 - Fine-tuning: {finetune_status}")
            if finetune_models:
                print(f"   📊 已完成步數: {', '.join(finetune_models[-5:])}")
            
            # 檢查可視化
            viz_files = glob.glob(os.path.join(latest_dir, "**/*.gif"), recursive=True)
            viz_files.extend(glob.glob(os.path.join(latest_dir, "**/*.mp4"), recursive=True))
            
            if viz_files:
                print(f"🎬 可視化文件: {len(viz_files)} 個")
                for viz in viz_files:
                    print(f"   📹 {os.path.basename(viz)}")
            else:
                print("🎬 可視化: ⏳ 等待生成")
            
            # 進度總結
            print("\n" + "="*40)
            if pretrain_models and finetune_models:
                print("🎉 狀態: 兩階段都在進行")
                completion = (len(pretrain_models) + len(finetune_models)) / 50 * 100  # 假設總共50步
                print(f"📈 大致進度: {completion:.1f}%")
            elif pretrain_models:
                print("🔄 狀態: 預訓練階段進行中")
                completion = len(pretrain_models) / 25 * 100  # 假設預訓練25步
                print(f"📈 預訓練進度: {completion:.1f}%")
            else:
                print("🚀 狀態: 實驗剛開始")
            
            print(f"⏰ 下次更新: 10秒後")
            print("💡 按 Ctrl+C 停止監控")
            
            time.sleep(10)
            
        except KeyboardInterrupt:
            print("\n\n👋 監控已停止")
            print("🔍 查看最終狀態: python check_experiment_status.py")
            break
        except Exception as e:
            print(f"\n❌ 監控錯誤: {e}")
            time.sleep(10)

if __name__ == "__main__":
    monitor_experiment()
 
"""
實時監控課程學習實驗進度
"""

import os
import glob
import time
from datetime import datetime

def monitor_experiment():
    """實時監控實驗"""
    print("📡 課程學習實驗實時監控")
    print("=" * 60)
    print("⏰ 監控開始時間:", datetime.now().strftime("%H:%M:%S"))
    print("🔄 每10秒更新一次...\n")
    
    last_status = ""
    
    while True:
        try:
            # 清除屏幕內容，保持標題
            os.system('cls' if os.name == 'nt' else 'clear')
            print("📡 課程學習實驗實時監控")
            print("=" * 60)
            print("⏰ 當前時間:", datetime.now().strftime("%H:%M:%S"))
            print()
            
            # 查找最新實驗目錄
            curriculum_dirs = glob.glob("logs/curriculum_*")
            
            if not curriculum_dirs:
                print("❌ 沒有找到實驗目錄")
                time.sleep(10)
                continue
            
            curriculum_dirs.sort()
            latest_dir = curriculum_dirs[-1]
            
            print(f"📁 監控目錄: {latest_dir}")
            print()
            
            # 檢查階段1: 預訓練
            pretrain_dir = os.path.join(latest_dir, "pretrain")
            pretrain_status = "❌ 未開始"
            pretrain_models = []
            
            if os.path.exists(pretrain_dir):
                models_dir = os.path.join(pretrain_dir, "models")
                if os.path.exists(models_dir):
                    pretrain_models = [d for d in os.listdir(models_dir) if d.isdigit()]
                    if pretrain_models:
                        pretrain_models.sort(key=int)
                        pretrain_status = f"🔄 進行中 (步數: {pretrain_models[-1]})"
                    else:
                        pretrain_status = "🔄 已創建，等待模型"
                else:
                    pretrain_status = "🔄 目錄已創建"
            
            print(f"📚 階段1 - 預訓練: {pretrain_status}")
            if pretrain_models:
                print(f"   📊 已完成步數: {', '.join(pretrain_models[-5:])}")  # 顯示最近5個
            
            # 檢查階段2: Fine-tuning
            finetune_dir = os.path.join(latest_dir, "finetune")
            finetune_status = "⏳ 等待預訓練完成"
            finetune_models = []
            
            if os.path.exists(finetune_dir):
                models_dir = os.path.join(finetune_dir, "models")
                if os.path.exists(models_dir):
                    finetune_models = [d for d in os.listdir(models_dir) if d.isdigit()]
                    if finetune_models:
                        finetune_models.sort(key=int)
                        finetune_status = f"🔄 進行中 (步數: {finetune_models[-1]})"
                    else:
                        finetune_status = "🔄 已創建，等待模型"
                else:
                    finetune_status = "🔄 目錄已創建"
            
            print(f"🎓 階段2 - Fine-tuning: {finetune_status}")
            if finetune_models:
                print(f"   📊 已完成步數: {', '.join(finetune_models[-5:])}")
            
            # 檢查可視化
            viz_files = glob.glob(os.path.join(latest_dir, "**/*.gif"), recursive=True)
            viz_files.extend(glob.glob(os.path.join(latest_dir, "**/*.mp4"), recursive=True))
            
            if viz_files:
                print(f"🎬 可視化文件: {len(viz_files)} 個")
                for viz in viz_files:
                    print(f"   📹 {os.path.basename(viz)}")
            else:
                print("🎬 可視化: ⏳ 等待生成")
            
            # 進度總結
            print("\n" + "="*40)
            if pretrain_models and finetune_models:
                print("🎉 狀態: 兩階段都在進行")
                completion = (len(pretrain_models) + len(finetune_models)) / 50 * 100  # 假設總共50步
                print(f"📈 大致進度: {completion:.1f}%")
            elif pretrain_models:
                print("🔄 狀態: 預訓練階段進行中")
                completion = len(pretrain_models) / 25 * 100  # 假設預訓練25步
                print(f"📈 預訓練進度: {completion:.1f}%")
            else:
                print("🚀 狀態: 實驗剛開始")
            
            print(f"⏰ 下次更新: 10秒後")
            print("💡 按 Ctrl+C 停止監控")
            
            time.sleep(10)
            
        except KeyboardInterrupt:
            print("\n\n👋 監控已停止")
            print("🔍 查看最終狀態: python check_experiment_status.py")
            break
        except Exception as e:
            print(f"\n❌ 監控錯誤: {e}")
            time.sleep(10)

if __name__ == "__main__":
    monitor_experiment()
 
 
 
 