import os
import glob
import time
from datetime import datetime

exp_dir = "logs/quick_restart_20250806_215903"
print("📊 開始監控實驗...")

for i in range(20):  # 監控10分鐘
    pretrain_models = glob.glob(os.path.join(exp_dir, "pretrain", "**", "*.pkl"), recursive=True)
    print(f"⏰ {datetime.now().strftime('%H:%M:%S')} - 預訓練模型: {len(pretrain_models)} 個")
    
    if pretrain_models:
        print("✅ 預訓練完成！")
        print(f"📦 模型文件: {pretrain_models}")
        break
        
    time.sleep(30)  # 30秒檢查一次
else:
    print("⏰ 監控超時")
