# 強健的數據日誌和離線繪圖系統

## 🎯 系統概述

本系統提供了一個完整的離線數據分析解決方案，可以記錄multi-agent episodes的所有關鍵數據並生成準確的分析圖表。這是評估模型性能的**單一真實來源**。

## 📁 文件結構

```
gcbfplus/utils/episode_logger.py    # 數據記錄器核心類
plot_results.py                     # 獨立繪圖腳本
evaluate_with_logging.py            # 評估腳本（帶數據記錄）
test_logging_system.py              # 測試腳本
```

## 🚀 快速開始

### 1. 使用訓練好的模型評估

```bash
# 評估模型並生成episode數據
python evaluate_with_logging.py \
    --model-dir logs/bptt/models/9500 \
    --config config/alpha_medium_obs.yaml \
    --episodes 5 \
    --auto-plot
```

### 2. 分析單個episode數據

```bash
# 生成圖表並顯示
python plot_results.py episode_logs/eval_episode_001_20250807_123456.npz

# 保存圖表到文件
python plot_results.py episode_logs/eval_episode_001_20250807_123456.npz \
    --save-plots --output-dir analysis_results
```

### 3. 在訓練中啟用數據記錄

在您的配置文件中添加：

```yaml
# config/your_config.yaml
enable_episode_logging: true
```

然後在訓練代碼中使用：

```python
from gcbfplus.trainer.bptt_trainer import BPTTTrainer

# 創建trainer時啟用記錄
trainer = BPTTTrainer(env, policy_network, cbf_network, config)

# 使用帶記錄的評估
metrics = trainer.evaluate_with_logging(num_episodes=3, log_episodes=True)
```

## 📊 記錄的數據

每個episode文件 (.npz) 包含：

### 時間序列數據
- **positions**: 智能體位置 `[timesteps, batch, n_agents, pos_dim]`
- **velocities**: 智能體速度 `[timesteps, batch, n_agents, vel_dim]`
- **actions**: 安全濾波後的動作 `[timesteps, batch, n_agents, action_dim]`
- **raw_actions**: 原始策略動作 `[timesteps, batch, n_agents, action_dim]`
- **alpha_values**: CBF alpha參數 `[timesteps, batch, n_agents, 1]`
- **h_values**: CBF障礙函數值 `[timesteps, batch, n_agents, 1]`
- **min_distances**: 到障礙物/其他智能體的最小距離 `[timesteps, batch, n_agents]`
- **goal_distances**: 到目標的距離 `[timesteps, batch, n_agents]`
- **rewards**: 步驟獎勵 `[timesteps, batch, n_agents]`
- **costs**: 步驟代價 `[timesteps, batch, n_agents]`

### 元數據
- **episode_id**: 唯一episode標識符
- **final_status**: 結果 ("SUCCESS", "COLLISION", "TIMEOUT")
- **total_steps**: 總步數
- **obstacles**: 障礙物位置和半徑
- **goals**: 目標位置
- **safety_radius**: 安全半徑

## 🎨 生成的圖表

### 1. 3D軌跡圖
- 顯示每個智能體的完整軌跡
- 障礙物（圓柱體）
- 目標位置（星形標記）
- 起始/結束點標記

### 2. 安全距離圖
- 每個智能體到最近障礙物的距離
- 安全半徑閾值線
- 碰撞區域高亮

### 3. CBF分析圖（如果可用）
- CBF h-values隨時間變化
- Alpha參數變化
- 安全邊界標記

### 4. 綜合分析圖
- 四面板視圖：
  - 2D軌跡概覽
  - 安全距離
  - 目標距離
  - 動作大小

## 🔧 API使用

### EpisodeLogger 類

```python
from gcbfplus.utils.episode_logger import EpisodeLogger

# 創建記錄器
logger = EpisodeLogger(log_dir="my_logs", prefix="experiment")

# 開始episode
episode_id = logger.start_episode(
    batch_size=1, n_agents=3,
    obstacles=obstacle_tensor,
    goals=goal_tensor
)

# 記錄每一步
for step in simulation_loop:
    logger.log_step(
        positions=positions,
        velocities=velocities,
        actions=actions,
        # ... 其他數據
    )

# 結束episode
filename = logger.end_episode("SUCCESS")
```

### EpisodePlotter 類

```python
from plot_results import EpisodePlotter

# 創建繪圖器
plotter = EpisodePlotter("episode_data.npz")

# 生成單個圖表
fig = plotter.plot_3d_trajectories()
fig = plotter.plot_safety_distances()
fig = plotter.plot_cbf_analysis()
fig = plotter.plot_comprehensive_analysis()

# 保存所有圖表
plotter.save_all_plots("output_directory")
```

## ✅ 測試系統

運行完整的測試套件：

```bash
python test_logging_system.py
```

測試包括：
1. 合成episode數據生成
2. 所有繪圖功能
3. CLI腳本功能
4. 文件I/O操作

## 📈 使用場景

### 1. 模型性能評估
```bash
# 評估最新的checkpoint
python evaluate_with_logging.py \
    --model-dir logs/bptt/models/latest \
    --config config/alpha_medium_obs.yaml \
    --episodes 10 \
    --auto-plot
```

### 2. 失敗案例分析
```bash
# 分析碰撞episode
python plot_results.py logs/collision_episode_042.npz --plot-type comprehensive
```

### 3. 訓練進度監控
在訓練腳本中定期運行：
```python
# 每1000步記錄一個評估episode
if step % 1000 == 0:
    metrics = trainer.evaluate_with_logging(num_episodes=1)
```

### 4. 研究論文數據
```bash
# 生成高質量圖表用於發表
python plot_results.py best_performance_episode.npz \
    --save-plots --output-dir paper_figures
```

## 🎯 優勢

1. **準確性**: 直接從模型執行記錄，無近似或模擬
2. **完整性**: 記錄所有關鍵變量和元數據
3. **可重現**: 離線分析可重複執行
4. **靈活性**: 多種圖表類型和分析選項
5. **效率**: 壓縮存儲，快速加載
6. **標準化**: 一致的數據格式和API

## 🔍 故障排除

### 常見問題

1. **Unicode錯誤**: 系統已處理Windows中文環境的編碼問題
2. **內存使用**: 大型episodes使用壓縮格式
3. **圖表顯示**: 使用 `--save-plots` 避免GUI問題

### 調試技巧

```python
# 檢查episode數據
import numpy as np
data = np.load("episode.npz")
print("Available keys:", list(data.keys()))
print("Episode status:", data['final_status'])
print("Total steps:", data['total_steps'])
```

## 📝 下一步

1. 集成到您的現有訓練pipeline
2. 為不同實驗創建專用配置
3. 設置自動化分析腳本
4. 建立性能基準數據庫

---

**注意**: 此系統提供了比實時可視化更可靠和詳細的分析能力，是評估multi-agent系統性能的推薦方法。
