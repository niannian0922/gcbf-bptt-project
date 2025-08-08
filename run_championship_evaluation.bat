@echo off
echo.
echo ====================================================================
echo      欢迎来到"王者争霸赛"！我们将对所有关键模型进行最终对决...
echo ====================================================================
echo.

REM --- 选手1: Rebalance C (我们的第一个稳定基准) ---
echo [1/4] 正在评估 "稳定基准" 模型 (Rebalance C)...
python evaluate_with_logging.py ^
    --model-dir logs/rebalance_C_training/models/2000 ^
    --config config/rebalance_C.yaml ^
    --episodes 10 ^
    --auto-plot ^
    --output-dir CHAMPIONSHIP_RESULTS/Rebalance_C
echo.
echo      "稳定基准" 评估完成。报告已生成至 "CHAMPIONSHIP_RESULTS/Rebalance_C"。
echo.

REM --- 选手2: Innovation 1 (安全门控Alpha) ---
echo [2/4] 正在评估 "安全门控" 模型 (Innovation 1)...
python evaluate_with_logging.py ^
    --model-dir logs/innovation_safety_gated/models/2000 ^
    --config config/innovation_safety_gated.yaml ^
    --episodes 10 ^
    --auto-plot ^
    --output-dir CHAMPIONSHIP_RESULTS/Innovation_Safety_Gated
echo.
echo      "安全门控" 评估完成。报告已生成至 "CHAMPIONSHIP_RESULTS/Innovation_Safety_Gated"。
echo.

REM --- 选手3: Innovation 2 (自适应裕度 - 2000步) ---
echo [3/4] 正在评估 "双重创新 (2k步)" 模型 (Innovation 2)...
python evaluate_with_logging.py ^
    --model-dir logs/innovation_adaptive_margin/models/2000 ^
    --config config/innovation_adaptive_margin.yaml ^
    --episodes 10 ^
    --auto-plot ^
    --output-dir CHAMPIONSHIP_RESULTS/Innovation_Adaptive_Margin_2k
echo.
echo      "双重创新 (2k步)" 评估完成。报告已生成至 "CHAMPIONSHIP_RESULTS/Innovation_Adaptive_Margin_2k"。
echo.

REM --- 选手4: Curriculum Learning (课程学习最终模型) ---
echo [4/4] 正在评估 "课程学习" 模型 (Curriculum Master)...
python evaluate_with_logging.py ^
    --model-dir logs/curriculum_phase_3_mastery/models/8000 ^
    --config config/curriculum_phase_3.yaml ^
    --episodes 10 ^
    --auto-plot ^
    --output-dir CHAMPIONSHIP_RESULTS/Curriculum_Master
echo.
echo      "课程学习" 评估完成。报告已生成至 "CHAMPIONSHIP_RESULTS/Curriculum_Master"。
echo.

echo ====================================================================
echo      "王者争霸赛"已结束！所有性能仪表盘已生成完毕。
echo ====================================================================
echo.
pause
