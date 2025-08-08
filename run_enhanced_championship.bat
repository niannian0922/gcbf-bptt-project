@echo off
setlocal enabledelayedexpansion

echo.
echo ========================================================================
echo     🏆 终极王者争霸赛 - 智能多Agent系统冠军评估 🏆
echo ========================================================================
echo     🎯 目标: 通过量化KPI找出真正的终极冠军模型
echo     📊 评估方式: 每模型10轮评估 + 自动性能仪表盘生成
echo     🏅 评判标准: 成功率、鲁棒性、效率、安全性综合评分
echo ========================================================================
echo.

REM 创建结果目录
if not exist "CHAMPIONSHIP_RESULTS" mkdir "CHAMPIONSHIP_RESULTS"

echo 🔥 开始王者争霸赛！准备评估 6 个关键模型...
echo.

REM --- 选手1: Rebalance C (稳定基准) ---
echo ========================================================================
echo 🥊 [1/6] 选手: "稳定基准王" (Rebalance C)
echo     📝 描述: 经过精心调参的平衡型基准模型
echo     🎯 预期: 稳定但可能不够激进
echo ========================================================================
python evaluate_with_logging.py ^
    --model-dir logs/rebalance_C_training/models/2000 ^
    --config config/rebalance_C.yaml ^
    --episodes 10 ^
    --auto-plot ^
    --output-dir CHAMPIONSHIP_RESULTS/01_Rebalance_C_Stable_King
if errorlevel 1 (
    echo ❌ "稳定基准王" 评估失败！
) else (
    echo ✅ "稳定基准王" 评估完成！
)
echo.

REM --- 选手2: Innovation 1 (安全门控Alpha) ---
echo ========================================================================
echo 🥊 [2/6] 选手: "安全守护者" (Safety-Gated Alpha)
echo     📝 描述: 创新的安全门控Alpha正则化模型
echo     🎯 预期: 高安全性，可能牺牲部分效率
echo ========================================================================
python evaluate_with_logging.py ^
    --model-dir logs/innovation_safety_gated/models/2000 ^
    --config config/innovation_safety_gated.yaml ^
    --episodes 10 ^
    --auto-plot ^
    --output-dir CHAMPIONSHIP_RESULTS/02_Innovation_Safety_Guardian
if errorlevel 1 (
    echo ❌ "安全守护者" 评估失败！
) else (
    echo ✅ "安全守护者" 评估完成！
)
echo.

REM --- 选手3: Innovation 2 (自适应裕度 - 2000步) ---
echo ========================================================================
echo 🥊 [3/6] 选手: "双重创新者" (Dual Innovation 2K)
echo     📝 描述: 安全门控 + 自适应裕度的双重创新组合
echo     🎯 预期: 理论上的完美平衡，实战待验证
echo ========================================================================
python evaluate_with_logging.py ^
    --model-dir logs/innovation_adaptive_margin/models/2000 ^
    --config config/innovation_adaptive_margin.yaml ^
    --episodes 10 ^
    --auto-plot ^
    --output-dir CHAMPIONSHIP_RESULTS/03_Dual_Innovation_2K
if errorlevel 1 (
    echo ❌ "双重创新者" 评估失败！
) else (
    echo ✅ "双重创新者" 评估完成！
)
echo.

REM --- 选手4: Enhanced Diversity (抗过拟合) ---
echo ========================================================================
echo 🥊 [4/6] 选手: "多样化大师" (Enhanced Diversity)
echo     📝 描述: 在高难度多样化环境中训练的抗过拟合模型
echo     🎯 预期: 强泛化能力，适应复杂环境
echo ========================================================================
python evaluate_with_logging.py ^
    --model-dir logs/enhanced_diversity_anti_overfitting/models/10000 ^
    --config config/enhanced_diversity_training.yaml ^
    --episodes 10 ^
    --auto-plot ^
    --output-dir CHAMPIONSHIP_RESULTS/04_Enhanced_Diversity_Master
if errorlevel 1 (
    echo ❌ "多样化大师" 评估失败！
) else (
    echo ✅ "多样化大师" 评估完成！
)
echo.

REM --- 选手5: Curriculum Learning (课程学习) ---
echo ========================================================================
echo 🥊 [5/6] 选手: "课程学习精英" (Curriculum Master)
echo     📝 描述: 通过三阶段课程学习训练的精英模型
echo     🎯 预期: 循序渐进的学习可能带来最佳综合性能
echo ========================================================================
python evaluate_with_logging.py ^
    --model-dir logs/curriculum_phase_3_mastery/models/8000 ^
    --config config/curriculum_phase_3.yaml ^
    --episodes 10 ^
    --auto-plot ^
    --output-dir CHAMPIONSHIP_RESULTS/05_Curriculum_Elite
if errorlevel 1 (
    echo ❌ "课程学习精英" 评估失败！
) else (
    echo ✅ "课程学习精英" 评估完成！
)
echo.

REM --- 选手6: 检查是否有其他候选模型 ---
if exist "CHAMPION_MODEL" (
    echo ========================================================================
    echo 🥊 [6/6] 选手: "预设冠军" (Pre-selected Champion)
    echo     📝 描述: 之前被标记为最优的冠军模型
    echo     🎯 预期: 需要验证是否仍能保持冠军地位
    echo ========================================================================
    python evaluate_with_logging.py ^
        --model-dir CHAMPION_MODEL ^
        --config CHAMPION_MODEL/config.yaml ^
        --episodes 10 ^
        --auto-plot ^
        --output-dir CHAMPIONSHIP_RESULTS/06_Previous_Champion
    if errorlevel 1 (
        echo ❌ "预设冠军" 评估失败！
    ) else (
        echo ✅ "预设冠军" 评估完成！
    )
) else (
    echo ========================================================================
    echo 🥊 [6/6] 跳过: "预设冠军" (目录不存在)
    echo ========================================================================
)
echo.

echo ========================================================================
echo 🏆 王者争霸赛已结束！正在生成总结报告...
echo ========================================================================
echo.

REM 生成总结
echo 📊 评估结果总结:
echo.
echo 🏅 性能仪表盘位置:
for /d %%d in ("CHAMPIONSHIP_RESULTS\*") do (
    if exist "%%d\CHAMPION_PERFORMANCE_DASHBOARD.png" (
        echo    ✅ %%~nd: %%d\CHAMPION_PERFORMANCE_DASHBOARD.png
    ) else (
        echo    ❌ %%~nd: 仪表盘生成失败
    )
)
echo.

echo 📈 详细分析报告:
for /d %%d in ("CHAMPIONSHIP_RESULTS\*") do (
    if exist "%%d\plots" (
        echo    📁 %%~nd: %%d\plots\
    )
)
echo.

echo 🏆 下一步建议:
echo    1. 查看各模型的 CHAMPION_PERFORMANCE_DASHBOARD.png
echo    2. 比较成功率、鲁棒性得分、完成时间等关键指标
echo    3. 选择综合性能最优的模型作为最终冠军
echo    4. 可运行单独的评估命令获取更多详细信息
echo.

echo ========================================================================
echo 🎊 王者争霸赛完毕！愿最优模型胜出！🎊
echo ========================================================================
echo.
pause
