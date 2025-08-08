@echo off
setlocal enabledelayedexpansion

echo.
echo ========================================================================
echo     🏆🔥 终极王者争霸赛 + 智能分析系统 🔥🏆
echo ========================================================================
echo     🎯 完整流程: 模型评估 → 性能仪表盘 → 横向对比 → 冠军加冕
echo     🚀 自动化程度: 100%% 全自动，无需人工干预
echo     📊 分析深度: 定量KPI + 可视化对比 + 等级评定
echo ========================================================================
echo.

echo 🤖 请选择评估模式:
echo.
echo [1] 🏃‍♂️ 快速评估 (每模型5轮，适合快速验证)
echo [2] 🏋️‍♂️ 标准评估 (每模型10轮，推荐模式)  
echo [3] 🔬 深度评估 (每模型20轮，最高精度)
echo [4] 📊 仅运行分析 (跳过评估，直接分析现有结果)
echo.
set /p choice="请输入选择 [1-4]: "

if "%choice%"=="4" goto analysis_only

REM 设置评估轮数
set episodes=10
if "%choice%"=="1" set episodes=5
if "%choice%"=="3" set episodes=20

echo.
echo 🎯 已选择模式: %choice% (每模型 %episodes% 轮评估)
echo.

REM 创建结果目录
if not exist "CHAMPIONSHIP_RESULTS" mkdir "CHAMPIONSHIP_RESULTS"

echo 🔥 启动终极王者争霸赛！
echo.

REM ==================== 选手 1 ====================
echo ========================================================================
echo 🥊 [1/6] 选手: "稳定基准王" (Rebalance C)
echo     📝 特色: 精心平衡的基准模型，稳定可靠
echo     🏆 目标: 证明传统方法的持久价值
echo ========================================================================

if exist "logs\rebalance_C_training\models\2000" (
    python evaluate_with_logging.py ^
        --model-dir logs/rebalance_C_training/models/2000 ^
        --config config/rebalance_C.yaml ^
        --episodes %episodes% ^
        --auto-plot ^
        --output-dir CHAMPIONSHIP_RESULTS/01_Rebalance_C_Stable_King
    if errorlevel 1 (
        echo ❌ "稳定基准王" 评估遇到错误，但继续进行下一个...
    ) else (
        echo ✅ "稳定基准王" 完成评估！
    )
) else (
    echo ⚠️ "稳定基准王" 模型文件不存在，跳过...
)
echo.

REM ==================== 选手 2 ====================
echo ========================================================================
echo 🥊 [2/6] 选手: "安全守护神" (Safety-Gated Alpha)
echo     📝 特色: 创新安全门控技术，安全至上
echo     🏆 目标: 证明安全创新的实用价值
echo ========================================================================

if exist "logs\innovation_safety_gated\models\2000" (
    python evaluate_with_logging.py ^
        --model-dir logs/innovation_safety_gated/models/2000 ^
        --config config/innovation_safety_gated.yaml ^
        --episodes %episodes% ^
        --auto-plot ^
        --output-dir CHAMPIONSHIP_RESULTS/02_Innovation_Safety_Guardian
    if errorlevel 1 (
        echo ❌ "安全守护神" 评估遇到错误，但继续进行下一个...
    ) else (
        echo ✅ "安全守护神" 完成评估！
    )
) else (
    echo ⚠️ "安全守护神" 模型文件不存在，跳过...
)
echo.

REM ==================== 选手 3 ====================
echo ========================================================================
echo 🥊 [3/6] 选手: "双创新融合者" (Dual Innovation)
echo     📝 特色: 安全门控 + 自适应裕度双重创新
echo     🏆 目标: 展现理论与实践的完美结合
echo ========================================================================

if exist "logs\innovation_adaptive_margin\models\2000" (
    python evaluate_with_logging.py ^
        --model-dir logs/innovation_adaptive_margin/models/2000 ^
        --config config/innovation_adaptive_margin.yaml ^
        --episodes %episodes% ^
        --auto-plot ^
        --output-dir CHAMPIONSHIP_RESULTS/03_Dual_Innovation_Master
    if errorlevel 1 (
        echo ❌ "双创新融合者" 评估遇到错误，但继续进行下一个...
    ) else (
        echo ✅ "双创新融合者" 完成评估！
    )
) else (
    echo ⚠️ "双创新融合者" 模型文件不存在，跳过...
)
echo.

REM ==================== 选手 4 ====================
echo ========================================================================
echo 🥊 [4/6] 选手: "环境适应专家" (Enhanced Diversity)
echo     📝 特色: 高难度多样化环境训练，泛化能力强
echo     🏆 目标: 验证复杂环境训练的优势
echo ========================================================================

if exist "logs\enhanced_diversity_anti_overfitting\models\10000" (
    python evaluate_with_logging.py ^
        --model-dir logs/enhanced_diversity_anti_overfitting/models/10000 ^
        --config config/enhanced_diversity_training.yaml ^
        --episodes %episodes% ^
        --auto-plot ^
        --output-dir CHAMPIONSHIP_RESULTS/04_Enhanced_Diversity_Expert
    if errorlevel 1 (
        echo ❌ "环境适应专家" 评估遇到错误，但继续进行下一个...
    ) else (
        echo ✅ "环境适应专家" 完成评估！
    )
) else (
    echo ⚠️ "环境适应专家" 模型文件不存在，跳过...
)
echo.

REM ==================== 选手 5 ====================
echo ========================================================================
echo 🥊 [5/6] 选手: "课程学习大师" (Curriculum Elite)
echo     📝 特色: 三阶段渐进式学习，系统化训练
echo     🏆 目标: 证明科学训练方法论的威力
echo ========================================================================

if exist "logs\curriculum_phase_3_mastery\models\8000" (
    python evaluate_with_logging.py ^
        --model-dir logs/curriculum_phase_3_mastery/models/8000 ^
        --config config/curriculum_phase_3.yaml ^
        --episodes %episodes% ^
        --auto-plot ^
        --output-dir CHAMPIONSHIP_RESULTS/05_Curriculum_Learning_Master
    if errorlevel 1 (
        echo ❌ "课程学习大师" 评估遇到错误，但继续进行下一个...
    ) else (
        echo ✅ "课程学习大师" 完成评估！
    )
) else (
    echo ⚠️ "课程学习大师" 模型文件不存在，跳过...
)
echo.

REM ==================== 选手 6 ====================
echo ========================================================================
echo 🥊 [6/6] 选手: "卫冕冠军" (Previous Champion)
echo     📝 特色: 之前被认定的最优模型
echo     🏆 目标: 验证是否能够卫冕成功
echo ========================================================================

if exist "CHAMPION_MODEL" (
    python evaluate_with_logging.py ^
        --model-dir CHAMPION_MODEL ^
        --config CHAMPION_MODEL/config.yaml ^
        --episodes %episodes% ^
        --auto-plot ^
        --output-dir CHAMPIONSHIP_RESULTS/06_Previous_Champion_Defense
    if errorlevel 1 (
        echo ❌ "卫冕冠军" 评估遇到错误，但继续分析...
    ) else (
        echo ✅ "卫冕冠军" 完成评估！
    )
) else (
    echo ⚠️ "卫冕冠军" 目录不存在，跳过...
)
echo.

:analysis_only

echo ========================================================================
echo 🧠 启动智能分析系统...
echo ========================================================================

REM 检查是否安装了必要的Python包
python -c "import pandas" 2>nul
if errorlevel 1 (
    echo 📦 安装必要的分析包...
    pip install pandas matplotlib numpy
)

REM 运行智能分析
echo 🔍 正在进行横向对比分析...
python championship_analyzer.py

echo.
echo ========================================================================
echo 🏆 终极王者争霸赛 + 智能分析 完成！
echo ========================================================================
echo.

echo 📊 生成的报告和文件:
echo.
echo 🏅 个人性能仪表盘:
for /d %%d in ("CHAMPIONSHIP_RESULTS\*") do (
    if exist "%%d\CHAMPION_PERFORMANCE_DASHBOARD.png" (
        echo    🏆 %%~nd: %%d\CHAMPION_PERFORMANCE_DASHBOARD.png
    )
)
echo.

echo 📈 横向对比分析:
if exist "CHAMPIONSHIP_RESULTS\CHAMPIONSHIP_COMPARISON_ANALYSIS.png" (
    echo    📊 综合对比图: CHAMPIONSHIP_RESULTS\CHAMPIONSHIP_COMPARISON_ANALYSIS.png
) else (
    echo    ⚠️ 横向对比图生成失败，请检查分析器输出
)
echo.

echo 📁 详细分析数据:
for /d %%d in ("CHAMPIONSHIP_RESULTS\*") do (
    if exist "%%d\plots" (
        echo    📂 %%~nd: %%d\plots\
    )
)
echo.

echo ========================================================================
echo 🎉 恭喜！终极王者争霸赛圆满结束！
echo.
echo 🏆 下一步建议:
echo    1️⃣ 查看各模型的性能仪表盘进行初步比较
echo    2️⃣ 查看横向对比分析图获取全局洞察
echo    3️⃣ 根据KPI指标确定最终冠军
echo    4️⃣ 保存冠军模型到 CHAMPION_MODEL 目录
echo.
echo 🎊 愿实力最强的模型胜出！🎊
echo ========================================================================
echo.
pause
