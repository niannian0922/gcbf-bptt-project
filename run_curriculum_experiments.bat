@echo off
REM 兩階段課程學習實驗管道
REM Phase 1: 預訓練（無障礙物）
REM Phase 2: Fine-tuning（有障礙物）

echo 🎓 開始兩階段課程學習實驗管道
echo ==================================================

REM 設置實驗參數
set DEVICE=cpu
set SEED=42

REM 創建結果目錄
if not exist results mkdir results
if not exist logs\curriculum mkdir logs\curriculum

REM ===== Phase 1: 預訓練階段 =====
echo.
echo 📚 Phase 1: 預訓練階段（無障礙物環境）
echo --------------------------------------------------

REM 實驗1: Alpha Medium 預訓練
echo 🔄 開始 Alpha Medium 預訓練...
python train_bptt.py --config config/alpha_medium_pretrain.yaml --device %DEVICE% --log_dir logs/curriculum/alpha_medium_pretrain --seed %SEED%

if errorlevel 1 (
    echo ❌ Alpha Medium 預訓練失败
    pause
    exit /b 1
) else (
    echo ✅ Alpha Medium 預訓練完成
)

REM 實驗2: Simple Collaboration 預訓練
echo.
echo 🔄 開始 Simple Collaboration 預訓練...
python train_bptt.py --config config/simple_collaboration_pretrain.yaml --device %DEVICE% --log_dir logs/curriculum/simple_collaboration_pretrain --seed %SEED%

if errorlevel 1 (
    echo ❌ Simple Collaboration 預訓練失败
    pause
    exit /b 1
) else (
    echo ✅ Simple Collaboration 預訓練完成
)

REM ===== Phase 2: Fine-tuning階段 =====
echo.
echo 🎯 Phase 2: Fine-tuning階段（有障礙物環境）
echo --------------------------------------------------

REM 實驗3: Alpha Medium Fine-tuning
echo 🔄 開始 Alpha Medium Fine-tuning...
python train_bptt.py --config config/alpha_medium_obs.yaml --device %DEVICE% --log_dir logs/curriculum/alpha_medium_finetune --load_pretrained_model_from logs/curriculum/alpha_medium_pretrain --seed %SEED%

if errorlevel 1 (
    echo ❌ Alpha Medium Fine-tuning失败
    pause
    exit /b 1
) else (
    echo ✅ Alpha Medium Fine-tuning完成
)

REM 實驗4: Simple Collaboration Fine-tuning
echo.
echo 🔄 開始 Simple Collaboration Fine-tuning...
python train_bptt.py --config config/simple_collaboration.yaml --device %DEVICE% --log_dir logs/curriculum/simple_collaboration_finetune --load_pretrained_model_from logs/curriculum/simple_collaboration_pretrain --seed %SEED%

if errorlevel 1 (
    echo ❌ Simple Collaboration Fine-tuning失败
    pause
    exit /b 1
) else (
    echo ✅ Simple Collaboration Fine-tuning完成
)

REM ===== 生成最終可視化 =====
echo.
echo 🎨 生成最終協作可視化
echo --------------------------------------------------

REM 可視化最佳Fine-tuning結果
echo 🔄 生成Alpha Medium課程學習可視化...
python visualize_bptt.py --model_dir logs/curriculum/alpha_medium_finetune --output results/curriculum_alpha_medium_collaboration.gif --device %DEVICE%

echo.
echo 🔄 生成Simple Collaboration課程學習可視化...
python visualize_bptt.py --model_dir logs/curriculum/simple_collaboration_finetune --output results/curriculum_simple_collaboration.gif --device %DEVICE%

REM ===== 實驗總結 =====
echo.
echo 📊 課程學習實驗完成總結
echo ==================================================
echo ✅ Phase 1 預訓練模型:
echo    - Alpha Medium: logs/curriculum/alpha_medium_pretrain
echo    - Simple Collaboration: logs/curriculum/simple_collaboration_pretrain
echo.
echo ✅ Phase 2 Fine-tuning模型:
echo    - Alpha Medium: logs/curriculum/alpha_medium_finetune
echo    - Simple Collaboration: logs/curriculum/simple_collaboration_finetune
echo.
echo 🎨 生成的可視化文件:
echo    - results/curriculum_alpha_medium_collaboration.gif
echo    - results/curriculum_simple_collaboration.gif
echo.
echo 🎓 課程學習實驗管道執行完成！
echo 展示了從簡單環境到複雜障礙環境的學習過程
echo.
pause
 
REM 兩階段課程學習實驗管道
REM Phase 1: 預訓練（無障礙物）
REM Phase 2: Fine-tuning（有障礙物）

echo 🎓 開始兩階段課程學習實驗管道
echo ==================================================

REM 設置實驗參數
set DEVICE=cpu
set SEED=42

REM 創建結果目錄
if not exist results mkdir results
if not exist logs\curriculum mkdir logs\curriculum

REM ===== Phase 1: 預訓練階段 =====
echo.
echo 📚 Phase 1: 預訓練階段（無障礙物環境）
echo --------------------------------------------------

REM 實驗1: Alpha Medium 預訓練
echo 🔄 開始 Alpha Medium 預訓練...
python train_bptt.py --config config/alpha_medium_pretrain.yaml --device %DEVICE% --log_dir logs/curriculum/alpha_medium_pretrain --seed %SEED%

if errorlevel 1 (
    echo ❌ Alpha Medium 預訓練失败
    pause
    exit /b 1
) else (
    echo ✅ Alpha Medium 預訓練完成
)

REM 實驗2: Simple Collaboration 預訓練
echo.
echo 🔄 開始 Simple Collaboration 預訓練...
python train_bptt.py --config config/simple_collaboration_pretrain.yaml --device %DEVICE% --log_dir logs/curriculum/simple_collaboration_pretrain --seed %SEED%

if errorlevel 1 (
    echo ❌ Simple Collaboration 預訓練失败
    pause
    exit /b 1
) else (
    echo ✅ Simple Collaboration 預訓練完成
)

REM ===== Phase 2: Fine-tuning階段 =====
echo.
echo 🎯 Phase 2: Fine-tuning階段（有障礙物環境）
echo --------------------------------------------------

REM 實驗3: Alpha Medium Fine-tuning
echo 🔄 開始 Alpha Medium Fine-tuning...
python train_bptt.py --config config/alpha_medium_obs.yaml --device %DEVICE% --log_dir logs/curriculum/alpha_medium_finetune --load_pretrained_model_from logs/curriculum/alpha_medium_pretrain --seed %SEED%

if errorlevel 1 (
    echo ❌ Alpha Medium Fine-tuning失败
    pause
    exit /b 1
) else (
    echo ✅ Alpha Medium Fine-tuning完成
)

REM 實驗4: Simple Collaboration Fine-tuning
echo.
echo 🔄 開始 Simple Collaboration Fine-tuning...
python train_bptt.py --config config/simple_collaboration.yaml --device %DEVICE% --log_dir logs/curriculum/simple_collaboration_finetune --load_pretrained_model_from logs/curriculum/simple_collaboration_pretrain --seed %SEED%

if errorlevel 1 (
    echo ❌ Simple Collaboration Fine-tuning失败
    pause
    exit /b 1
) else (
    echo ✅ Simple Collaboration Fine-tuning完成
)

REM ===== 生成最終可視化 =====
echo.
echo 🎨 生成最終協作可視化
echo --------------------------------------------------

REM 可視化最佳Fine-tuning結果
echo 🔄 生成Alpha Medium課程學習可視化...
python visualize_bptt.py --model_dir logs/curriculum/alpha_medium_finetune --output results/curriculum_alpha_medium_collaboration.gif --device %DEVICE%

echo.
echo 🔄 生成Simple Collaboration課程學習可視化...
python visualize_bptt.py --model_dir logs/curriculum/simple_collaboration_finetune --output results/curriculum_simple_collaboration.gif --device %DEVICE%

REM ===== 實驗總結 =====
echo.
echo 📊 課程學習實驗完成總結
echo ==================================================
echo ✅ Phase 1 預訓練模型:
echo    - Alpha Medium: logs/curriculum/alpha_medium_pretrain
echo    - Simple Collaboration: logs/curriculum/simple_collaboration_pretrain
echo.
echo ✅ Phase 2 Fine-tuning模型:
echo    - Alpha Medium: logs/curriculum/alpha_medium_finetune
echo    - Simple Collaboration: logs/curriculum/simple_collaboration_finetune
echo.
echo 🎨 生成的可視化文件:
echo    - results/curriculum_alpha_medium_collaboration.gif
echo    - results/curriculum_simple_collaboration.gif
echo.
echo 🎓 課程學習實驗管道執行完成！
echo 展示了從簡單環境到複雜障礙環境的學習過程
echo.
pause
 
 
 
 