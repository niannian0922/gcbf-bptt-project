@echo off
echo ===================================================================
echo           CURRICULUM LEARNING TRAINING PIPELINE
echo    From Foundation to Mastery: A 3-Phase Training Journey
echo ===================================================================
echo.

:: 设置错误处理
setlocal enabledelayedexpansion

:: 记录开始时间
echo Training started at: %date% %time%
echo.

echo ===================================================================
echo                        PHASE 1: FOUNDATION
echo              Building Core Skills (1-3 obstacles)
echo ===================================================================
echo Starting Phase 1: Foundation Training...
echo - Obstacle Count: 1-3 (Basic complexity)
echo - Training Steps: 8000
echo - Focus: Core obstacle avoidance and cooperation
echo.

python train_adaptive_margin.py --config config/curriculum_phase_1.yaml --pretrained CHAMPION_MODEL/models/2000

if errorlevel 1 (
    echo ERROR: Phase 1 training failed!
    echo Please check the logs and fix any issues before continuing.
    pause
    exit /b 1
)

echo.
echo Phase 1 completed successfully!
echo.

echo ===================================================================
echo                        PHASE 2: ADVANCEMENT
echo           Developing Strategic Skills (2-5 obstacles)
echo ===================================================================
echo Starting Phase 2: Advanced Training...
echo - Obstacle Count: 2-5 (Moderate complexity)
echo - Training Steps: 8000  
echo - Focus: Advanced navigation and strategic planning
echo - Pretrained from: Phase 1 model
echo.

python train_adaptive_margin.py --config config/curriculum_phase_2.yaml --pretrained logs/curriculum_phase_1_foundation/models/8000

if errorlevel 1 (
    echo ERROR: Phase 2 training failed!
    echo Please check the logs and fix any issues before continuing.
    pause
    exit /b 1
)

echo.
echo Phase 2 completed successfully!
echo.

echo ===================================================================
echo                        PHASE 3: MASTERY
echo         Ultimate Challenge Training (2-8 obstacles)
echo ===================================================================
echo Starting Phase 3: Mastery Training...
echo - Obstacle Count: 2-8 (Maximum complexity)
echo - Training Steps: 8000
echo - Focus: Expert-level performance under extreme conditions
echo - Pretrained from: Phase 2 model
echo.

python train_adaptive_margin.py --config config/curriculum_phase_3.yaml --pretrained logs/curriculum_phase_2_advanced/models/8000

if errorlevel 1 (
    echo ERROR: Phase 3 training failed!
    echo Please check the logs and fix any issues before continuing.
    pause
    exit /b 1
)

echo.
echo ===================================================================
echo                    CURRICULUM TRAINING COMPLETE!
echo ===================================================================
echo.
echo Training Summary:
echo - Phase 1 (Foundation): COMPLETED
echo - Phase 2 (Advanced): COMPLETED  
echo - Phase 3 (Mastery): COMPLETED
echo.
echo Total Training Steps: 24,000 (8,000 per phase)
echo Final Model Location: logs/curriculum_phase_3_mastery/models/8000
echo.
echo The AI agent has completed its journey from novice to expert!
echo Ready for ultimate performance evaluation.
echo.
echo Training completed at: %date% %time%
echo.

:: 可选：自动运行最终评估
echo ===================================================================
echo                      OPTIONAL: FINAL EVALUATION
echo ===================================================================
choice /C YN /M "Would you like to run a final evaluation of the master model"

if errorlevel 2 (
    echo Final evaluation skipped.
    echo You can run it manually later with:
    echo python evaluate_with_logging.py --model-dir logs/curriculum_phase_3_mastery/models/8000 --config config/curriculum_phase_3.yaml --episodes 5 --auto-plot --output-dir evaluation_CURRICULUM_MASTER
    goto :end
)

if errorlevel 1 (
    echo Starting final evaluation...
    python evaluate_with_logging.py --model-dir logs/curriculum_phase_3_mastery/models/8000 --config config/curriculum_phase_3.yaml --episodes 5 --auto-plot --output-dir evaluation_CURRICULUM_MASTER
    
    if errorlevel 1 (
        echo Warning: Final evaluation encountered errors.
    ) else (
        echo Final evaluation completed successfully!
        echo Results saved to: evaluation_CURRICULUM_MASTER
    )
)

:end
echo.
echo ===================================================================
echo                         MISSION ACCOMPLISHED!
echo        Your AI agent is now a certified obstacle avoidance expert!
echo ===================================================================
pause
