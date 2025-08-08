@echo off
echo.
echo ====================================================================
echo      正在启动最终的、带唯一路径的自动化训练与评估...
echo ====================================================================
echo.

REM --- 训练并评估配置 A ---
echo [1/3] 正在重新训练模型 A ...
python train_smooth_control.py --config config/rebalance_A.yaml
echo.
echo 正在评估模型 A ...
python evaluate_with_logging.py --model-dir logs/rebalance_A_training/models/2000 --config config/rebalance_A.yaml --episodes 5 --auto-plot --output-dir evaluation_A
echo.
echo      模型 A 处理完成。
echo.

REM --- 训练并评估配置 B ---
echo [2/3] 正在重新训练模型 B ...
python train_smooth_control.py --config config/rebalance_B.yaml
echo.
echo 正在评估模型 B ...
python evaluate_with_logging.py --model-dir logs/rebalance_B_training/models/2000 --config config/rebalance_B.yaml --episodes 5 --auto-plot --output-dir evaluation_B
echo.
echo      模型 B 处理完成。
echo.

REM --- 训练并评估配置 C ---
echo [3/3] 正在重新训练模型 C ...
python train_smooth_control.py --config config/rebalance_C.yaml
echo.
echo 正在评估模型 C ...
python evaluate_with_logging.py --model-dir logs/rebalance_C_training/models/2000 --config config/rebalance_C.yaml --episodes 5 --auto-plot --output-dir evaluation_C
echo.
echo      模型 C 处理完成。
echo.

echo ====================================================================
echo      所有训练和评估已成功完成！请检查 evaluation 文件夹。
echo ====================================================================
echo.
pause
