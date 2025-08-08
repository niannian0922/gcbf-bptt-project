@echo off
echo.
echo ==========================================================
echo      正在启动自动化批量训练...
echo ==========================================================
echo.

REM --- 训练配置 A (Rebalance A) ---
echo [1/3] 正在训练模型 A (Rebalance A)...
python train_smooth_control.py --config config/rebalance_A.yaml
echo.
echo      模型 A 训练完成。
echo.

REM --- 训练配置 B (Rebalance B) ---
echo [2/3] 正在训练模型 B (Rebalance B)...
python train_smooth_control.py --config config/rebalance_B.yaml
echo.
echo      模型 B 训练完成。
echo.

REM --- 训练配置 C (Rebalance C) ---
echo [3/3] 正在训练模型 C (Rebalance C)...
python train_smooth_control.py --config config/rebalance_C.yaml
echo.
echo      模型 C 训练完成。
echo.

echo ==========================================================
echo      所有训练已成功完成！
echo ==========================================================
echo.
pause
