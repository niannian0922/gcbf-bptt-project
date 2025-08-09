@echo off 
set PYTHONUTF8=1
setlocal enabledelayedexpansion 
 
mkdir comparison 2>nul

REM ===== TRAIN BASELINE (Safety-Gated, 5000 steps) =====
REM echo Skipping Baseline Training...
REM python train_innovation.py --config config/baseline_safety_gated_5000.yaml
REM IF %ERRORLEVEL% NEQ 0 (
REM   echo Baseline training failed. Exiting.
REM   exit /b 1
REM )

echo ===== TRAIN CANDIDATE (Probabilistic Safety Shield, 5000 steps) =====
python train_probabilistic_shield.py --config config/probabilistic_safety_shield_5000.yaml 
IF %ERRORLEVEL% NEQ 0 (
  echo Candidate training failed. Exiting. 
  exit /b 1 
) 

REM ===== EVALUATE BASELINE =====
REM echo Skipping Baseline Evaluation...
REM python evaluate_with_logging.py --model-dir logs/baseline_safety_gated_5000/models/5000 --config config/baseline_safety_gated_5000.yaml --episodes 20 --eval-horizon 300 > comparison\baseline_eval.txt

REM ===== EVALUATE CANDIDATE =====
REM echo You can evaluate the candidate after training if needed.
REM python evaluate_with_logging.py --model-dir logs/probabilistic_safety_shield_5000/models/5000 --config config/probabilistic_safety_shield_5000.yaml --episodes 20 --eval-horizon 300 > comparison\candidate_eval.txt

REM ===== SUMMARIZE COMPARISON =====
REM echo Skipping summary (only candidate trained in this run).
 
echo All done. See comparison\final_summary.txt 
exit /b 0
