@echo off 
setlocal enabledelayedexpansion 
 
mkdir comparison 2>nul 
 
echo ===== TRAIN BASELINE (Safety-Gated, 5000 steps) ===== 
python train_innovation.py --config config/baseline_safety_gated_5000.yaml 
IF 0 NEQ 0 ( 
  echo Baseline training failed. Exiting. 
  exit /b 1 
) 
 
echo ===== TRAIN CANDIDATE (Probabilistic Safety Shield, 5000 steps) ===== 
python train_probabilistic_shield.py --config config/probabilistic_safety_shield_5000.yaml 
IF 0 NEQ 0 ( 
  echo Candidate training failed. Exiting. 
  exit /b 1 
) 
 
echo ===== EVALUATE BASELINE ===== 
python evaluate_with_logging.py --model-dir logs/baseline_safety_gated_5000/models/5000 --config config/baseline_safety_gated_5000.yaml --episodes 20 --eval-horizon 300 > comparison\baseline_eval.txt 
 
echo ===== EVALUATE CANDIDATE ===== 
python evaluate_with_logging.py --model-dir logs/probabilistic_safety_shield_5000/models/5000 --config config/probabilistic_safety_shield_5000.yaml --episodes 20 --eval-horizon 300 > comparison\candidate_eval.txt 
 
echo ===== SUMMARIZE COMPARISON ===== 
python summarize_comparison.py --baseline comparison\baseline_eval.txt --candidate comparison\candidate_eval.txt --out comparison\final_summary.txt | cat 
 
echo All done. See comparison\final_summary.txt 
exit /b 0
