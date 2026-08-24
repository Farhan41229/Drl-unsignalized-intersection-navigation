@echo off
echo === Running Config B Exact Reproduction Pipeline ===
python train_b.py
python ..\common\evaluate_exact.py checkpoints\sac_model_ep_50000.pt --use_task_intent --output_dir .\eval_output
echo === Config B Pipeline Complete ===
pause