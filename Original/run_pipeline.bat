@echo off
echo === Running Original Exact Reproduction Pipeline ===
python train_original.py
python ..\common\evaluate_exact.py checkpoints\sac_model_ep_50000.pt --output_dir .\eval_output
echo === Original Pipeline Complete ===
pause