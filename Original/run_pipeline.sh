#!/bin/bash
set -e
echo "=== Running Original Exact Reproduction Pipeline ==="
# 1. Train 50,000 episodes
python train_original.py

# 2. Evaluate 1,000 episodes per task using final checkpoint
python ../common/evaluate_exact.py checkpoints/sac_model_ep_50000.pt --output_dir ./eval_output
echo "=== Original Pipeline Complete ==="