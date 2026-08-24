#!/bin/bash
set -e
echo "=== Running Config B Exact Reproduction Pipeline ==="
# 1. Train 50,000 episodes
python train_b.py

# 2. Evaluate 1,000 episodes per task using final checkpoint
python ../common/evaluate_exact.py checkpoints/sac_model_ep_50000.pt --use_task_intent --output_dir ./eval_output
echo "=== Config B Pipeline Complete ==="