#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "Launching Config Original Discrete SAC Training (P-core optimized, 4 threads)..."
setsid taskset -c 0,2,4,6 /home/ishmam/miniconda3/envs/flow/bin/python -u train_original.py > train_original.log 2>&1 < /dev/null &
PID=$!
echo $PID > train_original.pid
echo "Training started with PID $PID on CPU cores 0,2,4,6."
echo "Log file: $DIR/train_original.log"
echo "To check progress: python3 check_progress.py"
