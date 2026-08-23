#!/bin/bash
source /home/ishmam/miniconda3/etc/profile.d/conda.sh
conda activate flow
export SUMO_HOME="/home/ishmam/miniconda3/envs/flow/lib/python3.7/site-packages/sumo"
export PATH="/home/ishmam/miniconda3/envs/flow/bin:$PATH"

cd "/home/ishmam/flow/PROJECTS/1_MultiTask_Intersection/New Runs/B"
nohup python -u resume_train_b.py >> train_b.log 2>&1 &
PID=$!
echo $PID > train_b.pid
echo "Successfully resumed train_b with PID $PID"
