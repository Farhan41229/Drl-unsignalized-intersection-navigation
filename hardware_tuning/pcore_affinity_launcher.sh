#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
TARGET_SCRIPT="$1"
if [ -z "$TARGET_SCRIPT" ]; then echo "Usage: ./pcore_affinity_launcher.sh <script.py>"; exit 1; fi
shift
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
exec taskset -c 0,2,4,6 python -u "$TARGET_SCRIPT" "$@"
