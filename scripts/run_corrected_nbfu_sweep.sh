#!/usr/bin/env bash
# Headline ε=5 sweep with CORRECTED DP calibration (option B: client-level DP per
# round, no /n factor in noise std). Noise is now 8× larger than the prior runs.
# 3 attacks × 2 domains × 1 condition (ε=5) at seed=42.
set -euo pipefail

cd /raid/scratch/dzimmerman2021/geminio/Geminio

PY=/raid/scratch/dzimmerman2021/miniconda3/envs/geminio/bin/python
GPU=${GPU:-1}
ITERS=24000
SEED=42

LOG_DIR=./results/sweep_logs
mkdir -p "$LOG_DIR"

run() {
    local domain=$1 query=$2 attack=$3 name=$4
    local logfile="$LOG_DIR/${name}.log"
    echo "[$(date +%H:%M:%S)] START ${name}"
    "$PY" prototype/reconstruct_${domain}.py \
        --geminio-query "$query" \
        --attack "$attack" \
        --gpu "$GPU" \
        --max-iterations "$ITERS" \
        --batch-tag "nbfu5corr" \
        --seed "$SEED" \
        --dpsgd-epsilon 5.0 --dpsgd-max-grad-norm 1.0 2>&1 | tee "$logfile" | tail -1
    echo "[$(date +%H:%M:%S)] DONE  ${name}"
}

for attack in hfgradinv ig gradinversion; do
    run medical pneumonia_descriptive "$attack" "medical_${attack}_nbfu5corr"
    run uav solar_panels "$attack" "uav_${attack}_nbfu5corr"
done

echo "Corrected NBFU ε=5 sweep complete."
