#!/usr/bin/env bash
# Multi-seed NBFU at corrected 2C calibration (sensitivity = 2C, not C).
# 3 attacks x 2 domains x 3 seeds (42, 123, 256) = 18 runs.
set -euo pipefail

cd /raid/scratch/dzimmerman2021/geminio/Geminio

PY=/raid/scratch/dzimmerman2021/miniconda3/envs/geminio/bin/python
GPU=${GPU:-1}
ITERS=24000

LOG_DIR=./results/sweep_logs
mkdir -p "$LOG_DIR"

run() {
    local domain=$1 query=$2 attack=$3 seed=$4 name=$5
    local logfile="$LOG_DIR/${name}.log"
    echo "[$(date +%H:%M:%S)] START ${name}"
    "$PY" prototype/reconstruct_${domain}.py \
        --geminio-query "$query" \
        --attack "$attack" \
        --gpu "$GPU" \
        --max-iterations "$ITERS" \
        --batch-tag "nbfu5_2c_s${seed}" \
        --seed "$seed" \
        --dpsgd-epsilon 5.0 --dpsgd-max-grad-norm 1.0 2>&1 | tee "$logfile" | tail -1
    echo "[$(date +%H:%M:%S)] DONE  ${name}"
}

for SEED in 42 123 256; do
    for attack in hfgradinv ig gradinversion; do
        run medical pneumonia_descriptive "$attack" "$SEED" "medical_${attack}_nbfu5_2c_s${SEED}"
        run uav solar_panels "$attack" "$SEED" "uav_${attack}_nbfu5_2c_s${SEED}"
    done
done

echo "2C multi-seed sweep complete."
