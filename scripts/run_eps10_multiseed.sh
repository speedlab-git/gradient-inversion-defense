#!/usr/bin/env bash
# Multi-seed NBFU at epsilon=10 (2C calibration), for the "trainable + defended"
# regime requested by reviewer item #4. Seed 42 is already done; we add 123, 256.
# 1 attack (HF-GradInv) x 2 domains x 2 seeds = 4 runs.
set -euo pipefail

cd /raid/scratch/dzimmerman2021/geminio/Geminio
PY=/raid/scratch/dzimmerman2021/miniconda3/envs/geminio/bin/python
GPU=${GPU:-0}
ITERS=24000

LOG_DIR=./results/sweep_logs
mkdir -p "$LOG_DIR"

run() {
    local domain=$1 query=$2 seed=$3 name=$4
    local logfile="$LOG_DIR/${name}.log"
    echo "[$(date +%H:%M:%S)] START ${name}"
    "$PY" prototype/reconstruct_${domain}.py \
        --geminio-query "$query" \
        --attack hfgradinv \
        --gpu "$GPU" \
        --max-iterations "$ITERS" \
        --batch-tag "nbfu10_2c_s${seed}" \
        --seed "$seed" \
        --dpsgd-epsilon 10.0 --dpsgd-max-grad-norm 1.0 2>&1 | tee "$logfile" | tail -1
    echo "[$(date +%H:%M:%S)] DONE  ${name}"
}

for SEED in 123 256; do
    run medical pneumonia_descriptive "$SEED" "medical_hfgradinv_nbfu10_2c_s${SEED}"
    run uav solar_panels "$SEED" "uav_hfgradinv_nbfu10_2c_s${SEED}"
done

echo "Epsilon=10 multi-seed sweep complete."
