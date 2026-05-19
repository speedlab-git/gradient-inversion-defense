#!/usr/bin/env bash
# BN train-mode realism check. HF-GradInv, seed 42.
# 2 domains x 2 conditions (clean, NBFU eps=5) = 4 runs in train mode.
set -euo pipefail

cd /raid/scratch/dzimmerman2021/geminio/Geminio

PY=/raid/scratch/dzimmerman2021/miniconda3/envs/geminio/bin/python
GPU=${GPU:-1}
ITERS=24000
SEED=42

LOG_DIR=./results/sweep_logs
mkdir -p "$LOG_DIR"

run() {
    local domain=$1 query=$2 defense_args=$3 name=$4
    local logfile="$LOG_DIR/${name}.log"
    echo "[$(date +%H:%M:%S)] START ${name}"
    "$PY" prototype/reconstruct_${domain}.py \
        --geminio-query "$query" \
        --attack hfgradinv \
        --gpu "$GPU" \
        --max-iterations "$ITERS" \
        --batch-tag "bntrain" \
        --seed "$SEED" \
        --bn-train-mode \
        $defense_args 2>&1 | tee "$logfile" | tail -1
    echo "[$(date +%H:%M:%S)] DONE  ${name}"
}

run medical pneumonia_descriptive "" "medical_hfgradinv_bntrain_clean"
run medical pneumonia_descriptive "--dpsgd-epsilon 5.0 --dpsgd-max-grad-norm 1.0" "medical_hfgradinv_bntrain_nbfu5"
run uav solar_panels "" "uav_hfgradinv_bntrain_clean"
run uav solar_panels "--dpsgd-epsilon 5.0 --dpsgd-max-grad-norm 1.0" "uav_hfgradinv_bntrain_nbfu5"

echo "BN train-mode sweep complete."
