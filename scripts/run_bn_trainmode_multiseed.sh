#!/usr/bin/env bash
# Multi-seed BN train-mode runs, addressing reviewer item #5.
# Seed 42 was already done; we add 123 and 256.
# HF-GradInv x 2 domains x 2 seeds x {clean, NBFU eps=5} = 8 runs.
set -euo pipefail

cd /raid/scratch/dzimmerman2021/geminio/Geminio
PY=/raid/scratch/dzimmerman2021/miniconda3/envs/geminio/bin/python
GPU=${GPU:-7}
ITERS=24000

LOG_DIR=./results/sweep_logs
mkdir -p "$LOG_DIR"

run() {
    local domain=$1 query=$2 defense_args=$3 seed=$4 name=$5
    local logfile="$LOG_DIR/${name}.log"
    echo "[$(date +%H:%M:%S)] START ${name}"
    "$PY" prototype/reconstruct_${domain}.py \
        --geminio-query "$query" \
        --attack hfgradinv \
        --gpu "$GPU" \
        --max-iterations "$ITERS" \
        --batch-tag "bntrain_ms_s${seed}" \
        --seed "$seed" \
        --bn-train-mode \
        $defense_args 2>&1 | tee "$logfile" | tail -1
    echo "[$(date +%H:%M:%S)] DONE  ${name}"
}

for SEED in 123 256; do
    run medical pneumonia_descriptive ""                                     "$SEED" "medical_hfgradinv_bntrain_clean_s${SEED}"
    run medical pneumonia_descriptive "--dpsgd-epsilon 5.0 --dpsgd-max-grad-norm 1.0" "$SEED" "medical_hfgradinv_bntrain_nbfu5_s${SEED}"
    run uav     solar_panels         ""                                     "$SEED" "uav_hfgradinv_bntrain_clean_s${SEED}"
    run uav     solar_panels         "--dpsgd-epsilon 5.0 --dpsgd-max-grad-norm 1.0" "$SEED" "uav_hfgradinv_bntrain_nbfu5_s${SEED}"
done

echo "BN train-mode multi-seed sweep complete."
