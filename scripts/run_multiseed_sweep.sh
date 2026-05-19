#!/usr/bin/env bash
# Multi-seed sweep: validate 24k attack results across 2 additional seeds (123, 256).
# 3 competitive attacks × 2 domains × {clean, DP-SGD eps=5} × 2 seeds = 24 runs.
# Existing seed=42 results from run_attack_sweep_24k.sh provide the third seed.
set -euo pipefail

cd /raid/scratch/dzimmerman2021/geminio/Geminio

PY=/raid/scratch/dzimmerman2021/miniconda3/envs/geminio/bin/python
GPU=${GPU:-3}
ITERS=24000

LOG_DIR=./results/sweep_logs
mkdir -p "$LOG_DIR"

run() {
    local domain=$1 query=$2 attack=$3 defense_args=$4 seed=$5 name=$6
    local logfile="$LOG_DIR/${name}.log"
    echo "[$(date +%H:%M:%S)] START ${name}"
    "$PY" prototype/reconstruct_${domain}.py \
        --geminio-query "$query" \
        --attack "$attack" \
        --gpu "$GPU" \
        --max-iterations "$ITERS" \
        --batch-tag "seed${seed}" \
        --seed "$seed" \
        $defense_args 2>&1 | tee "$logfile" | tail -1
    echo "[$(date +%H:%M:%S)] DONE  ${name}"
}

for SEED in 123 256; do
    for attack in hfgradinv ig gradinversion; do
        run medical pneumonia_descriptive "$attack" "" "$SEED" \
            "medical_${attack}_clean_s${SEED}"
        run medical pneumonia_descriptive "$attack" \
            "--dpsgd-epsilon 5.0 --dpsgd-max-grad-norm 1.0" "$SEED" \
            "medical_${attack}_dpsgd5_s${SEED}"
        run uav solar_panels "$attack" "" "$SEED" \
            "uav_${attack}_clean_s${SEED}"
        run uav solar_panels "$attack" \
            "--dpsgd-epsilon 5.0 --dpsgd-max-grad-norm 1.0" "$SEED" \
            "uav_${attack}_dpsgd5_s${SEED}"
    done
done

echo "Multi-seed sweep complete."
