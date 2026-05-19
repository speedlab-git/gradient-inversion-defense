#!/usr/bin/env bash
# Multi-seed sigma=0 (clip-only) ablation. Seeds 123 and 256 only;
# seed 42 already done in run_cliponly_ablation.sh.
# 3 attacks x 2 domains x 2 seeds = 12 runs.
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
        --batch-tag "cliponly_s${seed}" \
        --seed "$seed" \
        --dpsgd-epsilon -1 --dpsgd-max-grad-norm 1.0 2>&1 | tee "$logfile" | tail -1
    echo "[$(date +%H:%M:%S)] DONE  ${name}"
}

for SEED in 123 256; do
    for attack in hfgradinv ig gradinversion; do
        run medical pneumonia_descriptive "$attack" "$SEED" "medical_${attack}_cliponly_s${SEED}"
        run uav solar_panels "$attack" "$SEED" "uav_${attack}_cliponly_s${SEED}"
    done
done

echo "Multi-seed sigma=0 ablation complete."
