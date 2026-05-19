#!/usr/bin/env bash
# Sigma=0 (pure clipping) ablation: separates clipping effect from noise effect.
# 3 attacks × 2 domains × 1 condition (clip-only) at seed=42.
set -euo pipefail

cd /raid/scratch/dzimmerman2021/geminio/Geminio

PY=/raid/scratch/dzimmerman2021/miniconda3/envs/geminio/bin/python
GPU=${GPU:-3}
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
        --batch-tag "cliponly" \
        --seed "$SEED" \
        --dpsgd-epsilon -1 --dpsgd-max-grad-norm 1.0 2>&1 | tee "$logfile" | tail -1
    echo "[$(date +%H:%M:%S)] DONE  ${name}"
}

for attack in hfgradinv ig gradinversion; do
    run medical pneumonia_descriptive "$attack" "medical_${attack}_cliponly"
    run uav solar_panels "$attack" "uav_${attack}_cliponly"
done

echo "Clip-only ablation complete."
