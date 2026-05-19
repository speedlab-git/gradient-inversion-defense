#!/usr/bin/env bash
# Sensitivity to clipping bound C at fixed eps=5. HF-GradInv, seed 42.
# C in {0.1, 0.5, 1.0, 2.0, 5.0} x 2 domains x clean(baseline at C=inf) + NBFU = 10 runs (NBFU only).
set -euo pipefail

cd /raid/scratch/dzimmerman2021/geminio/Geminio

PY=/raid/scratch/dzimmerman2021/miniconda3/envs/geminio/bin/python
GPU=${GPU:-2}
ITERS=24000
SEED=42
EPS=5.0

LOG_DIR=./results/sweep_logs
mkdir -p "$LOG_DIR"

run() {
    local domain=$1 query=$2 C=$3 name=$4
    local logfile="$LOG_DIR/${name}.log"
    echo "[$(date +%H:%M:%S)] START ${name}"
    "$PY" prototype/reconstruct_${domain}.py \
        --geminio-query "$query" \
        --attack hfgradinv \
        --gpu "$GPU" \
        --max-iterations "$ITERS" \
        --batch-tag "Csens_C${C}" \
        --seed "$SEED" \
        --dpsgd-epsilon "$EPS" --dpsgd-max-grad-norm "$C" 2>&1 | tee "$logfile" | tail -1
    echo "[$(date +%H:%M:%S)] DONE  ${name}"
}

for C in 0.1 0.5 1.0 2.0 5.0; do
    run medical pneumonia_descriptive "$C" "medical_Csens_C${C}"
    run uav solar_panels "$C" "uav_Csens_C${C}"
done

echo "C sensitivity sweep complete."
