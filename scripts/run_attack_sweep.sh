#!/usr/bin/env bash
# Defense sweep: DLG / IG / GradInversion vs calibrated DP-SGD (eps=5).
# Existing HF-GradInv results are reused from prior runs. Medical clean
# runs were produced during tuning (batch-tag tune2/tune3/tune4/tune).
set -euo pipefail

cd /raid/scratch/dzimmerman2021/geminio/Geminio

PY=/raid/scratch/dzimmerman2021/miniconda3/envs/geminio/bin/python
GPU=${GPU:-5}
ITERS=${ITERS:-8000}
SEED=42
TAG=sweep

LOG_DIR=./results/sweep_logs
mkdir -p "$LOG_DIR"

run() {
    local domain=$1 query=$2 attack=$3 defense_args=$4 name=$5
    local logfile="$LOG_DIR/${name}.log"
    echo "[$(date +%H:%M:%S)] START ${name}"
    "$PY" prototype/reconstruct_${domain}.py \
        --geminio-query "$query" \
        --attack "$attack" \
        --gpu "$GPU" \
        --max-iterations "$ITERS" \
        --batch-tag "$TAG" \
        --seed "$SEED" \
        $defense_args 2>&1 | tee "$logfile" | tail -1
    echo "[$(date +%H:%M:%S)] DONE  ${name}"
}

# Medical: DP-SGD eps=5 (clean already available from tuning)
for attack in dlg ig gradinversion; do
    run medical pneumonia_descriptive "$attack" \
        "--dpsgd-epsilon 5.0 --dpsgd-max-grad-norm 1.0" \
        "medical_${attack}_dpsgd5"
done

# UAV: clean + DP-SGD eps=5
for attack in dlg ig gradinversion; do
    run uav solar_panels "$attack" "" \
        "uav_${attack}_clean"
done
for attack in dlg ig gradinversion; do
    run uav solar_panels "$attack" \
        "--dpsgd-epsilon 5.0 --dpsgd-max-grad-norm 1.0" \
        "uav_${attack}_dpsgd5"
done

echo "Sweep complete."
