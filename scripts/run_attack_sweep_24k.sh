#!/usr/bin/env bash
# Paper-grade sweep: DLG / IG / GradInversion at 24k iterations.
# Clean + DP-SGD eps=5 on both medical and UAV.
set -euo pipefail

cd /raid/scratch/dzimmerman2021/geminio/Geminio

PY=/raid/scratch/dzimmerman2021/miniconda3/envs/geminio/bin/python
GPU=${GPU:-7}
ITERS=24000
SEED=42
TAG=24k

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

# Medical: clean + DP-SGD eps=5
for attack in dlg ig gradinversion; do
    run medical pneumonia_descriptive "$attack" "" \
        "medical_${attack}_clean_24k"
    run medical pneumonia_descriptive "$attack" \
        "--dpsgd-epsilon 5.0 --dpsgd-max-grad-norm 1.0" \
        "medical_${attack}_dpsgd5_24k"
done

# UAV: clean + DP-SGD eps=5
for attack in dlg ig gradinversion; do
    run uav solar_panels "$attack" "" \
        "uav_${attack}_clean_24k"
    run uav solar_panels "$attack" \
        "--dpsgd-epsilon 5.0 --dpsgd-max-grad-norm 1.0" \
        "uav_${attack}_dpsgd5_24k"
done

echo "24k sweep complete."
