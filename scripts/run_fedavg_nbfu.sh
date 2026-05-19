#!/usr/bin/env bash
# FedAvg multi-epoch + NBFU combined defense (reviewer #4 item 6).
# FedAvg with 5 local epochs, NBFU at eps=5 applied to the resulting update.
# 2 domains x 3 seeds = 6 runs.
set -euo pipefail

cd /raid/scratch/dzimmerman2021/geminio/Geminio
PY=/raid/scratch/dzimmerman2021/miniconda3/envs/geminio/bin/python
GPU=${GPU:-1}
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
        --batch-tag "fedavg5_nbfu5_s${seed}" \
        --seed "$seed" \
        --fedavg-epochs 5 --fedavg-lr 1e-3 \
        --dpsgd-epsilon 5.0 --dpsgd-max-grad-norm 1.0 2>&1 | tee "$logfile" | tail -2
    echo "[$(date +%H:%M:%S)] DONE  ${name}"
}

SEEDS=${SEEDS:-"42 123 256"}

for SEED in $SEEDS; do
    run medical pneumonia_descriptive "$SEED" "medical_fedavg5_nbfu5_s${SEED}"
    run uav     solar_panels         "$SEED" "uav_fedavg5_nbfu5_s${SEED}"
done

echo "FedAvg + NBFU sweep complete."
