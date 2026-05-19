#!/usr/bin/env bash
# Frozen-backbone ablation requested by reviewer #4.
# Setup: requires_grad=False on the ResNet-18 backbone, so the shared
# gradient covers only the 148K-param classifier head (vs 11.3M with backbone).
# 3 conditions (clean, NBFU eps=5, NBFU eps=10) x 2 domains x 3 seeds = 18 runs.
set -euo pipefail

cd /raid/scratch/dzimmerman2021/geminio/Geminio
PY=/raid/scratch/dzimmerman2021/miniconda3/envs/geminio/bin/python
GPU=${GPU:-1}
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
        --batch-tag "frozen_s${seed}" \
        --seed "$seed" \
        --freeze-backbone \
        $defense_args 2>&1 | tee "$logfile" | tail -2
    echo "[$(date +%H:%M:%S)] DONE  ${name}"
}

SEEDS=${SEEDS:-"42 123 256"}

for SEED in $SEEDS; do
    run medical pneumonia_descriptive ""                                                $SEED "medical_frozen_clean_s${SEED}"
    run medical pneumonia_descriptive "--dpsgd-epsilon 5.0  --dpsgd-max-grad-norm 1.0" $SEED "medical_frozen_nbfu5_s${SEED}"
    run medical pneumonia_descriptive "--dpsgd-epsilon 10.0 --dpsgd-max-grad-norm 1.0" $SEED "medical_frozen_nbfu10_s${SEED}"
    run uav     solar_panels         ""                                                $SEED "uav_frozen_clean_s${SEED}"
    run uav     solar_panels         "--dpsgd-epsilon 5.0  --dpsgd-max-grad-norm 1.0" $SEED "uav_frozen_nbfu5_s${SEED}"
    run uav     solar_panels         "--dpsgd-epsilon 10.0 --dpsgd-max-grad-norm 1.0" $SEED "uav_frozen_nbfu10_s${SEED}"
done

echo "Frozen-backbone ablation complete."
