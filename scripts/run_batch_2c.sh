#!/usr/bin/env bash
# Batch-size sweep at 2C NBFU calibration (HF-GradInv only).
# Batches 16 and 32 (B8 already at 2C from multi-seed sweep), x 2 domains, x clean+NBFU = 8 runs.
set -euo pipefail

cd /raid/scratch/dzimmerman2021/geminio/Geminio

PY=/raid/scratch/dzimmerman2021/miniconda3/envs/geminio/bin/python
GPU=${GPU:-3}
ITERS=24000
SEED=42

LOG_DIR=./results/sweep_logs
mkdir -p "$LOG_DIR"

run() {
    local domain=$1 query=$2 batch=$3 defense_args=$4 name=$5
    local data_dir="./assets/${domain}_batch${batch}/"
    local logfile="$LOG_DIR/${name}.log"
    echo "[$(date +%H:%M:%S)] START ${name}"
    "$PY" prototype/reconstruct_${domain}.py \
        --geminio-query "$query" \
        --attack hfgradinv \
        --gpu "$GPU" \
        --max-iterations "$ITERS" \
        --num-samples "$batch" \
        --data-dir "$data_dir" \
        --batch-tag "B${batch}_2c" \
        --seed "$SEED" \
        $defense_args 2>&1 | tee "$logfile" | tail -1
    echo "[$(date +%H:%M:%S)] DONE  ${name}"
}

for BATCH in 16 32; do
    run medical pneumonia_descriptive "$BATCH" "" "medical_hfgradinv_clean_B${BATCH}_2c"
    run medical pneumonia_descriptive "$BATCH" \
        "--dpsgd-epsilon 5.0 --dpsgd-max-grad-norm 1.0" \
        "medical_hfgradinv_nbfu5_B${BATCH}_2c"
    run uav solar_panels "$BATCH" "" "uav_hfgradinv_clean_B${BATCH}_2c"
    run uav solar_panels "$BATCH" \
        "--dpsgd-epsilon 5.0 --dpsgd-max-grad-norm 1.0" \
        "uav_hfgradinv_nbfu5_B${BATCH}_2c"
done

echo "Batch-size 2C sweep complete."
