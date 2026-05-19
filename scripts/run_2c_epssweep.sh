#!/usr/bin/env bash
# eps sweep at corrected 2C calibration, HF-GradInv only, seed 42.
set -euo pipefail

cd /raid/scratch/dzimmerman2021/geminio/Geminio

PY=/raid/scratch/dzimmerman2021/miniconda3/envs/geminio/bin/python
GPU=${GPU:-2}
ITERS=24000
SEED=42

LOG_DIR=./results/sweep_logs
mkdir -p "$LOG_DIR"

run() {
    local domain=$1 query=$2 eps=$3 name=$4
    local logfile="$LOG_DIR/${name}.log"
    echo "[$(date +%H:%M:%S)] START ${name}"
    "$PY" prototype/reconstruct_${domain}.py \
        --geminio-query "$query" \
        --attack hfgradinv \
        --gpu "$GPU" \
        --max-iterations "$ITERS" \
        --batch-tag "eps2c_e${eps}" \
        --seed "$SEED" \
        --dpsgd-epsilon "$eps" --dpsgd-max-grad-norm 1.0 2>&1 | tee "$logfile" | tail -1
    echo "[$(date +%H:%M:%S)] DONE  ${name}"
}

for EPS in 0.1 1.0 5.0 10.0 50.0; do
    run medical pneumonia_descriptive "$EPS" "medical_hfgradinv_eps${EPS}_2c"
    run uav solar_panels "$EPS" "uav_hfgradinv_eps${EPS}_2c"
done

echo "2C eps sweep complete."
