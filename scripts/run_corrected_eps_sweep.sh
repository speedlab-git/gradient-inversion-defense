#!/usr/bin/env bash
# Privacy-budget sweep at CORRECTED NBFU calibration (client-level DP, no /n).
# eps ∈ {0.1, 1.0, 10.0, 50.0} (eps=5 already done in run_corrected_nbfu_sweep.sh).
# HF-GradInv only on both domains, seed=42.
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
        --batch-tag "epscorr_e${eps}" \
        --seed "$SEED" \
        --dpsgd-epsilon "$eps" --dpsgd-max-grad-norm 1.0 2>&1 | tee "$logfile" | tail -1
    echo "[$(date +%H:%M:%S)] DONE  ${name}"
}

for EPS in 0.1 1.0 10.0 50.0; do
    run medical pneumonia_descriptive "$EPS" "medical_hfgradinv_eps${EPS}_corr"
    run uav solar_panels "$EPS" "uav_hfgradinv_eps${EPS}_corr"
done

echo "Corrected eps sweep complete."
