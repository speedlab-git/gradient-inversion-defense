#!/usr/bin/env bash
# Batch size 16/32 sweep: 3 attacks × {clean, DP-SGD eps=5} × {medical, uav}.
# DLG excluded — OOMs on batch≥16 and is non-competitive (F1≤0.375 at batch 8).
set -euo pipefail

cd /raid/scratch/dzimmerman2021/geminio/Geminio

PY=/raid/scratch/dzimmerman2021/miniconda3/envs/geminio/bin/python
GPU=${GPU:-3}
ITERS=24000
SEED=42

LOG_DIR=./results/sweep_logs
mkdir -p "$LOG_DIR"

run() {
    local domain=$1 query=$2 attack=$3 defense_args=$4 batch=$5 name=$6
    local data_dir="./assets/${domain}_batch${batch}/"
    local logfile="$LOG_DIR/${name}.log"
    echo "[$(date +%H:%M:%S)] START ${name}"
    "$PY" prototype/reconstruct_${domain}.py \
        --geminio-query "$query" \
        --attack "$attack" \
        --gpu "$GPU" \
        --max-iterations "$ITERS" \
        --num-samples "$batch" \
        --data-dir "$data_dir" \
        --batch-tag "B${batch}" \
        --seed "$SEED" \
        $defense_args 2>&1 | tee "$logfile" | tail -1
    echo "[$(date +%H:%M:%S)] DONE  ${name}"
}

for BATCH in 16 32; do
    for attack in hfgradinv ig gradinversion; do
        run medical pneumonia_descriptive "$attack" "" "$BATCH" \
            "medical_${attack}_clean_B${BATCH}"
        run medical pneumonia_descriptive "$attack" \
            "--dpsgd-epsilon 5.0 --dpsgd-max-grad-norm 1.0" "$BATCH" \
            "medical_${attack}_dpsgd5_B${BATCH}"
    done

    for attack in hfgradinv ig gradinversion; do
        run uav solar_panels "$attack" "" "$BATCH" \
            "uav_${attack}_clean_B${BATCH}"
        run uav solar_panels "$attack" \
            "--dpsgd-epsilon 5.0 --dpsgd-max-grad-norm 1.0" "$BATCH" \
            "uav_${attack}_dpsgd5_B${BATCH}"
    done
done

echo "Batch sweep complete."
