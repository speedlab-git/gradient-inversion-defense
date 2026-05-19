#!/usr/bin/env bash
# Resume batch sweep from IG B16 (HF-GradInv B16 already done).
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

# B16: remaining attacks (HF-GradInv already done)
for attack in ig gradinversion; do
    run medical pneumonia_descriptive "$attack" "" 16 \
        "medical_${attack}_clean_B16"
    run medical pneumonia_descriptive "$attack" \
        "--dpsgd-epsilon 5.0 --dpsgd-max-grad-norm 1.0" 16 \
        "medical_${attack}_dpsgd5_B16"
done
for attack in hfgradinv ig gradinversion; do
    run uav solar_panels "$attack" "" 16 \
        "uav_${attack}_clean_B16"
    run uav solar_panels "$attack" \
        "--dpsgd-epsilon 5.0 --dpsgd-max-grad-norm 1.0" 16 \
        "uav_${attack}_dpsgd5_B16"
done

# B32: all attacks
for attack in hfgradinv ig gradinversion; do
    run medical pneumonia_descriptive "$attack" "" 32 \
        "medical_${attack}_clean_B32"
    run medical pneumonia_descriptive "$attack" \
        "--dpsgd-epsilon 5.0 --dpsgd-max-grad-norm 1.0" 32 \
        "medical_${attack}_dpsgd5_B32"
done
for attack in hfgradinv ig gradinversion; do
    run uav solar_panels "$attack" "" 32 \
        "uav_${attack}_clean_B32"
    run uav solar_panels "$attack" \
        "--dpsgd-epsilon 5.0 --dpsgd-max-grad-norm 1.0" 32 \
        "uav_${attack}_dpsgd5_B32"
done

echo "Batch sweep complete."
