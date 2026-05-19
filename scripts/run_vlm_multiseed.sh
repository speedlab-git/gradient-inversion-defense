#!/usr/bin/env bash
# Multi-seed VLM end-to-end (HF-GradInv, no defense). Seeds 123, 256.
# Seed 42 already done in run_vlm_endtoend.sh. 7 VLMs x 2 domains x 2 seeds = 28 runs.
set -euo pipefail

cd /raid/scratch/dzimmerman2021/geminio/Geminio

PY=/raid/scratch/dzimmerman2021/miniconda3/envs/geminio/bin/python
GPU=${GPU:-1}
ITERS=24000

LOG_DIR=./results/sweep_logs
mkdir -p "$LOG_DIR"

run_med() {
    local vlm=$1 modelfile=$2 seed=$3
    local name="medical_vlm_${vlm}_s${seed}"
    local logfile="$LOG_DIR/${name}.log"
    echo "[$(date +%H:%M:%S)] START ${name}"
    "$PY" prototype/reconstruct_medical.py \
        --geminio-query pneumonia_descriptive \
        --model-path "malicious_models_medical_v2/$modelfile" \
        --attack hfgradinv \
        --gpu "$GPU" \
        --max-iterations "$ITERS" \
        --batch-tag "vlm_${vlm}_s${seed}" \
        --seed "$seed" 2>&1 | tee "$logfile" | tail -1
    echo "[$(date +%H:%M:%S)] DONE  ${name}"
}

run_uav() {
    local vlm=$1 modelfile=$2 seed=$3
    local name="uav_vlm_${vlm}_s${seed}"
    local logfile="$LOG_DIR/${name}.log"
    echo "[$(date +%H:%M:%S)] START ${name}"
    "$PY" prototype/reconstruct_uav.py \
        --geminio-query solar_panels \
        --model-path "malicious_models_uav/$modelfile" \
        --attack hfgradinv \
        --gpu "$GPU" \
        --max-iterations "$ITERS" \
        --batch-tag "vlm_${vlm}_s${seed}" \
        --seed "$seed" 2>&1 | tee "$logfile" | tail -1
    echo "[$(date +%H:%M:%S)] DONE  ${name}"
}

for SEED in 123 256; do
    run_med clip      "Any_chest_X-ray_showing_pneumonia_clip.pt"     "$SEED"
    run_med simclip2  "Any_chest_X-ray_showing_pneumonia_simclip2.pt" "$SEED"
    run_med simclip4  "Any_chest_X-ray_showing_pneumonia_simclip4.pt" "$SEED"
    run_med fare2     "Any_chest_X-ray_showing_pneumonia_fare2.pt"    "$SEED"
    run_med fare4     "Any_chest_X-ray_showing_pneumonia_fare4.pt"    "$SEED"
    run_med tecoa2    "Any_chest_X-ray_showing_pneumonia_tecoa2.pt"   "$SEED"
    run_med tecoa4    "Any_chest_X-ray_showing_pneumonia_tecoa4.pt"   "$SEED"
    run_uav clip      "aerial_drone_image_showing_solar_panels_on_rooftops.pt"          "$SEED"
    run_uav simclip2  "aerial_drone_image_showing_solar_panels_on_rooftops_simclip2.pt" "$SEED"
    run_uav simclip4  "aerial_drone_image_showing_solar_panels_on_rooftops_simclip4.pt" "$SEED"
    run_uav fare2     "aerial_drone_image_showing_solar_panels_on_rooftops_fare2.pt"    "$SEED"
    run_uav fare4     "aerial_drone_image_showing_solar_panels_on_rooftops_fare4.pt"    "$SEED"
    run_uav tecoa2    "aerial_drone_image_showing_solar_panels_on_rooftops_tecoa2.pt"   "$SEED"
    run_uav tecoa4    "aerial_drone_image_showing_solar_panels_on_rooftops_tecoa4.pt"   "$SEED"
done

echo "Multi-seed VLM end-to-end complete."
