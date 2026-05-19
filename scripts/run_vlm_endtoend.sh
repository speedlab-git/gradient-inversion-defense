#!/usr/bin/env bash
# End-to-end reconstruction across all 7 VLM variants (vanilla CLIP, SimCLIP-2/4,
# FARE-2/4, TeCoA-2/4). HF-GradInv, no defense, seed 42. Both domains.
# Uses --model-path to point at the VLM-specific malicious model.
set -euo pipefail

cd /raid/scratch/dzimmerman2021/geminio/Geminio

PY=/raid/scratch/dzimmerman2021/miniconda3/envs/geminio/bin/python
GPU=${GPU:-3}
ITERS=24000
SEED=42

LOG_DIR=./results/sweep_logs
mkdir -p "$LOG_DIR"

run_med() {
    local vlm=$1 modelfile=$2
    local name="medical_vlm_${vlm}"
    local logfile="$LOG_DIR/${name}.log"
    echo "[$(date +%H:%M:%S)] START ${name}"
    "$PY" prototype/reconstruct_medical.py \
        --geminio-query pneumonia_descriptive \
        --model-path "malicious_models_medical_v2/$modelfile" \
        --attack hfgradinv \
        --gpu "$GPU" \
        --max-iterations "$ITERS" \
        --batch-tag "vlm_${vlm}" \
        --seed "$SEED" 2>&1 | tee "$logfile" | tail -1
    echo "[$(date +%H:%M:%S)] DONE  ${name}"
}

run_uav() {
    local vlm=$1 modelfile=$2
    local name="uav_vlm_${vlm}"
    local logfile="$LOG_DIR/${name}.log"
    echo "[$(date +%H:%M:%S)] START ${name}"
    "$PY" prototype/reconstruct_uav.py \
        --geminio-query solar_panels \
        --model-path "malicious_models_uav/$modelfile" \
        --attack hfgradinv \
        --gpu "$GPU" \
        --max-iterations "$ITERS" \
        --batch-tag "vlm_${vlm}" \
        --seed "$SEED" 2>&1 | tee "$logfile" | tail -1
    echo "[$(date +%H:%M:%S)] DONE  ${name}"
}

# Medical pneumonia: 7 VLM variants
run_med clip      "Any_chest_X-ray_showing_pneumonia_clip.pt"
run_med simclip2  "Any_chest_X-ray_showing_pneumonia_simclip2.pt"
run_med simclip4  "Any_chest_X-ray_showing_pneumonia_simclip4.pt"
run_med fare2     "Any_chest_X-ray_showing_pneumonia_fare2.pt"
run_med fare4     "Any_chest_X-ray_showing_pneumonia_fare4.pt"
run_med tecoa2    "Any_chest_X-ray_showing_pneumonia_tecoa2.pt"
run_med tecoa4    "Any_chest_X-ray_showing_pneumonia_tecoa4.pt"

# UAV solar_panels: 7 VLM variants
run_uav clip      "aerial_drone_image_showing_solar_panels_on_rooftops.pt"
run_uav simclip2  "aerial_drone_image_showing_solar_panels_on_rooftops_simclip2.pt"
run_uav simclip4  "aerial_drone_image_showing_solar_panels_on_rooftops_simclip4.pt"
run_uav fare2     "aerial_drone_image_showing_solar_panels_on_rooftops_fare2.pt"
run_uav fare4     "aerial_drone_image_showing_solar_panels_on_rooftops_fare4.pt"
run_uav tecoa2    "aerial_drone_image_showing_solar_panels_on_rooftops_tecoa2.pt"
run_uav tecoa4    "aerial_drone_image_showing_solar_panels_on_rooftops_tecoa4.pt"

echo "VLM end-to-end sweep complete."
