# Gradient Inversion Defense

Evaluating defense mechanisms against VLM-guided gradient inversion attacks in federated learning.

## Key findings

1. **Robust VLMs do not defend against gradient inversion.** SimCLIP, FARE, and TeCoA encoders — designed for adversarial robustness — actually *strengthen* gradient inversion attacks by producing more stable, semantically meaningful gradients.

2. **Calibrated DP-SGD with gradient clipping effectively stops Geminio.** At ε=5 (δ=1e-5, C=1.0), attack F1 drops to 0.0 on UAV data and to 0.125 on medical data. Gradient clipping is the primary mechanism.

3. **Defense generalizes across multiple GIA algorithms.** Tested against four reconstruction methods — HF-GradInv (Ye et al., 2024), Inverting Gradients (Geiping et al., 2020), GradInversion (Yin et al., 2021), and DLG (Zhu et al., 2019). DP-SGD reduces attack F1 for every functional baseline across both domains.

Results:
- [VLM backfire analysis](results/vlm_backfire.md) — loss ratios across VLM variants
- [Defense comparison](results/defense_comparison.md) — DP-SGD, FedAvg, pruning, noise, VLM robustness
- [Attack generalization sweep](results/attack_sweep_summary/) — DP-SGD tested against 4 GIA algorithms
- [Batch size sweep](results/batch_size_sweep.md) — defense margin at batch 8, 16, 32
- [Analytical argument](results/analytical_argument.md) — why gradient clipping is sufficient (SNR derivation)
- [Lab notebook](lab_notebook.md) — full experimental chronology and methods

## Repository structure

```
core/                Our model architectures and VLM integration
prototype/           Training, reconstruction, and defense scripts
configs/attack/      Custom GIA configs (DLG, IG, GradInversion)
scripts/             Sweep automation and results collection
results/             Summary tables
vendor/Geminio/      Upstream Geminio framework (git submodule)
```

All novel code is at the top level. The upstream [Geminio](https://github.com/HKU-TASR/Geminio) framework (Shan et al., 2024) — including the `breaching` GIA library — is vendored as a git submodule under `vendor/`.

### Our contributions

| File | Description |
|------|-------------|
| `prototype/defenses.py` | Calibrated DP-SGD, FedAvg, gradient pruning, gradient noise |
| `prototype/reconstruct_{medical,uav}.py` | Cross-domain reconstruction with `--attack` and defense flags |
| `prototype/train_{medical,uav}.py` | Malicious model training with VLM variant selection |
| `prototype/vlm_simclip.py` | Unified adapter for SimCLIP/FARE/TeCoA evaluation |
| `configs/attack/{dlg,ig,gradinversion}.yaml` | Baseline GIA configs for defense generalization |
| `core/models.py` | ResNet18 + 3-layer classifier head |

## Setup

```bash
git clone --recursive git@github.com:speedlab-git/gradient-inversion-defense.git
cd gradient-inversion-defense
bash setup.sh

conda create -n geminio python=3.9
conda activate geminio
pip install torch torchvision open_clip_torch transformers lpips pytorch-wavelets
```

## Reproducing experiments

### Phase 1: Compute VLM embeddings
```bash
python prototype/compute_embeddings.py --domain medical --vlm clip
python prototype/compute_embeddings.py --domain uav --vlm clip
```

### Phase 2: Train malicious Geminio models
```bash
python prototype/train_medical.py --vlm clip --query "Any chest X-ray showing pneumonia"
python prototype/train_uav.py --vlm clip --query "aerial drone image showing solar panels on rooftops"
```

### Phase 3: Attack + defense evaluation
```bash
# Baseline attack (HF-GradInv, no defense)
python prototype/reconstruct_medical.py --geminio-query pneumonia_descriptive --gpu 0

# Alternative GIA algorithms: dlg, ig, gradinversion
python prototype/reconstruct_medical.py --geminio-query pneumonia_descriptive --attack ig --gpu 0

# With calibrated DP-SGD defense
python prototype/reconstruct_medical.py --geminio-query pneumonia_descriptive --attack ig --gpu 0 \
    --dpsgd-epsilon 5.0 --dpsgd-max-grad-norm 1.0
```

### Full defense generalization sweep
```bash
GPU=0 bash scripts/run_attack_sweep_24k.sh
python scripts/collect_attack_sweep.py --paper
```

## Attack configurations

| Config | Paper | Objective | Regularization |
|--------|-------|-----------|---------------|
| `hfgradinv` | Ye et al., 2024 | Dynamic-layer cosine similarity | TV + group consistency |
| `ig` | Geiping et al., 2020 | Cosine similarity | Total variation |
| `gradinversion` | Yin et al., 2021 | Euclidean | TV + L2 norm + BN stats |
| `dlg` | Zhu et al., 2019 | Euclidean | None |

## Defense mechanisms

Implemented in [`prototype/defenses.py`](prototype/defenses.py):

- **Calibrated DP-SGD** (`--dpsgd-epsilon`): Per-sample gradient clipping + calibrated Gaussian noise.
- **FedAvg** (`--fedavg-epochs`): Multi-epoch local training before gradient upload.
- **Gradient pruning** (`--prune-rate`): Zero out smallest-magnitude gradient entries.
- **Gradient noise** (`--noise-scale`): Additive Laplacian noise.

## References

- Shan et al., "Geminio: Language-Guided Gradient Inversion Attacks in Federated Learning," arXiv:2411.14937, 2024.
- Geiping et al., "Inverting Gradients — How easy is it to break privacy in federated learning?" NeurIPS, 2020.
- Yin et al., "See through Gradients: Image Batch Recovery via GradInversion," CVPR, 2021.
- Zhu et al., "Deep Leakage from Gradients," NeurIPS, 2019.
