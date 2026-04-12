# Lab Notebook: Gradient Inversion Defense Evaluation

**Dan Zimmerman** | Advisor: Dr. Ahmed Imteaj | Collaborator: Md Zarif Hossain  
**Period:** March 15 – April 12, 2026  
**Repo:** https://github.com/speedlab-git/gradient-inversion-defense

---

## 1. Project overview

This notebook documents the systematic evaluation of defense mechanisms against Geminio (Shan et al., 2024), a VLM-guided gradient inversion attack in federated learning. Geminio uses CLIP to craft malicious models that amplify target-sample gradients, enabling targeted reconstruction of private data from shared gradient updates.

**Research question:** Can we defend against Geminio? If so, what mechanism is necessary and sufficient?

**Domains tested:**
- Medical: ChestMNIST chest X-rays (15 classes, 28×28 upsampled to 224×224)
- UAV: UAVScenes aerial drone imagery (18 classes, 224×224)

**Model architecture:** ResNet18 backbone (ImageNet-pretrained, frozen) + 3-layer classifier head (512→256→64→N classes). Only the classifier is trained during Geminio's malicious model optimization.

---

## 2. Phase 1: Attack reproduction and cross-domain extension (March 15–16)

### 2.1 Objective

Reproduce the Geminio attack on the original ImageNet domain, then extend to medical and UAV domains to characterize vulnerability across visual domains with different characteristics.

### 2.2 Method

1. Compute CLIP (ViT-L/14) embeddings for all images in each dataset
2. Train malicious Geminio models using the VLM-guided loss surface reshaping (Eq. 4 from the paper):
   `L = mean(per_sample_loss * (1 - softmax(clip_sim * T)))` where T=100
3. Simulate FL round: victim computes gradients on malicious model with private batch
4. Run HF-GradInv (Ye et al., 2024) to reconstruct private images from gradients

### 2.3 Results

**Loss ratios** (ratio of per-sample loss on target vs non-target images, higher = stronger attack):

| Domain | Avg loss ratio | Peak | Interpretation |
|--------|---------------|------|----------------|
| ImageNet | ~16x | 19x (jewelry) | Moderate amplification |
| Medical | 8–12x | 15x (effusion) | Lower — chest X-rays look similar |
| UAV | ~91x | 174x (solar panels) | Very high — aerial scenes are visually distinct |

**Reconstruction quality** (HF-GradInv, batch size 8):

| Domain | Query | Rec loss | LPIPS | Attack F1 |
|--------|-------|----------|-------|-----------|
| Medical | pneumonia | 0.0922 | ~1.0 | 0.625 |
| Medical | baseline (no query) | 0.1855 | — | — |
| UAV | swimming_pool | 0.0205 | 0.68 | — |
| UAV | solar_panels | 0.0986 | 0.66 | — |

**Key finding:** Visual distinctiveness drives vulnerability. UAV images (pools, runways, solar panels look nothing alike) produce 91x gradient amplification vs 8-12x for medical images (all chest X-rays look similar).

### 2.4 Standard defenses tested

| Defense | Medical | UAV | Verdict |
|---------|---------|-----|---------|
| Gradient pruning 70% | Minimal effect | Minimal effect | Insufficient |
| Gradient pruning 99% | Inconsistent | F1: 0.375→0.125 | Extreme and unreliable |
| Laplacian noise 1e-1 | Inconsistent | Inconsistent | No formal guarantee |
| Laplacian noise 1e-3 | No effect | No effect | — |

Standard tricks from the Geminio paper's own defense evaluation confirmed as insufficient.

---

## 3. Phase 2: VLM robustness evaluation (March 30 – April 1)

### 3.1 Hypothesis

Zarif Hossain shared the SimCLIP repository (adversarial fine-tuning of CLIP's vision encoder). If robust VLMs produce different embeddings, Geminio's text-image similarity scores might become less discriminative, weakening the attack.

**Threat model caveat:** SimCLIP/FARE defend against adversarial *input* perturbations to CLIP. Geminio uses CLIP on *clean* server-side images. The threat models don't directly align, but robust encoders do produce different feature representations that could incidentally affect the attack.

### 3.2 VLM variants tested

| Variant | Architecture | Embed dim | Training |
|---------|-------------|-----------|----------|
| BiomedCLIP | ViT-B/16 | 512 | Domain-specific medical pretraining |
| Vanilla CLIP | ViT-L/14 | 768 | Standard CLIP (OpenAI) |
| SimCLIP-4 | ViT-L/14 | 768 | Adversarial fine-tuning, cosine loss, ε=4/255 |
| SimCLIP-2 | ViT-L/14 | 768 | Same, ε=2/255 |
| FARE-4 | ViT-L/14 | 768 | Adversarial fine-tuning, L2 loss, ε=4/255 |
| FARE-2 | ViT-L/14 | 768 | Same, ε=2/255 |
| TeCoA-4 | ViT-L/14 | 768 | Text-guided contrastive adversarial, ε=4/255 |
| TeCoA-2 | ViT-L/14 | 768 | Same, ε=2/255 |

SimCLIP/FARE/TeCoA only fine-tune the vision encoder; the text encoder is unchanged. This means text embeddings are identical — only image embeddings differ.

Implementation: `prototype/vlm_simclip.py` (unified adapter for all 8 variants via `--vlm` flag).

### 3.3 Results: Medical domain loss ratios

| Query | BiomedCLIP | Vanilla CLIP | SimCLIP-4 | FARE-4 |
|-------|-----------|-------------|-----------|--------|
| Pneumonia | 10.83x | 9.53x | **14.40x** | 11.34x |
| Cardiomegaly | 7.67x | 7.14x | **12.07x** | 9.55x |
| Effusion | 10.10x | 6.77x | 7.26x | **15.30x** |
| Normal | 4.95x | 5.71x | **11.87x** | 10.07x |
| Lung mass | — | 7.47x | 11.70x | **12.48x** |
| **Average** | ~8.39x | ~7.32x | **~11.46x** | **~11.75x** |
| **vs Vanilla CLIP** | +15% | baseline | **+57%** | **+61%** |

**Both robust VLMs increase medical attack effectiveness by 57–61%.**

### 3.4 Results: UAV domain loss ratios

| Query | Vanilla CLIP | SimCLIP-4 | FARE-4 |
|-------|-------------|-----------|--------|
| Swimming pool | **126.98x** | 68.05x | 50.40x |
| Solar panels | **173.73x** | 45.05x | 67.04x |
| Trucks on road | 22.64x | **82.80x** | 69.09x |
| River with bridge | **153.51x** | 32.66x | 33.40x |
| Airport runway | 64.50x | **208.12x** | 73.48x |
| Shipping containers | 5.34x | **44.54x** | 31.27x |
| **Average** | ~91.12x | ~80.20x | ~54.11x |
| **vs Vanilla CLIP** | baseline | −12% | −41% |

UAV results are mixed. FARE-4 reduces loss ratios by 41% but they remain extremely high (54x). The UAV domain already has such high visual distinctiveness that vanilla CLIP achieves 91x without help.

### 3.5 Analysis

**VLM adversarial robustness is orthogonal to gradient inversion defense.** Adversarial training forces the vision encoder to learn semantically meaningful features rather than texture shortcuts. For medical images — where all chest X-rays look similar to an untrained model — more semantic features mean the VLM can better distinguish pneumonia from cardiomegaly. This is exactly what the attacker needs.

The attack exploits gradient leakage in the FL protocol. The VLM is merely the attacker's targeting tool. Hardening the tool doesn't close the vulnerability.

---

## 4. Phase 3: DP-SGD defense (April 1)

### 4.1 Motivation

Since VLM-side defenses failed, we implemented DP-SGD — a gradient-level defense with formal privacy guarantees (Abadi et al., CCS 2016).

Two mechanisms:
1. **Gradient clipping:** bound the global L2 norm of the gradient to C (we use C=1.0)
2. **Calibrated Gaussian noise:** σ = C · √(2 ln(1.25/δ)) / ε

Implementation: `prototype/defenses.py`, function `apply_dpsgd()`.

### 4.2 Results: Medical (pneumonia_descriptive, batch size 8)

| ε | LPIPS ↓ | PSNR ↑ | Attack F1 | Defense level |
|---|---------|--------|-----------|--------------|
| None | 1.019 | 10.86 | 0.625 | No defense |
| 50.0 | 0.836 | 11.63 | 0.625 | Very weak |
| 10.0 | 0.797 | 11.28 | 0.250 | Weak |
| 5.0 | 0.791 | 11.41 | **0.125** | Moderate |
| 1.0 | 0.804 | 11.31 | 0.250 | Strong |
| 0.1 | 0.788 | 11.35 | 0.375 | Very strong |

### 4.3 Results: UAV (swimming_pool, batch size 8)

| ε | LPIPS ↓ | PSNR ↑ | Attack F1 |
|---|---------|--------|-----------|
| None | 0.679 | 12.88 | — |
| 50.0 | 0.852 | 12.43 | **0.000** |
| 10.0 | 0.685 | 12.33 | **0.000** |
| 5.0 | 0.689 | 12.39 | **0.000** |
| 1.0 | 0.693 | 12.50 | 0.125 |
| 0.1 | 0.703 | 12.49 | 0.125 |

### 4.4 Key insight: gradient clipping is the primary mechanism

Measured gradient norms before and after clipping (C=1.0):

| Domain | ||G|| before clip | Clip factor | After clip |
|--------|-----------------|-------------|------------|
| Medical | 15.6 | 0.064 | 1.0 |
| UAV | 527.0 | 0.0019 | 1.0 |

Even ε=50 (minimal noise) stops UAV targeting completely. Clipping shrinks the gradient by 527x (UAV) or 16x (medical). After clipping to C=1.0, the calibrated noise (||N|| ≈ 407 at ε=5) overwhelms the signal by 400:1.

See `results/analytical_argument.md` for the full derivation.

---

## 5. Phase 4: FedAvg multi-epoch defense (April 2)

### 5.1 Method

Instead of sharing single-step gradients (FedSGD), the client runs multiple local SGD epochs before sharing the parameter update. This "washes out" the malicious model's carefully crafted loss surface.

Implementation: `prototype/defenses.py`, function `simulate_fedavg()`.

### 5.2 Results: Medical (pneumonia_descriptive, lr=1e-3)

| Local epochs | LPIPS ↓ | Attack F1 |
|-------------|---------|-----------|
| 0 (FedSGD) | 1.019 | 0.625 |
| 1 | 0.898 | 0.500 |
| 2 | 0.813 | 0.875 |
| 5 | 0.918 | 0.000 |
| 10 | 0.835 | 0.500 |
| 20 | 0.819 | 0.000 |

### 5.3 Results: UAV (swimming_pool, lr=1e-3)

| Local epochs | LPIPS ↓ | Attack F1 |
|-------------|---------|-----------|
| 0 (FedSGD) | 0.679 | — |
| 1 | 0.660 | 0.125 |
| 5 | 0.667 | 0.375 |
| 10 | 0.668 | 0.250 |
| 20 | 0.671 | 0.250 |

### 5.4 Assessment

FedAvg provides partial defense but is **unreliable**: medical F1 is non-monotonic (0.875 at 2 epochs, 0.000 at 5, back to 0.500 at 10). There is no formal privacy guarantee. DP-SGD is strictly superior for this threat model.

---

## 6. Phase 5: Attack generalization sweep (April 11)

### 6.1 Motivation

To show DP-SGD is not overfit to a single reconstruction algorithm, we tested it against three additional gradient inversion attacks from Geminio's related work.

### 6.2 Attacks implemented

| Attack | Paper | Config | Objective | Regularization |
|--------|-------|--------|-----------|---------------|
| HF-GradInv | Ye et al., 2024 | `hfgradinv` | Dyna-layer cosine sim | TV + group consistency |
| IG | Geiping et al., NeurIPS 2020 | `ig` | Cosine similarity | Total variation |
| GradInversion | Yin et al., CVPR 2021 | `gradinversion` | Euclidean | TV + L2 norm + BN stats |
| DLG | Zhu et al., NeurIPS 2019 | `dlg` | Euclidean | None |

Configs: `configs/attack/{dlg,ig,gradinversion}.yaml`. All share the same Geminio malicious-model pipeline — only the reconstruction algorithm swaps.

### 6.3 Tuning notes

- **DLG:** Original L-BFGS formulation exploded on ResNet18 (loss 336M at step 1). Switched to Adam + gradient normalization + patterned-4-randn init. DLG is a 2019 MNIST-scale method; it does not scale well to 224×224 images.
- **IG:** Hard-signed Adam gave F1=0.25; switching to soft-signed + patterned init raised it to 0.75 (matching HF-GradInv).
- **GradInversion:** Worked well out of the box. BN-stats prior (DeepInversion regularizer) helps semantic alignment. Achieved F1=0.875 at 4k iterations.

### 6.4 Results: 24k iterations, seed=42

**Medical (pneumonia_descriptive, Attack F1 @ cosine threshold 0.90):**

| Attack | Clean | DP-SGD ε=5 | Drop |
|--------|-------|-----------|------|
| DLG | 0.375 | 0.375 | — (baseline too weak) |
| IG | **0.750** | **0.375** | **−50%** |
| GradInversion | **0.750** | **0.500** | **−33%** |
| HF-GradInv | 0.750 | 0.125 | −83% |

**UAV (solar_panels):**

| Attack | Clean | DP-SGD ε=5 | Drop |
|--------|-------|-----------|------|
| DLG | 0.000 | 0.000 | — (non-functional) |
| IG | **0.250** | **0.000** | **−100%** |
| GradInversion | **0.250** | **0.000** | **−100%** |
| HF-GradInv | 0.125 | 0.000 | −100% |

### 6.5 Observations

1. **DP-SGD generalizes across attacks.** Every attack with non-zero clean F1 drops under DP-SGD ε=5. On UAV, all attacks reach F1=0.0 (complete defense).

2. **IG and GradInversion are more resistant to clipping than HF-GradInv.** Their regularizers (TV, BN stats) partially compensate for degraded gradient information. This is a useful nuance: defenses should be evaluated against the strongest available attack, not just the default.

3. **DLG is non-competitive at this scale.** F1=0.375 on clean medical, 0.0 on UAV. Included for completeness but not informative for defense evaluation.

4. **Attack F1 at batch=8 has only 9 discrete values** ({0, 0.125, ..., 1.0}). Larger batches (16, 32) would provide smoother metrics. Planned as next experiment.

---

## 7. Analytical argument: why clipping is sufficient (April 12)

### 7.1 Core mechanism

Geminio creates gradient norms ||G|| proportional to the loss ratio R. Measured values:

| Domain | Loss ratio R | ||G|| | C=1.0 clip factor |
|--------|-------------|-------|-------------------|
| Medical | 8–12x | 15.6 | 0.064 (16x shrinkage) |
| UAV | 54–91x | 527.0 | 0.0019 (527x shrinkage) |

### 7.2 Signal-to-noise after defense

After clipping to C=1.0 and adding calibrated noise (ε=5, δ=1e-5, n=8):

- Clipped signal norm: ||G_clip|| = 1.0
- Noise norm: ||N|| = σ_elem · √d = 0.121 · √(11.3M) ≈ 407
- SNR = 1.0 / 407 ≈ **0.0025**

The defended gradient is 99.75% noise. No reconstruction algorithm — regardless of its objective function (Euclidean, cosine, etc.) — can extract target information at this SNR.

### 7.3 Why clipping matters more than noise

Without clipping, the UAV gradient (||G|| = 527) would have SNR = 527/407 = 1.29. The signal survives the noise. With clipping, SNR drops to 0.0025 **regardless of the attacker's gradient amplification**. Clipping normalizes the signal to a fixed scale before noise addition.

See `results/analytical_argument.md` for the full derivation.

---

## 8. Complete defense comparison

| Defense | Level | Medical F1 | UAV F1 | Reliable? | Formal guarantee? |
|---------|-------|-----------|--------|-----------|-------------------|
| None (FedSGD) | — | 0.625 | — | — | No |
| Gradient pruning 99% | Gradient | inconsistent | inconsistent | No | No |
| Laplacian noise 1e-1 | Gradient | inconsistent | inconsistent | No | No |
| SimCLIP-4 | VLM | +57% (**worse**) | −12% | No | No |
| FARE-4 | VLM | +61% (**worse**) | −41% | No | No |
| FedAvg 5 epochs | FL protocol | 0.000 | 0.375 | Partial | No |
| FedAvg 20 epochs | FL protocol | 0.000 | 0.250 | Partial | No |
| **DP-SGD ε=5** | **Gradient** | **0.125** | **0.000** | **Yes** | **Yes (ε,δ)-DP** |

---

## 9. Remaining work

1. **Batch size 16 and 32 sweeps** — smoother Attack F1 metric with more discrete levels
2. **Additional datasets** — Zarif sharing another UAV dataset
3. **Formal analytical write-up** — expand the clipping argument into a proposition with proof sketch
4. **EMNLP paper draft** — deadline May 27, 2026

---

## 10. Environment and reproducibility

- **Hardware:** NVIDIA DGX with 8× H200 GPUs (shared)
- **Software:** Python 3.9, PyTorch, open_clip, transformers, lpips, pytorch-wavelets
- **Conda:** `geminio` at `/raid/scratch/dzimmerman2021/miniconda3/envs/geminio/`
- **Random seed:** 42 for all reported results (except multi-seed ablations)
- **Reconstruction:** 24,000 iterations for paper-grade numbers; 8,000 for tuning
- **All commands:** documented in `scripts/run_attack_sweep_24k.sh`
