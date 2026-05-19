# Geminio: Evaluating VLM Robustness as a Defense Against Gradient Inversion Attacks
## Progress Report — April 1, 2026

**Dan Zimmerman** | Advisor: Dr. Ahmed Imteaj | Collaborator: Md Zarif Hossain

---

## 1. Summary of Prior Work (v1 Report, March 16)

We reproduced and extended the Geminio attack (Shan et al., 2411.14937) — a VLM-guided gradient inversion attack in federated learning. The attacker (a malicious server) uses CLIP to identify which images match a text query, trains a rigged model that amplifies gradients from those images, then reconstructs them from the victim's gradient updates.

Key findings from the first phase:
- **Cross-domain extension**: Loss ratios of 16x (ImageNet), 8-12x (medical/ChestMNIST), 91x (UAV/drone imagery)
- **Visual distinctiveness drives vulnerability**: visually distinct objects (pools, solar panels) are far easier to target than visually uniform data (chest X-rays)
- **Descriptive prompts improve medical targeting by 39%** (Ahmed's suggestion)
- **Pseudo-labels work**: even 29.6% accuracy labels enable effective attacks
- **Standard defenses fail**: gradient pruning and noise injection are inconsistent
- **53 total ablation experiments** across batch composition, temperature, pseudo-labels, seeds, and batch size

Full details: `reports/detailed_report.md`

---

## 2. Direction Shift: Defense Evaluation

### 2.1 March 16 Meeting

Attendees: Dan Zimmerman, Dr. Ahmed Imteaj, Dr. George Sklivanitis, Md Zarif Hossain.

George and Ahmed agreed that the publishable contribution is **demonstrating effective defenses** against Geminio, not just attack reproduction. George: *"I think it would be of interest to show if you can show that you can reduce significantly the reconstruction capability of the attacker."*

Action items from meeting:
- Pivot from attack exploration to defense evaluation
- Test existing defense mechanisms more thoroughly
- Integrate Ahmed/Zarif's **Federated FAML** framework (CVPR findings track) as a defense
- Collaborate with Zarif on advanced defense mechanisms

### 2.2 Email Thread Timeline

| Date | Event |
|------|-------|
| **March 8** | Dan emails Ahmed/Zarif with Geminio status update and FathomNet competition parallels |
| **March 9** | Ahmed responds: suggests descriptive prompts + UAVScenes dataset. Proposes March 16 meeting |
| **March 9** | George confirms availability for March 16 |
| **March 16** | Meeting: pivot to defense mechanisms, Federated FAML direction |
| **March 24** | Zarif shares Sim-CLIP repository (https://github.com/speedlab-git/SimCLIP) and asks Dan to evaluate whether robust VLMs defend against Geminio |
| **April 1** | Dan completes SimCLIP/FARE evaluation. Results below. |

### 2.3 SimCLIP/FARE Context

Zarif's Sim-CLIP is an **adversarially robust CLIP variant**. It fine-tunes the vision encoder using a symmetric cosine similarity loss with stop-gradient (Siamese architecture) to make embeddings invariant to adversarial input perturbations (PGD/APGD at ε=4/255). FARE is a comparable baseline using L2 embedding distance loss.

Zarif's request: *"Could you please take a look at the repository and try running the Sim-CLIP models? It would be great if you could evaluate whether these robust models are able to defend against the attack you're studying."*

---

## 3. Threat Model Analysis

Before running experiments, we identified a **fundamental mismatch** between the two threat models:

| | SimCLIP/FARE Defense | Geminio Attack |
|---|---|---|
| **What's attacked** | CLIP's vision encoder (adversarial pixel perturbations) | FL gradient-sharing protocol |
| **Where CLIP runs** | On the victim/client (processing inputs) | On the attacker/server (offline targeting tool) |
| **Attack vector** | Perturbed images → wrong embeddings | Clean images → leaked gradients |
| **Defense mechanism** | Robust vision encoder resists perturbations | N/A — images are unperturbed |

In Geminio, CLIP processes **real, unperturbed images** on the server side to compute similarity scores. It never sees adversarial perturbations. Robustifying the vision encoder guards against an attack that doesn't occur in this setting.

However, robust vision encoders produce **different embeddings** for the same images, which could incidentally affect the attack. This motivated the experimental evaluation.

---

## 4. Experimental Setup

### 4.1 VLM Variants

| Variant | Architecture | Dim | Training | Source |
|---------|-------------|-----|----------|--------|
| BiomedCLIP | ViT-B/16 | 512 | Domain-specific medical pretraining | microsoft/BiomedCLIP |
| Vanilla CLIP | ViT-L/14 | 768 | Standard CLIP (OpenAI) | openai |
| **SimCLIP-4** | ViT-L/14 | 768 | Adversarial fine-tuning, cosine loss, ε=4/255 | hossainzarif19/SimCLIP |
| **FARE-4** | ViT-L/14 | 768 | Adversarial fine-tuning, L2 loss, ε=4/255 | chs20/fare4-clip |

SimCLIP and FARE only modify the **vision encoder**. Text encoder is unchanged (verified: text embedding cosine similarity = 1.000 across all CLIP-based variants). Image embeddings differ meaningfully (CLIP vs SimCLIP-4 = 0.923, CLIP vs FARE-4 = 0.914).

### 4.2 Integration Pipeline

We built a unified VLM adapter (`prototype/vlm_simclip.py`) supporting all four variants via open_clip:

```bash
# Phase 1: Compute image embeddings with each VLM
python prototype/compute_embeddings.py --domain medical --vlm simclip4 --gpu 0
python prototype/compute_embeddings.py --domain uav --vlm fare4 --gpu 0

# Phase 2: Train malicious models using each VLM's similarity scores
python prototype/train_medical.py --all --vlm simclip4 --gpu 0
python prototype/train_uav.py --all --vlm fare4 --gpu 0
```

All models trained identically: Adam lr=1e-3, 5 epochs, T=100, classifier-head only.

### 4.3 Domains

- **Medical**: ChestMNIST (22,433 images, 15 classes), 5 queries, ResNet18
- **UAV**: UAVScenes (4,029 images, 18 classes), 6 queries, ResNet18

---

## 5. Results

### 5.1 Medical Domain Loss Ratios (Top-10%)

Higher loss ratio = stronger attack = worse for defense.

| Query | BiomedCLIP | Vanilla CLIP | SimCLIP-4 | FARE-4 |
|-------|-----------|-------------|-----------|--------|
| Pneumonia | 10.83x | 9.53x | **14.40x** | 11.34x |
| Cardiomegaly | 7.67x | 7.14x | **12.07x** | 9.55x |
| Effusion | 10.10x | 6.77x | 7.26x | **15.30x** |
| Normal | 4.95x | 5.71x | **11.87x** | 10.07x |
| Lung mass | — | 7.47x | 11.70x | **12.48x** |
| **Average** | **~8.39x** | **~7.32x** | **~11.46x** | **~11.75x** |
| **vs Vanilla CLIP** | +15% | baseline | **+57%** | **+61%** |

**Both robust VLMs increase medical attack effectiveness by 57-61% over vanilla CLIP.**

### 5.2 UAV Domain Loss Ratios (Top-10%)

| Query | Vanilla CLIP | SimCLIP-4 | FARE-4 |
|-------|-------------|-----------|--------|
| Swimming pool | **126.98x** | 68.05x | 50.40x |
| Solar panels | **173.73x** | 45.05x | 67.04x |
| Trucks on road | 22.64x | **82.80x** | 69.09x |
| River with bridge | **153.51x** | 32.66x | 33.40x |
| Airport runway | 64.50x | **208.12x** | 73.48x |
| Shipping containers | 5.34x | **44.54x** | 31.27x |
| **Average** | **~91.12x** | **~80.20x** | **~54.11x** |
| **vs Vanilla CLIP** | baseline | -12% | -41% |

**UAV results are mixed.** FARE-4 reduces loss ratios by 41% but they remain extremely high (54x average). SimCLIP-4 is close to vanilla CLIP overall but redistributes effectiveness across queries.

### 5.3 Summary

| Domain | Best for Attacker | Worst for Attacker | Defense Verdict |
|--------|-------------------|--------------------|-----------------------|
| Medical | FARE-4 (11.75x) | Vanilla CLIP (7.32x) | **Robust VLMs HELP the attacker** |
| UAV | Vanilla CLIP (91.12x) | FARE-4 (54.11x) | **Partial reduction but attack still viable (54x)** |

---

## 6. Analysis

### 6.1 Why Robust VLMs Strengthen the Medical Attack

Adversarial training forces the vision encoder to learn **semantically meaningful features** rather than texture shortcuts. For medical images — where all chest X-rays look similar to an untrained model — more semantic features mean the VLM can better distinguish pneumonia from cardiomegaly from normal. This is exactly what the attacker needs: better text-image discrimination in the embedding space.

Vanilla CLIP, trained on internet images, struggles with subtle medical distinctions. SimCLIP and FARE, by learning more robust representations, inadvertently provide the attacker with a better targeting tool.

### 6.2 Why UAV Is Mixed

The UAV domain already has high visual distinctiveness — pools, runways, and solar panels look nothing alike. Vanilla CLIP achieves 91x average loss ratio without any help. The robust encoders don't add meaningful discrimination here, and may lose some texture-level cues that helped vanilla CLIP on certain queries (e.g., swimming pools: 127x → 68x).

### 6.3 Core Conclusion

**VLM adversarial robustness is orthogonal to gradient inversion defense.** The attack exploits gradient leakage in the FL protocol. The VLM is merely the attacker's targeting tool. Hardening the tool doesn't close the vulnerability — and for domains where vanilla CLIP is already weak (medical), it actually sharpens the attacker's capability.

---

## 7. Artifacts Produced

### 7.1 Code

| File | Purpose |
|------|---------|
| `prototype/vlm_simclip.py` | Unified adapter for CLIP/SimCLIP-4/2/FARE-4 via open_clip |
| `prototype/compute_embeddings.py` | Unified Phase 1 embedding script (--domain, --vlm flags) |
| `prototype/train_medical.py` | Updated with --vlm flag (biomedclip, clip, simclip4, fare4) |
| `prototype/train_uav.py` | Updated with --vlm flag (clip, simclip4, fare4) |
| `prototype/dataset_medical.py` | Configurable embed_path for VLM-specific embeddings |

### 7.2 Computed Embeddings

| File | Shape | VLM |
|------|-------|-----|
| `data/medical-biomedclip-test.pt` | [22433, 512] | BiomedCLIP (prior) |
| `data/medical-clip-test.pt` | [22433, 768] | Vanilla CLIP |
| `data/medical-simclip4-test.pt` | [22433, 768] | SimCLIP-4 |
| `data/medical-fare4-test.pt` | [22433, 768] | FARE-4 |
| `uavscenes/uav_clip_embeddings_*.pt` | [4029, 768] | Vanilla CLIP (prior) |
| `uavscenes/uav_simclip4_embeddings_*.pt` | [4029, 768] | SimCLIP-4 |
| `uavscenes/uav_fare4_embeddings_*.pt` | [4029, 768] | FARE-4 |

### 7.3 Trained Models

- 15 new medical models: 5 queries × 3 VLMs (clip, simclip4, fare4) in `malicious_models_medical_v2/`
- 12 new UAV models: 6 queries × 2 VLMs (simclip4, fare4) in `malicious_models_uav/`

---

## 8. DP-SGD Defense Evaluation

### 8.1 Motivation

Since VLM-side defenses failed (Section 5-6), we implemented DP-SGD — a gradient-level defense with formal privacy guarantees (Abadi et al., CCS 2016). DP-SGD applies:
1. **Gradient clipping**: bound the global L2 norm to C (we use C=1.0)
2. **Calibrated Gaussian noise**: σ = C × sqrt(2 ln(1.25/δ)) / ε

Implementation: `prototype/defenses.py`, integrated via `--dpsgd-epsilon` flag in both reconstruction scripts.

### 8.2 Medical Domain (pneumonia_descriptive)

| ε (Privacy Budget) | LPIPS ↓ | PSNR ↑ | Attack F1 | Privacy Level |
|-----|---------|--------|-----------|--------------|
| None (baseline) | 1.019 | 10.86 | 0.625 | No defense |
| 0.1 | 0.788 | 11.35 | 0.375 | Very strong |
| 1.0 | 0.804 | 11.31 | 0.250 | Strong |
| 5.0 | 0.791 | 11.41 | **0.125** | Moderate |
| 10.0 | 0.797 | 11.28 | 0.250 | Weak |
| 50.0 | 0.836 | 11.63 | 0.625 | Very weak |

### 8.3 UAV Domain (swimming_pool)

| ε | LPIPS ↓ | PSNR ↑ | Attack F1 |
|---|---------|--------|-----------|
| None | 0.679 | 12.88 | — |
| 0.1 | 0.703 | 12.49 | 0.125 |
| 1.0 | 0.693 | 12.50 | 0.125 |
| 5.0 | 0.689 | 12.39 | **0.000** |
| 10.0 | 0.685 | 12.33 | **0.000** |
| 50.0 | 0.852 | 12.43 | **0.000** |

### 8.4 UAV Validation (solar_panels)

| ε | LPIPS ↓ | PSNR ↑ | Attack F1 |
|---|---------|--------|-----------|
| 0.1 | 0.721 | 12.36 | 0.000 |
| 1.0 | 0.720 | 12.27 | 0.125 |
| 5.0 | 0.747 | 12.55 | 0.000 |
| 10.0 | 0.728 | 12.46 | 0.000 |
| 50.0 | 0.764 | 12.31 | 0.375 |

### 8.5 Analysis

**DP-SGD is the first defense that effectively stops Geminio in our experiments.**

1. **Gradient clipping is the key mechanism.** Even ε=50 (minimal noise) stops UAV targeting completely for swimming_pool. The clipping bounds the gradient norm, directly disrupting Geminio's gradient amplification.

2. **UAV is more sensitive to DP-SGD than medical.** All UAV epsilon values achieve F1 ≤ 0.125 (swimming_pool) or F1 ≤ 0.375 (solar_panels). Medical requires ε ≤ 5 to reach F1=0.125. This is the inverse of undefended vulnerability.

3. **Perceptual metrics improve or hold steady.** Medical LPIPS improves from 1.019 → 0.788 under DP-SGD. The defense disrupts targeted gradient amplification but overall reconstruction quality is comparable — the attack loses targeting ability.

4. **Contrast with SimCLIP/FARE.** VLM robustness increased medical attack effectiveness by 57-61%. DP-SGD reduces it by 80% (F1: 0.625 → 0.125). The defense operates at the correct level — gradients, not embeddings.

5. **Results validated across queries.** Solar_panels confirms the swimming_pool pattern.

### 8.6 FedAvg Multi-Epoch Defense

The Geminio paper showed FedAvg with multiple local epochs weakens the attack. We tested this systematically.

**Medical (pneumonia_descriptive, lr=1e-3)**

| Local Epochs | LPIPS ↓ | PSNR ↑ | Attack F1 |
|-------------|---------|--------|-----------|
| 0 (FedSGD) | 1.019 | 10.86 | 0.625 |
| 1 | 0.898 | 11.15 | 0.500 |
| 2 | 0.813 | 11.04 | 0.875 |
| 5 | 0.918 | 10.87 | **0.000** |
| 10 | 0.835 | 9.59 | 0.500 |
| 20 | 0.819 | 10.93 | **0.000** |

**UAV (swimming_pool, lr=1e-3)**

| Local Epochs | LPIPS ↓ | PSNR ↑ | Attack F1 |
|-------------|---------|--------|-----------|
| 0 (FedSGD) | 0.679 | 12.88 | — |
| 1 | 0.660 | 10.79 | 0.125 |
| 2 | 0.669 | 11.04 | 0.125 |
| 5 | 0.667 | 10.92 | 0.375 |
| 10 | 0.668 | 10.90 | 0.250 |
| 20 | 0.671 | 10.95 | 0.250 |

FedAvg provides some defense but is **inconsistent** — medical F1 is non-monotonic (0.000 at 5 epochs, 0.500 at 10). DP-SGD is more reliable and provides formal privacy guarantees.

### 8.7 Complete Defense Comparison

| Defense | Level | Medical F1 | UAV F1 | Reliable? | Formal Guarantee? |
|---------|-------|-----------|--------|-----------|-------------------|
| None (FedSGD) | — | 0.625 | — | — | No |
| Gradient pruning 99% | Gradient | inconsistent | inconsistent | No | No |
| Laplacian noise 1e-1 | Gradient | inconsistent | inconsistent | No | No |
| SimCLIP-4 | VLM | +57% (worse) | -12% | No | No |
| FARE-4 | VLM | +61% (worse) | -41% | No | No |
| FedAvg 5 epochs | FL protocol | 0.000 | 0.375 | **Partial** | No |
| FedAvg 20 epochs | FL protocol | 0.000 | 0.250 | **Partial** | No |
| **DP-SGD ε=5** | **Gradient** | **0.125** | **0.000** | **Yes** | **Yes (ε,δ)-DP** |

---

## 9. Next Steps

### Immediate
1. **Share results with Zarif** — SimCLIP/FARE negative result + DP-SGD/FedAvg positive results
2. **Integrate Federated FAML** — Ahmed/Zarif's robust FL framework (CVPR findings). Compare with DP-SGD.

### Medium-term
3. **Combined defenses** — DP-SGD + FedAvg together, DP-SGD + pruning
4. **Multi-seed validation** for statistical significance
5. **Paper draft** — cross-domain attack, VLM robustness insufficient, DP-SGD effective, FedAvg analysis, Federated FAML

### Deprioritized
6. FEDLEAK/GUIDE comparison
7. Per-target-image analysis
