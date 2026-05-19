# EMNLP 2026 Long Paper — Outline

**Working title:** Norm-Bounded Federated Updates Stop Language-Guided Gradient Inversion Attacks

**Tagline:** Robust VLMs backfire; calibrated gradient clipping suffices.

**Target:** 8 pages content (long paper). ARR deadline May 25, 2026.

---

## 1. Abstract (~200 words)

- Federated learning + gradient inversion threat
- Geminio uses VLMs to target valuable data via natural language
- Two findings:
  (a) Adversarially robust VLMs (SimCLIP, FARE) **strengthen** the attack by 57–61% on medical data
  (b) Calibrated gradient clipping (NBFU/DP-SGD) is sufficient to neutralize Geminio across multiple GIA algorithms
- Analytical argument: clipping normalizes the gradient signal independently of the attacker's loss-ratio amplification
- Cross-domain validation on medical and UAV imagery

## 2. Introduction (~1 page)

**Hook:** Federated learning promises privacy. Geminio (Shan et al., 2024) breaks that promise using natural language queries.

**Threat:** Adversary describes targets in plain English ("show me chest X-rays with pneumonia"); VLM-guided malicious model amplifies target gradients; standard GIAs reconstruct from amplified gradients.

**Why it matters for EMNLP:** Attack interface IS natural language. The attacker's discrimination ability scales with VLM quality. As VLMs improve, this attack class strengthens.

**Contributions:**
1. **Negative result** — VLM adversarial robustness backfires for FL privacy
2. **Positive result** — Calibrated DP-SGD with gradient clipping (NBFU) is sufficient
3. **Generalization** — defense works across 4 GIA reconstruction algorithms
4. **Analysis** — gradient clipping operates orthogonally to attacker amplification

**Roadmap.**

## 3. Background and Threat Model (~1 page)

### 3.1 Federated learning under FedSGD

Standard FedSGD setup. Server sends θ_t; clients send G_t = (1/n) Σ ∇L.

### 3.2 Gradient inversion attacks

Optimization-based GIAs match candidate gradients to victim gradients. Survey: DLG (Zhu 2019), IG (Geiping 2020), GradInversion (Yin 2021), HF-GradInv (Ye 2024).

### 3.3 Geminio: VLM-guided malicious models

Eq. 4 from Geminio paper. Loss-surface reshaping. Result: ||∇L_target|| >> ||∇L_other||.

### 3.4 Threat model

Active server, can modify global model parameters. VLM is server-side (operates on clean inputs, not adversarially perturbed).

## 4. Method: Norm-Bounded Federated Updates (~1.5 pages)

### 4.1 Defense definition

Pre-share clipping + calibrated noise:
- G_clip = G · min(1, C/||G||)
- G_def = G_clip + N, N ~ N(0, σ²I), σ = C·√(2 ln(1.25/δ))/(εn)

### 4.2 Analytical argument: why clipping suffices

**Proposition (informal):** For Geminio with loss ratio R, the post-defense SNR is C/(σ√d), independent of R.

Proof sketch:
- ||G|| ∝ R (by Geminio's mechanism)
- Clipping: ||G_clip|| = C regardless of ||G||
- Noise: ||N|| ≈ σ√d
- SNR = ||G_clip|| / ||N|| = C / (σ√d) — does not depend on R

**Implication:** As VLMs improve and R grows, NBFU effectiveness is preserved. This contrasts with VLM-side defenses (Section 5.2) whose effectiveness depends on R.

### 4.3 Numerical SNR

For our setup (C=1, ε=5, n=8, d=11.3M): SNR ≈ 0.0025. Defended gradient is 99.75% noise.

## 5. Experiments

### 5.1 Setup (~0.5 page)

- **Datasets:** ChestMNIST (medical), UAVScenes (aerial)
- **Model:** ResNet18 + 3-layer classifier, ImageNet-pretrained backbone
- **VLMs:** vanilla CLIP ViT-L/14, BiomedCLIP, SimCLIP-2/4, FARE-2/4, TeCoA-2/4
- **GIAs:** HF-GradInv (default), IG, GradInversion, DLG
- **Metrics:** Attack F1 @ cosine threshold 0.90 (output-layer gradient match), LPIPS, PSNR
- **Hyperparameters:** 24k iterations, batch size 8 (ablations: 16, 32), seed 42

### 5.2 Robust VLMs backfire (~0.75 page)

**Result:** Loss ratios across VLM variants. Table from `vlm_backfire.md`:
- Medical: vanilla CLIP 7.32x → SimCLIP-4 11.46x (+57%) → FARE-4 11.75x (+61%)
- UAV: mixed, vanilla CLIP already 91x

**Interpretation:** Robust VLMs learn semantically meaningful features → better text-image discrimination → stronger targeting tool. The attack improves as VLM quality improves.

### 5.3 NBFU stops the attack (~0.75 page)

**Result:** DP-SGD ε sweep:
- Medical pneumonia: ε=5 → F1 0.625→0.125 (80% reduction)
- UAV swimming pool: ε=5 → F1 →0.0 (full defense)
- Even ε=50 (minimal noise) stops UAV: clipping is what matters

**Comparison with other defenses:** DP-SGD vs FedAvg vs pruning vs noise vs robust VLMs (table from `defense_comparison.md`).

### 5.4 Defense generalizes across attacks (~0.75 page)

**Result:** Defense sweep at 24k iterations (table from `attack_sweep_summary/`):

| Attack | Medical clean→DP-SGD | UAV clean→DP-SGD |
|--------|---------------------|------------------|
| HF-GradInv | 0.75 → 0.125 | 0.125 → 0.000 |
| IG | 0.75 → 0.375 | 0.25 → 0.000 |
| GradInversion | 0.75 → 0.50 | 0.25 → 0.000 |
| DLG | 0.375 → 0.375 (too weak) | 0.0 → 0.0 |

**Interpretation:** Every functional attack drops under NBFU. UAV reaches F1=0 across all three competitive attacks.

### 5.5 Batch size sweep (~0.5 page)

**Result:** B8/B16/B32 (HF-GradInv only — full-layer attacks OOM at larger batches):
- Medical defense margin: −83% at B8, −15% at B16, −21% at B32
- UAV defense margin: maintained across all batches

**Interpretation:** At larger batches, target gradient is diluted in the average; clipping is less aggressive. Practical guidance: tune ε per batch size.

## 6. Discussion (~0.5 page)

- Why clipping > noise: noise alone has no formal guarantee; clipping bounds sensitivity. Even minimal noise (ε=50) suffices when clipping is in place.
- Connection to gradient stability literature (Slate / Robey et al.): robust encoders → stable gradients → easier inversion. Inverse of intended robustness goal.
- Practical guidance for FL deployments: NBFU with C calibrated to expected gradient scale, ε chosen for batch size.

## 7. Limitations (~0.25 page)

- Single seed for headline results (multi-seed in appendix)
- Only ResNet18 architecture
- DLG and IG don't scale to batch≥16 due to OOM (batch sweep limited to HF-GradInv)
- No formal proof of the SNR proposition — empirical SNR matches but bounded analysis not rigorous
- Robust VLM threat model mismatch acknowledged (input-perturbation defenses tested in clean-input attack setting)

## 8. Conclusion (~0.25 page)

NBFU defends Geminio. Robust VLMs are not a substitute for gradient-level defense.

---

## Appendix candidates

- A: Full DP-SGD ε sweep (5 values × 2 domains)
- B: Full FedAvg epoch sweep
- C: Full VLM variant comparison (8 VLMs × 5 medical queries × 6 UAV queries)
- D: Multi-seed ablation (if we run it)
- E: Detailed attack hyperparameters
- F: Reproducibility checklist (links to lab repo)

---

## Page budget summary

| Section | Pages |
|---------|-------|
| Abstract + Title | 0.25 |
| 1. Introduction | 1.0 |
| 2. Background + Threat Model | 1.0 |
| 3. NBFU Method + Analysis | 1.5 |
| 4. Experiments | 3.25 |
| 5. Discussion | 0.5 |
| 6. Limitations | 0.25 |
| 7. Conclusion | 0.25 |
| **Total content** | **8.0** |
| References | overflow allowed |
| Appendix | overflow allowed |

---

## What's missing / deadline-critical

1. **Multi-seed validation** — single-seed F1 has ±0.125 noise at batch 8. Need 3+ seeds for reportable error bars. ~6 hours of GPU time.
2. **Additional dataset** (Zarif's UAV) — not yet shared. Paper works without it but stronger with.
3. **Formal proposition wording** — proof sketch is informal; reviewers may want tighter bounds.
4. **Figures** — currently all results are tables. Need at least: (a) loss ratio bar chart by VLM, (b) F1 vs ε curve, (c) F1 vs batch size, (d) gradient norm distributions.
5. **Related work writeup** — currently no Section 2.5 on prior FL defenses.

## Suggested writing order

1. Methods + Analysis (4) — load-bearing technical content
2. Experiments (5) — straight from results files
3. Introduction (1) — last, after the story is concrete
4. Related work (2.5) — fill in citations
5. Discussion + Limitations + Conclusion (6-8) — short, last
