# Why gradient clipping stops Geminio: an analytical argument

## Setup

Let θ denote the parameters of a model with d parameters (d = 11,325,263 for our ResNet18). A victim client with batch B of n samples computes and shares the averaged gradient:

    G = (1/n) Σ_{(x,y) ∈ B} ∇_θ L(F_θ(x); y)

## Geminio's gradient amplification

Geminio crafts a malicious model θ_Q such that per-sample loss on target samples (matching the attacker's query Q) is amplified by a factor R relative to non-targets:

    L(F_{θ_Q}(x_target)) ≈ R · L(F_{θ_Q}(x_other))

where R is the **loss ratio** — our measured values:

| Domain | R (average) | ||G|| measured |
|--------|-------------|---------------|
| Medical | 8–12x | 15.6 |
| UAV | 54–91x | 527.0 |

Since gradient magnitude is proportional to loss, the target sample's gradient dominates the averaged gradient. If only one sample in B matches the query:

    G ≈ (1/n) · ∇L_target

This is what makes reconstruction possible: the averaged gradient carries mostly single-sample information.

## DP-SGD: clipping + noise

Our defense applies two steps:

**Step 1 — Clipping.** Scale G so that ||G|| ≤ C:

    G_clip = G · min(1, C / ||G||)

With C = 1.0 (our default), the shrinkage factor is:

| Domain | ||G|| | C / ||G|| (shrinkage) |
|--------|-------|----------------------|
| Medical | 15.6 | 0.064 (16x reduction) |
| UAV | 527.0 | 0.0019 (527x reduction) |

**Step 2 — Calibrated noise.** Add Gaussian noise N with per-element std:

    σ_elem = (C · √(2 ln(1.25/δ))) / (ε · n)

For ε = 5, δ = 1e-5, n = 8: σ_elem = 0.121.

## The signal-to-noise ratio after defense

The defended gradient is G_def = G_clip + N. Consider the norms:

**Clipped signal norm:**

    ||G_clip|| = C = 1.0

**Noise norm (expected):**

    ||N|| = σ_elem · √d = 0.121 · √(11,325,263) ≈ 407

The noise norm exceeds the signal norm by a factor of **407×**. The defended gradient is 99.75% noise. The cosine similarity between G_def and the true (clipped) gradient is approximately:

    cos(G_def, G_clip) ≈ ||G_clip|| / ||G_def|| ≈ 1.0 / 407 ≈ 0.0025

A reconstruction algorithm — whether it uses Euclidean distance (DLG, GradInversion) or cosine similarity (IG, HF-GradInv) — cannot distinguish the target image from random noise at this SNR.

## Why clipping is the key mechanism (not noise)

The noise magnitude σ_elem is calibrated to C, not to ||G||. Without clipping, the SNR would be:

    SNR_no_clip = ||G|| / ||N|| = ||G|| / (σ_elem · √d)

| Domain | ||G|| | ||N|| | SNR without clipping |
|--------|-------|-------|---------------------|
| Medical | 15.6 | 407 | 0.038 |
| UAV | 527.0 | 407 | 1.29 |

Without clipping, the UAV gradient (||G|| = 527) would have SNR > 1, meaning the signal survives the noise. With clipping, SNR drops to 0.0025 regardless of the original gradient magnitude. **Clipping normalizes the signal to a fixed scale C before noise addition, making the defense independent of the attacker's gradient amplification factor R.**

This explains two experimental observations:

1. **Even ε = 50 (minimal noise) stops UAV targeting.** At ε = 50: σ_elem = 0.012, ||N|| = 42. With clipping: SNR = 1.0/42 = 0.024 (gradient is 97.7% noise). Without clipping: SNR = 527/42 = 12.5 (gradient is 92% signal). Clipping is what turns a survivable noise level into a destructive one.

2. **UAV is more sensitive to DP-SGD than medical.** Counterintuitively, the domain where Geminio is strongest (R = 91x, ||G|| = 527) is also where DP-SGD is most effective. This is because clipping shrinks the UAV gradient by 527x (vs 16x for medical), removing more of the attacker's amplification.

## Connection to robust VLMs

Robust VLMs (SimCLIP, FARE) increase the loss ratio R by improving the attacker's text-image discrimination. On medical data, R increases from 7.3x (vanilla CLIP) to 11.5–11.8x (robust VLMs), a 57–61% improvement for the attacker.

However, clipping is **invariant to R**: the gradient is always clipped to ||G|| = C regardless of how large R makes the original gradient. The defense operates at a layer (gradient norms) that VLM robustness cannot influence. This is why robust VLMs strengthen the undefended attack but cannot circumvent DP-SGD.

## Summary

Geminio's power comes from creating ||G|| >> 1 through loss-ratio amplification. Calibrated DP-SGD neutralizes this by:

1. **Clipping** — normalizes ||G|| → C, eliminating the amplification regardless of R
2. **Noise** — at the clipped scale, even moderate DP noise (ε = 5–50) produces ||N|| >> C, drowning the signal

The defense is attack-agnostic (works against DLG, IG, GradInversion, HF-GradInv) because all gradient inversion algorithms require gradient information, and clipping + noise destroys that information at the source. The effectiveness scales with model size d: larger models have higher ||N|| = σ√d for the same ε, making them naturally harder to invert under DP-SGD.
