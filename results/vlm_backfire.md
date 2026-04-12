# VLM robustness backfires on gradient inversion defense

Adversarially robust VLMs (SimCLIP, FARE) — designed to resist adversarial examples — actually **strengthen** Geminio's gradient inversion attack. This is because robust encoders learn more semantically meaningful features, which gives the attacker better text-image discrimination for targeting.

## Medical domain loss ratios (Top-10%)

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

Both robust VLMs increase medical attack effectiveness by 57-61% over vanilla CLIP.

## UAV domain loss ratios (Top-10%)

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

UAV results are mixed. FARE-4 reduces loss ratios by 41% but they remain extremely high (54x average). The UAV domain already has such high visual distinctiveness that even vanilla CLIP achieves 91x.

## Why robust VLMs help the attacker

Adversarial training forces the vision encoder to learn **semantically meaningful features** rather than texture shortcuts. For medical images — where all chest X-rays look similar to an untrained model — more semantic features mean the VLM can better distinguish pneumonia from cardiomegaly from normal. This is exactly what the attacker needs: better text-image discrimination in the embedding space.

**VLM adversarial robustness is orthogonal to gradient inversion defense.** The attack exploits gradient leakage in the FL protocol. The VLM is merely the attacker's targeting tool. Hardening the tool doesn't close the vulnerability — and for domains where vanilla CLIP already struggles (medical), it actually sharpens the attacker's capability.
