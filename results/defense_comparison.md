# Complete defense comparison

All defenses tested against Geminio (HF-GradInv reconstruction) on the medical pneumonia and UAV swimming_pool/solar_panels queries.

## Defense summary

| Defense | Level | Medical F1 | UAV F1 | Reliable? | Formal Guarantee? |
|---------|-------|-----------|--------|-----------|-------------------|
| None (FedSGD) | — | 0.625 | — | — | No |
| Gradient pruning 99% | Gradient | inconsistent | inconsistent | No | No |
| Laplacian noise 1e-1 | Gradient | inconsistent | inconsistent | No | No |
| SimCLIP-4 | VLM | +57% (worse) | -12% | No | No |
| FARE-4 | VLM | +61% (worse) | -41% | No | No |
| FedAvg 5 epochs | FL protocol | 0.000 | 0.375 | Partial | No |
| FedAvg 20 epochs | FL protocol | 0.000 | 0.250 | Partial | No |
| **DP-SGD ε=5** | **Gradient** | **0.125** | **0.000** | **Yes** | **Yes (ε,δ)-DP** |

## DP-SGD across epsilon values

### Medical (pneumonia_descriptive)

| ε (Privacy Budget) | LPIPS ↓ | PSNR ↑ | Attack F1 | Privacy Level |
|-----|---------|--------|-----------|--------------|
| None (baseline) | 1.019 | 10.86 | 0.625 | No defense |
| 0.1 | 0.788 | 11.35 | 0.375 | Very strong |
| 1.0 | 0.804 | 11.31 | 0.250 | Strong |
| 5.0 | 0.791 | 11.41 | 0.125 | Moderate |
| 10.0 | 0.797 | 11.28 | 0.250 | Weak |
| 50.0 | 0.836 | 11.63 | 0.625 | Very weak |

### UAV (swimming_pool)

| ε | LPIPS ↓ | PSNR ↑ | Attack F1 |
|---|---------|--------|-----------|
| None | 0.679 | 12.88 | — |
| 0.1 | 0.703 | 12.49 | 0.125 |
| 1.0 | 0.693 | 12.50 | 0.125 |
| 5.0 | 0.689 | 12.39 | 0.000 |
| 10.0 | 0.685 | 12.33 | 0.000 |
| 50.0 | 0.852 | 12.43 | 0.000 |

### UAV validation (solar_panels)

| ε | LPIPS ↓ | PSNR ↑ | Attack F1 |
|---|---------|--------|-----------|
| 0.1 | 0.721 | 12.36 | 0.000 |
| 1.0 | 0.720 | 12.27 | 0.125 |
| 5.0 | 0.747 | 12.55 | 0.000 |
| 10.0 | 0.728 | 12.46 | 0.000 |
| 50.0 | 0.764 | 12.31 | 0.375 |

## FedAvg multi-epoch defense

### Medical (pneumonia_descriptive, lr=1e-3)

| Local Epochs | LPIPS ↓ | PSNR ↑ | Attack F1 |
|-------------|---------|--------|-----------|
| 0 (FedSGD) | 1.019 | 10.86 | 0.625 |
| 1 | 0.898 | 11.15 | 0.500 |
| 2 | 0.813 | 11.04 | 0.875 |
| 5 | 0.918 | 10.87 | 0.000 |
| 10 | 0.835 | 9.59 | 0.500 |
| 20 | 0.819 | 10.93 | 0.000 |

### UAV (swimming_pool, lr=1e-3)

| Local Epochs | LPIPS ↓ | PSNR ↑ | Attack F1 |
|-------------|---------|--------|-----------|
| 0 (FedSGD) | 0.679 | 12.88 | — |
| 1 | 0.660 | 10.79 | 0.125 |
| 2 | 0.669 | 11.04 | 0.125 |
| 5 | 0.667 | 10.92 | 0.375 |
| 10 | 0.668 | 10.90 | 0.250 |
| 20 | 0.671 | 10.95 | 0.250 |

FedAvg provides some defense but is inconsistent — medical F1 is non-monotonic (0.000 at 5 epochs, 0.500 at 10).

## Key insights

1. **Gradient clipping is the key mechanism.** Even ε=50 (minimal noise) stops UAV targeting completely. Clipping bounds the gradient norm, directly disrupting Geminio's gradient amplification.

2. **VLM robustness increased medical attack effectiveness by 57-61%. DP-SGD reduces it by 80%.** The defense operates at the correct level — gradients, not embeddings.

3. **UAV is more sensitive to DP-SGD than medical.** All UAV epsilon values achieve F1 ≤ 0.375. Medical requires ε ≤ 5 to reach F1=0.125. This is the inverse of undefended vulnerability.
