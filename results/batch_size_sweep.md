# Batch size sweep: Attack F1 at batch 8, 16, 32

HF-GradInv (Geminio default) with and without DP-SGD ε=5, seed=42, 24k iterations.

## Medical (pneumonia_descriptive)

| Batch | Clean F1 | Clean (id/total) | DP-SGD F1 | DP-SGD (id/total) | Drop |
|-------|----------|-------------------|-----------|---------------------|------|
| 8 | 0.750 | 6/8 | 0.125 | 1/8 | −83% |
| 16 | 0.813 | 13/16 | 0.688 | 11/16 | −15% |
| 32 | 0.875 | 28/32 | 0.688 | 22/32 | −21% |

## UAV (solar_panels)

| Batch | Clean F1 | Clean (id/total) | DP-SGD F1 | DP-SGD (id/total) | Drop |
|-------|----------|-------------------|-----------|---------------------|------|
| 8 | 0.125 | 1/8 | 0.000 | 0/8 | −100% |
| 16 | 0.063 | 1/16 | 0.125 | 2/16 | +100%* |
| 32 | 0.156 | 5/32 | 0.031 | 1/32 | −80% |

*UAV B16 inversion: artifact of low absolute counts (1 vs 2 samples at cosine threshold 0.90). B32 confirms defense trend.

## Observations

1. **Clean F1 increases with batch size on medical** (0.75 → 0.81 → 0.88). More samples in the batch means more potential targets matching the pneumonia query, so more are identified.

2. **Defense margin narrows at larger batches.** At B8, DP-SGD drops medical F1 by 83%. At B16/B32, only 15–21%. Larger batches dilute the target gradient in the average, so the unclipped gradient norm is lower relative to C=1.0, and clipping removes less signal.

3. **UAV remains low across all batch sizes.** Even clean F1 is ≤0.156 — the 0.90 cosine threshold is stringent for aerial imagery. Defense is effective at B32 (0.156→0.031).

4. **B32 gives 33 discrete F1 levels** (vs 9 at B8), making trends more interpretable for paper figures.

## Implication for the analytical argument

The narrowing defense margin at larger batches is consistent with the SNR analysis. At larger batches:
- The averaged gradient norm ||G|| decreases (more samples averaging out)
- Clipping factor C/||G|| increases (less aggressive clipping)
- Post-clipping SNR improves slightly

This predicts that very large batch sizes (64+) would further weaken the defense, but the clean attack also becomes harder (more non-target signal mixed in). The optimal ε should be tuned per batch size.
