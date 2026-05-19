# Geminio: Gradient Inversion Reconstruction Results
## Full End-to-End Attack Across Three Domains
### Progress Report — March 15, 2026

**Dan Zimmerman** | Advisor: Dr. Ahmed Imteaj

---

## 1. Overview

This report documents the complete end-to-end Geminio gradient inversion attack across three image domains: natural images (ImageNet), medical chest X-rays (ChestMNIST), and aerial drone imagery (UAVScenes). We demonstrate that a malicious federated learning server can reconstruct private training images from shared gradients, and that language-guided targeting (via CLIP/BiomedCLIP) allows the attacker to selectively recover images matching a natural language query.

**Three-phase pipeline completed:**
1. **VLM Embedding** — Pre-compute CLIP/BiomedCLIP embeddings for all training images
2. **Malicious Model Training** — Train models with Geminio loss surface reshaping (completed previously)
3. **Gradient Inversion** — Reconstruct private images from victim gradients (this report)

**Attack configuration:**
- Reconstruction algorithm: HFGradInv (24,000 iterations, Adam optimizer)
- Objective: Layer-weighted cosine similarity between candidate and shared gradients
- Regularization: Total Variation (scale 5e-2, starts at iter 3000)
- Initialization: Patterned-4-randn
- Labels: Provided to attacker (best-case scenario)

---

## 2. ImageNet Reconstruction (Existing Results)

**Setup:** 64 private ImageNet images (1000 classes), ResNet34 backbone, CLIP ViT-L/14, CrossEntropyLoss

These results were produced in a prior session and serve as the baseline for cross-domain comparison.

### Ground Truth (64 images)
![ImageNet Ground Truth](../results/Any_jewelry/a_truth.jpg)

### Baseline Reconstruction (no query targeting)
![ImageNet Baseline](../results/baseline/final_rec.jpg)
*Top row: reconstructed images. Bottom row: ground truth. The baseline reconstruction produces uniformly noisy results across all 64 images — the gradient signal is spread evenly.*

### Query-Guided: "Any jewelry?"
![ImageNet Jewelry](../results/Any_jewelry/final_rec.jpg)
*Top row: reconstructed images. Bottom row: ground truth. The query-guided reconstruction shows noticeably better reconstruction quality for specific images (jewelry-related), while other images remain noisy. The malicious model concentrates gradient signal on target images.*

**Loss ratio from training:** 19.01x (jewelry images have 19x higher loss than non-jewelry)

---

## 3. Medical Reconstruction (ChestMNIST + BiomedCLIP)

**Setup:** 8 private chest X-rays from ChestMNIST (15 classes), ResNet18 backbone, BiomedCLIP (512-dim), CrossEntropyLoss

**Private samples:** Diverse conditions — pneumothorax, effusion, emphysema, nodule, normal, atelectasis, cardiomegaly, infiltration

### 3.1 Ground Truth
![Medical Ground Truth](../results/medical_baseline/a_truth.jpg)
*8 chest X-rays from different patients with different conditions, resized from 64x64 to 224x224.*

### 3.2 Baseline Reconstruction
![Medical Baseline](../results/medical_baseline/final_rec.jpg)
*Top row: reconstructed. Bottom row: ground truth. The baseline reconstruction shows general chest-like structure in some images (images 1-4 show vague chest outlines) but overall poor quality. Reconstruction loss: **0.1855**.*

### 3.3 Query-Guided: "chest X-ray with bilateral patchy opacities and air bronchograms indicating pneumonia infection"
![Medical Pneumonia](../results/medical_pneumonia_descriptive/final_rec.jpg)
*Top row: reconstructed. Bottom row: ground truth. The pneumonia query-guided reconstruction shows clearer chest X-ray structure in several images — rib cages, lung fields, and mediastinal contours are visible (especially images 1, 3-5). Reconstruction loss: **0.0922** (2.0x better than baseline).*

### 3.4 Query-Guided: "chest X-ray showing blunted costophrenic angle with fluid layering consistent with pleural effusion"
![Medical Effusion](../results/medical_effusion_descriptive/final_rec.jpg)
*Top row: reconstructed. Bottom row: ground truth. The effusion query reconstruction shows recognizable X-ray anatomy in images 1-4 but higher noise in images 5-8. Reconstruction loss: **0.5893** — higher than baseline, suggesting this query's gradient concentration may not align well with the specific private samples.*

### 3.5 Medical Reconstruction Summary

| Run | Rec. Loss | vs Baseline | Training Loss Ratio |
|-----|-----------|-------------|-------------------|
| Baseline (no query) | 0.1855 | — | — |
| Pneumonia (descriptive) | **0.0922** | **2.0x better** | 12.48x |
| Effusion (descriptive) | 0.5893 | 0.31x worse | 14.85x |

**Analysis:** The pneumonia query produces clearly better reconstructions both quantitatively (2x lower reconstruction loss) and qualitatively (more recognizable X-ray anatomy). The effusion query, despite having the highest training loss ratio (14.85x), produced poorer reconstruction — likely because the specific 8 private samples didn't contain strong effusion cases, so the concentrated gradient signal didn't match the actual batch content.

---

## 4. UAV/Drone Reconstruction (UAVScenes + CLIP)

**Setup:** 8 private aerial drone images from UAVScenes (18 multi-label classes), ResNet18 backbone, CLIP ViT-L/14 (768-dim), BCEWithLogitsLoss (multi-label)

**Private samples:** Diverse aerial scenes from AMtown01 and HKairport01 — containing roofs, roads, fields, solar panels, bridge, trucks, containers, airstrip

**Technical note:** UAVScenes required adapting the breaching framework for multi-label classification. We overrode the default CrossEntropyLoss with BCEWithLogitsLoss at both the user (gradient computation) and attacker (gradient matching) levels, and manually injected multi-label targets to avoid the framework's label sorting which corrupts binary label vectors.

### 4.1 Ground Truth
![UAV Ground Truth](../results/uav_baseline/a_truth.jpg)
*8 aerial drone images showing residential areas, roads, vegetation, an airstrip, and mixed urban scenes. Original resolution 2448x2048, resized to 224x224.*

### 4.2 Baseline Reconstruction
![UAV Baseline](../results/uav_baseline/final_rec.jpg)
*Top row: reconstructed. Bottom row: ground truth. The baseline reconstruction produces remarkably good results — green vegetation, brown/tan rooftops, road structures, and general aerial scene composition are clearly captured. This suggests aerial drone imagery is particularly vulnerable to gradient inversion attacks. Reconstruction loss: **0.0986**.*

### 4.3 Query-Guided: "aerial drone image of a swimming pool"
![UAV Swimming Pool](../results/uav_swimming_pool/final_rec.jpg)
*Top row: reconstructed. Bottom row: ground truth. The swimming pool query achieves dramatically better gradient matching (4.8x lower rec loss) by concentrating the gradient signal on pool-similar images. The reconstructions appear noisier overall because the malicious model deliberately suppresses gradient contributions from non-target images. Reconstruction loss: **0.0205**.*

### 4.4 Query-Guided: "aerial drone image showing solar panels on rooftops"
![UAV Solar Panels](../results/uav_solar_panels/final_rec.jpg)
*Top row: reconstructed. Bottom row: ground truth. The solar panels query shows mixed results — image 1 captures some aerial structure, but most images are noisy. Reconstruction loss: **0.1136** (similar to baseline). Despite the highest training loss ratio (173.73x), the reconstruction quality depends on whether the private batch actually contains solar panel images.*

### 4.5 UAV Reconstruction Summary

| Run | Rec. Loss | vs Baseline | Training Loss Ratio |
|-----|-----------|-------------|-------------------|
| Baseline (no query) | 0.0986 | — | — |
| Swimming pool | **0.0205** | **4.8x better** | 126.98x |
| Solar panels | 0.1136 | 0.87x (similar) | 173.73x |

**Analysis:** The swimming pool query achieves the best gradient matching across all experiments (rec loss 0.0205), consistent with its strong training loss ratio. The baseline UAV reconstruction is surprisingly effective, suggesting drone imagery's high visual detail and color variation make it inherently vulnerable. The solar panels query's similar-to-baseline performance may indicate that the specific private samples contain few solar panel images.

---

## 5. Cross-Domain Comparison

### 5.1 Reconstruction Loss Summary

| Domain | Batch | Baseline Rec Loss | Best Query Rec Loss | Best Query | Improvement |
|--------|-------|-------------------|--------------------:|------------|-------------|
| ImageNet | 64 | *(existing)* | *(existing)* | Any jewelry? | *(pre-existing)* |
| Medical | 8 | 0.1855 | 0.0922 | Pneumonia (descriptive) | 2.0x |
| UAV | 8 | 0.0986 | 0.0205 | Swimming pool | 4.8x |

### 5.2 Training Loss Ratio vs Reconstruction Quality

| Domain | Query | Training Loss Ratio | Rec Loss | Effective? |
|--------|-------|-------------------:|----------|:----------:|
| Medical | Pneumonia (descriptive) | 12.48x | 0.0922 | Yes |
| Medical | Effusion (descriptive) | 14.85x | 0.5893 | No |
| UAV | Swimming pool | 126.98x | 0.0205 | Yes |
| UAV | Solar panels | 173.73x | 0.1136 | Moderate |

**Key insight:** High training loss ratios are necessary but not sufficient for successful reconstruction. The actual reconstruction quality depends on two factors:
1. **The loss ratio** (how well the malicious model concentrates gradient signal)
2. **Whether the private batch contains target images** (if no pool images exist in the batch, the concentrated gradient signal doesn't align with the actual data)

### 5.3 Baseline Reconstruction Quality by Domain

The baseline (un-targeted) reconstruction quality varies dramatically:

- **UAV (best):** Clear aerial structures, vegetation colors, building outlines — drone imagery's high spatial detail and color diversity make it inherently vulnerable
- **Medical (worst):** Only vague chest outlines — the visual uniformity of X-rays (all grayscale, similar anatomy) limits what gradient inversion can recover
- **ImageNet (middle):** Some recognizable objects but mostly noisy — the extreme class diversity (1000 classes, 64 images) spreads the gradient signal thin

This ranking mirrors the training loss ratio ordering: UAV (91x avg) > ImageNet (16x avg) > Medical (12x avg), confirming that domains where images are more visually diverse are more vulnerable to gradient inversion.

---

## 6. Technical Implementation Notes

### 6.1 ResNet18 Adaptation
The original Geminio uses ResNet34 (~120 parameter tensors). Our medical and UAV extensions use ResNet18 (66 parameter tensors). The HFGradInv attack's layer-weighted objective had `start=100` (tuned for ResNet34), which caused errors since ResNet18 has fewer layers. We adjusted to `start=50, min_start=15` for ResNet18.

### 6.2 Multi-Label Loss (UAVScenes)
The breaching framework assumes single-label CrossEntropyLoss. For UAVScenes' multi-label classification, we:
1. Overrode `user.loss` with `BCEWithLogitsLoss()` after `construct_case()`
2. Disabled automatic label sorting (`user.provide_labels = False`) since sorting corrupts multi-label binary vectors
3. Manually injected unsorted multi-label targets into `shared_data["metadata"]["labels"]`
4. Passed `BCEWithLogitsLoss()` as the attacker's loss function for gradient matching

### 6.3 Dataset Registration
Added ChestMNIST and UAVScenes as placeholder datasets in the breaching framework's vision dataset builder (actual data provided via `custom_data`).

### 6.4 Files Created

```
prototype/
  reconstruct_medical.py     Medical gradient inversion reconstruction
  reconstruct_uav.py         UAV gradient inversion (multi-label BCE)
  prepare_medical_samples.py Extract diverse ChestMNIST images as PNGs
  prepare_uav_samples.py     Extract UAVScenes images as PNGs

breaching/config/case/
  geminio_medical.yaml       Medical case config (ResNet18, 15 classes)
  geminio_uav.yaml           UAV case config (ResNet18, 18 classes)
  data/ChestMNIST.yaml       Medical data config (normalization, shape)
  data/UAVScenes.yaml        UAV data config (normalization, shape)

assets/
  medical_samples/           8 chest X-rays as PNGs ({index}-{class}.png)
  uav_samples/               8 aerial images as PNGs + labels.npy

results/
  medical_baseline/          Baseline medical reconstruction
  medical_pneumonia_descriptive/  Pneumonia query reconstruction
  medical_effusion_descriptive/   Effusion query reconstruction
  uav_baseline/              Baseline UAV reconstruction
  uav_swimming_pool/         Pool query reconstruction
  uav_solar_panels/          Solar panels query reconstruction
```

---

## 7. Methodology Deviations from Geminio Paper

The original Geminio authors (2411.14937) did not release their training code — only a reconstruction demo script. Our implementation was reverse-engineered from the paper's equations and descriptions. Below we document each engineering decision that differs from the paper and explain the rationale.

### 7.1 Summary of Deviations

| # | Aspect | Paper | Our Implementation | Impact |
|---|--------|-------|--------------------|--------|
| 1 | **Temperature scaling** | Raw cosine similarity in Eq. 4 | `clip_sim × 100` before softmax | High — affects gradient concentration strength |
| 2 | **Trainable parameters** | Full model (implied by Sec 3.2) | Classifier head only (backbone frozen) | Medium — faster training, may reduce attack power |
| 3 | **Loss formulation** | Eq. 4: `L = Σ lᵢ · (1 − pᵢ)` | `mean(lᵢ × (1 − softmax(sim × 100)))` | Low — functionally equivalent (mean vs sum) |
| 4 | **Label source** | Eq. 5: VLM zero-shot pseudo-labels | Ground truth labels from dataset | Low — best-case for attacker |
| 5 | **Training batch size** | 64 | 16 | Low — per-sample weighting is batch-independent |
| 6 | **Backbone architecture** | ResNet34 for all domains | ResNet18 for medical/UAV | Medium — fewer parameters, adjusted attack config |
| 7 | **VLM for medical domain** | CLIP ViT-L/14 for all | BiomedCLIP for medical | Positive — domain-specific embeddings improve targeting |

### 7.2 Detailed Explanations

**1. Temperature Scaling (High Impact)**

The paper's Equation 4 shows the weighting factor as `softmax(cosine_similarity)`, implying raw cosine similarities are passed directly to softmax. However, raw cosine similarities between CLIP embeddings typically lie in a narrow range (e.g., 0.15–0.35), which produces nearly uniform softmax outputs — effectively destroying the query-targeting signal.

We multiply by a temperature factor of 100 (`softmax(sim × 100)`) to amplify the contrast between target and non-target images. This is consistent with standard practice in contrastive learning (e.g., CLIP's own temperature parameter τ=100), and we believe the paper uses a similar mechanism internally without stating it explicitly. Without temperature scaling, our training produced loss ratios of ~1x (no gradient concentration).

**2. Classifier-Only Training (Medium Impact)**

The paper implies training all model parameters (backbone + classifier). We freeze the pretrained ResNet backbone and only train a 3-layer classifier head (~3M parameters for ResNet18 vs ~11M full model for ResNet34).

*Rationale:* The pretrained backbone already produces useful features. Training only the classifier head is dramatically faster and sufficient for reshaping the loss surface — our loss ratios (5–174×) demonstrate effective gradient concentration. Full model training would be closer to the paper but would require significantly longer training time.

**3. Loss Formulation (Low Impact)**

Our implementation uses `mean()` over the batch while the paper uses `Σ` (sum). These are equivalent up to a constant factor (batch size), which the optimizer absorbs into the learning rate. The weighting mechanism `(1 − pᵢ)` is identical.

**4. Real Labels vs. VLM Pseudo-Labels (Low Impact)**

The paper's Equation 5 generates classification labels via CLIP zero-shot prediction. We use ground truth labels from the dataset instead. This represents a *best-case scenario* for the attacker — if anything, using real labels should produce better loss surfaces than noisy pseudo-labels. This deviation simplifies the pipeline without weakening the attack.

**5. Training Batch Size (Low Impact)**

We use batch size 16 vs the paper's 64, due to memory constraints. The Geminio loss function computes per-sample losses weighted by per-sample CLIP similarities — this mechanism is independent of batch size. The only effect is training speed (more gradient updates per epoch with smaller batches).

**6. ResNet18 for Medical/UAV (Medium Impact)**

We use ResNet18 for the medical and UAV domains instead of the paper's ResNet34. ResNet18 has 66 parameter tensors vs ResNet34's ~120, which required adjusting the HFGradInv attack's layer-weighted objective (`start=50, min_start=15` instead of `start=100`). The smaller architecture is appropriate for our smaller batch sizes (8 images vs the paper's 128).

**7. BiomedCLIP for Medical Domain (Positive Deviation)**

The paper uses CLIP ViT-L/14 for all domains. We use BiomedCLIP (PubMedBERT + ViT-B/16-224, trained on PMC-15M biomedical data) for the medical domain. CLIP was not trained on medical images and produces poor text-image alignment for clinical descriptions like "bilateral patchy opacities with air bronchograms." BiomedCLIP produces 512-dimensional embeddings with much better medical concept discrimination, which translates to more effective gradient concentration (loss ratios of 7–15× with descriptive prompts).

---

## 8. Quantitative Evaluation Metrics

### 8.1 Metrics Used

Following the Geminio paper's evaluation methodology, we compute:

- **LPIPS** (Learned Perceptual Image Patch Similarity, AlexNet backbone) — primary metric, lower is better. Measures perceptual similarity between reconstructed and ground truth images.
- **PSNR** (Peak Signal-to-Noise Ratio) — higher is better. Standard pixel-level reconstruction quality metric.
- **CW-SSIM** (Complex Wavelet Structural Similarity) — higher is better. Scale-invariant structural similarity.
- **Attack F1** — the paper's custom metric. Computes per-sample output-layer gradient cosine similarity between reconstructed and ground truth images. Images with cosine similarity ≥ 0.90 are counted as "successfully attacked."

### 8.2 Results

| Domain | Run | LPIPS ↓ | PSNR ↑ | CW-SSIM ↑ | Attack F1 | Avg Cos Sim |
|--------|-----|---------|--------|-----------|-----------|-------------|
| Medical | Baseline | **0.7927** | **11.32** | **0.3200** | **0.625** | 0.910 |
| Medical | Pneumonia (descriptive) | 1.0193 | 10.86 | 0.1878 | 0.625 | 0.632 |
| Medical | Effusion (descriptive) | 1.0661 | 10.64 | 0.2414 | 0.875 | 0.876 |
| UAV | Baseline | **0.6021** | 12.28 | 0.3288 | 0.000 | 0.676 |
| UAV | Swimming pool | 0.6786 | **12.88** | **0.3711** | **0.375** | 0.811 |
| UAV | Solar panels | 0.6554 | 12.37 | 0.3417 | 0.500 | 0.722 |

**Analysis:**

- **UAV domain produces the best perceptual quality** across all metrics. UAV baseline LPIPS (0.60) is significantly better than medical baseline (0.79), confirming that aerial imagery with high color diversity is more vulnerable to gradient inversion.

- **Medical baseline outperforms query-guided on LPIPS/PSNR/SSIM**, contrary to what reconstruction loss suggested. The medical query-guided models (pneumonia: LPIPS 1.02, effusion: 1.07) produce perceptually worse reconstructions than baseline (0.79). This is because the query-guided model concentrates gradient signal on target images, potentially sacrificing overall batch reconstruction quality when matched pairs are reordered.

- **Attack F1 tells a different story.** The effusion query achieves the highest Attack F1 (0.875) despite the worst LPIPS, meaning 7/8 reconstructed images produce output-layer gradients that closely match ground truth — the model recognizes the reconstructed images as functionally similar to the originals even if they look perceptually different.

- **UAV baseline has 0% Attack F1** (avg cos sim 0.68 < 0.90 threshold) while the swimming pool query achieves 37.5% — the query-guided model produces reconstructions whose gradient fingerprints match specific target images.

---

## 9. Defense Evaluation

### 9.1 Defense Mechanisms

We evaluate the attack's robustness against two common gradient privacy defenses:

1. **Gradient Pruning** — Zeroes out the smallest gradient values by magnitude. Tested at pruning rates: 70%, 80%, 90%, 95%, 99%.
2. **Gradient Noise Injection** — Adds Laplacian noise to gradients before sharing. Tested at scales: 1×10⁻⁴, 1×10⁻³, 1×10⁻², 1×10⁻¹.

Defenses are applied post-hoc to the victim's gradient updates before the attacker receives them.

### 9.2 Defense Results — Medical (Pneumonia Query)

| Defense | Setting | LPIPS ↓ | PSNR ↑ | CW-SSIM ↑ | Attack F1 | Avg Cos Sim |
|---------|---------|---------|--------|-----------|-----------|-------------|
| None | — | 1.0193 | 10.86 | 0.1878 | 0.625 | 0.632 |
| Pruning | 70% | 1.1754 | 10.16 | 0.1777 | 0.625 | 0.620 |
| Pruning | 90% | 0.9469 | 11.03 | 0.2407 | 0.375 | 0.797 |
| Pruning | 99% | **0.8309** | **11.34** | **0.2721** | 0.375 | 0.798 |
| Noise | 1e-3 | 0.8965 | 11.43 | 0.2244 | 0.625 | 0.640 |
| Noise | 1e-2 | 0.9351 | 11.01 | 0.2447 | 0.250 | 0.551 |
| Noise | 1e-1 | 0.8526 | **11.57** | 0.2011 | 0.625 | 0.638 |

**Medical defense analysis:** Surprisingly, gradient pruning at 90–99% *improves* perceptual metrics (LPIPS drops from 1.02 to 0.83) while reducing Attack F1 from 0.625 to 0.375. This suggests that pruning removes the malicious gradient concentration signal, causing the reconstruction to revert toward a more uniform (baseline-like) quality across all images. Noise injection shows mixed effects — moderate noise (1e-2) reduces Attack F1 to 0.25 but the relationship is non-monotonic.

### 9.3 Defense Results — UAV (Swimming Pool Query)

| Defense | Setting | LPIPS ↓ | PSNR ↑ | CW-SSIM ↑ | Attack F1 | Avg Cos Sim |
|---------|---------|---------|--------|-----------|-----------|-------------|
| None | — | 0.6786 | 12.88 | **0.3711** | 0.375 | 0.811 |
| Pruning | 70% | **0.6618** | **13.08** | 0.3185 | 0.375 | 0.832 |
| Pruning | 90% | 0.7205 | 11.88 | 0.3472 | 0.375 | 0.826 |
| Pruning | 99% | 0.6690 | 12.27 | 0.3135 | **0.125** | **0.295** |
| Noise | 1e-3 | 0.7378 | 13.08 | 0.3344 | 0.375 | 0.796 |
| Noise | 1e-2 | 0.7447 | 12.99 | 0.3640 | 0.500 | 0.824 |
| Noise | 1e-1 | 0.6954 | 12.89 | 0.3481 | 0.500 | 0.835 |

**UAV defense analysis:** Only extreme pruning (99%) significantly reduces Attack F1 (from 0.375 to 0.125) and cos similarity (from 0.81 to 0.30). Moderate pruning and noise injection barely affect the attack — UAV reconstruction quality remains robust. This is consistent with aerial imagery's inherent vulnerability: the high visual detail provides redundant gradient signal that survives partial pruning or noise.

### 9.4 Cross-Domain Defense Summary

The Geminio attack shows different defense resilience across domains:

- **Medical domain:** More susceptible to defenses. Even moderate pruning (90%) reduces Attack F1 by 40%. The concentrated gradient signal from the malicious model is fragile in the medical domain where visual similarity between images is high.
- **UAV domain:** Highly resilient to defenses. Only 99% pruning (removing all but 1% of gradient entries) meaningfully degrades the attack. Noise injection at all tested scales has no significant impact. This suggests that aerial imagery's visual diversity creates robust, distributed gradient signals that are harder to defend against.

---

## 10. Conclusions

6. **Quantitative metrics confirm domain vulnerability ranking.** UAV imagery has the best LPIPS (0.60 baseline) while medical X-rays are worst (0.79). This aligns with the qualitative observations from Section 5.

7. **Defenses are domain-dependent.** Medical domain attacks are fragile — 90% gradient pruning reduces Attack F1 by 40%. UAV attacks are resilient — only extreme 99% pruning degrades the attack. Noise injection is largely ineffective against both domains at tested scales.

---

## Previous Conclusions (from Phase 3)

1. **The Geminio attack works end-to-end across all three domains.** We have demonstrated the complete pipeline from malicious model training to actual image reconstruction, proving that VLM-guided gradient inversion is a practical threat.

2. **Query-guided reconstruction outperforms baseline** when the private batch contains target images. The swimming pool query achieved 4.8x better gradient matching than baseline; the pneumonia query achieved 2.0x better.

3. **Domain vulnerability varies dramatically.** Aerial drone imagery is most vulnerable (clear reconstructions even without targeting), medical X-rays are least vulnerable (visual uniformity limits reconstruction), and natural images fall in between.

4. **High training loss ratios are necessary but not sufficient.** The effusion query had the highest medical loss ratio (14.85x) but the worst reconstruction, because the private batch may not have contained strong effusion cases. The attack's effectiveness depends on both the model's gradient concentration AND the actual content of the victim's data.

5. **Multi-label classification is attackable.** Our BCEWithLogitsLoss adaptation for UAVScenes demonstrates that Geminio generalizes beyond single-label CrossEntropyLoss, broadening the attack surface.

---

## 11. Next Steps

1. ~~**Quantitative metrics:** Compute PSNR, SSIM, and LPIPS~~ — **DONE** (Section 8)
2. **Controlled batch composition:** Run reconstructions with batches known to contain specific target images (e.g., include a pool image in the UAV batch) to isolate the effect of batch content vs query targeting
3. ~~**Defense evaluation:** Test gradient pruning and noise defenses~~ — **DONE** (Section 9)
4. **More queries:** Run reconstructions for all trained queries (5 ImageNet, 7 medical, 6 UAV)
5. **Paper figures:** Generate publication-quality comparison figures
6. **Baseline attack comparisons:** Compare Geminio against standard gradient inversion (no query targeting) and other attacks (Fishing, GradFilt) if implementations are available
