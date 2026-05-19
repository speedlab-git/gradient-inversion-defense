# Language analysis sweep — plan (#1 + #2)

## Goal

Turn "language" from interface into object of study. Two deliverables:

- **#1 Empirical sensitivity:** how do paraphrase / composition / negation choices change Geminio's amplification (loss ratio) and end-to-end attack success?
- **#2 Discriminability-leakage analysis:** define a VLM-based "query discriminability" margin on the auxiliary set; correlate it with loss ratio and attack F1 across queries. Show when correlation breaks (domain-dependent).

## Query design — 12 medical + 12 UAV = 24 new malicious models

(Expanded from 9+9 per user direction: added 3 language-diversity stress queries per domain — multilingual, lexical noise, extra-verbose — addressing reviewer item #6 simultaneously.)

### Medical (base concept: pneumonia, base text already trained as `pneumonia_descriptive`)

| Tag | Query | Class |
|---|---|---|
| `pna_base` | "Any chest X-ray showing pneumonia" | paraphrase (anchor) |
| `pna_terse` | "pneumonia" | paraphrase (terse) |
| `pna_verbose` | "A frontal chest radiograph demonstrating airspace consolidation consistent with pneumonia" | paraphrase (verbose) |
| `pna_synonym` | "Chest imaging revealing signs of bacterial pneumonia" | paraphrase (lexical) |
| `pna_indirect` | "Lung infection visible on chest radiograph" | paraphrase (indirect) |
| `pna_compAND` | "Any chest X-ray showing pneumonia and cardiomegaly" | composition (AND) |
| `pna_compOR` | "Any chest X-ray showing either pneumonia or pleural effusion" | composition (OR) |
| `pna_negEX` | "Any chest X-ray showing pneumonia without pleural effusion" | negation (exclusion) |
| `pna_negPURE` | "Any chest X-ray that does not show pneumonia" | negation (pure) |
| `pna_es` | "Cualquier radiografía de tórax que muestre neumonía" | multilingual (Spanish) |
| `pna_typo` | "Any chst Xray showing pneumon" | lexical noise / typos |
| `pna_xlong` | "A diagnostic chest X-ray, presented in standard posteroanterior projection, demonstrating focal or multilobar airspace consolidation with air bronchograms consistent with community- or hospital-acquired pneumonia" | extra-verbose (radiologist-style) |

### UAV (base concept: solar panels, base text already trained as `solar_panels`)

| Tag | Query | Class |
|---|---|---|
| `sol_base` | "aerial drone image showing solar panels on rooftops" | paraphrase (anchor) |
| `sol_terse` | "solar panels" | paraphrase (terse) |
| `sol_verbose` | "high-altitude aerial photograph capturing photovoltaic panel installations atop residential and commercial rooftops" | paraphrase (verbose) |
| `sol_synonym` | "drone view of rooftop photovoltaic installations" | paraphrase (lexical) |
| `sol_indirect` | "Solar power infrastructure visible from above" | paraphrase (indirect) |
| `sol_compAND` | "aerial drone image showing solar panels on residential rooftops in a suburban neighborhood" | composition (AND, specificity) |
| `sol_compOR` | "aerial drone image showing either solar panels or wind turbines" | composition (OR) |
| `sol_negEX` | "aerial drone image showing solar panels without trees blocking the view" | negation (exclusion) |
| `sol_negPURE` | "aerial drone image that does not show solar panels" | negation (pure) |
| `sol_es` | "Imagen aérea de dron mostrando paneles solares en tejados" | multilingual (Spanish) |
| `sol_typo` | "aerial dron image showing solar panles on rofotops" | lexical noise / typos |
| `sol_xlong` | "A high-resolution aerial photograph taken from a quadcopter drone at altitude, capturing photovoltaic solar panel arrays installed atop residential single-family homes and commercial flat-roof buildings in a suburban neighborhood" | extra-verbose |

## Measurements per query

1. **Loss ratio** $R(Q)$ — emitted by `train_*.py` at end of Phase 2 training (mean high-sim loss / mean low-sim loss). Free.
2. **Discriminability margin** $\Delta(Q)$ — on the auxiliary embeddings, define
   $$\Delta(Q) = \mathbb{E}_{i \in \text{top-}k} \cos(e_i, t_Q) - \mathbb{E}_{i \in \text{bottom-}k} \cos(e_i, t_Q)$$
   where $t_Q$ is the VLM text embedding, $e_i$ are image embeddings, $k = 10\%$. Cheap, post-hoc.
3. **Clean attack F1** (seed 42 only for screening) on a batch of 8 reconstructed at 24k iterations.
4. **NBFU defense F1** ($\varepsilon{=}5$, sensitivity $2C$) on a *subset* (base + 1 hardest paraphrase + 1 negation, both domains = 6 conditions) — confirms defense invariance to language.

## Compute estimate (updated for 24 queries × 3 seeds × clean+defense)

| Stage | Per cell | Total cells | Wall-clock parallel (4 GPUs) |
|---|---|---|---|
| Train malicious model (5 epochs, CLF head only) | ~5 min | 24 | ~30-45 min |
| Clean reconstruction (24k iters) | ~12 min | 24 × 3 seeds = 72 | ~4-5 hr |
| NBFU defense reconstruction (24k iters) | ~15 min | 24 × 3 seeds = 72 | ~5-6 hr |
| Discriminability metric | post-hoc | 24 | ~10 min |
| **Total** | | | **~10-12 hr parallel** |

GPU access is currently tight (most at 90+ GiB used out of 143). Reconstruction needs ~15-20 GiB; training needs ~6-8 GiB. Realistic wall-clock 1-2 days given shared-GPU pressure. Will need to launch in stages and monitor.

## Launch staging

1. **Stage 1 (now)**: Train all 24 malicious models. Run 4 in parallel on different GPUs. Estimate 45 min.
2. **Stage 2 (overnight)**: Clean reconstruction × 72 runs. Stage in batches of 4 parallel.
3. **Stage 3**: NBFU defense reconstruction × 72 runs.
4. **Stage 4 (post-hoc)**: Discriminability + analysis script + figure.

## Outputs

- **Table 10**: per-query (loss ratio, discriminability $\Delta$, mean clean F1 ± std across 3 seeds) for all 24 queries.
- **Figure 6**: scatter of discriminability $\Delta$ vs.\ loss ratio (or attack F1), one point per query, colored by domain and shape-coded by query class (paraphrase/composition/negation/diversity) — visualizes when correlation holds and breaks.
- **Table 11**: NBFU $\varepsilon{=}5$ defense F1 (mean ± std) across all 24 queries grouped by class — confirms uniform attack-F1 suppression regardless of language regime.
- **New Section 5.X / 6.X**: "Language properties of the attack surface" (~3/4 page + table + figure).

## Risks / pitfalls

- Negation queries (`negPURE`) may produce essentially baseline (untargeted) attacks if the VLM can't represent "not X" properly. That's a *finding*, not a failure.
- Lossy paraphrases may produce very low loss ratios (training basically fails); also a finding.
- ChestMNIST batch reconstruction at 24k iters with seed 42 has F1 variance ~0.07 across seeds — single-seed screening accepts this noise.
- Risk that 5-epoch Phase 2 isn't enough for verbose/indirect paraphrases to converge. Mitigate: monitor final loss-ratio; if `R < 5` for a query, mark as "weak amplification" and report as such rather than re-training.

## Decision points before launch

1. **Query set** as above — accept or modify?
2. **NBFU defense subset**: confirm 6 conditions (3 queries × 2 domains) is enough, or extend to all 18?
3. **Seed count**: stick with seed 42 only for screening (current plan), or run 3-seed for robustness?
4. **Output dir**: `malicious_models_language_sweep/` (separate from main models)?
