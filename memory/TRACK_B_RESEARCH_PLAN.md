# Track B Research Plan: Brain-Specialized Cell Foundation Models as Dynamics-Aware Embedding Systems

Generated: 2026-05-18  
Status: DRAFT v1  
Scope: Track B only. This plan intentionally ignores the Track A phenomena tour except where it supplies reusable infrastructure.

---

## Executive Summary

The independent project should not be framed as "train a small scGPT clone." That is too weak and too benchmark-shaped. The stronger project is:

> Train from-scratch brain-specialized cell foundation models on Allen ABC whole-brain scRNA-seq, then test whether their learned representations preserve biologically meaningful cell-state dynamics better than supervised classifiers and classical baselines.

The core scientific tension is:

- Supervised cell-type learning encourages **neural collapse**: same-label cells become tight points.
- Biological systems are not all discrete points: OPCs cycle, developing neurons move through trajectories, and adult cell states can have flux.
- A useful biological embedding should preserve both **stable identities** and **dynamic axes**.

The project's novelty is therefore not "better cell-type classification." It is:

1. **Training dynamics** of a from-scratch brain-specialized cell FM.
2. **Representation geometry** under different sparse-vector encodings.
3. **Dynamics preservation** versus collapse, using Allen datasets with known stable and dynamic populations.

---

## Premise Challenge

### Premise 1: A brain-specialized cell FM is worth training from scratch

**Verdict: keep, but sharpen.**

Training from scratch is justified only if the object of study is the training process and representation geometry. If the goal were downstream accuracy, the better baseline would be to fine-tune scGPT, Geneformer, scFoundation, UCE, or scPRINT. The project should explicitly say:

> I train from scratch because released checkpoints do not expose training dynamics, scaling behavior, collapse trajectories, or tokenization-controlled representation evolution.

### Premise 2: ABC WMB is the primary corpus

**Verdict: keep.**

Allen WMB 10x scRNA-seq is the only dataset in scope with the needed scale: ~4M QC cells, whole-brain diversity, rich taxonomy, and enough adult populations to test stable versus dynamic structure. MERFISH and Dev VIS are not substitutes.

### Premise 3: MERFISH should be part of the core training corpus

**Verdict: reject.**

MERFISH is a spatial validation modality, not a standard cell-FM corpus. It has ~500 genes and spatial coordinates. Standard scRNA-shaped tokenizers break on it. Use MERFISH to validate anatomical localization of learned clusters or axes, not for pretraining in v1.

### Premise 4: Dev VIS Multiome belongs in the core

**Verdict: reject for v1, keep as Research Aim 3 / stretch.**

Dev VIS Multiome is ideal for developmental RNA+ATAC trajectory questions, but it is smaller, nuclei-based, and region-specific. It should be used after the WMB cell-FM pipeline works. Its best role is testing whether FM embeddings preserve developmental continuity or whether ATAC improves trajectory prediction.

### Premise 5: Rank encoding is the default because it is easiest

**Verdict: partially reject.**

Rank encoding is a good first smoke-test because it is simple and comparable to Geneformer. But the project's main biological question needs expression magnitudes. The serious comparison must include a value-aware tokenizer: binned expression or continuous value embeddings.

---

## Refined Research Question

### Main Question

Do self-supervised brain-specialized cell foundation models learn embeddings that preserve biologically meaningful stable and dynamic cell-state structure, or do they collapse toward discrete taxonomy labels?

### Subquestions

1. **Training dynamics:** How do loss, downstream accuracy, representation variance, attention patterns, and scaling behavior change as model size increases from 10M to 100M parameters?
2. **Encoding choice:** Does rank-only tokenization erase dynamic structure compared with value-aware tokenization?
3. **Collapse versus dynamics:** Does self-supervised masked gene modeling preserve within-type variation better than supervised cell-type classification?
4. **Biological geometry:** Do FM embeddings recover stable interneuron classes and dynamic OPC cycling axes better than PCA, scVI, and diffusion-map baselines?
5. **Developmental transfer:** When evaluated on Dev VIS, do learned embeddings preserve developmental continuity, or does sc/sn distribution shift dominate?

---

## Recommended Project Title Options

1. **Training Dynamics and Representation Geometry of Brain-Specialized Cell Foundation Models**
2. **Do Cell Foundation Models Preserve Biological Dynamics? A From-Scratch Study on the Allen Brain Cell Atlas**
3. **Sparse Tokenization, Neural Collapse, and Dynamics Preservation in Brain Cell Foundation Models**

Recommended: **Option 2** for external readability. It states the question plainly.

---

## Datasets

### Dataset A: Allen WMB 10x scRNA-seq

**Role:** primary pretraining and adult geometry corpus.

**Why:** ~4M QC cells, whole-brain adult diversity, 5k+ clusters, rich cell taxonomy. This supports scaling experiments and stable/dynamic population tests.

**Use in this project:**

- Start with **isocortex** for model/debugging.
- Expand to **whole brain or multi-region subset** for 30M/100M training if compute permits.
- Pull specific analysis subsets:
  - Stable inhibitory populations: SST, PV, VIP.
  - Dynamic glial populations: OPC / oligodendrocyte lineage.
  - Optional: microglia activation-like states if metadata supports it.

### Dataset B: Allen WMB MERFISH

**Role:** spatial validation only.

**Why:** 500-gene panel + CCFv3 coordinates. It can test whether clusters or embedding neighborhoods correspond to anatomical structure.

**Use in this project:**

- Do not use in v1 pretraining.
- Map WMB-10X clusters / subclasses to MERFISH spatial distributions.
- Use for final figures:
  - where stable interneuron populations reside,
  - whether dynamic OPC neighborhoods align with white/gray matter structure,
  - anatomical sanity checks for learned axes.

### Dataset C: Allen Dev VIS scRNA + Multiome

**Role:** developmental transfer / stretch experiment.

**Why:** 568k scRNA cells plus 200k Multiome nuclei with paired RNA+ATAC and developmental ages E11.5 to P56. This is the right dataset for trajectory and epigenetic predictive-information claims.

**Use in this project:**

- First use RNA-only Dev VIS as an out-of-domain evaluation set.
- Test trajectory continuity in FM embedding:
  - nearest-neighbor age monotonicity,
  - pseudotime correlation,
  - subclass-by-age smoothness.
- Only after core WMB results: add ATAC via scVI / peak module features and compare RNA-only versus RNA+ATAC trajectory inference.

### Dataset Relationship Rules

1. WMB 10x = pretraining corpus.
2. WMB MERFISH = spatial validation.
3. Dev VIS = developmental trajectory / epigenetic extension.
4. Do not pool all three into one training matrix in v1.
5. Treat cell versus nucleus differences as a real batch/domain shift.

---

## Methods Under Consideration

### Input Encoding Families

| Family | Method | Role in project |
|--------|--------|-----------------|
| A | HVG + dense MLP | minimal supervised baseline |
| B | Weighted EmbeddingBag | strong sparse non-transformer baseline |
| C1 | Rank tokens | Geneformer-style FM v1 |
| C2 | Binned expression tokens | scGPT-style FM v2 |
| C3 | Continuous value tokens | value-aware FM v3 if time permits |
| D | PCA / TruncatedSVD / scVI | classical latent baselines |
| D/E | Diffusion maps / anisotropic diffusion | biology-motivated geometry baseline |
| E | Graph / peak-gene encoders | stretch for Dev VIS Multiome, not core v1 |

### Locked Method Decisions

1. **Implement weighted EmbeddingBag baseline early.** It is mathematically sparse linear and sets a serious non-transformer floor.
2. **Use rank tokens for first FM smoke test.** This reduces implementation risk.
3. **Use value-aware tokens for the main biology result.** Rank-only tokenization likely drops the magnitude information needed for OPC cycling and developmental gradients.
4. **Pick sequence length from data.** Use the 95th percentile of nonzero genes per cell, not a published default.
5. **Always compare to PCA/scVI/diffusion baselines.** Without classical baselines, the FM embedding result has no biological meaning.

---

## Model Families

### Baseline 0: Classical Latent Models

- HVG + PCA / TruncatedSVD.
- scVI latent model.
- Diffusion maps / anisotropic diffusion maps.

**Purpose:** establish how much geometry is recoverable without transformers.

### Baseline 1: Supervised Cell-Type Classifier

- Input: HVG dense vector or EmbeddingBag.
- Objective: classify subclass / cluster.
- Metrics: accuracy, F1, calibration, within-class variance, collapse score.

**Purpose:** controlled neural-collapse comparison. This model should classify well but may destroy within-type dynamics.

### Model 1: Rank-Token Cell FM

- Input: top-L expressed genes sorted by rank.
- Objective: masked gene modeling.
- Architecture: BERT-style transformer encoder.
- Scales: 1M smoke test, 10M serious, 30M, 100M.

**Purpose:** simple from-scratch Geneformer-like baseline.

### Model 2: Value-Aware Cell FM

- Input: gene ID + binned or continuous expression value.
- Objective: masked gene/value prediction.
- Architecture: transformer encoder with gene embedding + value embedding.
- Scales: 10M and 30M minimum; 100M if rank-token result is promising.

**Purpose:** main biological comparison. Hypothesis: value-aware models better preserve dynamics.

### Optional Model 3: Dynamics-Regularized Cell FM

Only add if Models 1 and 2 are complete.

Possible regularizers:

- preserve local neighborhoods from diffusion map,
- contrastive positives from same subclass but nearby along pseudotime/cell-cycle phase,
- within-class variance floor for dynamic populations,
- graph-aware term using gene modules.

**Purpose:** intervention if standard self-supervision collapses dynamic axes.

---

## Core Hypotheses

### H1: Self-supervised FM embeddings collapse less than supervised classifiers

**Prediction:** supervised classifiers show lower within-class variance and worse OPC-cycle preservation than masked-gene FMs.

**Test:**

- Train supervised subclass classifier.
- Train rank-token and value-aware FMs.
- Compare within-subclass covariance spectra, participation ratio, and nearest-neighbor continuity.

### H2: Value-aware tokenization preserves dynamic structure better than rank encoding

**Prediction:** rank-token models recover discrete cell type well but underperform on continuous axes such as OPC cycling and development.

**Test:**

- Same architecture/size, rank versus binned/continuous value tokens.
- Compare:
  - cell-type classification,
  - OPC cell-cycle score correlation,
  - diffusion pseudotime correlation,
  - kNN graph preservation,
  - topological/cyclic signal in OPCs.

### H3: Scaling improves loss and classification before it improves biological geometry

**Prediction:** larger models improve masked-gene loss and held-out labels earlier than they improve dynamic-axis preservation.

**Test:**

- Train 10M, 30M, 100M.
- Fit scaling curves for validation loss and downstream metrics separately.
- Look for metric divergence: accuracy saturates while geometry keeps changing or degrades.

### H4: Classical dynamics-aware embeddings remain competitive on small biological subsets

**Prediction:** diffusion maps outperform FMs on local dynamic axes unless FM objective is value-aware or dynamics-regularized.

**Test:**

- Compare PCA, scVI, diffusion map, rank FM, value FM on same OPC and interneuron subsets.

### H5: Dev VIS exposes cell-versus-nucleus transfer failure

**Prediction:** WMB-trained models transfer imperfectly to Dev VIS Multiome snRNA because of sc/sn shift. Some failure will be technical domain shift, not biology.

**Test:**

- Evaluate on Dev VIS RNA-only.
- Compare with and without batch/modality token or normalization.
- Measure age monotonicity and subclass continuity.

---

## Metrics

### Training Metrics

- Training loss and validation loss.
- Masked gene prediction top-k accuracy.
- Masked value-bin accuracy or MSE for value-aware model.
- Gradient norm, update-to-weight ratio.
- Attention entropy by layer/head.
- Token frequency bias.
- Runtime throughput: cells/sec, tokens/sec, GPU memory.

### Downstream Task Metrics

- Subclass classification accuracy.
- Cluster classification accuracy.
- Macro-F1 for rare classes.
- Marker-gene recovery.
- Batch/donor prediction leakage.

### Geometry Metrics

- Within-class covariance trace.
- Participation ratio / effective dimension.
- Between-class versus within-class variance.
- Neural collapse ETF-style class-center geometry where applicable.
- kNN graph preservation against PCA/scVI/diffusion reference.
- Trustworthiness / continuity.
- Silhouette score for stable types.
- Pseudotime or cell-cycle score correlation for dynamic populations.
- Persistent homology / loop score for OPC cycling, if practical.

### Biological Validation Metrics

- SST/PV/VIP separation and marker agreement.
- OPC cycling axis correlation with cell-cycle gene scores.
- Oligodendrocyte lineage ordering.
- Dev VIS age monotonicity and subclass-by-age smoothness.
- MERFISH anatomical coherence for selected subclasses.

---

## Experimental Plan

### Phase 0: Research Spec and Data Probe (Week 1)

**Goal:** prove the data and hypotheses are concrete before model work.

Tasks:

1. Download/access one WMB-10X package, likely isocortex.
2. Inspect AnnData structure:
   - `.X`, `.obs`, `.var`,
   - cell type labels,
   - donor/library metadata,
   - nnz distribution per cell.
3. Pick first analysis populations:
   - stable: SST/PV/VIP,
   - dynamic: OPC / oligodendrocyte lineage.
4. Compute:
   - nnz histogram,
   - top expressed genes,
   - class balance,
   - HVG list,
   - cell-cycle score feasibility.
5. Write `notes/week01_cell_fm_spec.md`.

Exit criteria:

- one batch of cells loaded,
- top-L choice justified from nnz distribution,
- stable/dynamic subsets identified,
- no model code beyond data inspection.

### Phase 1: Serious Baselines (Weeks 2-3)

**Goal:** make the transformer earn its place.

Models:

1. HVG + logistic/MLP classifier.
2. Weighted EmbeddingBag classifier.
3. PCA/TruncatedSVD latent.
4. scVI latent, if setup time is reasonable.
5. Diffusion map on selected subsets.

Outputs:

- baseline accuracy table,
- baseline geometry table,
- first OPC/interneuron plots,
- compute estimate for FM training.

Exit criteria:

- supervised classifier working,
- collapse metrics implemented,
- diffusion/PCA baselines available for comparison.

### Phase 2: Rank-Token FM v1 (Weeks 4-5)

**Goal:** end-to-end from-scratch FM training.

Implementation:

- tokenization: top-L rank genes,
- architecture: transformer encoder,
- objective: masked gene modeling,
- scale: 1M smoke, 10M serious.

Measurements:

- train/val loss,
- cell-type eval from frozen embeddings,
- attention entropy,
- collapse metrics,
- geometry on stable/dynamic subsets.

Exit criteria:

- 10M model converges,
- embeddings extractable,
- rank-token limitations measured, not guessed.

### Phase 3: Value-Aware FM v2 (Weeks 6-7)

**Goal:** test the main biology-relevant method.

Implementation:

- gene embedding + value-bin embedding, or continuous value projection,
- same architecture budget as rank-token model,
- objective: masked gene/value prediction.

Experiments:

- 10M value-aware model.
- If successful, 30M value-aware model.
- Same eval suite as rank-token model.

Exit criteria:

- rank versus value comparison complete at matched scale,
- evidence for or against H2.

### Phase 4: Scaling and Training Dynamics (Weeks 8-9)

**Goal:** turn the project into a training-dynamics study.

Models:

- 10M, 30M, 100M rank-token and/or value-aware depending on Phase 3 outcome.

Analysis:

- scaling curves for loss,
- scaling curves for downstream classification,
- scaling curves for geometry metrics,
- attention evolution across steps and scale,
- rare-cell performance across scale.

Exit criteria:

- at least three model sizes for one tokenizer,
- one controlled tokenizer comparison,
- figure-ready scaling plots.

### Phase 5: Biological Geometry Study (Weeks 10-11)

**Goal:** answer the main research question.

Stable-state tests:

- SST/PV/VIP embeddings,
- class-center separation,
- marker recovery,
- MERFISH anatomical validation if time permits.

Dynamic-state tests:

- OPC cell-cycle axis,
- oligodendrocyte lineage ordering,
- cyclic/trajectory continuity,
- comparison to diffusion maps and scVI.

Exit criteria:

- stable versus dynamic contrast written as a result,
- clear pass/fail for H1-H4.

### Phase 6: Dev VIS Transfer / Multiome Stretch (Week 12+)

Only start if the WMB study is coherent.

Minimum Dev VIS experiment:

- load Dev VIS RNA-only,
- map labels,
- evaluate WMB-trained encoder,
- test age monotonicity and subclass continuity,
- report sc/sn shift explicitly.

Optional Multiome experiment:

- use scVI or peak modules for RNA+ATAC latent,
- compare RNA-only versus RNA+ATAC trajectory smoothness,
- do not attempt a full multimodal transformer unless everything else is done.

Exit criteria:

- one developmental transfer result,
- clear statement of whether Dev VIS supports or complicates the WMB result.

---

## Ablation Matrix

### Required Ablations

| Ablation | Why |
|----------|-----|
| supervised classifier vs self-supervised FM | tests neural collapse hypothesis |
| rank vs value-aware tokenization | tests whether magnitudes matter |
| 10M vs 30M vs 100M | scaling and training dynamics |
| FM vs PCA/scVI/diffusion | prevents transformer-only storytelling |
| stable types vs dynamic populations | central biological contrast |

### Optional Ablations

| Ablation | When to add |
|----------|-------------|
| sequence length L | if truncation appears large |
| mask ratio | if MGP loss unstable |
| rare-cell sampling | if model ignores rare types |
| batch/modality token | if Dev VIS transfer fails |
| dynamics regularizer | if standard FMs collapse dynamic axes |

---

## Figures to Produce

### Minimum Figure Set

1. **Project schematic:** WMB pretraining → embeddings → stable/dynamic tests → Dev VIS stretch.
2. **Dataset table:** WMB-10X, WMB-MERFISH, Dev VIS, roles and exclusions.
3. **Input encoding diagram:** sparse vector → rank tokens / value tokens / EmbeddingBag.
4. **Training curves:** 10M/30M/100M loss curves.
5. **Scaling plot:** validation loss and biological metrics versus parameters.
6. **Collapse plot:** supervised vs FM within-class variance over training.
7. **Stable-state panel:** SST/PV/VIP embeddings across PCA/scVI/diffusion/FM.
8. **Dynamic-state panel:** OPC cycle across PCA/scVI/diffusion/FM.
9. **Tokenizer comparison:** rank vs value-aware on the same biological metrics.
10. **Optional Dev VIS panel:** developmental age continuity in embedding space.

### Strong Final Storyboard

1. Existing cell FMs hide training dynamics.
2. Sparse encoding choice changes what biology the model can preserve.
3. Supervised models classify well but collapse dynamic variation.
4. Self-supervised FMs preserve more structure, but only with value-aware tokenization.
5. Classical diffusion methods remain strong on local dynamics.
6. Scaling improves loss before it improves biological geometry.

---

## Success Criteria

### Minimum Viable Research Project

- WMB subset pipeline works.
- Weighted EmbeddingBag and supervised classifier baselines work.
- Rank-token 10M FM trains from scratch.
- One stable population and one dynamic population analyzed.
- FM embeddings compared to PCA/diffusion.
- Written result: "what worked, what failed, what this says about cell FM embeddings."

### Strong Project

- Rank and value-aware tokenizers compared.
- 10M/30M/100M scaling data.
- Neural-collapse metrics tracked during training.
- OPC dynamic structure convincingly evaluated.
- MERFISH spatial validation used for one figure.

### Excellent Project

- Dev VIS transfer result included.
- RNA+ATAC trajectory stretch included.
- Dynamics-preserving regularizer proposed and tested.
- Repo reads like a small research paper, not a tutorial.

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Data access too slow / huge | stalls project | start with one WMB package; cache preprocessed token files |
| Transformer does not beat baselines | weak if framed as accuracy | frame around training dynamics and geometry, not benchmark wins |
| Rank tokenizer destroys dynamics | expected failure | use as controlled negative result; value-aware tokenizer is main |
| scVI/diffusion beats FM | not fatal | this is a real finding about dynamics-aware embeddings |
| Dev VIS sc/sn shift confounds result | false biological claim | explicitly model as domain shift; keep Dev VIS stretch |
| 100M too expensive | scaling story weakened | 1M/10M/30M still enough if analysis is deep |
| Too many directions | project dies | WMB dynamics first; Dev VIS and MERFISH are validation/stretch |

---

## Not In Scope

- Beating scGPT/Geneformer on public benchmarks.
- Full multimodal RNA+ATAC transformer.
- Training on MERFISH as if it were scRNA-seq.
- Whole-brain 300M run before small WMB subset works.
- Direction II cortical plasticity modeling.
- Production-quality package or API.

---

## Concrete Week-by-Week Plan

### Week 1: Data and research spec

- Inspect WMB-10X isocortex package.
- Write `notes/week01_cell_fm_spec.md`.
- Compute nnz distribution and pick provisional L.
- Identify SST/PV/VIP and OPC subsets.

### Week 2: Baseline encoders

- HVG + dense classifier.
- Weighted EmbeddingBag classifier.
- PCA/TruncatedSVD latent.
- First collapse metrics.

### Week 3: Classical biology baselines

- scVI if feasible.
- Diffusion maps on stable/dynamic subsets.
- First stable/dynamic geometry report.

### Week 4: Rank-token FM smoke and 10M

- Implement rank tokenizer.
- Train 1M smoke.
- Train 10M rank-token FM.

### Week 5: Rank-token evaluation

- Extract embeddings.
- Evaluate classification, collapse, stable/dynamic geometry.
- Decide if rank-token limitations are severe.

### Week 6: Value-aware tokenizer

- Implement binned or continuous-value tokenizer.
- Train 10M value-aware FM.
- Matched comparison against 10M rank model.

### Week 7: Value-aware 30M

- Train 30M if 10M is stable.
- Compare scaling across loss and geometry.

### Week 8: Scaling push

- Train 100M for the winning tokenizer if compute allows.
- Otherwise expand dataset coverage and keep 30M analysis deep.

### Week 9: Neural collapse and geometry synthesis

- Supervised vs self-supervised collapse.
- SST/PV/VIP stable-state study.
- OPC dynamic-state study.

### Week 10: MERFISH validation

- Pull MERFISH labels/coordinates for selected subclasses.
- Spatial sanity-check final embeddings and clusters.

### Week 11: Dev VIS transfer

- Load Dev VIS RNA-only.
- Evaluate WMB encoder on developmental cells/nuclei.
- Test age and pseudotime continuity.

### Week 12: Write-up

- Figures.
- README.
- Research memo.
- Decide whether the next phase is Multiome RNA+ATAC or dynamics regularization.

---

## Decision Gates

### Gate 1: End of Week 1

Proceed only if:

- WMB subset loads,
- labels are available,
- stable/dynamic subsets are identified,
- L and tokenization candidates are data-backed.

### Gate 2: End of Week 3

Proceed to transformer only if:

- baselines run,
- geometry metrics run,
- collapse metrics run.

### Gate 3: End of Week 5

Proceed to value-aware model if:

- rank-token model trains,
- embeddings extract cleanly,
- rank limitations are measurable.

### Gate 4: End of Week 7

Proceed to 100M only if:

- 10M/30M curves are stable,
- value-aware model beats or meaningfully differs from rank model on biological geometry.

### Gate 5: End of Week 10

Proceed to Dev VIS only if:

- WMB result can already be summarized in 3 figures.

---

## Autoplan Review Summary

### Best Current Approach

Use a **two-model FM comparison**:

1. rank-token FM as the simple Geneformer-like baseline,
2. value-aware FM as the biologically serious model.

Compare both against supervised, PCA/scVI, and diffusion baselines on WMB stable/dynamic populations.

### Rejected Approaches

1. **Train one giant FM first.** Too likely to produce expensive curves without biological meaning.
2. **Start with Dev VIS Multiome.** Too much multimodal complexity before the FM baseline exists.
3. **Use MERFISH in pretraining.** Wrong modality for v1.
4. **Optimize downstream cell-type accuracy.** Wrong target; not novel enough.

### Taste Decision

Whether to prioritize **rank-token 100M scaling** or **value-aware 30M biology** if compute is limited.

Recommendation: **value-aware 30M biology**. A smaller model that answers the biological question is stronger than a larger model that only improves loss.

---

## The Assignment

Write `notes/week01_cell_fm_spec.md` with these exact sections:

1. **Primary question:** one paragraph.
2. **Dataset choice:** WMB subset, stable subset, dynamic subset.
3. **Sparse encoding plan:** EmbeddingBag baseline, rank tokenizer, value-aware tokenizer.
4. **Pretraining objective:** masked gene/value modeling.
5. **Evaluation suite:** classification, collapse, geometry, biology.
6. **Compute plan:** model sizes, expected runtime, fallback if 100M is too expensive.
7. **Kill criteria:** when to stop scaling and switch to analysis.

The sentence that must appear near the top:

> This project studies whether brain-specialized self-supervised cell foundation models preserve biological dynamics that supervised cell-type models tend to collapse.

