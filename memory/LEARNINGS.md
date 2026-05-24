# Project Learnings: Cell FM on Allen ABC Atlas

Captured 2026-05-18 from office-hours session.

These are durable insights about the project domain that future sessions
should reference before making implementation decisions. Confidence scores
reflect how settled each insight is.

---

## Architecture

### `merfish-vs-scrnaseq` (confidence: 8/10)

Standard cell foundation models (Geneformer, scGPT, scFoundation, UCE) are
**scRNA-seq-shaped**. They impose artificial token-sequence order via rank
encoding (sort genes by relative expression) or value binning across the
full ~25K-gene transcriptome.

**MERFISH is a different data modality**, not just a different dataset:
- ~500-gene panel (vs ~25K genes for scRNA-seq) → ~97% vocabulary mismatch
- Spatial (x, y) coordinates per cell → information absent from scRNA-seq
- Cell-cell tissue context → requires graph/spatial-aware attention

**Implication for project:** Use scRNA-seq from ABC Atlas for Core/Bridge
phases. Standard cell FM architecture (rank or binned encoding) works
directly. MERFISH integration requires real architecture changes (smaller
vocab, spatial position encoding, neighbor-aware attention) and should
remain a stretch option only if the project has time after Phase 3.

**Why this matters:** Choosing "use MERFISH because ABC has it" without
recognizing it's a different architecture problem would burn 4+ weeks on
multi-modal infrastructure instead of producing the dynamics-study results
that drive the project's novelty claim.

### `novelty-stack-cell-fm` (confidence: 9/10)

Existing cell foundation models (Geneformer 2023, scGPT 2024, scFoundation,
UCE, scPRINT) have **real gaps in the literature**, none of which require
beating SOTA on benchmarks to address:

1. **Training dynamics never published** — no loss curves, no scaling laws,
   no attention pattern evolution analysis. The cell FM papers are entirely
   downstream-benchmark-driven.
2. **All existing models are tissue-agnostic** — trained on broad-tissue
   corpora (CELLxGENE). No brain-specialized cell FM exists as a primary
   focus.
3. **Embedding-dynamics analysis missing** — no published study of whether
   cell FM embeddings preserve cycles (OPCs), trajectories (developmental
   cortex), or stable manifolds (mature interneurons).
4. **No comparison to dynamics-aware classical methods** — anisotropic
   diffusion maps vs cell FM embeddings on the same data is not published.
5. **Tokenization underspecified** — Geneformer says "rank encoding," scGPT
   says "binning," no controlled comparison with downstream task analysis
   exists.

**Project novelty stack (ranked by load-bearing weight):**

- **PRIMARY (must claim):** Training-dynamics study of from-scratch
  brain-specialized cell FM. Loss curves, scaling laws, attention pattern
  evolution. This is the contribution that justifies "from-scratch" framing.
- **SECONDARY (must do):** Dynamics-preservation analysis vs neural collapse.
  Specifically: does the cell FM embedding preserve OPC cycle structure,
  SST/PV/VIP separation, and developmental trajectory continuity? Compare
  to PCA and diffusion-map baselines.
- **TERTIARY (optional, stretch):** Methodological intervention to preserve
  dynamics — within-class variance regularizer, contrastive loss on
  developmental pairs, or graph-aware attention if MERFISH integrated.

**Reviewer/PI defense for "why not fine-tune scGPT?":**
Studying training dynamics requires owning the training run, not just the
released checkpoint. scGPT is a fine model but it's a closed artifact for
the questions this project asks.

**Why this matters:** Without this framing, the project drifts toward
benchmark-paper territory (compute-impossible at this scale) or pure
educational ML (no research output). The novelty stack threads the needle.

---

## Patterns

### `neural-collapse-vs-dynamic-embedding` (confidence: 8/10)

**Neural collapse** (within-class variance → 0, classes maximally separated)
is **optimal for cell-type classification** but **destructive for dynamic
embedding**.

For biologically dynamic populations:
- **Cycling cells (OPCs):** collapse destroys the cell-cycle axis. All
  cycling OPCs become a single point in latent space instead of a loop.
- **Mature stable types (SST, PV, VIP interneurons):** collapse is fine
  here — these ARE near fixed points biologically. Should see tight clusters.
- **Developmental trajectories (E12.5 → E16.5 cortex):** collapse destroys
  the developmental gradient. Cells at different stages should be
  continuously distributed along a trajectory.
- **Transition states (mid-differentiation cells):** collapse snaps these
  to whichever class is closest, erasing transition signal entirely.

**Key open question this project tests:** Does self-supervised cell FM
pretraining (masked gene prediction) show less collapse than supervised
cell-type classification? Hypothesis: yes, because masked-gene prediction
requires within-class variation to predict masked genes, whereas
classification only requires between-class separation.

**Week 9 experiments designed to test this:**
1. Measure within-class variance over training for supervised classifier
   on scRNA-seq.
2. Measure same metrics for self-supervised cell FM on labeled subsets.
3. For OPCs specifically: do PCA / persistent homology on embeddings,
   look for cyclic structure (supervised should fail, foundation model
   should preserve).
4. For SST/PV/VIP interneurons: confirm tight stable clusters in both.
5. For developing cortex (if dev visual cortex atlas included): test
   trajectory continuity via nearest-neighbor consistency along
   developmental time.

**Publishable result if hypothesis holds:** Self-supervised cell FM
pretraining produces embeddings that preserve biological dynamics that
supervised cell-type classification destroys. Direct contribution to cell
FM literature, direct connection to Direction I Line 2 of research proposal.

**Why this matters:** This is the load-bearing scientific claim of the
entire project. Every other experiment supports or qualifies this one.

---

## Datasets

Three Allen datasets are in scope for this project. Each has a different
construction pipeline, scale, and role. They are **not interchangeable**.

### `wmb-10x-scrnaseq` (confidence: 9/10)

**Allen Brain Cell Atlas — Whole Mouse Brain 10x scRNA-seq.**
Primary reference: Yao et al., *Nature* 2023
(https://www.nature.com/articles/s41586-023-06812-z).

**Construction:**
1. **781 10x Chromium libraries** (v2 + v3) from anatomically defined,
   **CCFv3-guided** tissue microdissections across the adult mouse brain.
2. ~7.0M single-cell transcriptomes profiled → **stringent QC** (pilot-
   clustering-informed thresholds, doublet/debris filters) → **~4.3M cells
   pass**.
3. **Iterative clustering** with Allen's `scrattch.bigcat`: 10xv2 and 10xv3
   clustered separately, then integrated → initial taxonomy of 5,283 clusters.
4. Pairwise cluster comparisons → **8,108 differentially expressed genes**.
5. Noise-cluster removal (doublets/mixed debris that escaped initial QC) →
   **~5,200 final high-quality clusters** containing ~4.1M cells (paper
   figures; current portal release lists 5,322 clusters after subsequent
   curation).
6. Hierarchy: **7 divisions / 32 classes / 306 subclasses / 1,045 supertypes
   / 5,200 clusters** (paper) — portal version: 34 / 338 / 1,201 / 5,322.

**Shape on disk:**
- AnnData / H5AD, **sparse counts** in `.X`, ~20k genes in `.var`.
- Per-cell `.obs`: `cluster`, `subclass`, `supertype`, `class`, `division`,
  `neurotransmitter`, brain region, QC scores, library/donor IDs.
- Distributed as **multiple packages** (e.g. by region) via S3 + manifest.

**Access:** `abc_atlas_access` Python package + AWS Public Dataset on S3.
License: **CC BY-NC 4.0**.

**Role in this project:**
- **PRIMARY corpus for cell FM pretraining (Track B).** Scale (~4M cells)
  + label richness + region diversity make it the natural target for the
  10M → 30M → 100M → 300M parameter ladder.
- **PRIMARY corpus for adult dynamics-aware embeddings (Direction I.2).**
  OPC cycling, SST/PV/VIP stable types, microglia at population scale.
- Use **regional subsets** (e.g. isocortex, white matter) for early work;
  do not load whole brain at once.

**Why this matters:** This is the only Allen dataset with the scale needed
for a brain-specialized cell FM. Without it, Track B has no corpus.

### `wmb-merfish-spatial` (confidence: 8/10)

**Allen Brain Cell Atlas — Whole Mouse Brain MERFISH (AIBS panel,
MERFISH-C57BL6J-638850).** Same paper as `wmb-10x-scrnaseq`.

**Construction:**
1. **DEG-driven panel design:** 500 marker genes selected from the 8,108
   scRNA-seq DEGs to maximally discriminate WMB clusters (a second 1,147-
   gene panel was used by the Zhuang lab in a companion study).
2. **Imaging:** Vizgen MERSCOPE, **59 serial coronal sections** at 200 μm
   intervals, **single male C57BL/6J brain**.
3. **Segmentation + QC** → **~4.3M MERFISH cells**.
4. **Registration** of every section to **Allen CCFv3** → 3D coordinates
   per cell.
5. **Mapping to taxonomy:** each MERFISH cell scored against scRNA-seq
   cluster profiles on the 500-gene overlap → best-match cluster +
   correlation score.
6. **Spatial annotation per cluster:** computed from high-confidence
   MERFISH assignments only.

**Shape on disk:**
- AnnData / H5AD, **dense ~500-gene** expression in `.X`.
- `.obs`: x/y/z (or `obsm['spatial']`), CCF region, mapped `cluster`,
  correlation score, section ID.
- Median expression correlation 10xv3 vs MERFISH ≈ **0.91** for matched
  types (paper).

**Access:** `abc_atlas_access`, S3 (separate package from WMB-10X). License:
CC BY-NC 4.0.

**Role in this project:**
- **Spatial validation only.** Cluster locations, anatomical figures,
  sanity-check that embedding axes respect brain geography.
- **NOT a pretraining corpus.** See `merfish-vs-scrnaseq` learning above:
  500-gene panel + spatial coords are a different modality from scRNA-seq,
  not just a smaller version of it. Using MERFISH as a cell FM corpus
  requires architecture changes (smaller vocab, spatial position
  encoding, neighbor attention).

**Why this matters:** Tempting to combine "all ABC data" into one corpus.
Doing so silently breaks the standard cell-FM tokenizer and conflates two
modalities that answer different questions.

### `devvis-multiome` (confidence: 9/10)

**Allen Developing Mouse Visual Cortex — 10x scRNA-seq + 10x Multiome.**
Primary reference: Yao et al., *Nature* 2025
(https://www.nature.com/articles/s41586-025-09644-1). Code:
https://github.com/AllenInstitute/MouseDevVIS.

**Construction:**
1. **53 mice**, **35 synchronized ages** E11.5 → ~P56 (whole brain at
   E11.5–E12.5, cerebrum + brainstem at E13.5–E14.5, **VIS dissected via
   CCFv3** from E15.5 onward).
2. **Two parallel lines on the same biology:**
   - **Line A — 10x GEX-only:** 913k cells profiled → **568,654 pass QC**.
     Defines the developmental taxonomy and pseudotime.
   - **Line B — 10x Multiome:** 331k nuclei profiled across **35 libraries
     / 13 embryonic + postnatal time points** → **200,061 nuclei pass QC**.
     Provides paired snRNA + snATAC **from the same nucleus barcode**.
3. **Taxonomy** built on scRNA: iterative clustering + **scVI integration
   across ages** → **15 classes / 40 subclasses / 148 clusters /
   714 subclusters**. Broadly consistent with WMB at subclass level, with
   finer time-resolved subclusters.
4. **Multiome labels:** transferred from scRNA reference via **scVI**
   integration (Multiome nuclei are **not** independently re-clustered into
   714 types from ATAC alone).
5. **ATAC processing:** **882,075 peaks** called with **ArchR** on
   pseudobulks (subclass × age and finer) → pairwise DA peak tests →
   **peak modules** with shared subclass/temporal specificity →
   peak↔gene linking within **5 Mb** by correlation across subclass-by-age
   groups → **SCENIC+** gene regulatory networks.
6. Cross-modal check at **P0**: Multiome snRNA-seq integrated with
   whole-brain MERFISH at P0 (the one place this dataset meets WMB
   spatially).

**Shape on disk:**
- **scRNA line:** via `abc_atlas_access`, ~14 GB matrices + ~180 MB
  metadata. `.obs` includes `donor_age` (35 values), `synchronized_age`,
  `age_bin` (18 bins like `E11.5_E12.5`, `P11`, `54_68`), `class`,
  `subclass`, `cluster`, `subcluster`.
- **Multiome line:** three separate S3 objects (not abc_atlas_access):
  - `DevVIS_multiome_snRNA_processed.h5ad` — 200k nuclei × ~20k genes.
  - `DevVIS_multiome_snATAC_processed.h5ad` — 200k nuclei × up to
    ~882k peaks.
  - `ATAC_fragment.tar.gz` — fragment-level data for ArchR workflows.
- **Pairing:** snRNA and snATAC share `obs_names` (nucleus barcode).

**Access:** scRNA via `abc_atlas_access`. Multiome direct from
`allen-developmental-mouse-atlas.s3...`. Raw data on **BICAN** portal +
**NeMO archive** (separate IDs per modality). License: CC BY-NC 4.0.

**Role in this project:**
- **PRIMARY corpus for Direction I.1** (neural OT for developmental
  trajectories + epigenetic prediction). The only Allen product with
  paired RNA+ATAC across dense developmental time.
- **Optional time-conditioned evaluation** for cell FM (Direction I.2 /
  Track B Week 9): test whether FM embeddings preserve developmental
  trajectory continuity.
- **NOT a substitute for WMB** for adult dynamics work — region (VIS only)
  and scale (200k nuclei) are wrong for whole-brain dynamics claims.

**Why this matters:** "Cortical Multiome" in the research proposal points
specifically to this dataset. OT + epigenetics is the load-bearing claim
for Direction I.1 and cannot be tested on WMB alone.

### `datasets-relationship` (confidence: 9/10)

The three datasets share a **taxonomy backbone** but cover different
**scopes**:

| Axis | WMB-10X | WMB-MERFISH | Dev VIS |
|------|---------|-------------|---------|
| Region | Whole brain | Whole brain | Visual cortex |
| Time | Adult | Adult | E11.5 → P56 |
| Modalities | scRNA | Spatial + 500-gene panel | scRNA + Multiome (RNA + ATAC) |
| Cells / nuclei | ~4.1M cells | ~4.3M cells | 568k cells + 200k nuclei |
| Defines taxonomy of | ~5,200–5,322 adult clusters | Mapped onto WMB-10X | 148 clusters / 714 subclusters |
| Spatial | None | CCFv3 x/y/z | None (P0 cross-link only) |
| Epigenetics | None | None | snATAC peaks (882k) |

**Shared:**
- **CCFv3** anatomical reference frame.
- **Subclass-level cell-type vocabulary** (`Sst`, `Pvalb`, `Vip`, `L2/3 IT`,
  `OPC`, etc.) aligns across WMB and Dev VIS.
- **Allen tooling**: `abc_atlas_access` for primary distribution; AWS S3.

**Connections that exist:**
1. Dev VIS adult endpoint (~P56) cells map onto WMB cortical subclasses
   via shared taxonomy → use **MapMyCells** or scVI projection.
2. Dev VIS Multiome P0 nuclei integrate with **WMB MERFISH at P0** in the
   2025 paper (single cross-link point between dev and adult spatial data).
3. **Consensus-WMB** taxonomy (separate product) integrates Allen scRNA
   with Broad snRNA via scVI → 6,721 consensus clusters. Useful if you
   want labels comparable across BICCN centers.

**Connections that do NOT exist:**
- No developmental MERFISH product (only the P0 cross-check).
- No joint MuData / paired-modality object on S3 — Multiome ships as two
  AnnData files joined by barcode.
- No shared raw cells between WMB and Dev VIS (different mice).

**Why this matters:** Treating "ABC Atlas" as one dataset hides the actual
data product split. Direction I.1 and Direction I.2 need different primary
corpora; the cell FM pretraining target is WMB-10X, not Dev VIS or MERFISH.

### `cells-vs-nuclei-shift` (confidence: 7/10)

WMB 10x is **single cells (sc)**. Dev VIS Multiome is **single nuclei
(sn)**. The Dev VIS companion scRNA line is also **cells**, but the
Multiome line is **nuclei**.

**Distributions differ in ways that are not biology:**
- snRNA underrepresents cytoplasmic transcripts and lncRNAs.
- Cell-cycle genes (often cytoplasmic) appear weaker in sn data → OPC
  cycle signal may look smaller in Multiome than in WMB sc data even for
  the same biological population.
- Total UMI counts and gene-detection rates shift between sc and sn.

**Implication for project:**
- If a cell FM is pretrained on **WMB cells** and evaluated on **Dev VIS
  Multiome nuclei**, expect a covariate shift that is not the biology
  being tested. Mitigate with batch tokens, per-modality normalization,
  or a paired sn benchmark (Consensus-WMB has snRNA from Broad/Macosko).
- Mixing sc and sn pretraining data without a batch-correction signal
  can wash out real cell-type structure.

**Why this matters:** "Did the FM fail on Dev VIS?" might just mean "the
FM never saw sn data." Worth controlling for before claiming a biological
result.

---

## Sparse vector encoding for ML/DL

How to turn a high-dimensional sparse vector (e.g. scRNA-seq counts,
$G \approx 20\text{k}$, ~90–95% zeros) into an input $z$ for a neural
network. This is the core design choice for Track B cell FM and for
baselines in Direction I.

### `sparse-encoding-overview` (confidence: 9/10)

**Problem:** $x \in \mathbb{R}^G$ with $\text{nnz}(x) \ll G$. Want
encoder $f(x) \to z \in \mathbb{R}^d$ (or token sequence) for MLP,
transformer, GNN, etc.

**Three trade-offs:**
1. **Compute** — dense $Wx$ is $O(Gd)$; sparse/embedding-bag is
   $O(\text{nnz} \cdot d)$.
2. **Inductive bias** — permutation invariance? feature order? values vs
   presence only? pairwise interactions?
3. **Information loss** — counts vs ranks vs bins; top-$L$ truncation vs
   full vocabulary.

**Five families (pick by which trade-off you need to win):**

| Family | Core operation | Typical cost |
|--------|----------------|--------------|
| **A. Dense / sparse linear** | $z = Wx + b$ | $O(Gd)$ or $O(\text{nnz} \cdot d)$ |
| **B. Embedding bag / set** | $z = \sum_i x_i e_i$ over nonzero $i$ | $O(\text{nnz} \cdot d)$ |
| **C. Tokenize → sequence model** | top-$L$ gene IDs → transformer | $O(L^2 d)$ attention |
| **D. Compress then dense** | PCA, SVD, VAE → $z \in \mathbb{R}^k$ | $O(Gk)$ or amortized |
| **E. Structured** | GNN on gene graph, FM, ZINB-VAE (scVI) | graph / likelihood dependent |

**Why this matters:** Cell FM papers hide this choice behind "rank
encoding" or "binning." Every experiment in this project should name
which family it uses and why.

### `sparse-linear-equals-weighted-bag` (confidence: 9/10)

**Sparse linear** and **weighted EmbeddingBag** are the same operation:

$$z = Wx = \sum_{i \in \text{nz}(x)} x_i \, W_{:,i}$$

- PyTorch dense: `nn.Linear(G, d)` — wastes work on zeros.
- PyTorch sparse: `nn.EmbeddingBag(G, d, mode='sum', sparse=True)` with
  `indices`, `offsets`, `per_sample_weights=values`.

**When to use:** Strongest **non-transformer** baseline for scRNA-seq.
Usually within 1–2% of a small MLP on cell-type tasks at a fraction of
FLOPs. Week 1 tiny MLP can be beaten or matched fairly with HVG (~2k
genes) + EmbeddingBag before investing in a transformer.

**Why this matters:** Choosing "embedding bag" vs "linear layer" is an
engineering choice, not a modeling one. Do not treat them as different
methods in ablations.

### `rank-vs-value-tokenization` (confidence: 9/10)

**Token-sequence encoders (Family C)** dominate cell FMs. Two main variants:

| Variant | How | Keeps magnitudes? | Examples |
|-------|-----|-------------------|----------|
| **Rank encoding** | Sort genes by expression, take top-$L$ **gene IDs** only | **No** (order only) | Geneformer |
| **Value binning** | Per gene: bin count into $K$ levels; token = gene + bin | **Binned** | scGPT |
| **Continuous value** | Gene embedding + MLP/bin embedding on $x_i$ | **Yes** | scGPT variant, scFoundation |

**Rank encoding:** input shape `(B, L)` integer gene IDs; embed to
`(B, L, d_model)`. Loses how highly each gene is expressed beyond rank.

**Implications for this project:**
- **Cell-type classification:** rank often sufficient (subclass is
  mostly "which genes are on," not exact counts).
- **Dynamics (OPC cycle, dev trajectories):** magnitudes matter — cycle
  is a continuous gradient in expression space. **Prefer binned or
  continuous-value tokens** for Direction I.2 and dev VIS work.
- **Neural collapse analysis (Week 9):** rank-only models may collapse
  differently than value-aware models; compare explicitly.

**Why this matters:** "Use Geneformer-style encoding" silently drops
information your dynamics hypotheses need.

### `sequence-length-L-from-nnz` (confidence: 8/10)

For token-sequence encoders, **$L$ = sequence length** (positions per
cell), **not** batch size, **not** $G$ (vocabulary size).

- Each cell → top-$L$ genes (by rank or detection), pad to fixed $L$
  for batching.
- Choose $L$ from the **95th percentile of detected genes per cell** in
  the training corpus, not from a published default (e.g. Geneformer
  2048).
- Truncation drops biology for high-UMI cells; padding wastes compute on
  low-UMI cells.

**Shapes after embedding:**
- Token IDs: `(B, L)` → `nn.Embedding` → `(B, L, d_model)`.
- Transformer expects `(B, L, d_model)` + attention mask for pad tokens.

**Why this matters:** Wrong $L$ is a silent hyperparameter bug — looks
like "model doesn't learn" when you are truncating half the transcriptome.

### `compress-then-dense-encoders` (confidence: 8/10)

**Family D:** project $x$ to low-dimensional dense $z$ before any DL head.

| Method | Notes | Project use |
|--------|-------|-------------|
| **HVG subset** | Keep top ~2k variable genes; then dense linear | Week 1 MLP baseline |
| **TruncatedSVD / PCA** | Linear; works on `scipy.sparse` | Classical baseline vs FM latent |
| **Random projection** | JL lemma; no training | Quick sanity check |
| **scVI / VAE (ZINB)** | Count-aware generative model; latent $z$ | OT on Dev VIS; dynamics baselines |
| **Diffusion map / PHATE** | Nonlinear manifold; not a neural encoder | Direction I.2 primary classical tool |

scVI latent is the standard **continuous** representation for optimal
transport and trajectory methods on count data.

**Why this matters:** Week 9 compares **cell FM latent vs PCA vs
diffusion map** on the same ABC subset — Family D is the fair classical
side of that comparison.

### `set-and-interaction-encoders` (confidence: 7/10)

**Family B extensions** when you need interactions without full
transformer cost:

- **Deep Sets:** $z = \rho(\sum_i \phi(i, x_i))$ — permutation invariant,
  universal approximator for sets (with enough width).
- **Set Transformer:** self-attention over the **set** of expressed genes
  — $O(L^2)$ but $L$ = nnz per cell, not $G$.
- **Factorization Machine (FM):** pairwise interactions
  $\langle e_i, e_j \rangle$ for features $i,j$ with nonzero product;
  DeepFM stacks FM + MLP.
- **GNN:** genes as nodes, edges from pathway / coexpression graph; $x$
  as node features — uses structure scRNA-seq flat vectors ignore.

**When to use:** Multimodal Dev VIS (RNA + ATAC different vocabs) may use
**per-modality EmbeddingBag + concat** or **peak–gene graph** before a
joint head. MERFISH spatial stretch would need neighbor-aware encoders
(Family E + spatial), not standard cell FM.

### `project-encoder-choices` (confidence: 9/10)

**Locked recommendations for this repo (revisit at Week 2 tokenization
gate):**

| Goal | Encoder choice |
|------|----------------|
| Week 1 tiny baseline | HVG (~2k) + dense `Linear` or **weighted EmbeddingBag** |
| Track B cell FM v1 | **Rank tokens** (Geneformer-style) for scaling-law ladder |
| Track B cell FM v2 | **Binned or continuous-value tokens** (scGPT-style) — compare v1 |
| Direction I.2 dynamics | **scVI / diffusion map / PCA** baselines + FM latent; prefer **value-aware** FM tokens |
| Direction I.1 OT + ATAC | **scVI latent** or continuous RNA vector + ATAC peak matrix; Multiome paired by barcode |
| Week 9 collapse vs dynamics | Same data, three encodings: supervised linear, rank-FM, value-FM |

**PyTorch primitives to implement first:**
```python
# Weighted bag (Family B = sparse linear)
nn.EmbeddingBag(num_embeddings=G, embedding_dim=d, mode='sum', sparse=True)

# Rank / binned tokens (Family C)
gene_emb = nn.Embedding(G, d)
value_emb = nn.Embedding(n_bins, d)  # optional

# Classical (Family D)
from sklearn.decomposition import TruncatedSVD  # on sparse .X
# scvi.model.SCVI(adata)  # count-aware
```

**Why this matters:** Single reference table prevents re-debating
tokenization every session and ties encoder choice to research question
(classification vs dynamics vs multimodal OT).
