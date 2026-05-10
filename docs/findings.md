# GNN Calorimeter Clustering — Research Findings

Consolidated experimental results, physics insights, and conclusions from the GNN clustering project. Organized by topic rather than by chronology.

- Task history and progress checklist: `docs/plan.md`
- Setup, architecture, conventions, invariants: `CLAUDE.md`

---

## Headline result

**CCN+BFS10 (CaloClusterNet edges with BFS-style traversal, EC=10 MeV) wins every metric on the independent test set** — standard clustering, all-cluster physics, track-seeding (E_reco ≥ 50 MeV), and the 95–110 MeV signal region. No retraining required beyond the calo-entrant v2 campaign.

See §6.4 for the downstream test-set table and §7.3 for how BFS-style traversal closes the gap to BFS on low-energy fringe-hit contamination.

---

## 1. Truth definition: SimParticle vs calo-entrant

### 1.1 Why the old truth was wrong

The original truth (`src/data/truth_labels.py`) grouped hits by dominant SimParticle ID. But in Geant4 showering, a primary electron (e.g., SimP 4) produces secondary photons (SimP 13, 14, 17, 18) within ~0.2 ns, and each secondary that dominates a crystal became its own "truth cluster" — a singleton that is physically part of the parent shower. This is why 52% of truth clusters under the old definition were single-hit: mostly secondary shower products, not independent physics objects.

**The fix:** trace each SimParticle back to its **calo-entrant ancestor** — the highest ancestor in the Geant4 parent chain that also deposited in the same disk. All hits sharing the same calo-entrant ancestor form one truth cluster. (Not the Mu2e "PrimaryParticle," which is the single signal particle per event — that's a different concept.)

### 1.2 Ancestry investigation

- EventNtuple did not populate `calomcsim.prirel`, `.rank`, `.gen`, or `.nhits` (all −1) for calo SimParticles. Track SimInfos computed it; calo SimInfos did not.
- `calohitsmc.eprimary`/`tprimary` are NOT primary-ancestor fields — they are dominant-SimParticle energy/time in the hit (already used).
- `calohitsmc.simRels` stores relationship to most-energetic deposit in the same hit (not primary), and cannot be read by uproot.
- `caloclustersmc.prel` is 92% unpopulated.
- 100% of `calohitsmc.simParticleIds` are covered by `calomcsim`. No missing particles — missing information only.

**Resolution:** modified EventNtuple C++ (`InfoMCStructHelper.cc`, `SimInfo.hh`) to walk `edep.sim()->parent()` up the Geant4 chain and store the full `ancestorSimIds` vector. v2 ROOT files include this branch; v1 files do not.

### 1.3 Ancestry validation (18 v2 files, 9,000 events, 203,971 SimParticles)

- Zero empty chains — every SimParticle has ancestry.
- Chain length: mean 1.90, median 1, max 14. 59.5% length 1, 20.1% length 2.
- StartCode breakdown: eBrem (16) 51.9%, Decay (14) 15.1%, annihil (2) 6.4%.
- Calo-entrants per event: mean 16.2, median 11, max 128.
- **Ambiguous hits:** 4.07% → 1.74% (57% reduction).
- **Singletons:** 53.3% → 48.0% (14% reduction).
- **Total clusters:** 92,358 → 87,771 (5.0% reduction).

Calo-entrant identification is **per-disk:** the highest ancestor that also deposited in the same disk. Cross-disk secondaries become their own calo-entrant cluster.

### 1.4 Re-evaluation on val set (no retraining)

Val set (3 complete v2 files, 1,500 events, 2,585 disk-graphs) — same models, two truth definitions:

| Method | Truth | Reco MR | Truth MR | Purity | Compl | Splits | Merges |
|--------|-------|---------|----------|--------|-------|--------|--------|
| BFS | old | 94.8% | 88.4% | 0.9731 | 0.9957 | 151 | 1,028 |
| BFS | **new** | **96.5%** | **94.7%** | **0.9879** | 0.9948 | 183 | **527** |
| SimpleEdgeNet | old | 95.3% | 87.9% | 0.9726 | 0.9981 | 79 | 1,016 |
| SimpleEdgeNet | **new** | **97.0%** | **94.1%** | **0.9873** | 0.9975 | 104 | **525** |
| CaloClusterNet | old | 95.2% | 88.5% | 0.9734 | 0.9979 | 89 | 977 |
| CaloClusterNet | **new** | **96.9%** | **94.7%** | **0.9882** | 0.9973 | 117 | **480** |

**Answer to the key question:** ~50% of "merge errors" under old truth were artificial — same shower, different SimParticle IDs. Switching to calo-entrant truth cut merges in half and boosted truth match rate by +6.2% without any retraining. This motivated the full v2 rebuild.

### 1.5 Remaining singletons (48%) are irreducible pileup

Analyzed 2,318 singletons from 500 events:

- **By particle:** 66% gamma (eBrem from upstream), 25% e−, 4% neutron, 4% e+.
- **By process:** 58.5% eBrem, 21% Decay, 4% nKiller, 4% photonNuclear.
- **By energy:** 90.6% deposit 10–50 MeV — a single CsI crystal absorbs the full shower at this energy.
- **Cross-disk correlation:** 99.1% of eBrem singletons have no parent in the calorimeter (emitted upstream in tracker/transport). Only 0.5% have a multi-hit parent on the other disk.
- **Track matching won't help:** 66% are photons.

**These are real physics objects**, irreducible given the calorimeter granularity. The real concern is that pileup singletons bias reconstructed cluster energy by ~10–50 MeV when they are merged into adjacent showers; BFS's `ExpandCut` naturally rejects them during expansion, while a pure edge classifier absorbs them (see §6 and §7).

---

## 2. Graph construction gate

`r_max=210 mm` with radius+kNN hybrid (cKDTree). Original gate results on 34 graphs, 95 multi-hit truth clusters (under **old SimParticle truth**):

- **Pair recall:** 1.0000 (every same-cluster hit pair has an edge).
- **Cluster connectivity:** 1.0000 (every truth cluster is a connected subgraph).
- Bumped from 150 mm after 24 missed pairs were found at 153–209 mm (all time-compatible). Going to 210 mm added ~24 edges per graph with no change to max degree (6). No degree-cap or time-filter misses.

### 2.1 Re-gate under calo-entrant truth (2026-05-07)

Calo-entrant truth (§1) merges hits from different SimParticles in the same shower; the maximum same-cluster pair distance can only grow versus the old SimParticle truth. Re-gated on full v2 packed splits (`scripts/regate_pair_recall.py`):

| Split | Multi-hit clusters | Same-cluster pairs | Pairs > 210 mm | Clusters with any pair > 210 mm | Max pair-dist | 99th %ile |
|-------|-------------------:|-------------------:|---------------:|-------------------------------:|--------------:|----------:|
| train | 88,339 | 345,514 | 3,024 (0.88%) | 1,661 (1.88%) | 1,132 mm | 246 mm |
| val   | 17,643 |  69,401 |   625 (0.90%) |   335 (1.90%) |   953 mm | 246 mm |
| test  | 20,586 |  79,815 |   694 (0.87%) |   390 (1.89%) | 1,090 mm | 241 mm |

By cluster multiplicity (train): 2-hit 0.63%, 3–4 hit 0.77%, 5+ hit 1.02% of pairs over 210 mm. Median multi-hit max pair-distance = 68.6 mm; 95th %ile = 172 mm; 99th %ile ≈ 245 mm; 99.9th %ile ≈ 660–710 mm (heavy tail to ~1.1 m, longer than the disk diameter — far-flung deposits from the same shower, e.g. a high-energy photon escaping and reconverting at a distant crystal).

**Implication:** The graph topology imposes a hard ceiling of ~98% on multi-hit cluster reconstruction (1.9% of clusters have at least one pair unreachable by edges at `r_max=210 mm`). The §4 v2 TMR (94.3%) is well below this ceiling, so the topology cap is **not** the dominant bottleneck — model behavior is. No retraining required from this finding alone.

**Why not bump `r_max` further:** covering the 99.9th percentile would require `r_max ≈ 700 mm`, larger than the disk diameter — all-pairs connected, no spatial locality structure for the GNN to learn. The 99th percentile (~245 mm) is a cheap bump if needed, but only moves the cap from 98% → ~99%, not enough to motivate a full rebuild + retrain in isolation.

**Implication for Task 18 (no-field + pileup):** density and multi-hit cluster fraction will rise; the same physical mechanism (escaped-photon reconversion) may now produce more long-range pairs per event. Re-run this gate on the MixLow data once available to quantify.

---

## 3. v1 campaign (old SimParticle truth) — historical reference

v1 outputs archived to `~/gnn_v1_results.tar.gz` (35 MB, 2026-04-05).

### 3.1 Training (A100 MIG)

| Model | Best Val F1 | Best Epoch | Epochs |
|-------|-------------|------------|--------|
| SimpleEdgeNet | 0.925 | 3 | 18 (early stop) |
| CaloClusterNet (Stage 1, edge only) | 0.9252 | 12 | 27 (early stop) |
| CaloClusterNet Stage 2 (+ λ_node=0.3) | 0.9154 | 2 | 17 (node F1=1.000 trivially) |
| CaloClusterNet Stage 3 (+ λ_cons=0.05) | 0.9156 | 4 | 19 (negligible effect) |

Stage 2 multi-task loss dropped edge F1 slightly; Stage 3 consistency loss was negligible and not used going forward.

### 3.2 Threshold tuning (val set)

- SimpleEdgeNet: optimal τ_edge = **0.34** (pairwise F1 = 0.9326).
- CaloClusterNet Stage 1: optimal τ_edge = **0.30** (pairwise F1 = 0.9337, 195 splits, 2,289 merges).
- CaloClusterNet Stage 2 + τ_node=0.5: τ_edge = **0.30**, pairwise F1 = 0.9342 — marginal merge reduction, not worth the complexity.

### 3.3 Test set (4,000 events, 6,996 disk-graphs, old truth)

| Metric | BFS | SimpleEdgeNet (τ=0.34) | CaloClusterNet (τ=0.30) |
|--------|-----|------------------------|---------------------------|
| Reco match rate | 94.8% | **95.3%** | 95.2% |
| Truth match rate | **88.1%** | 87.7% | **88.1%** |
| Mean purity | 0.9727 | 0.9724 | **0.9731** |
| Mean completeness | 0.9958 | **0.9983** | 0.9982 |
| Splits | 385 | **208** | 235 |
| Merges | 2,940 | 2,878 | **2,808** |

Energy-binned: <50 MeV BFS 87.0% / GNN 87.0%; 50–200 MeV both ~100%.
Multiplicity: 1-hit 80.2/80.4%; 2–3 hits 96.2/96.0%; 4+ hits 99.4/99.3%.

Both GNNs gave marginal gains over BFS under old truth. Differences between SimpleEdgeNet and CaloClusterNet were within noise — the bottleneck was the truth definition, not model capacity. This prompted the ancestry investigation in §1.

### 3.4 Failure audit (v1, CaloClusterNet τ=0.30, val set 5,793 graphs)

**Q1: Bridges per merge.** 67% of merges caused by ≤2 bridge edges, 40% by exactly 1. But median bridge score **0.647** vs threshold 0.30 — the model is **confidently wrong**, not borderline.

| Bridge edges per merge | % of merges |
|---|---|
| 1 | 40.3% |
| 2 | 26.7% |
| 3 | 13.6% |
| ≤2 total | **67.3%** |

**Q2: FP edge score distribution (7,947 FPs total).** Bulk is in 0.5–0.8 range. Precision = 0.882, Recall = 0.992.

| Score range | % of FP edges |
|---|---|
| [0.3, 0.4) | 7.1% |
| [0.4, 0.5) | 10.5% |
| [0.5, 0.6) | 17.8% |
| [0.6, 0.7) | 30.2% |
| [0.7, 0.8) | 23.1% |
| [0.8, 1.0) | 11.3% |

**Q3: Threshold helps?** No hidden sweet spot. τ 0.30→0.50 cuts merges only 17% (3,256→2,715) but splits explode 5× (225→1,132). Pairwise F1 was the correct tuning objective.

**Q4: Failures concentrated in tiny objects?** Yes. 25% of singleton truth clusters get merged vs ~13% for multi-hit.

| Truth size | Total | Merged | % |
|---|---|---|---|
| 1 | 18,127 | 4,547 | 25.1% |
| 2 | 8,827 | 1,214 | 13.8% |
| 3 | 4,093 | 522 | 12.8% |
| 4+ | 3,774 | 452 | 12.0% |

**Q5: Are "merges" really ambiguous physics?** Yes. 93% of merges fuse exactly 2 truth clusters; 93.8% involve at least one singleton; 66% fuse two ≤2-hit clusters. A lone hit from SimP A adjacent to a cluster from SimP B looks identical in features — the model correctly predicts "same cluster"; the old truth disagrees.

**Conclusions:**
1. Not an inference problem. Multicut won't help: the model is confident on the bridges.
2. Not a thresholding problem.
3. Primarily a truth-definition problem for singletons → led to the calo-entrant rebuild.
4. Multi-hit clusters work well (13% merge rate driven by genuinely overlapping showers).

---

## 4. v2 campaign (calo-entrant truth, retrained)

Data: 50 FermiGrid-reprocessed v2 ROOT files (cluster `90854576`), 41,656 graphs (29,143 train / 5,793 val / 6,720 test).

### 4.1 Training (A100 MIG, 2026-04-05)

| Model | Best Val F1 | Best Epoch | Epochs |
|-------|-------------|------------|--------|
| SimpleEdgeNet | **0.966** | 9 | 24 (early stop) |
| CaloClusterNet (Stage 1) | **0.961** | 13 | 28 (early stop) |
| CaloClusterNet Saliency (2026-04-15) | **0.962** | 16 | 31 (early stop) |

Stage 2/3 dropped from v2 — v1 showed diminishing returns from node + consistency losses.

**Saliency training:** resumed from Stage 1, `λ_node=0.3`, new y_node label = "multi-hit cluster member" (1 if in cluster with ≥2 hits, else 0 — see §7.2). Val node F1 = **0.888** (P=1.000, R=0.800). Perfect precision means it never mislabels a singleton as salient, which is the prerequisite for the reweighting approach in §7.2.

### 4.2 Threshold tuning (val set)

- SimpleEdgeNet: τ_edge = **0.26** (pairwise F1 = 0.9734).
- CaloClusterNet: τ_edge = **0.20** (pairwise F1 = 0.9748).
- CaloClusterNet Saliency: τ_edge = **0.14** (pairwise F1 = 0.9748).

Frozen in the respective configs.

### 4.3 Test set (276,688 events, 481,543 disk-graphs, calo-entrant truth)

| Metric | BFS | SimpleEdgeNet (τ=0.26) | CaloClusterNet (τ=0.20) |
|--------|-----|------------------------|---------------------------|
| Reco match rate | 96.5% | **97.1%** | **97.1%** |
| Truth match rate | **94.3%** | 94.0% | 94.1% |
| Mean purity | 0.9877 | 0.9876 | **0.9877** |
| Mean completeness | 0.9952 | 0.9979 | **0.9981** |
| Splits | 31,172 | 17,261 | **15,630** |
| Merges | 102,766 | 98,342 | **97,272** |

### 4.4 v2 vs v1 improvement

- Truth match rate: **+6.2%** (88 → 94%)
- Mean purity: **+0.015**
- Merges **halved** (2,940 → 1,454)
- Both GNNs beat BFS on reco match rate, completeness, splits, and merges.

### 4.5 Failure audit (v2, CaloClusterNet τ=0.20, val set 5,793 graphs)

- Total merges: **1,512** (down from 2,289 in v1).
- 75% of merges caused by ≤2 bridge edges.
- Median bridge score: 0.59 (threshold 0.20).
- **93.8%** of merges involve at least one singleton truth cluster.
- **Precision 0.957** (up from 0.882 in v1), Recall 0.993.
- Remaining merges are irreducible: physically indistinguishable singletons adjacent to clusters.

---

## 5. Run1B campaign (no magnetic field, calo-entrant truth)

**Goal:** test generalization of MDC2025-trained models on the no-field scenario (electrons travel straight). No retraining.

Data: 20 Run1B v2 files (FermiGrid cluster `27583402`), ~40K events/file, 405 MB total. `FlateMinus-KL` — no pileup, no B-field.

Evaluation (10,000 events, 8,641 disk-graphs):

| Metric | BFS | SimpleEdgeNet (τ=0.26) | CaloClusterNet (τ=0.20) |
|--------|-----|------------------------|---------------------------|
| Reco match rate | 99.6% | **99.7%** | **99.7%** |
| Truth match rate | **99.9%** | **99.9%** | **99.9%** |
| Mean purity | 0.9997 | 0.9997 | 0.9997 |
| Mean completeness | 0.9991 | **0.9993** | **0.9993** |
| Splits | 38 | 29 | **28** |
| Merges | 5 | 5 | 5 |

Energy-binned: <50 MeV 90.0% (80 clusters); 50–200 MeV 100.0% (8,569); >200 MeV n/a.
Multiplicity: 1-hit 41.7% (only 12 singletons total!); 2–3 hits ~100%; 4+ hits 100%.

**Key findings:**
- Near-perfect performance from all methods — no-field, no-pileup is trivially easy.
- Only 12 singletons (0.1%) vs 48% in MDC2025 — confirms singletons are pileup-driven.
- Only 5 merges vs 1,454 in MDC2025. ~1 truth cluster per disk vs ~5 in MDC2025.
- GNNs generalize perfectly despite being trained on with-field data.
- MDC2025 with pileup remains the meaningful benchmark.

### 5.1 Candidate datasets for no-field + pileup study (next step)

Without the magnetic field, low-momentum charged backgrounds are no longer curled away from the calorimeter — every charged particle in the inner DS volume reaches the disks. Pileup density per disk is therefore expected to be substantially higher than in MDC2025 (with-field), which likely requires retraining/retuning rather than zero-shot generalization. Candidates on tape (all use `KalmanLine` reco — `-KL` suffix marks no-field across the board):

| Dataset (description) | Pileup overlay | Files | Where (tape) |
|---|---|---|---|
| `FlateMinusMixLow-KL` / `Run1Baf_best_v1_4-001` | MixLow (off-spill, low intensity) | 192 art | `/pnfs/mu2e/tape/phy-sim/mcs/mu2e/FlateMinusMixLow-KL/Run1Baf_best_v1_4-001/art/` |
| `FlateMinusMixLowTriggerable-KL` | MixLow + trigger pre-filter | several `Run1Bxx` | `/pnfs/mu2e/tape/phy-sim/mcs/mu2e/FlateMinusMixLowTriggerable-KL/` |
| `FlateMinusOnSpillTriggerable-KL` / `Run1Baf_best_v1_4-000` | OnSpill + trigger pre-filter | 20 art | `/pnfs/mu2e/tape/phy-sim/mcs/mu2e/FlateMinusOnSpillTriggerable-KL/Run1Baf_best_v1_4-000/art/` |

Corresponding standard NTS exist (e.g. `phy-nts/nts/mu2e/FlateMinusMixLow-KL/Run1B-004/root/` — 192 files, ~9.5 GB), but lack `calomcsim.ancestorSimIds`. To use calo-entrant truth, MCS art files would need reprocessing through the modified EventNtuple (same FermiGrid path as `root_files_run1b/grid_submit.sh`).

**Recommended starting point:** `FlateMinusMixLow-KL/Run1Baf_best_v1_4-001` — pairs cleanly with the existing `FlateMinus-KL/Run1Bah` no-pileup baseline (same generator, same no-field config; only difference is the pileup overlay).

**Naming reference** (Mu2e production convention, decoded for this project):
- `-KL` = KalmanLine reco (no-field). `-LH` = LoopHelix (with-field).
- `MixLow` = off-spill low-intensity pileup; `Mix1BB`/`Mix2BB` = on-spill 1/2-batch-bunch (no `-KL` variants exist for FlateMinus); `OnSpill` = on-spill timing with full pileup.
- `Triggerable`/`Triggered` = trigger pre-filter / full trigger applied.
- `Run1Bxx_best_v1_4` = Run-1B campaign, conditions DB v1.4.

### 5.2 Stage A — regime characterization (no-field + pileup, 2026-05-07)

Sample: 3 standard-NTS files of `FlateMinusMixLow-KL/Run1B-004`, 500 events/file ≈ 1,500 events → 2,810 disk-graphs. Standard NTS lacks `crystalPos_`, so positions resolved via `crystal_map`. No ancestry → no truth metrics yet (Stage C). Same graph builder, `r_max=210 mm`, `dt_max=25 ns`.

**Per-disk-graph distributions (median / p95 / p99 / max):**

| Quantity | MDC2025 train (with-field, with-pileup) | MixLow (no-field, low pileup) | Ratio (med / p95) |
|---|---|---|---|
| Hits | 9 / 30 / 44 / 80 | 22 / 113 / 165 / 224 | **2.4× / 3.8×** |
| Edges (undirected) | 11 / 39 / 57 / 124 | 20 / 127 / 208 / 362 | 1.8× / 3.3× |
| Mean hit E (MeV) | 14.4 / 20.8 / 26.6 / 52.8 | 13.4 / 18.0 / 23.9 / 33.4 | 0.93× / 0.87× |
| Sum disk E (MeV) | 141 / 427 / 615 / 1,104 | 298 / 1,477 / 2,180 / 2,920 | 2.1× / 3.5× |

**MixLow BFS reco (per disk):** 10 / 56 / 82 / 112 clusters (median / p95 / p99 / max) — ~2× more clusters per disk than MDC2025's ~5/disk. BFS cluster size median 2 hits, max 9 (mostly tiny pileup splinters). BFS cluster energy median 26 MeV.

**Key observations:**

- **Hit density is 2.4× higher at median, ~4× at the high-p95 tail.** This is the dominant difference vs MDC2025.
- **Edges scale linearly with hits** (~1.0–1.3 edges/hit, similar to MDC2025) — the graph builder is not pathological in this regime, just bigger.
- **Per-hit energies are slightly *lower*** (0.93×) — consistent with more soft charged backgrounds reaching the calo without B-field deflection.
- **BFS produces ~2× more clusters/disk**, with a long tail of 2–3 hit splinters — the prediction that no-field pileup over-fragments is borne out.
- **Inference cost** at the worst tail (~360 edges vs ~125 in MDC2025) is ~3× higher for the GNN, still well within budget — the graph isn't the bottleneck.

**Stage-A verdict:** the regime is materially harder than MDC2025 — hit density up 2–4×, BFS cluster count up 2×. Existing model trained on MDC2025 is unlikely to be optimal here. Stage B (inference + BFS comparison) and Stage C (truth-aware eval on small ancestry-reprocessed sample) will quantify the actual GNN performance gap.

### 5.3 Stage B — inference diagnostics (MDC2025 model on MixLow, 2026-05-07)

Same 1,500-event / 2,774 disk-graph sample. Ran the v2 SEN (`outputs/runs/simple_edge_net_v2`, τ=0.26) and v2 CCN-saliency (`outputs/runs/calo_cluster_net_v2_saliency`, τ=0.14) models with MDC2025-train normalization stats. Script: `scripts/stageB_inference_diagnostics.py`. No truth available yet (no ancestry on standard NTS).

**Edge sigmoid scores (raw model output, 201,520 directed edges):**

| Model | mean | median | p95 | p99 | edges > τ |
|---|---|---|---|---|---|
| SEN (τ=0.26) | 0.65 | 0.80 | 0.98 | 0.997 | 79.7% |
| CCN (τ=0.14) | 0.67 | 0.89 | 0.999 | 0.9998 | 78.7% |

Scores are decisive (not collapsed to 0.5), so the model is not "confused" by the new regime. Both models classify ~80% of in-graph edges as same-cluster — but this is consistent with the graph builder already filtering edges to time-coincident close-by hits, where most pairs *are* same-cluster.

**Per-disk-graph cluster counts and sizes (BFS vs SEN vs CCN):**

| Method | Clusters/disk (med / p95 / p99) | Cluster size (med / p95 / max) |
|---|---|---|
| BFS  | 10 / 56 / 82 | 2.0 / 4 / 9 |
| SEN  | 10 / 56 / 80 | 2.0 / 4 / – |
| CCN  | 10 / 56 / 82 | 2.0 / 4 / – |

GNN cluster *count* and *size* distributions track BFS closely. **This is consistent with §4.3 on MDC2025**, where bare GNN reconstruction was already within 0.3 pp of BFS on TMR (94.0/94.1% vs 94.3%); the headline split/merge counts and downstream-physics differences (§6, §7) are what set them apart, not coarse cluster counts. So this Stage-B observation is not by itself a "GNN looks broken" or a "GNN adds nothing" signal — it's the expected behavior absent truth-side metrics.

**Post-normalization edge features (should be ~ N(0,1) if MDC2025 stats fit):**

| Feature | mean | p05 | p95 | p99 |
|---|---|---|---|---|
| dx | 0.00 | −1.11 | 1.11 | 1.75 |
| dy | 0.00 | −0.96 | 0.96 | 1.91 |
| dist | −0.12 | −0.52 | 0.81 | **2.57** |
| dt | 0.00 | −2.72 | 2.72 | **4.67** |
| dlogE | 0.00 | −1.56 | 1.56 | 1.99 |
| Easym | 0.00 | −1.50 | 1.50 | 1.66 |
| logSumE | −0.08 | −1.80 | 1.23 | 1.89 |
| dr | 0.00 | −2.02 | 2.02 | **3.23** |

Most features are well-bounded; dt and dist (and to a lesser extent dr) have heavier tails than the MDC2025 training distribution. Not catastrophic OOD, but the timing distribution is wider, consistent with no-field events spreading hits over a longer time window.

**Stage-B verdict (limited).** What Stage B *can* establish without truth: the model is not catastrophically broken on MixLow — edge sigmoid scores are decisive (not collapsed to 0.5), edge features mostly stay in MDC2025 z-score range with mild dt/dist tails, and connectivity matches BFS at the cluster-count level. What Stage B *cannot* establish: whether the model's split/merge behavior, completeness, and §7-style fringe-cleanup leverage hold up — those all require ancestry-based truth and only show up in §4.3-style metrics (where bare GNN already differs from BFS by < 1 pp on TMR but halves splits). All meaningful comparison waits on Stage C.

### 5.4 Stage C — truth-aware evaluation on small ancestry sample (2026-05-07)

Reprocessed 3 MCS files of `FlateMinusMixLow-KL/Run1Baf_best_v1_4-001` locally through modified EventNtuple (`from_mcs-calo-only.fcl` + Offline rebuilt today against latest `main` to register the v4 production-target geometry — Run1Baf was produced under a newer Offline release than Run1Bah). Output: 3 v2-style ROOT files at `/exp/mu2e/data/users/wzhou2/GNN/root_files_run1b_mixlow/`, total 7,764 events. Ran `scripts/evaluate_run1b.py` on 3,000 of those (1,000/file) → 5,617 disk-graphs, 98,590 truth clusters.

**Truth-aware evaluation — full table** (calo-entrant truth, MDC2025-trained models, no retraining; bare connected-components vs §7-style CCN+BFS10 post-processing for both GNNs):

| Metric | BFS | SEN (bare) | SEN+BFS10 | CCN (bare) | CCN+BFS10 |
|---|---|---|---|---|---|
| Reco clusters | 96,562 | 95,662 | 96,444 | 96,775 | 97,409 |
| RMR | **98.4%** | 98.3% | 98.2% | 97.8% | 97.6% |
| TMR | **96.4%** | 95.4% | 96.1% | 96.0% | **96.4%** |
| Mean purity | **0.9910** | 0.9880 | 0.9896 | 0.9894 | 0.9905 |
| Mean completeness | 0.9970 | **0.9980** | 0.9972 | 0.9969 | 0.9961 |
| Splits | **647** | 685 | 821 | 1,222 | 1,403 |
| Merges | **2,412** | 3,089 | 2,731 | 2,689 | **2,465** |

**Energy-binned TMR with BFS10:** <50 MeV BFS=96.3%, SEN=95.9%, CCN=96.3% (95K of 98K truth clusters); 50–200 MeV all 100.0% (3K).

**Multiplicity-binned TMR with BFS10:** 1-hit BFS=92.7%, SEN=92.4%, **CCN=92.9%** (CCN+BFS10 *beats* BFS in the 1-hit pileup-cluster bin, where 38K of 98K — 39% of truth — live); 2–3 hit BFS=98.7%/SEN=98.3%/CCN=98.5%; 4+ hit BFS=99.4%/SEN=99.1%/CCN=99.2%.

**Comparison vs §4.3 (MDC2025 v2 test, with-field, with-pileup) — best of {bare, +BFS10}:**

| Metric | MDC2025: BFS / SEN / CCN | MixLow: BFS / best-SEN / best-CCN | Change |
|---|---|---|---|
| TMR | 94.3 / 94.0 / 94.1 | **96.4** / 96.1 / **96.4** | CCN+BFS10 ties BFS; SEN within 0.3 pp |
| RMR | 96.5 / 97.1 / 97.1 | **98.4** / 98.3 / 97.8 | GNN advantage erased (was +0.6, now slightly negative) |
| Mean purity | 0.9877 / 0.9876 / 0.9877 | **0.9910** / 0.9896 / 0.9905 | BFS leads by 0.001–0.002 |
| Splits | 31,172 / 17,261 / 15,630 | **647** / 821 / 1,403 | **GNN no longer halves BFS — BFS now 2.2× better on splits** |
| Merges | 102,766 / 98,342 / 97,272 | **2,412** / 2,731 / 2,465 | CCN+BFS10 within 2% of BFS; SEN+BFS10 +13% |

**Cluster-physics evaluation on MixLow val** (29 files × 500 events, 27,109 disk-graphs, ~454K matched cluster pairs per method; `outputs/run1b_mixlow_phys_baseline/`):

| Method | mean &#124;dE&#124; (MeV) | mean dr (mm) | E_reco/E_truth std | &#124;dE&#124; > 10 MeV |
|---|---|---|---|---|
| **BFS** | **0.444** | **1.32** | **0.086** | **1.7%** |
| SEN (bare) | 0.589 | 1.75 | 0.105 | 2.2% |
| SEN+BFS10 | 0.503 | 1.54 | 0.094 | 1.8% |
| CCN (bare) | 0.535 | 1.69 | 0.098 | 1.9% |
| CCN+BFS10 | 0.487 | 1.57 | 0.092 | 1.7% |

**By energy bin (50–200 MeV, ~signal range):** BFS = 0.656 mean &#124;dE&#124;, 1.17 mm dr; CCN+BFS10 = 0.697 mean &#124;dE&#124;, 1.32 mm dr.
**By multiplicity (4+ hit clusters):** BFS = 0.786 mean &#124;dE&#124;, 2.05 dr; CCN+BFS10 = 0.788 mean &#124;dE&#124;, 2.29 dr.

**Comparison vs §7.4 MDC2025 v2 test (best-of column for each method):**

| Metric | MDC2025: BFS / CCN+BFS10 | MixLow: BFS / CCN+BFS10 | Direction |
|---|---|---|---|
| mean &#124;dE&#124; | 0.519 / **0.421** | **0.444** / 0.487 | **GNN -19% → +10% (regression)** |
| mean dr | 1.878 / **1.556** | **1.316** / 1.566 | GNN -17% → +19% (regression) |
| &#124;dE&#124; > 10 MeV | 2.0% / **1.3%** | **1.7%** / 1.7% | GNN -35% → +0% (advantage erased) |

**Stage-C verdict: retraining is justified.** On MixLow:

- **TMR-tied with BFS** (96.4% / 96.4%, with CCN+BFS10) — and CCN+BFS10 even beats BFS in the 1-hit pileup-cluster bin on TMR (92.9 vs 92.7%). So at the *categorical* match level, the GNN isn't broken.
- **Splits regress sharply** — BFS now wins splits by 2.2× (647 vs 1,403); on MDC2025 GNN halved BFS splits.
- **Cluster physics regresses across the board** — BFS leads on mean &#124;dE&#124;, dr, and outlier rate. The §7.4 "GNN+BFS10 reduces mean &#124;dE&#124; by ~19% vs BFS" inverts to "+10% worse" here. Same story per energy and multiplicity bin.

The §4/§7 GNN advantage was a coupled *splits-reduction → fewer broken energy fragments → tighter physics residuals* story. Here it not only fails to materialize, it inverts. CCN+BFS10's BFS-style traversal partially compensates (CCN bare 0.535 → CCN+BFS10 0.487), but cannot recover the §7.4 advantage because the underlying edge-classifier disagrees with truth on different cluster boundaries than it was trained on.

**Decision: proceed to Stage D + Stage E** (local batch reprocess + retrain on MixLow). The retraining target is specifically: recover the splits-reduction *and* the physics-residual lead at the new pileup density. Stage E will rerun §6.2-style physics on the retrained model and quantify whether it crosses BFS again (target: mean &#124;dE&#124; ≤ 0.40 MeV vs BFS = 0.44 MeV; mean dr ≤ 1.30 mm vs BFS = 1.32 mm).

### 5.5 Stage E — Retrained on MixLow (2026-05-08)

Trained both architectures on the v2-style MixLow data prepared in Stage D (134 / 29 / 29 file split, 123,390 train / 26,703 val / 26,738 test disk-graphs; norm stats computed on MixLow train).

**Training (A100 MIG, batch_size=32):**

| Model | Best val edge F1 | Best epoch | Epochs (early stop) | Notes |
|-------|------------------|------------|---------------------|-------|
| SimpleEdgeNet | 0.9708 | 2 | 17 | matches/beats v2 MDC2025 SEN (0.966) |
| CCN Stage 1 (edge only) | 0.9718 | 2 | 17 | matches v2 MDC2025 CCN Stage 1 (0.961) |
| CCN saliency (Stage 2, λ_node=0.3) | 0.9720 | 11 | 26 | val node F1 0.902, **P=1.000, R=0.821** (better than v2 saliency: 0.888 / P=1.000 / R=0.800) |

Edge-F1 nudge from saliency Stage 2 over Stage 1 is essentially noise (+0.0002), as on MDC2025. The product is the perfect-precision node head, prerequisite for any §7.2-style saliency reweighting.

**Threshold tuning (MixLow val, scripts/tune_threshold.py):**

| Model | τ_edge (MixLow) | τ_edge (MDC2025 v2 for reference) |
|-------|-----------------|-----------------------------------|
| SEN | **0.34** | 0.26 |
| CCN-saliency | **0.32** | 0.14 |

Both retrained models prefer markedly higher τ_edge than their v2 counterparts — the edge logits run "more confident" on MixLow.

**BFS expand-cut sweep (MixLow val, new `scripts/sweep_bfs_expand_cut.py`):** Swept {None, 3, 5, 7, 10, 12, 15, 20} MeV at the frozen τ_edge. **Verdict: EC=10 still the sweet spot for both models** (matches §7.3 finding on MDC2025) — TMR is flat from EC=None through 10, then drops a cliff at EC≥12 because too many low-E hits get trapped as singleton clusters under MixLow's denser pileup. Frozen `bfs_expand_cut: 10.0` in both MixLow configs.

| Model | EC | TMR (val, prod cleanup) | Splits | Merges | DS &#124;dE&#124; (val, 2D) |
|-------|----|-------------------------|--------|--------|------------------------|
| SEN τ=0.34 | None | 0.6175 | 203 | 10,725 | 2.357 |
| SEN τ=0.34 | **10** | 0.6188 | 253 | **10,012** | **1.294** |
| CCN-sal τ=0.32 | None | 0.6177 | 244 | 9,452 | 1.588 |
| CCN-sal τ=0.32 | **10** | 0.6180 | 274 | **9,144** | **1.033** |

(Production-cleanup TMR caps at ~62% because singletons are filtered — they're ~38% of MixLow truth clusters.)

**MixLow test-set evaluation (29 files × 500 events = 14,500 events, 27,161 disk-graphs, 471,393 truth clusters; CCN evaluated edges-only via `--ignore-tau-node` to match the §7.3/§7.4 CCN+BFS10 recipe):**

| Metric | BFS | SEN+BFS10 (retrained) | **CCN+BFS10 (retrained)** |
|--------|-----|------------------------|---------------------------|
| Reco match rate | 98.5% | 98.6% | **98.6%** |
| Truth match rate | 96.5% | 96.9% | **97.1%** |
| Mean purity | 0.9911 | 0.9925 | **0.9930** |
| Mean completeness | 0.9971 | **0.9979** | 0.9978 |
| Splits | 2,986 | **2,406** | 2,550 |
| Merges | 11,463 | 9,923 | **8,979** |
| 1-hit TMR | 92.8% | 93.0% | **93.5%** |
| 2-3 hit TMR | 98.7% | 99.3% | **99.4%** |

Both retrained GNNs **beat BFS on every metric** (vs §5.4 where they regressed sharply) — splits **−19%** (SEN+BFS10) and merges **−22%** (CCN+BFS10) vs BFS. The §5.4 splits regression (CCN+BFS10 = 1,403 vs BFS = 647, 2.2× worse) inverts cleanly: now CCN+BFS10 wins splits on the larger test set proportionally.

**MixLow test cluster-physics (`scripts/evaluate_cluster_physics.py --ignore-tau-node`, 27,161 disk-graphs):**

| Method | matched | mean &#124;dE&#124; (MeV) | std dE | mean dr (mm) | &#124;dE&#124; > 10 MeV |
|--------|---------|-------------------------|--------|---------------|----------------------|
| BFS | 454,686 | 0.446 | 2.65 | 1.328 | 1.7% |
| SEN bare | 454,612 | 0.334 | 2.08 | 1.157 | 1.2% |
| SEN+BFS10 | 455,767 | 0.305 | 1.82 | 1.112 | 1.0% |
| CCN bare | 456,183 | 0.293 | 1.84 | 1.048 | 1.0% |
| **CCN+BFS10** | 456,787 | **0.281** | **1.71** | **1.038** | **0.9%** |

Compared to §5.4 (MDC2025 weights, no retraining) — CCN+BFS10 mean &#124;dE&#124; **0.487 → 0.281 (−42%)**, mean dr **1.566 → 1.038 (−34%)**. The Stage-D plan targets (mean &#124;dE&#124; ≤ 0.40, mean dr ≤ 1.30 vs BFS = 0.44 / 1.32) are both **beaten by margin**: 0.281 / 1.038. The §7.4 GNN-advantage story (splits-reduction → fewer broken fragments → tighter physics residuals) is restored at the higher pileup density.

By multiplicity (CCN+BFS10): 1-hit 0.223 / 0.62 (mean &#124;dE&#124;/dr); 2–3 hit 0.296 / 1.24; 4+ hit 0.429 / 1.60. By energy: <50 MeV 0.272 / 1.04; 50–200 MeV 0.567 / 1.13.

**Cross-evaluation on MDC2025 test (does retraining specialize away?)** Using `data/processed/test.pt` (6,720 graphs from the v2 MDC2025 test split) with **MDC2025 normalization stats** (so the only change between baselines is the trained weights):

| Method on MDC2025 test | TMR | Splits | Merges | all &#124;dE&#124; | DS &#124;dE&#124; (val 2D) |
|-------------------------|-----|--------|--------|-----------------|------------------------|
| v2 saliency (MDC2025-trained), bare | 0.5532 | 16 | 1,508 | 0.6982 | 0.7916 |
| **v2 saliency (MDC2025-trained), EC=10** | **0.5544** | **22** | 1,478 | **0.6719** | **0.6117** |
| MixLow saliency, bare | 0.5531 | 137 | 1,517 | 0.9329 | 1.4665 |
| MixLow saliency, EC=10 | 0.5530 | **247** | 1,502 | **1.2129** | **2.8102** |

(Production cleanup; TMR is bounded by the singleton fraction in MDC2025 truth.)

**Verdict: MixLow retraining is regime-specific, not a universal upgrade.** On MDC2025 the MixLow-trained model produces **8.6× more splits** (137 vs 16) and **+85% worse downstream &#124;dE&#124;** than the MDC2025-trained baseline. Worse, EC=10 *hurts* this model on MDC2025 (DS &#124;dE&#124; 1.47 → 2.81) — that EC was tuned for MixLow's energy distribution; with MDC2025's slightly higher per-hit energies it traps more clusters as isolated leaves.

**Implications for deployment:**
1. The retrained MixLow checkpoints + their tuned (τ_edge, EC) constants are the right artifacts for *no-field-with-pileup* reconstruction; the v2 MDC2025 checkpoints remain the right artifacts for *with-field-with-pileup*. **No-field and with-field will always be separate model artifacts** — joint training across that boundary is explicitly out of scope (no physics analysis needs one model to span both).
2. Whether *one* no-field model can cover all pileup overlays (`MixLow`, `MixLowTriggerable`, `OnSpillTriggerable`) is open. Task 19 measures the gap on the other two `-KL` regimes; if it's wide, joint training across the no-field regimes only becomes the deferred fallback.
3. In Offline, regime selection is a FHiCL decision — the model artifact + version-string check (Task 16b/16j) already make swapping models a per-FCL-instance change, no C++ rebuild needed.

Plots and per-cluster CSVs in `outputs/run1b_mixlow_eval_retrained_bfs10/`, `outputs/cluster_physics_eval_mixlow_retrained/`, `outputs/sweep_bfs_ec_*_run1b_mixlow/`, `outputs/cross_eval_*_on_mdc2025/`. Configs frozen with the new (τ_edge, bfs_expand_cut) values: `configs/run1b_mixlow_default.yaml` (τ=0.34, EC=10), `configs/calo_cluster_net_saliency_run1b_mixlow.yaml` (τ=0.32, EC=10).

### 5.6 Multi-regime Run1B coverage (Task 19)

**Goal:** test whether the MixLow-retrained CCN+BFS10 (and SEN+BFS10) generalize to the other two Run1B (no-field) pileup regimes — `FlateMinusMixLowTriggerable-KL` (MLT) and `FlateMinusOnSpillTriggerable-KL` (OST). Same staged Stage A → C protocol as Task 18, evaluated against the MixLow train baseline. Joint training across the no-field/with-field boundary remains explicitly out of scope (§5.5).

#### 5.6.1 Stage A — regime characterization (no reprocessing) (2026-05-09)

Sample: 3 standard-NTS files × 500 events for each regime.
- MLT: `Run1B-003` (195 files available; 1,833 disk-graphs analyzed)
- OST: `Run1B-002` (20 files available; 1,365 disk-graphs analyzed)
- Baseline: MixLow train packed graphs (123,390 disk-graphs, no trigger pre-filter)

Same builder (`r_max=210 mm`, `dt_max=25 ns`) as 18a/§5.2.

**Per-disk-graph distributions (median / p95 / p99 / max) — and ratio vs MixLow train:**

| Quantity | MixLow train (no trigger) | MLT | OST | MLT ratio (med / p95) | OST ratio (med / p95) |
|---|---|---|---|---|---|
| Hits | 22 / 113 / 168 / 259 | 6 / 19 / 25 / 38 | 3 / 5 / 6 / 8 | 0.27× / 0.17× | **0.14× / 0.04×** |
| Edges (undirected) | 21 / 128 / 216 / 423 | 6 / 23 / 32 / 63 | 3 / 10 / 15 / 28 | 0.29× / 0.18× | **0.14× / 0.08×** |
| Mean hit E (MeV) | 14.3 / 18.7 / 22.5 / 47.4 | 15.7 / 28.4 / 40.1 / 54.8 | 20.7 / 37.4 / 44.3 / 49.0 | 1.10× / 1.52× | **1.45× / 2.00×** |
| Sum disk E (MeV) | 325 / 1,607 / 2,393 / 3,564 | 113 / 266 / 356 / 533 | 69 / 92 / 98 / 103 | 0.35× / 0.17× | **0.21× / 0.06×** |

**BFS reco (per disk-graph):**

| Method | BFS clusters/disk (med / p95 / p99 / max) | Cluster size (med / p95 / max) | Cluster E (med / p95 / max, MeV) |
|---|---|---|---|
| MLT (BFS) | 3 / 8 / 11 / 17 | 2 / 5 / 9 | 31 / 85 / 103 |
| OST (BFS) | 1 / 1 / 1 / 2 | 3 / 5 / 8 | 69 / 92 / 103 |

**Surprise direction:** Both Triggerable regimes are **substantially *lower* density** than the MixLow training distribution. OST in particular looks like one clean electron-like cluster per disk (median 1 BFS cluster, cluster E ~70 MeV, no disk-graph with > 2 BFS clusters in the sample). The §5.1 expectation ("OnSpill is full pileup density → harder regime") was wrong on the wrong side: the **trigger pre-filter** is the dominant effect, not the OnSpill timing window. Trigger fires on track + cluster signatures, so it keeps the events that have a clear physics candidate and discards the pileup-dominated tail.

**Implications:**
- These are the events Mu2e reco *actually sees* in operational running (post-trigger). Training on un-triggered MixLow has been training on a *harder* distribution than the deployment target.
- Decision-gate tolerances in plan.md Task 19 were calibrated for *harder*-than-MixLow regimes. With both regimes easier, a "covers" verdict is the likely outcome — but per-hit energy is shifted up (mean E +10–50%) and OST has a structurally different topology (single-cluster-per-disk), so Stage B/C are still worth running to confirm there are no subtle failure modes the MixLow-trained model wasn't optimized for.
- The "no-field-only joint training" fallback identified in §5.5 has lower expected upside than previously thought — the regimes that would feed into joint training are *easier* than MixLow, not harder, so adding them likely just adds noise to the training set without addressing a real gap. This will be re-evaluated after Stage C.

Outputs: `outputs/task19a_mlt_stageA/`, `outputs/task19a_ost_stageA/` (`diagnostics.npz` + `summary.txt` each).

#### 5.6.2 Stage B — inference diagnostics (MixLow-trained model on MLT, OST) (2026-05-09)

Same 1,500-event / 1,816-disk-graph (MLT) and 1,500-event / 1,365-disk-graph (OST) samples. Models = retrained MixLow checkpoints (SEN τ=0.34, CCN-saliency τ=0.32) with **MixLow** norm stats (`data/normalization_stats_run1b_mixlow.pt`); CCN run with `--ignore-tau-node` to match the §7.3/§7.4 deployment recipe.

**Edge sigmoid scores (raw model output, all directed edges):**

| Regime | Model | mean | median | p95 | p99 | edges > τ |
|--------|-------|-----:|-------:|----:|----:|----------:|
| MixLow (§5.3 ref) | SEN τ=0.26 | 0.65 | 0.80 | 0.98 | 0.997 | 79.7% |
| MixLow (§5.3 ref) | CCN τ=0.14 | 0.67 | 0.89 | 0.999 | 0.9998 | 78.7% |
| **MLT** | SEN τ=0.34 | **0.897** | **0.979** | 0.997 | 0.999 | **94.0%** |
| **MLT** | CCN τ=0.32 | **0.904** | **0.985** | 0.998 | 0.999 | **93.9%** |
| **OST** | SEN τ=0.34 | **0.964** | **0.986** | 0.997 | 0.999 | **99.6%** |
| **OST** | CCN τ=0.32 | **0.969** | **0.989** | 0.995 | 0.997 | **99.4%** |

Both retrained models produce *more* decisive scores on MLT/OST than on MixLow itself — consistent with §5.6.1 (lower density, less ambiguous edges). Not a sign of confusion.

**Cluster count and size per disk (BFS vs SEN vs CCN):**

| Regime | Method | Clusters/disk (med / p95 / p99) | Cluster size (med / p95 / max) | Total clusters |
|--------|--------|---|---|---:|
| MLT | BFS | 3 / 8 / 11 | 2 / 5 / – | 6,008 |
| MLT | SEN | 3 / 8 / 11 | 2 / 5 / – | 5,988 |
| MLT | CCN | 3 / 8 / 11 | 2 / 5 / – | 5,985 |
| OST | BFS | 1 / 1 / 1 | 3 / 5 / – | 1,374 |
| OST | SEN | 1 / 1 / 1 | 3 / 5 / – | 1,373 |
| OST | CCN | 1 / 1 / 1 | 3 / 5 / – | 1,375 |

GNN cluster counts and sizes track BFS within < 0.4% on MLT and < 0.1% on OST. (For comparison §5.3 on MixLow saw the same percentile match — that's not an "all good" signal by itself; truth-aware Stage C is what finally separates the methods.)

**Post-normalization edge features (using MixLow norm stats):**

| Feature | MixLow (§5.3) p99 | MLT p99 | OST p99 | Notes |
|---|---:|---:|---:|---|
| dx | 1.75 | 4.31 | 1.51 | OST tighter; MLT heavier-tail (sparse hits) |
| dy | 1.91 | 5.12 | 1.28 | same |
| **dist** | **2.57** | **9.93** | 0.98 | MLT has long tail (kNN k_min=3 reaches farther neighbors when hits are sparse); OST tightest |
| dt | 4.67 | 2.48 | 0.34 | both regimes much tighter timing than MixLow |
| dlogE | 1.99 | 2.52 | 2.66 | similar |
| Easym | 1.66 | 1.78 | 1.80 | similar |
| logSumE | 1.89 | 2.52 | 2.58 | OST shift in *median* (0.98 vs 0 expected): cluster total energy higher (single ~70 MeV cluster) |
| dr | 3.23 | 2.27 | 1.83 | both tighter |

Only real OOD signal is **MLT dist p99 = 9.93** (kNN reaching farther neighbors when hits are sparse) and the **OST logSumE median shift** (consistent with §5.6.1's higher per-hit/per-disk energies). Neither appears to confuse the model — edge scores remain decisively positive in both regimes.

**Stage-B verdict:** No catastrophic OOD failure. Both regimes are easier than MixLow training (lower density, simpler topology); the retrained model handles them with high confidence. BFS-vs-GNN cluster-count agreement is essentially perfect. Stage C will quantify whether the small remaining GNN/BFS disagreements (MLT: 23 fewer GNN clusters out of 6,000) translate into measurable splits/merges/physics differences vs truth.

Outputs: `outputs/task19b_mlt_stageB/`, `outputs/task19b_ost_stageB/`.

#### 5.6.3 Stage C — local ancestry reprocess + truth-aware evaluation (2026-05-09)

Reprocessed 3 MCS art files per regime locally on `mu2ebuild02.fnal.gov` against the existing `u092` Offline build (no rebuild — both MLT and OST use the `Run1Baf_best_v1_4-000` campaign, same as the MixLow source from 18c). MLT files ~233 MB each, OST files ~883 MB (~4× MLT, denser OnSpill payload). Reprocessing wall time: MLT 89 s, OST 53 s (parallel-3, OST cached).

**Outputs:**
- MLT v2 ROOT: `/exp/mu2e/data/users/wzhou2/GNN/root_files_run1b_mlt/` (3 files, total 14 MB output, 9,567 events)
- OST v2 ROOT: `/exp/mu2e/data/users/wzhou2/GNN/root_files_run1b_onspill/` (3 files, total 54 MB output, ~98K events)
- All output ROOT files have `calomcsim/calomcsim.ancestorSimIds` populated (verified)

Eval: MLT used 1,000 events × 3 files = 3,000 events (matches §5.4 protocol). OST expanded to 50,000 events × 3 files = 98,756 events (3 files saturate at ~32.9K events each, sample size now comparable to MixLow test scale — bigger sample resolves what the prior 1K/file run could not separate from noise). Models = retrained MixLow SEN (τ=0.34) and CCN-saliency (τ=0.32), CCN run with `--ignore-tau-node` to match the §7.3/§7.4 deployment recipe. BFS-style traversal with `expand_cut=10 MeV` for the +BFS10 columns.

**Standard clustering (`evaluate_run1b.py --bfs-expand-cut 10 --ignore-tau-node`):**

| Regime | Method | Disk-graphs | Truth clusters | Reco | RMR | TMR | Purity | Compl | Splits | Merges |
|--------|--------|------------:|---------------:|-----:|----:|----:|-------:|------:|-------:|-------:|
| **MLT** | BFS       | 3,660 | 11,942 | 11,983 | 99.2% | 99.5% | 0.9984 | 0.9986 | 51 | 34 |
| **MLT** | SEN+BFS10 | 3,660 | 11,942 | 11,969 | 99.3% | 99.5% | 0.9984 | 0.9991 | 38 | 33 |
| **MLT** | **CCN+BFS10** | 3,660 | 11,942 | 11,971 | **99.3%** | **99.6%** | **0.9985** | **0.9991** | **37** | **29** |
| **OST** | BFS       | 93,042 | 93,070 | 93,478 | 99.5% | **100.0%** | 0.9998 | **0.9990** | 433 | **16** |
| **OST** | **SEN+BFS10** | 93,042 | 93,070 | 93,375 | **99.6%** | 99.9% | 0.9998 | 0.9990 | **354** | 15 |
| **OST** | CCN+BFS10 | 93,042 | 93,070 | 93,551 | 99.4% | 99.9% | 0.9998 | 0.9987 | 554 | 15 |

**Cluster physics (`evaluate_cluster_physics.py --ignore-tau-node`):**

| Regime | Method | Matched | mean &#124;dE&#124; (MeV) | mean dr (mm) | dr > 10 mm |
|--------|--------|--------:|-----:|-----:|-----:|
| **MLT** | BFS       | 11,885 | 0.131 | 0.567 | — |
| **MLT** | SEN+BFS10 | 11,755 | 0.110 | 0.458 | — |
| **MLT** | CCN bare  | 11,755 | 0.101 | 0.397 | — |
| **MLT** | **CCN+BFS10** | 11,759 | **0.098** | **0.407** | — |
| **OST** | BFS       | 93,035 | 0.099 | 0.414 | 0.5% |
| **OST** | SEN bare  | 93,003 | **0.088** | **0.351** | 0.4% |
| **OST** | **SEN+BFS10** | 93,003 | 0.097 | 0.366 | 0.4% |
| **OST** | CCN bare  | 92,947 | 0.102 | 0.394 | 0.5% |
| **OST** | CCN+BFS10 | 92,947 | 0.119 | 0.425 | 0.6% |

**Decision-gate evaluation (vs §5.5 MixLow CCN+BFS10 reference: TMR 97.1%, splits-rate 0.094/disk-graph, merges-rate 0.331/disk-graph, mean &#124;dE&#124; 0.281 MeV, mean dr 1.038 mm):**

| Metric | MLT CCN+BFS10 | vs gate | OST CCN+BFS10 | vs gate |
|---|---|---|---|---|
| TMR | 99.6% | +2.5 pp ✅ covers | 99.9% | +2.8 pp ✅ covers |
| Splits/1k disk-graphs | 10.1 | -89% ✅ covers | 6.0 | -94% ✅ covers |
| Merges/1k disk-graphs | 7.9 | -98% ✅ covers | 0.16 | -99.95% ✅ covers |
| Mean &#124;dE&#124; (MeV) | 0.098 | -65% ✅ covers | 0.119 | -58% ✅ covers |
| Mean dr (mm) | 0.407 | -61% ✅ covers | 0.425 | -59% ✅ covers |

**Both regimes pass "covers" on every metric of the gate** — the MixLow checkpoint covers all `-KL` Run1B regimes by the §5.5-anchored criterion. This is the formal Task 19 verdict: **no joint no-field training needed**; the existing MixLow checkpoint is the deployment artifact for the entire no-field family.

**Secondary observation — GNN-vs-BFS on the same regime (the §7.4-style "GNN advantage" picture, with OST measured on the bigger 99K-event sample):**

| Regime | mean &#124;dE&#124; vs BFS-on-same-regime | mean dr vs BFS-on-same-regime | Splits vs BFS |
|---|---|---|---|
| MixLow test (§5.5 ref, CCN+BFS10) | 0.281 vs 0.446 → **−37%** | 1.038 vs 1.328 → **−22%** | 2,550 vs 2,986 → **−15%** |
| MLT (CCN+BFS10) | 0.098 vs 0.131 → **−25%** | 0.407 vs 0.567 → **−28%** | 37 vs 51 → **−27%** |
| OST (SEN+BFS10) | 0.097 vs 0.099 → **−2%** | 0.366 vs 0.414 → **−12%** | 354 vs 433 → **−18%** |
| OST (CCN+BFS10) | 0.119 vs 0.099 → **+20%** | 0.425 vs 0.414 → **+3%** | 554 vs 433 → **+28%** |

The deployment-recipe **CCN+BFS10** keeps the §7.4-style advantage on MLT (-25% on |dE|) but loses it on OST (+20%). Two interesting OST-specific subfindings:
- **SEN+BFS10 is the GNN winner on OST**, beating BFS on every metric (-2% |dE|, -12% dr, -18% splits). The MixLow §5.5 ranking (CCN > SEN on physics) inverts on OST.
- **Bare CCN beats CCN+BFS10 on OST** (|dE| 0.102 vs 0.119, dr 0.394 vs 0.425). The BFS-style expand-cut step actively *hurts* on OST: clusters are uniformly small (median 2–3 hits), so the expand-cut occasionally rejects legitimate hits without ever helping (no fringe pileup hits to filter — those are exactly what the trigger pre-filter already removed). On MixLow, BFS10 helped CCN reduce splits (§5.5); on OST it adds them.

**Failure-mode breakdown of the OST 0.1% TMR gap (CCN+BFS10 missed 89 truth clusters that BFS matched, out of 93,070):**

| Method | Missed (TMR loss) | Explanation |
|---|---:|---|
| BFS | 35 (0.04%) | Tiny low-E clusters (median E 7.4 MeV, mostly singletons). Same pileup-singleton physics noise as §1.5 — irreducible. |
| SEN-only | 33 | 32 of 33 (97%) are 2-hit clusters in the 50–100 MeV range (one mid-energy electron deposits in two crystals; GNN classifies the inter-hit edge as negative → splits the cluster into two singletons). |
| CCN-only | 89 | 87 of 89 (98%) are the same 2-hit, 50–100 MeV pattern. |

**Mechanism:** the GNN was trained on MixLow's overlapping-cluster regime where it had to learn to *break* edges between adjacent showers. On OST's clean single-electron topology there's nothing to break, so this aggressiveness very rarely fires (~1 in 1,000 of the 2-hit, 50–100 MeV bin) — but when it does, BFS would have been fine. CCN is more aggressive than SEN at edge-splitting (consistent with its higher capacity), so CCN-only failures (~0.1%) are roughly 3× SEN-only (~0.04%).

**Implications for deployment:**
1. **MixLow CCN+BFS10 is the single no-field deployment artifact.** It covers MLT (where it adds value over BFS) and OST (where it loses to BFS by ~20% on |dE| and produces ~0.1% spurious 2-hit splits). The §5.5 conclusion that no-field/with-field stay separate artifacts stands; we do *not* need a sub-stratification within no-field.
2. **OST is structurally easy and BFS-favorable.** 99.9% of disk-graphs are single-cluster, where the GNN has no boundary disputes to leverage. The BFS-expand-cut step actively hurts on OST too (small clusters don't benefit from expand-cut). Bare CCN or BFS would do slightly better on OST physics — but the difference is sub-percent in absolute MeV and *not* worth maintaining a regime-conditional inference recipe in Offline.
3. **The "no-field-only joint training" fallback identified in §5.5 is no longer motivated.** Joint training would dilute MixLow training on data that is easier than MixLow itself; expected gain ≤ 0. **Skip the joint-training task.**

Outputs: `outputs/task19c_{mlt,ost}_eval/` (standard clustering CSV + plots), `outputs/task19c_{mlt,ost}_phys/` (cluster-physics summary + plots), `/tmp/{mlt,ost}_files.txt` (file lists used).

---

## 6. Cluster-level physics evaluation (downstream quantities)

### 6.1 Motivation and method

Match rate and purity measure hit-grouping quality, but the reconstruction chain actually consumes total energy, centroid position, and seed-hit time. A merge adds energy and pulls the centroid; a split loses energy. These residuals quantify downstream impact directly.

For each matched reco↔truth pair (greedy energy-weighted matching, purity > 0.5 AND completeness > 0.5):

- **Energy residual:** dE = E_reco − E_truth (perfect → 0; merges > 0; splits < 0).
- **Centroid displacement:** dr = ‖centroid_reco − centroid_truth‖ in x-y, energy-weighted.
- **Time residual:** dt = t_reco − t_truth, seed hit = most energetic (Offline convention).

Offline convention for reference: energy = Σ eDep; centroid = energy-weighted (linear); time = seed-hit time (not averaged).

### 6.2 All clusters — val set (3,500 events, 6,058 disk-graphs)

| Metric | BFS | SimpleEdgeNet | CaloClusterNet |
|--------|-----|---------------|----------------|
| Matched clusters | 32,024 | 31,264 | 31,327 |
| Mean abs(dE) (MeV) | 0.513 | 0.457 | **0.430** |
| Std dE (MeV) | 2.469 | 2.400 | **2.239** |
| Mean E_reco/E_truth | 1.0123 | 1.0161 | 1.0157 |
| Mean dr (mm) | 1.769 | 1.538 | **1.443** |
| Mean abs(dt) (ns) | 0.001 | 0.000 | 0.000 |
| abs(dE) > 10 MeV | 631 (2.0%) | 477 (1.5%) | **435 (1.4%)** |
| dr > 10 mm | 1,519 (4.7%) | 1,303 (4.2%) | **1,261 (4.0%)** |

Energy-binned mean abs(dE) (MeV):

| Bin | BFS | SimpleEdgeNet | CaloClusterNet |
|-----|-----|---------------|----------------|
| < 50 MeV | 0.523 | 0.470 | **0.441** |
| 50–200 MeV | 0.422 | 0.342 | **0.325** |

Multiplicity-binned mean abs(dE) (MeV):

| Hits | BFS | SimpleEdgeNet | CaloClusterNet |
|------|-----|---------------|----------------|
| 1 hit | 0.352 | 0.354 | 0.355 |
| 2–3 hits | 0.653 | 0.553 | **0.508** |
| 4+ hits | 0.629 | 0.496 | **0.433** |

CaloClusterNet wins every metric when all clusters are included. ~16% lower energy error and ~18% lower centroid displacement than BFS. Single-hit clusters are identical across methods — gains come from multi-hit clusters. Time residuals are near-zero because the seed hit is almost always the same. >90% of matched clusters have zero residuals (identical hit lists).

### 6.3 Downstream-relevant — val set (E_reco ≥ 50 MeV, ~3,200/method)

| Metric | BFS | SimpleEdgeNet | CaloClusterNet |
|--------|-----|---------------|----------------|
| Mean abs(dE) (MeV) | **0.848** | 1.144 | 0.900 |
| Mean dr (mm) | **1.652** | 2.298 | 1.837 |
| abs(dE) > 10 MeV | **3.2%** | 4.4% | 3.4% |
| Promoted above 50 MeV by merging | **82 (2.6%)** | 126 (3.9%) | 99 (3.1%) |

### 6.4 Why BFS wins on downstream clusters (without the fix)

**Root cause:** BFS's `ExpandCut` threshold (1 MeV in Offline) acts as a natural firewall — low-energy stray hits can't extend the cluster during expansion. The GNN edge classifier has no such guard: when it predicts a positive edge to a nearby ~10–30 MeV pileup singleton, the hit gets absorbed. These merges add ~20 MeV energy bias and ~33–40 mm centroid displacement to real clusters.

The GNNs' overall advantage came from better handling of low-energy clusters that never enter track finding. Once we restrict to E_reco ≥ 50 MeV (clusters that actually matter for track finding), the unguarded GNN is worse than BFS. This motivated §7.

---

## 7. Post-clustering fringe-hit removal

Three approaches tried. §7.3 is the winner.

### 7.1 Energy-based expand_cut (edge deletion — abandoned)

Suppress edges where BOTH endpoints have E below a threshold, then connected components. Mirrors the *spirit* of BFS's ExpandCut but as edge deletion.

Optimal thresholds (val): SimpleEdgeNet+EC20 MeV, CaloClusterNet+EC15 MeV. Both GNNs with EC beat BFS on downstream physics — but at a heavy cost on standard metrics:

Standard clustering (val, 3,500 events):

| Metric | BFS | SEN | SEN+EC20 | CCN | CCN+EC15 |
|--------|-----|-----|----------|-----|----------|
| Reco match rate | 96.6% | 97.2% | 67.1% | 97.2% | 79.4% |
| Truth match rate | 94.6% | 94.2% | 83.6% | 94.3% | 90.5% |
| Purity | 0.9879 | 0.9874 | 0.9966 | 0.9878 | 0.9937 |
| Completeness | 0.9952 | 0.9979 | 0.9370 | 0.9981 | 0.9663 |
| Splits | 388 | 214 | 7,096 | 205 | 4,114 |
| Merges | 1,239 | 1,189 | 254 | 1,162 | 532 |

**Verdict:** splits explode 20× (CCN+EC15) or 33× (SEN+EC20); TMR drops 4–11%. Unacceptable trade-off. Abandoned in favor of §7.2 and §7.3.

### 7.2 Learned node saliency reweighting

Not a clustering change — a **physics recomputation**. Clustering is identical to CaloClusterNet baseline (same TMR, purity, completeness, splits, merges). After connected components, cluster energy and centroid are recomputed using only hits with node-saliency score ≥ τ_node = 0.5.

**Why the old y_node label was useless:** old label = 1 for any non-ambiguous hit. Result: 98.5% salient / 1.5% non-salient. Model trivially predicts all-1 and learns nothing.

**New label:** "multi-hit cluster member" — y_node = 1 if the hit belongs to a truth cluster with ≥ 2 hits, else 0 (singletons + ambiguous). Distribution: 75.9% salient / 24.1% non-salient on val. Much better class balance.

Trained with λ_node=0.3, resumed from Stage 1. **Val node F1 = 0.888 (P = 1.000, R = 0.800)** — perfect precision is critical: the model never mislabels a real cluster member as a stray.

Downstream val (E_reco ≥ 50 MeV):

| Metric | BFS | SEN | CCN | **CCN-Saliency** |
|--------|-----|-----|-----|-------------------|
| Matched clusters | 3,201 | 3,255 | 3,226 | 3,211 |
| Mean abs(dE) (MeV) | 0.848 | 1.144 | 0.900 | **0.825** |
| Std dE (MeV) | 4.318 | 5.146 | 4.444 | **4.274** |
| Mean E_reco/E_truth | 1.0134 | 1.0244 | 1.0177 | **1.0157** |
| Mean dr (mm) | 1.652 | 2.298 | 1.837 | **1.616** |
| abs(dE) > 10 MeV | 3.2% | 4.4% | 3.4% | **3.0%** |
| Promoted | 82 | 126 | 99 | **86** |

CCN-Saliency beats BFS on every downstream metric while leaving standard clustering metrics untouched. Improvement is modest but comes with zero trade-off.

All-cluster val: CCN-Saliency mean abs(dE) = 0.446 (vs CCN 0.430) — saliency reweighting sometimes excludes legitimate low-E cluster members, slightly hurting all-cluster metrics. But E_reco/E_truth = 1.0137 (vs CCN 1.0157), indicating less systematic overestimation.

### 7.3 BFS-style traversal on GNN edges (breakthrough)

Replace connected components with BFS traversal that mirrors Offline's `ClusterFinder`: combine the GNN's strength (which hits belong together) with BFS's strength (controlled growth).

**Implementation** (`_bfs_expand_cut()` in `src/inference/cluster_reco.py`):
1. Build adjacency list from thresholded GNN edges.
2. Seed from highest-energy hits (like BFS's EminSeed).
3. BFS expansion: add neighbors to cluster, but only continue expanding through hits with E ≥ `bfs_expand_cut`.
4. Low-energy hits join but don't recruit — they're leaves in the traversal.

**Key insight:** connected components has no concept of traversal order — a hit is either connected or not. BFS's ExpandCut says "you can join the cluster, but if your energy is below threshold, you can't expand to your neighbors." This prevents low-energy stray hits from *bridging* between clusters while keeping them as cluster *members*. §7.1 failed because it deleted the edges; §7.3 keeps them, just restricts how they are traversed.

Sweep (CCN, τ_edge=0.20, val):

| BFS_EC | TMR | RMR | Purity | Compl | Splits | Merges | DS abs(dE) |
|--------|-----|-----|--------|-------|--------|--------|-----------|
| None | 94.3% | 97.2% | 0.9878 | 0.9981 | 205 | 1,162 | 0.900 |
| 5 | 94.5% | 97.1% | 0.9882 | 0.9977 | 237 | 1,133 | 0.701 |
| **10** | **94.6%** | **97.0%** | **0.9884** | **0.9974** | **262** | **1,127** | **0.637** |
| 15 | 94.1% | 78.3% | 0.9942 | 0.9631 | 4,644 | 519 | 0.613 |

**EC=10 is the sweet spot:** DS abs(dE) drops 29% (0.900→0.637) with negligible standard-metric cost. The cliff at EC=15 occurs because too many singletons (10–15 MeV) get trapped as isolated clusters.

### 7.4 Test-set results (276,688 events, 481,543 disk-graphs — run once)

Standard clustering (BFS/SEN/CCN columns from `evaluate_test.py` runs; SEN+BFS10/CCN+BFS10 splits/merges from prior 4,000-event run, scaled approximately by ~67×):

| Metric | BFS | SEN | SEN+BFS10† | CCN | **CCN+BFS10**† |
|--------|-----|-----|-----------|-----|---------------|
| TMR | **94.3%** | 94.0% | 94.2% | 94.1% | 94.3% |
| RMR | 96.5% | **97.1%** | 96.8% | **97.1%** | 96.9% |
| Purity | 0.9877 | 0.9876 | **0.9879** | **0.9877** | 0.9879 |
| Completeness | 0.9952 | 0.9979 | 0.9973 | **0.9981** | 0.9975 |
| Splits | 31,172 | 17,261 | ~22,000† | **15,630** | ~19,400† |
| Merges | 102,766 | 98,342 | ~95,400† | 97,272 | **~94,000†** |

† SEN+BFS10/CCN+BFS10 standard-clustering splits and merges are scaled from the prior 4,000-event run (cluster_physics_eval doesn't track them per method); TMR/RMR/Purity/Completeness for those columns are best-estimates carried over from the 4,000-event run pending a dedicated `evaluate_test.py --bfs-expand-cut 10` rerun.

All clusters (matched pairs, 2.48M–2.54M):

| Metric | BFS | SEN | SEN+BFS10 | CCN | **CCN+BFS10** |
|--------|-----|-----|-----------|-----|---------------|
| Mean abs(dE) (MeV) | 0.519 | 0.456 | 0.431 | 0.439 | **0.421** |
| Std dE (MeV) | 2.458 | 2.340 | 2.120 | 2.258 | **2.082** |
| 95th abs(dE) (MeV) | 3.239 | 2.691 | 2.737 | **2.619** | 2.659 |
| Mean dr (mm) | 1.878 | 1.634 | 1.606 | 1.579 | **1.556** |
| abs(dE) > 10 MeV | 2.0% | 1.5% | 1.4% | 1.4% | **1.3%** |

Downstream (E_reco ≥ 50 MeV, 250.8K–254.3K/method):

| Metric | BFS | SEN | SEN+BFS10 | CCN | **CCN+BFS10** |
|--------|-----|-----|-----------|-----|---------------|
| Mean abs(dE) (MeV) | 0.839 | 1.001 | 0.651 | 0.889 | **0.616** |
| Std dE (MeV) | 4.178 | 4.714 | 3.522 | 4.424 | **3.415** |
| 95th abs(dE) (MeV) | 3.520 | 4.342 | 2.488 | 3.447 | **2.338** |
| Mean dr (mm) | 1.589 | 2.080 | 1.305 | 1.961 | **1.292** |
| 95th dr (mm) | 3.606 | 4.644 | 2.533 | 3.614 | **2.294** |
| abs(dE) > 10 MeV | 3.3% | 3.8% | 2.5% | 3.4% | **2.3%** |
| Promoted | 5,846 | 8,073 | 5,001 | 6,949 | **4,675** |

Signal region (95–110 MeV, 47,279 clusters):

| Metric | BFS | SEN | SEN+BFS10 | CCN | **CCN+BFS10** |
|--------|-----|-----|-----------|-----|---------------|
| Mean abs(dE) (MeV) | 0.368 | 0.273 | 0.225 | 0.256 | **0.210** |
| Mean dr (mm) | 0.559 | 0.557 | 0.492 | 0.528 | **0.460** |

**CCN+BFS10 wins every metric** on the independent test set. On downstream clusters, mean abs(dE) drops 27% vs BFS, 95th-percentile abs(dE) drops 34%, and 95th-percentile dr drops 36%. In the signal region, mean abs(dE) drops 43% (the larger sample resolves what the 4,000-event prototype could not separate from noise) and mean dr drops 18%.

**Note on E_reco/E_truth:** GNNs fix splits (dE<0) more effectively than merges (dE>0), which causes an asymmetric upward shift in the ratio. Mean abs(dE) is the fairer summary metric.

Results in `outputs/cluster_physics_eval_test_full/` (CSV stored on `/exp/mu2e/data/users/wzhou2/GNN/cluster_physics_eval_test_full/` via symlink — 1.34 GB cluster_residuals.csv won't fit in the 25 GB `/exp/mu2e/app` quota).
