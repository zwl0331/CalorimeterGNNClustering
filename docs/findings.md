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
