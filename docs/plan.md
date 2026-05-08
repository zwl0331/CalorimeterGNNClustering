# Plan: GNN Calorimeter Clustering for Mu2e

> **Note:** Experimental results, result tables, physics insights, and conclusions live in `docs/findings.md`. This file is the task history and progress checklist — it describes *what was done*, not the numerical outcomes. Section references point to `findings.md`.

## Context

Sam is a Caltech grad student (advisor: David Hitlin, co-supervisor: Sophie Middleton) building a GNN-based clustering algorithm for the Mu2e electromagnetic calorimeter. The calorimeter has two crystal-disk geometry layers. The existing algorithm (`CaloClusterMaker` in Offline) uses a seed-based BFS approach with hard energy/time thresholds. The goal is a GNN prototype that can eventually match or outperform the existing algorithm.

Sophie has an existing working prototype (training data in EventNtuple format). Sam will start fresh and benchmark against Sophie's results after their planned meeting.

**Dataset (from Sophie Middleton):**
- **Mixed v2** (with pileup + MC truth):
  `/pnfs/mu2e/tape/phy-nts/nts/mu2e/FlateMinusMix1BBTriggered/MDC2025-002/root/`
  ~50 ROOT files, copied locally to `/exp/mu2e/data/users/wzhou2/GNN/root_files/` (97 GB)

EventNtuple format (`EventNtuple/ntuple` TTree). MC truth via `calohitsmc` branches.

**Engineering choices made:**
- Task: **Edge classification** — predict for each pair of nearby hits whether they belong to the same true cluster
- Framework: **PyTorch Geometric (PyG)**
- Approach: Start fresh in `/exp/mu2e/app/users/wzhou2/projects/calorimeter/GNN/`
- Immediate goal: **Working end-to-end prototype**

---

## Progress Checklist

### Task 0: Crystal Geometry Lookup Table

**Goal:** Build a one-time `crystalId → (x, y, disk)` lookup table, required for all hit node features.

**Why needed:** The `calohits` branch stores only `crystalId_` — there are no (x,y) position fields in older datasets. Crystal positions live in the Mu2e geometry service (C++ only). MDC2025-002 does provide `calohits.crystalPos_` directly.

- [x] ~~Write C++ art analyzer~~ — bypassed: user provided complete crystal map directly
- [x] Save raw crystal map as `data/crystal_map_raw.csv` (2740 rows, all SiPM channels)
- [x] Generate `data/crystal_geometry.csv` from raw map (1348 crystals: 1344 CAL + 4 CAPHRI, 674 per disk)
- [x] Produce `data/crystal_neighbors.csv` — computed from physical positions using cKDTree (1.5× pitch threshold)
- [x] Write Python loader `src/geometry/crystal_geometry.py`: `load_crystal_map()` and `load_neighbor_map()`
- [x] Fix path bug in loader (`parents[3]` → `parents[2]`)
- [x] Note: 4 CAPHRI crystals (IDs 582, 609, 610, 637) are not CsI — all on disk 0

**Confirmed C++ API:**
```cpp
const Calorimeter& cal = *(GeomHandle<Calorimeter>());
const Crystal& cry = cal.crystal(crystalId);
int diskId = cry.diskID();
CLHEP::Hep3Vector localPos = cry.localPosition();  // disk-local frame (mm)
const std::vector<int>& neighbors = cal.neighbors(crystalId);
const std::vector<int>& nextNeighbors = cal.nextNeighbors(crystalId);
```

**Output:** `data/crystal_geometry.csv`, `data/crystal_neighbors.csv`
**Blocker for:** Tasks 4, 5, and all event displays

---

### Task 1: Environment Setup

**Goal:** Confirm the Mu2e Python environment with PyG is available and set up the project structure.

- [x] Verify environment: Python 3.12.13, torch 2.5.1, torch_geometric 2.7.0, uproot 5.7.2, scipy 1.17.1
- [x] Confirm `torch-cluster` is NOT installed — use `scipy.spatial.cKDTree` for graph construction
- [x] Create `setup_env.sh` (sources `setupmu2e-art.sh`, calls `pyenv ana 2.6.1`)
- [x] Create `scripts/smoke_test_env.py` (imports packages, builds synthetic graph, runs forward pass)
- [x] Create project directory structure (`src/`, `scripts/`, `tests/`, `configs/`, `data/`, `splits/`, etc.)
- [x] Create `configs/default.yaml` with all hyperparameter sections (data, graph, model, train, inference, output)
- [x] Create `.gitignore` (exclude processed data, checkpoints, outputs; keep crystal geometry CSV)

---

### Task 2: Understand the Existing Algorithm

**Goal:** Know what the BFS algorithm produces so you can define ground-truth labels and comparison metrics.

- [x] Study `ClusterFinder.cc` — BFS graph traversal
- [x] Study `CaloProtoClusterMaker_module.cc` — seed selection
- [x] Study `CaloClusterMaker_module.cc` — cluster association
- [x] Document key outputs: cluster disk ID, time, energy, CoG, hit list
- [x] Note: E1, E9, E25 shape variables are NOT in EventNtuple — drop from baseline metrics

**BFS algorithm summary:**
- **Seeds:** hits with E > EminSeed threshold
- **BFS expansion:** neighbors (1st/2nd ring) included if |Δt| < deltaTime; expand further only if E > ExpandCut
- **Split-offs:** remaining hits re-clustered, then attached to nearest main cluster (distance + time cut)
- **Merging:** main clusters merged if close enough (maxDistMain) and time-compatible
- **Outputs per cluster:** diskID, total energy, time (from most energetic hit), CoG (3D), hit list (sorted by E)
- **Truth matching** (`CaloHitTruthMatch`): MC showers matched by time window, grouped by SimParticle

**Key source files:**
- `CaloCluster/src/ClusterFinder.cc`
- `CaloCluster/src/CaloProtoClusterMaker_module.cc`
- `CaloCluster/src/CaloClusterMaker_module.cc`

---

### Task 3: Baseline — Extract BFS Cluster Results

**Goal:** Extract BFS cluster-level outputs already stored in EventNtuple files as a baseline.

- [x] Write `scripts/baseline_existing.py` — BFS vs MC truth benchmark on MDC2025-002
- [x] Save per-event results to `data/baseline/bfs_benchmark.csv`
- [x] Generate summary histograms to `data/baseline/plots/` — `scripts/baseline_plots.py` produces 6 plots + `summary.txt`
- [x] Extract per-cluster detail (disk, energy, time, CoG, hits) to `data/baseline/bfs_cluster_summary.csv` — `scripts/baseline_cluster_detail.py`

BFS baseline numbers (old truth, 3 files): see `docs/findings.md` §3.

**Confirmed EventNtuple branches (MDC2020/MDC2025-001):**

| Field | Branch |
|-------|--------|
| Cluster disk | `caloclusters.diskID_` |
| Cluster energy | `caloclusters.energyDep_` |
| Cluster time | `caloclusters.time_` |
| Cluster CoG | `caloclusters.cog_.data[3]` |
| Cluster n_hits | `caloclusters.size_` |
| Cluster hit indices | `caloclusters.hits_` |
| Hit crystal ID | `calohits.crystalId_` |
| Hit energy | `calohits.eDep_` |
| Hit time | `calohits.time_` |
| Hit → cluster index | `calohits.clusterIdx_` |

**Cluster matching rule (freeze now):** Greedy max energy-weighted overlap. Purity = E_shared / E_reco; Completeness = E_shared / E_truth. Matched if purity > 0.5 AND completeness > 0.5.

---

### Task 3b: Truth Label Specification

**Goal:** Define truth labels deterministically before any training.

**Status:** UNBLOCKED — MDC2025-002 includes full MC truth via `calohitsmc` and `caloclustersmc` branches.

- [x] Implement `src/data/truth_labels.py` using MC truth:
  - [x] Use `calohitsmc.simParticleIds` + `calohitsmc.eDeps` for per-hit SimParticle contributions
- [x] Define truth cluster = all non-ambiguous hits sharing the same primary SimParticle (per disk)
- [x] Define ambiguous hit = dominant SimParticle purity < 0.7
- [x] Edge labels: `y_edge = 1` if both non-ambiguous AND same truth cluster; mask if either ambiguous
- [x] Write unit tests in `tests/test_truth_labels.py` — 11 tests passing (unittest):
  - [x] Two hits with same truth particle → edge label 1
  - [x] Ambiguous hit → masked from loss
  - [x] Hit with purity < threshold → `is_ambiguous = True`
  - [x] Additional: boundary thresholds, different disks, empty/zero-energy hits, multi-hit clusters, dtypes

**MC truth branches (MDC2025-002):**
- `calohitsmc.simParticleIds` — per-hit SimParticle IDs
- `calohitsmc.eDeps` — per-hit energy deposits per SimParticle
- `calohitsmc.nsim` — number of contributing SimParticles
- `caloclustersmc.*` — cluster-level MC truth

**⚠ Note:** MDC2025-002 CoG format differs: `caloclusters.cog_.fCoordinates.{fX,fY,fZ}` instead of `cog_.data[3]`.

---

### Task 3c: Train/Val/Test Split Policy

**Goal:** Establish frozen, leak-free dataset splits before any normalization or training.

- [x] Write `scripts/make_splits.py` (deterministic, seed=42)
- [x] Produce `splits/train_files.txt` (35), `splits/val_files.txt` (7), `splits/test_files.txt` (8) — 50 MDC2025-002 files
- [x] Commit split files and freeze — never re-split after this point
- [x] Verify: file-level split guarantees per-disk graphs from the same event stay in the same split

**Rules:**
- Normalization stats computed on train split only → `data/normalization_stats.pt`
- Threshold tuning on val only
- Final physics metrics reported once on test

---

### Task 4: Data Extraction

**Goal:** Read EventNtuple ROOT files and extract per-disk CaloHit data as PyG Data objects.

**Requires:** Task 0 (crystal geometry) or MDC2025-002 (has `calohits.crystalPos_` in-file).

- [x] Implement `src/data/dataset.py` — `extract_events_from_file()` + `CaloGraphDataset` PyG Dataset
- [x] Extract per-disk graphs with:
  - [x] Node features (6-dim): log energy, time, x, y, radial distance, relative energy
  - [x] Edge features (8-dim): Δx, Δy, distance, Δt, Δ log energy, energy asymmetry, log summed energy, Δr
  - [x] Node labels `y_node` and edge labels `y_edge` via `truth_labels.py`
- [x] Implement `src/data/normalization.py` — compute/save/load/apply z-score normalization from train split
- [x] Save to `data/processed/{source_stem}_evt{N}_disk{D}.pt`
- [x] Implement `src/data/graph_builder.py` — hybrid radius+kNN, time filter, degree cap, feature computation
- [x] Write `scripts/build_graphs.py` — CLI entry point with split-aware file loading
- [x] Verified on 20 events: 34 graphs, 6-dim nodes, 8-dim edges, correct labels, no NaNs

**Graph unit:** One graph per disk per event (Disk 0 and Disk 1 are separate graphs).

**Node features (6-dim, globally normalized from training set):**

| # | Feature | Formula |
|---|---------|---------|
| 1 | log energy | `log(1 + E)` |
| 2 | time | `t` |
| 3 | x | position in disk-local frame |
| 4 | y | position in disk-local frame |
| 5 | radial distance | `sqrt(x² + y²)` |
| 6 | relative energy | `E / E_max` (per-graph) |

**Edge features (8-dim, directed i→j):**

| # | Feature | Formula |
|---|---------|---------|
| 1 | Δx | `x_i - x_j` |
| 2 | Δy | `y_i - y_j` |
| 3 | distance | `sqrt(Δx² + Δy²)` |
| 4 | Δt | `t_i - t_j` |
| 5 | Δ log energy | `log(1+E_i) - log(1+E_j)` |
| 6 | energy asymmetry | `(E_i - E_j) / (E_i + E_j)` |
| 7 | log summed energy | `log(1 + E_i + E_j)` |
| 8 | Δr | `r_i - r_j` |

---

### Task 5: Graph Construction

**Goal:** Build per-disk radius graphs with a kNN fallback to prevent isolated tail hits.

- [x] Implement `src/data/graph_builder.py` — done in Task 4
- [x] Hard constraints: same disk only (graph built per disk), `|Δt| < 25 ns`
- [x] Hybrid edge strategy:
  - [x] Radius graph: `d < r_max = 210 mm` via cKDTree.query_pairs (bumped from 150mm — see gate results)
  - [x] kNN fallback: isolated nodes get edges to `k_min = 3` nearest time-compatible neighbors
- [x] Max degree cap: `k_max = 20` nearest neighbors after radius + time cut
- [x] Save graph diagnostics per graph: n_nodes, n_edges, avg_degree, isolated nodes, min/max degree
- [x] Pass graph construction gate (`scripts/graph_gate.py`):
  - [x] 100% of same-cluster hit pairs connected by an edge (target >99%)
  - [x] 100% of truth clusters fully connected as a subgraph (target >95%)
  - [x] Recall stratified by cluster energy, hit multiplicity, radial position — all bins 100%
- [x] Write unit tests in `tests/test_graph_builder.py` — 22 tests passing (unittest)

**r_max tuning note:** At r_max=150mm, 24 missed pairs (all spatial, 153–209mm apart, all time-compatible). Bumping to 210mm captures all pairs with minimal edge count increase (832→856 edges, max degree unchanged at 6). No degree cap or time filter misses.

Gate results (pair recall 1.0000, cluster connectivity 1.0000): see `docs/findings.md` §2. Raw results in `outputs/graph_gate/`.

---

### Task 6a: Simple Baseline GNN (SimpleEdgeNet)

**Goal:** Prove the data pipeline and labels work before committing to the full architecture.

**Code (done — ready to use):**
- [x] Implement `src/models/simple_edge_net.py` (215K params, 3 MP rounds, sum aggregation)
- [x] Loss: class-balanced BCE (`pos_weight = neg/pos`) on masked edges only
- [x] Metrics: edge P/R/F1/AUC, cluster purity/completeness from predicted edges
- [x] Trainer: AdamW + ReduceLROnPlateau + early stopping + checkpointing + JSON history
- [x] Write `scripts/train_gnn.py`, `scripts/build_all_graphs.sh`, `scripts/run_training.sh`
- [x] Write `scripts/evaluate_head_to_head.py` — GNN vs BFS both evaluated against MC truth
- [x] Write `notebooks/train_simple_edge_net.ipynb` — full pipeline notebook
- [x] All 50 ROOT files (97 GB) available at `/exp/mu2e/data/users/wzhou2/GNN/root_files/`

**Execution:**
- [x] Build graphs with MC truth: `bash scripts/build_all_graphs.sh` — 41,656 graphs (29,143 train / 5,793 val / 6,720 test), 276 MB, ~5 min
- [x] Pack graphs into single files per split: `python3 scripts/pack_graphs.py` — train.pt (114 MB), val.pt (22 MB), test.pt (26 MB). Eliminates NFS I/O bottleneck (25 min → ~10s load).
- [x] Train on GPU: 18 epochs on A100 MIG, ~6s/epoch, best val F1 = 0.925 (epoch 3), early stopped
- [x] Run head-to-head evaluation: `python3 scripts/evaluate_head_to_head.py` — 6,058 disk-graphs from val split
- [x] Training visualizations: `python3 scripts/plot_training.py` — 5-panel overview saved to run dir
- [x] Acceptance criteria:
  - [x] Edge F1 = 0.925 on validation (target >0.7)
  - [x] GNN purity 0.977 > BFS purity 0.974 (evaluated against MC truth)
  - [x] No NaN losses, no degenerate outputs

Head-to-head val results (old truth): see `docs/findings.md` §3.2.

---

### Task 7: Inference Pipeline (SimpleEdgeNet)

**Goal:** Reconstruct clusters from SimpleEdgeNet edge predictions; tune thresholds on val; evaluate on test.

**Note:** SimpleEdgeNet has no node saliency head — skip `τ_node` prefiltering for now. Focus on edge threshold tuning and cluster reconstruction.

#### 7a: Cluster Reconstruction Module ✓

- [x] Implement `src/inference/cluster_reco.py`:
  - [x] `symmetrize_edge_scores()` — average directed p_ij/p_ji into undirected scores (canonical i<j ordering)
  - [x] `reconstruct_clusters()` — sigmoid → symmetrize → threshold → connected components → cleanup → contiguous relabeling
  - [x] `predict_clusters()` — convenience wrapper: model forward pass + reconstruction
  - [x] Cleanup: remove clusters with `n_hits < min_hits` or `E_total < min_energy_mev`
  - [x] Return cluster assignments as integer array (per-node cluster ID, -1 for unclustered)
- [x] Write unit tests: `tests/test_inference.py` — 12 tests passing
  - [x] Two-cluster graph, threshold separation, min_hits/min_energy cleanup
  - [x] Symmetrization: bidirectional averaging, single-direction passthrough, canonical ordering
  - [x] Contiguous relabeling, torch tensor input, no-symmetrize mode, all-below-threshold

#### 7b: Cluster Postprocessing ✓

- [x] Implement `src/inference/postprocess.py`:
  - [x] `compute_cluster_features()` — per-cluster: hit list, n_hits, total energy, energy-weighted centroid (x, y), energy-weighted time, RMS spatial width, max-hit energy fraction
  - [x] `compute_summary_statistics()` — n_clusters, mean/median cluster size, mean/median energy
- [x] Write unit tests: `tests/test_postprocess.py` — 10 tests passing
  - [x] Single/multiple clusters, energy-weighted centroid, RMS width, unclustered exclusion, hit indices, empty/single-hit edge cases, summary stats

#### 7c: Threshold Tuning on Validation Set ✓

- [x] Write `scripts/tune_threshold.py`:
  - [x] Load trained model checkpoint + val split graphs (5,793 packed graphs, 14s inference on GPU)
  - [x] Sweep `τ_edge` over [0.1, 0.2, ..., 0.9] (coarse) + finer grid ±0.1 around optimum (step 0.02)
  - [x] For each threshold, run cluster reconstruction and compute:
    - [x] Cluster-level purity, completeness (energy-weighted matching)
    - [x] Truth match rate, reco match rate
    - [x] Number of merges and splits
    - [x] Pairwise F1 (edge-level, for consistency)
  - [x] Select optimal `τ_edge` = **0.34** (maximizes pairwise F1)
  - [x] Save sweep results to `outputs/threshold_sweep/sweep_results.csv`
  - [x] Generate 6-panel threshold sweep plot to `outputs/threshold_sweep/threshold_sweep.png`
  - [x] **Frozen** optimal threshold in `configs/default.yaml` → `inference.tau_edge: 0.34`

v1 threshold sweep numbers: see `docs/findings.md` §3.2.

**Design note:** Sweep uses `min_hits=1, min_energy=0` (no cleanup) to match head-to-head evaluation conditions and get meaningful truth match rate variation. 52.2% of val truth clusters are single-hit — production cleanup (`min_hits=2`) removes them, capping truth match rate at ~50%. Pairwise F1 is unaffected by cleanup and is the primary optimization metric. Production cleanup metrics also reported at optimal threshold.

#### 7d: Test Set Evaluation (Run Once)

- [x] Write `scripts/evaluate_test.py`:
  - [x] Load model + frozen threshold (τ_edge=0.34) + test split ROOT files
  - [x] Run inference → `reconstruct_clusters()` → matching against MC truth
  - [x] Also run BFS baseline on same test events for direct comparison
  - [x] Compute full metrics suite: purity, completeness, truth/reco match rate, merges, splits
  - [x] Energy-binned metrics (low E < 50 MeV, mid 50–200 MeV, high > 200 MeV)
  - [x] Hit-multiplicity-binned metrics (1-hit, 2-3 hits, 4+ hits)
  - [x] Save results to `outputs/test_eval/test_results.csv` + `truth_cluster_detail.csv`
  - [x] Generate 9-panel comparison plot to `outputs/test_eval/test_evaluation.png`
- [x] **Test set evaluated once.** 4,000 events (500/file × 8 files), 6,996 disk-graphs, 42.5s on CPU.

v1 SimpleEdgeNet test-set results (old truth): see `docs/findings.md` §3.3. GNN's main advantage was ~46% fewer splits and higher completeness, competitive elsewhere.

#### 7e + 7f: Debug Visualization & GNN Cluster Display ✓

Combined into single script `scripts/plot_gnn_clusters.py`:

- [x] 3-panel layout per event: MC Truth | BFS Reco | GNN Predicted (side-by-side on crystal map)
- [x] Crystal map background (reuses pattern from `plot_event_display.py`)
- [x] Hit crystals colored by cluster assignment; unclustered hits in gray
- [x] GNN panel: edges drawn with RdYlGn color gradient by predicted probability
- [x] Annotate per-panel: n_clusters, n_hits, total energy, unclustered count
- [x] Merged clusters highlighted with red border; split clusters with orange border
- [x] `--find-failures` mode: scans N events, auto-selects worst failure cases (merges/splits)
- [x] CLI flags: `--n-events`, `--event-indices`, `--checkpoint`, `--tau-edge`, `--split {val,test}`, `--find-failures`, `--n-scan`, `--device`
- [x] Reads ROOT files for BFS labels + MC truth; builds graphs on-the-fly for GNN inference
- [x] Normal displays saved to `outputs/gnn_cluster_display/` (6 events)
- [x] Debug displays saved to `outputs/debug/` (6 worst failure cases from 200 scanned events, 110/200 had failures)

---

### Task 8: GNN Model — CaloClusterNet-v1 ✓

**Goal:** Multi-task edge-centric message-passing GNN with auxiliary node saliency head.
**Prerequisite:** Complete Task 7 (inference pipeline + test evaluation) first. Decide whether to proceed based on SimpleEdgeNet test-set results.

- [x] Implement `src/models/layers.py` — `EdgeAwareResBlock` with residual + LayerNorm + gated aggregation + global context
- [x] Implement `src/models/heads.py` — `NodeSaliencyHead` + `EdgeClusteringHead`
- [x] Implement `src/models/calo_cluster_net.py`:
  - [x] Node encoder (6→96) + Edge encoder (8→96) with GELU activations
  - [x] 4× EdgeAwareResBlock (edge update, gated aggregation, node update, global context)
  - [x] Node saliency head → q_i logit (raw, no sigmoid)
  - [x] Edge clustering head → s_ij logit ([h_i, h_j, e_ij, |h_i-h_j|] 384-dim input)
  - [x] Forward returns `{"edge_logits": Tensor(E,), "node_logits": Tensor(N,)}`
- [x] Update training infrastructure:
  - [x] `src/training/losses.py` — `node_saliency_loss`, `consistency_loss`, `multitask_loss` (backward-compatible with SimpleEdgeNet tensor output)
  - [x] `src/training/trainer.py` — handles dict model output, logs per-component losses, reads `lambda_edge/node/cons` from config
  - [x] `scripts/train_gnn.py` — `build_model()` in `src/models/__init__.py`, shared by all scripts; `--resume` flag for staged training
  - [x] `src/inference/cluster_reco.py` — `predict_clusters()` handles dict output
  - [x] `src/models/__init__.py` — exports both models + `build_model()` factory
- [x] Write unit tests: `tests/test_calo_cluster_net.py` — 18 tests passing (73 total)
  - [x] EdgeAwareResBlock: output shapes, residual changes, gradient flow
  - [x] NodeSaliencyHead / EdgeClusteringHead: output shapes, unbounded logits
  - [x] CaloClusterNet: dict output, shapes, default config, param count, gradients, single-node graph, eval determinism
  - [x] Inference compatibility: `reconstruct_clusters` and `predict_clusters` work with dict output
  - [x] Multi-task loss: edge-only stage, full multi-task stage, SimpleEdgeNet backward compatibility
- [x] CPU dry-run: 2 epochs on full dataset, val edge F1=0.893, node F1=0.687, no NaN — `outputs/runs/dry_run_v1/`

**CaloClusterNet: 676,550 params (3.1× SimpleEdgeNet's 215,553)**

**Architecture detail — each message-passing block:**
- A: edge update (residual): `m_ij = MLP([h_i, h_j, e_ij])` (288→192→96); `e_ij ← LayerNorm(e_ij + m_ij)`
- B: gated aggregation: `g_ij = σ(Linear(e_ij))`; `a_i = Σ g_ji · e_ji`
- C: node update (residual): `u_i = MLP([h_i, a_i])` (192→192→96); `h_i ← LayerNorm(h_i + u_i)`
- D: global context: `c = mean(h_i)`; `h_i ← h_i + Linear(c)`

**Output heads:**
- Node saliency: Linear(96,64) → GELU → Dropout(0.1) → Linear(64,1)
- Edge clustering: [h_i, h_j, e_ij, |h_i−h_j|] (384-dim) → Linear(384,192) → GELU → Dropout → Linear(192,96) → GELU → Dropout → Linear(96,1)

**Multi-task loss:** `L = λ_edge·L_edge + λ_node·L_node + λ_cons·L_cons`
- L_edge: masked BCE with class-balanced pos_weight (same as SimpleEdgeNet)
- L_node: BCE on node saliency (y_node >= 0 = signal)
- L_cons: `mean(σ(s_ij) · (1-σ(q_i)) · (1-σ(q_j)))` — penalizes positive edges between non-salient nodes

**To train:** Set `model.name: CaloClusterNet`, `model.hidden_dim: 96`, `model.n_mp_layers: 4` in config. Control staged training via `train.lambda_node` and `train.lambda_cons`.

---

### Task 9: Training & Evaluation (CaloClusterNet-v1)

**Goal:** Train CaloClusterNet-v1 with staged multi-task loss; tune thresholds; evaluate on test set; produce event displays.
**Prerequisite:** Task 8 model implementation ✓. Requires GPU node.

#### 9a: Script Infrastructure ✓

- [x] Shared `build_model()` in `src/models/__init__.py` (used by all scripts)
- [x] `--resume` flag in `scripts/train_gnn.py` (loads model weights, resets optimizer for staged training)
- [x] Stage configs: `configs/calo_cluster_net.yaml` (Stage 1), `_stage2.yaml`, `_stage3.yaml`
- [x] `node_saliency_metrics()` in `src/training/metrics.py`
- [x] Trainer logs val node saliency metrics when model returns `node_logits`
- [x] `reconstruct_clusters()` and `predict_clusters()` support `node_logits`/`tau_node`
- [x] Update `scripts/tune_threshold.py` — use `build_model()`, handle dict output, add `--tau-node`, model-specific output dir
- [x] Update `scripts/evaluate_test.py` — use `build_model()`, handle dict output, pass `node_logits`/`tau_node`, model-specific output dir
- [x] Update `scripts/plot_gnn_clusters.py` — use `build_model()`, handle dict output, pass `tau_node`, model-specific output dir

#### 9b: Staged GPU Training

- [x] Stage 1: edge-only (λ_node=0, λ_cons=0), 27 epochs (~16s/epoch on A100 MIG), early stopped
  - Checkpoint: `outputs/runs/calo_cluster_net_stage1/checkpoints/best_model.pt`
- [x] Stage 2: add node saliency (λ_node=0.3), 17 epochs, resumed from Stage 1, node F1 = 1.000 trivially
  - Checkpoint: `outputs/runs/calo_cluster_net_stage2/checkpoints/best_model.pt`
  - Edge F1 slightly lower than Stage 1 due to multi-task loss; node head learned immediately
- [x] Stage 3: add consistency (λ_cons=0.05), 19 epochs — essentially identical to Stage 2
  - Checkpoint: `outputs/runs/calo_cluster_net_stage3/checkpoints/best_model.pt`
  - Consistency loss had negligible effect; not used going forward

v1 CaloClusterNet training numbers: see `docs/findings.md` §3.1.

#### 9c: Threshold Tuning (Val Set) ✓

- [x] Sweep τ_edge on val set for Stage 1 (no τ_node — node head untrained in Stage 1); frozen τ_edge = **0.30**. Results: `outputs/threshold_sweep_caloclusternet_stage1/`
- [x] Sweep τ_edge on val set for Stage 2 with τ_node=0.5 (trained node head); τ_node=0.5 gave marginal merge reduction; not worth the complexity. Results: `outputs/threshold_sweep_caloclusternet_stage2/`
- [x] Frozen τ_edge = 0.30 in `configs/calo_cluster_net.yaml`

v1 threshold tuning numbers: see `docs/findings.md` §3.2.

**Note:** τ_node filtering should only be applied when the node head was explicitly trained (lambda_node > 0). Scripts updated to not auto-apply τ_node for untrained node heads.

#### 9d: Test-Set Evaluation (Run Once) ✓

- [x] Run once: CaloClusterNet Stage 1 (τ_edge=0.30) vs BFS, both vs MC truth
- [x] 4,000 events (500/file × 8 files), 6,996 disk-graphs
- [x] Results: `outputs/test_eval_caloclusternet/`

v1 CaloClusterNet test-set results: see `docs/findings.md` §3.3. CaloClusterNet gave marginal improvements over SimpleEdgeNet (fewest merges, highest purity); differences within noise — the bottleneck was the truth definition, not model capacity, which motivated Task 11.

#### 9e: Event Displays ✓

- [x] 6 normal event displays: `outputs/gnn_cluster_display_caloclusternet/`
- [x] 6 worst failure cases from 200 scanned events: `outputs/debug_caloclusternet/`

---

### Task 10: Failure Audit (Root-Cause Analysis) ✓

**Goal:** Before building new models, determine whether remaining failures are caused by bridge edges, thresholding, tiny objects, or the truth definition itself.

- [x] Write `scripts/failure_audit.py` — comprehensive audit on val set (5,793 graphs)
- [x] Run on CaloClusterNet (τ_edge=0.30), results in `outputs/failure_audit/audit_summary.json`

Full v1 audit results and Q1–Q5 analysis: see `docs/findings.md` §3.4.

**Headline conclusions:**
1. Not an inference problem (bridge edges are confidently wrong, not borderline).
2. Not a thresholding problem (no better trade-off exists).
3. Primarily a truth-definition issue for singletons → led to Task 11.
4. Multi-hit clusters work well.

---

### Task 11: Primary-Level Truth Redefinition

**Goal:** Fix the truth definition so that all hits from the same electromagnetic shower are assigned to the same truth cluster, instead of being split by individual SimParticle ID.

**Root cause discovered:** The current truth (`truth_labels.py`) groups hits by dominant SimParticle ID. But a primary particle (e.g., electron SimP 4) produces secondary photons (SimP 13, 14, 17, 18) during showering. These secondaries deposit small amounts of energy in nearby crystals at the same time (~0.2 ns). Each secondary that dominates a crystal becomes its own "truth cluster" — a singleton that is physically part of the parent shower.

This is why 52% of truth clusters are single-hit: they are mostly secondary shower products, not independent physics objects.

**The fix:** Trace each SimParticle back to its **calo-entrant ancestor** — the particle that entered the calorimeter from outside and initiated the shower. Group all hits from the same calo-entrant shower as one truth cluster. This requires adding parent chain info to the ROOT files (currently missing), then using it in Python truth labeling.

**Key distinction:** The Mu2e "PrimaryParticle" is the single signal particle per event (e.g., mu2eFlateMinus). That's NOT what we want. We want the calo-entrant: the particle that entered the calo volume from outside. A single primary can produce multiple calo-entrants (e.g., signal electron enters disk 0, its brem photon enters disk 1 → two separate showers/truth clusters).

**Expected impact:**
- Singleton truth clusters should largely disappear (collapsed into parent shower clusters)
- K (objects per graph) should decrease, object sizes should grow
- The model's current "merge errors" on singletons should become correct predictions
- Metrics should improve without retraining — same model, better truth

#### 11a: Investigate SimParticle Ancestry in ROOT Files ✓

- [x] Read `calomcsim` branches — ancestry fields are **not populated**:
  - `calomcsim.prirel._rel` = -1 for all SimParticles
  - `calomcsim.prirel._rem` = -1 for all SimParticles
  - `calomcsim.gen` = -1, `calomcsim.rank` = -1, `calomcsim.nhits` = -1
  - Only useful fields: `calomcsim.id`, `calomcsim.pdg`, `calomcsim.time`, `calomcsim.startCode`, `calomcsim.mom`
- [x] `calohitsmc.eprimary`/`tprimary` are NOT the primary ancestor — they are just the energy/time from the dominant SimParticle in each hit (same info we already use)
- [x] Confirmed shower co-occurrence: secondaries always deposit in the same hits as their parent (within ~0.2 ns), so a co-occurrence heuristic would mostly work — but it's not the real Geant4 parent-child relationship
- [x] Investigated C++ source code in EventNtuple, Offline, and Production repos:
  - **Root cause:** `fillCaloSimInfos()` in `InfoMCStructHelper.cc` never computes `prirel` or stores parent IDs (unlike the track version `fillAllSimInfos()` which does)
  - `SimParticle::parent()` IS available in the art event — the code just doesn't use it for calo
  - `CaloHitTruthMatch` (Offline) computes MCRelationship to primary particles in `CaloEDepMC.rel_`, but EventNtuple ignores this field for calo
  - `calohitsmc.simRels` stores relationship to most-energetic deposit in the same hit (not to primary) — and can't be read by uproot due to serialization
  - `caloclustersmc.prel` is 92% unpopulated (-1), only 8% have `same` (cluster IS the signal primary)
  - 100% of `calohitsmc.simParticleIds` are covered by `calomcsim` — no missing particles
  - `startCode` IS populated (eBrem=16 dominates at 50%, Decay=14, annihil=2, etc.) but alone is insufficient to match secondaries to specific parents
- [x] Identified Mu2e "PrimaryParticle" concept: this is the single signal particle per event (e.g., mu2eFlateMinus), NOT what we need. We need the **root ancestor** of each SimParticle's parent chain (many per event: signal + all pileup sources)
- [x] Better clustering target: the **calo-entrant ancestor** — the particle that entered the calorimeter from outside, identified by walking up the parent chain

**Resolution:** Modify EventNtuple C++ to add `ancestorSimIds` field storing the full parent chain, then reprocess ROOT files. See Tasks 11a2–11a4.

#### 11a2: Modify EventNtuple C++ to Store Ancestry

**Goal:** Add a `std::vector<int> ancestorSimIds` field to `SimInfo` and populate it for calo SimParticles by walking `edep.sim()->parent()` up the Geant4 parent chain.

**Source code:** `/exp/mu2e/app/users/wzhou2/working_dir/EventNtuple/`

- [x] In `inc/SimInfo.hh`: added `std::vector<int> ancestorSimIds;` field (line 32). `reset()` uses `*this = SimInfo()` which clears it automatically.
- [x] In `src/InfoMCStructHelper.cc`, `fillCaloSimInfos()` (lines 421-426): after `fillSimInfo()`, walks `parent()` chain and collects all ancestor IDs. Loop stops when `hasParent()` returns false (root) or parent pointer is null (compression).
- [x] In `src/CrvInfoHelper.cc` (line 250-251): removed extra `ewmh` argument from `CrvMCHelper::GetInfoFromCrvRecoPulse()` call to fix CRV API mismatch with CVMFS Offline (function expects 8 args, EventNtuple was passing 9). Unrelated to our calo changes.
- [x] No dictionary changes needed — `std::vector<int>` inside already-registered `SimInfo` is handled by ROOT automatically
- [x] `InfoMCStructHelper.os` compiles cleanly (verified)
- [x] Full build and link — `muse build -j64` on build node (2026-04-04)
  - Fixed pre-existing `getTrigPathNameByIndex` → `getTrigPathName` API mismatch in `EventNtupleMaker_module.cc` (lines 1072, 1078)
  - Cloned `Production` repo into working dir (needed for `epilog.fcl` dependency)
  - Build: `al9-prof-e29-p087` with EventNtuple + Offline + Production

**Files modified:**
1. `EventNtuple/inc/SimInfo.hh` — added `ancestorSimIds` field (line 32)
2. `EventNtuple/src/InfoMCStructHelper.cc` — walk parent chain in `fillCaloSimInfos()` (lines 421-426)
3. `EventNtuple/src/CrvInfoHelper.cc` — CRV API fix (line 250-251, unrelated to calo)
4. `EventNtuple/src/EventNtupleMaker_module.cc` — trigger API fix (`getTrigPathNameByIndex` → `getTrigPathName`, lines 1072, 1078)
5. `EventNtuple/fcl/from_mcs-calo-only.fcl` — custom minimal FCL that skips TrkQual/TrkPID (avoids ArtAnalysis dependency)

**Build instructions (non-interactive shell):**
`setupmu2e-art.sh` on al9 handles spack + muse setup automatically (no separate `$SPACK_ROOT` sourcing needed):
```bash
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh
cd /exp/mu2e/app/users/wzhou2/working_dir
muse setup
muse build -j64
```

**Working dir contents:** `EventNtuple/`, `Offline/`, `Production/` (all three required for build).

#### 11a3: Build Modified EventNtuple and Reprocess ROOT Files

**Goal:** Produce new NTS ROOT files with the `calomcsim.ancestorSimIds` branch.

**Input MCS art files:** `/pnfs/mu2e/persistent/datasets/phy-sim/mcs/mu2e/FlateMinusMix1BBTriggered/MDC2025af_best_v1_1/art/` (50 files, run 001430)

**FHiCL:** `EventNtuple/fcl/from_mcs-calo-only.fcl` (custom minimal FCL, skips TrkQual/TrkPID)

- [x] Build modified EventNtuple against Offline + Production using `muse build -j64`
- [x] Test on 1 MCS art file (5 events) — `calomcsim.ancestorSimIds` branch populated with full parent chains
  - Example: signal particle chain `[4, 3, 2, 1, 0]`; secondary with single parent `[478]`
- [x] Batch reprocess all 50 MCS files — tmux session `reprocess`
  - Script: `/exp/mu2e/data/users/wzhou2/GNN/root_files_v2/run_all.sh`
  - Log: `/exp/mu2e/data/users/wzhou2/GNN/root_files_v2/batch_output.log` (run 1), `batch_output_v2.log` (run 2)
  - Per-file logs: `/exp/mu2e/data/users/wzhou2/GNN/root_files_v2/logs/`
  - Output: `/exp/mu2e/data/users/wzhou2/GNN/root_files_v2/mcs.*.root`
  - **Gotcha fixed:** must use `-T` (not `--TFileName`) for art TFileService output path
  - Monitor: `tmux attach -t reprocess` or `ls root_files_v2/mcs.*.root | wc -l`
  - **Run 1 (2026-04-04 07:43):** 20 parallel jobs. Login node killed all processes after ~10 min (resource limits). 20/50 files completed, 18 partial (62-717 MB vs ~1.9 GB expected), 12 never started.
  - **Run 2 (2026-04-04 09:00):** Reduced to `MAX_PARALLEL=10`. Deleted 18 partial files, rerunning 30 remaining. Still killed after ~3-4 min — 10 new partials (735-851 MB), 0 new completions. Login node resource limits too aggressive for parallel mu2e jobs.
  - **Status:** 20 complete v2 files (18 valid, 2 corrupt: 00000008, 00000029). Remaining 30 files need reprocessing with lower parallelism (MAX_PARALLEL=5 or sequential) or on a different node.
- [x] Reprocess remaining 30 files — superseded by FermiGrid submission (all 50 files produced via grid jobs, cluster `90854576`)
- [x] Verify 18 valid files have all existing branches plus `ancestorSimIds` — confirmed by validate_ancestry.py

#### 11a4: Validate Ancestry Data in New ROOT Files ✓

**Goal:** Confirm the ancestor chains are correct and usable before building truth labels.

- [x] Write `scripts/validate_ancestry.py`
- [x] Validated on all 18 valid v2 files (9,000 events, 203,971 SimParticles)
- [x] Calo-entrant identification: **per-disk** — highest ancestor that also deposited in the same disk. Cross-disk secondaries become their own calo-entrant.

Full ancestry validation numbers (chain stats, StartCode breakdown, ambiguity/singleton reductions): see `docs/findings.md` §1.3.

#### 11b: Implement Calo-Entrant Truth Definition ✓

**Goal:** Redefine truth clusters at the calo-entrant level using the new ancestry data.

- [x] Created `src/data/truth_labels_primary.py` (old `truth_labels.py` unchanged for comparison)
  - `build_calo_root_map()` — per-disk: highest ancestor in calomcsim that also deposited on the same disk
  - `assign_mc_truth_primary()` — groups hit energy deposits by calo-root before computing purity
  - Fallback: SimParticle not in calo_root_map → uses pid itself (safe for missing ancestry)
- [x] 15 unit tests in `tests/test_truth_labels_primary.py`:
  - `TestBuildCaloRootMap`: no ancestors, same-disk ancestor, cross-disk secondary, deep chain, gap in chain, both disks
  - `TestAssignMcTruthPrimary`: same shower positive, different showers negative, ambiguity resolved by grouping, still ambiguous with different roots, cross-disk separate, fallback, empty hit, shapes/dtypes, multi-hit shower

#### 11c: Compare Truth Cluster Statistics (Old vs New) ✓

Truth comparison is built into `scripts/validate_ancestry.py` — prints side-by-side old vs new truth stats. See 11a4 validation results above for the full summary.

#### 11d: Re-Evaluate Existing Models with New Truth (No Retraining) ✓

- [x] Wrote `scripts/evaluate_new_truth.py` — reads v2 ROOT files, builds graphs on the fly, evaluates BFS + both GNN models against old and new truth definitions side-by-side
- [x] Run on val split (3 complete v2 files, 1,500 events, 2,585 disk-graphs, 22s on CPU)
- [x] Results saved to `outputs/new_truth_eval/new_truth_comparison.csv`

Side-by-side old-vs-new truth table and analysis: see `docs/findings.md` §1.4. Headline: truth match rate +6.2%, merges halved, no retraining — motivated the full v2 rebuild.

#### 11e: Singleton Origin Analysis ✓

**Goal:** Understand the physics origin of the remaining 48% singletons under calo-entrant truth.

- [x] Analyzed 2,318 singletons from 500 events (1 v2 file)
- [x] Documented in Notion: "GNN Calorimeter Clustering: Singleton Truth Cluster Analysis"

Full singleton-origin characterization (particle breakdown, energy distribution, cross-disk correlation) and conclusions (irreducible pileup, motivation for §7 fringe-hit cleanup): see `docs/findings.md` §1.5.

#### 11f: Full v2 Rebuild (ACTIVE — replaces low-priority retrain)

**Decision (2026-04-05):** Full project rebuild using v2 ROOT files and calo-entrant truth exclusively. All v1 outputs purged from working tree and archived to `~/gnn_v1_results.tar.gz` (35 MB). Source code, scripts, configs, and tests retained.

**Rule: calo-entrant truth (`truth_labels_primary.py`) is the exclusive truth definition.** Old SimParticle-level truth (`truth_labels.py`) is retained for reference only — never used for new work.

**What was purged:**
- `data/processed/` — 41,662 old graph files (437 MB)
- `data/normalization_stats.pt` — old normalization
- `outputs/` — all old runs, evaluations, plots, failure audit, threshold sweeps
- `logs/train.log` — old training log

**What was kept:**
- All source code (`src/`), scripts, configs, tests
- Crystal geometry CSVs (unchanged)
- Split file structure (will be regenerated for v2 paths)

**Rebuild steps:**

#### Step 1: v2 ROOT file reprocessing
- [x] Build modified EventNtuple with `ancestorSimIds` branch (already done in 11a2)
- [x] Create muse tarball: `/exp/mu2e/data/users/wzhou2/museTarball/tmp.Vq0XbRhzu6/Code.tar.bz2`
- [x] Generate 50 per-job FCL files via `generate_fcl` on grid scratch
- [x] Submit 50 grid jobs via `mu2eprodsys` (cluster `90854576`, `--disk=25GB`)
- [x] All 50 jobs completed successfully — 50 ROOT files, 46 GB total
- [x] Copy output to `/exp/mu2e/data/users/wzhou2/GNN/root_files_v2/`
- [x] Document grid workflow in Notion: "How to Submit Jobs to Fermilab's Grid (FermiGrid)"

#### Step 2: Validate v2 files
- [x] All 50 files copied cleanly (2.0 GB each, 97 GB total, 0 empty)
- [x] Verified readable by uproot — extracted 5 events with calo-entrant truth successfully
- [x] v1 ROOT files deleted to free quota (97 GB freed, 150 GB per-user quota on `/exp/mu2e/data/`)

#### Step 3: Data pipeline setup
- [x] Regenerate `splits/{train,val,test}_files.txt` pointing to v2 ROOT file paths
- [x] Update `scripts/build_all_graphs.sh` to use `root_files_v2`
- [x] Update `src/data/dataset.py` to use `truth_labels_primary` (reads `calomcsim.id` + `ancestorSimIds`, builds `calo_root_map`)
- [x] Run unit tests: 88 tests passing
- [x] Committed: `2066e79 Switch data pipeline to v2 ROOT files with calo-entrant truth`

#### Step 4: Build graphs with calo-entrant truth
- [x] Built all graphs: 41,656 total (29,143 train / 5,793 val / 6,720 test) in 310s
- [x] Graph counts match v1 (41,656 vs 41,656) — same events, new truth labels
- [x] Normalization stats computed from train split: `data/normalization_stats.pt`
- [x] Packed: train.pt (119 MB), val.pt (23 MB), test.pt (27 MB)
- [x] Graph gate skipped — graph construction unchanged from v1 (100% pair recall confirmed), only truth labels differ

#### Step 5: Train SimpleEdgeNet (GPU node) ✓
- [x] Train: `python3 scripts/train_gnn.py --config configs/default.yaml --device cuda --epochs 100 --batch-size 64`
- [x] Checkpoint: `outputs/runs/simple_edge_net_v2/` (git hash 9c2cbc9, 2026-04-05)

#### Step 6: Train CaloClusterNet (GPU node) ✓
- [x] Stage 1 (edge only): `python3 scripts/train_gnn.py --config configs/calo_cluster_net.yaml --device cuda --run-name calo_cluster_net_v2_stage1`
- [x] Stage 2/3 not pursued for v2 — v1 showed diminishing returns from node saliency + consistency
- [x] Checkpoint: `outputs/runs/calo_cluster_net_v2_stage1/` (git hash 9c2cbc9, 2026-04-05)

v2 training numbers (best val F1, epochs): see `docs/findings.md` §4.1.

#### Step 7: Threshold tuning (val set) ✓
- [x] Sweep τ_edge for SimpleEdgeNet → frozen **τ=0.26**. Results: `outputs/threshold_sweep_simpleedgenet/`
- [x] Sweep τ_edge for CaloClusterNet → frozen **τ=0.20**. Results: `outputs/threshold_sweep_caloclusternet/`
- [x] Frozen in `configs/default.yaml` (τ=0.26) and `configs/calo_cluster_net.yaml` (τ=0.20)

v2 threshold-tuning numbers: see `docs/findings.md` §4.2.

#### Step 8: Test set evaluation (run ONCE) ✓
- [x] Evaluated both GNNs + BFS on test set: 276,688 events (full test set, 8 files), 481,543 disk-graphs
- [x] SimpleEdgeNet results: `outputs/test_eval_simpleedgenet/`
- [x] CaloClusterNet results: `outputs/test_eval_caloclusternet/`

v2 test-set results and v2-vs-v1 improvement (TMR +6.2%, merges halved): see `docs/findings.md` §4.3–4.4.

#### Step 9: Visualization and analysis ✓
- [x] 3-panel event displays (6 each): `outputs/gnn_cluster_display_simpleedgenet/`, `outputs/gnn_cluster_display_caloclusternet/`
- [x] Failure case displays (6 each): `outputs/debug_simpleedgenet/`, `outputs/debug_caloclusternet/`
- [x] Success case displays (6 each): `outputs/success_simpleedgenet/`, `outputs/success_caloclusternet/`
- [x] Training curve plots in each run dir (`training_curves.png`)
- [x] Failure audit (CaloClusterNet τ=0.20, val set, 5,752 graphs): `outputs/failure_audit/audit_summary.json`

v2 failure audit headline (1,512 merges down from 2,289 in v1, precision 0.957, 93.8% involve singletons): see `docs/findings.md` §4.5.

#### Step 10: Documentation and wrap-up ✓
- [x] CLAUDE.md updated with v2 results (commit `eaad801`)
- [x] v2 optimal thresholds frozen in configs (commit `f94419b`)
- [x] All changes committed on `new-mc-truth` branch (clean working tree)
---

## Milestones


### Milestone A — Data & Graph Gate (must pass before any model training)
- [x] Environment verified and smoke test passing
- [x] Project scaffold and config system in place
- [x] Crystal geometry lookup table built and committed (`data/crystal_geometry.csv`)
- [x] MC truth label strategy implemented — 11 unit tests passing
- [x] BFS baseline metrics extracted — plots in `data/baseline/plots/`, per-cluster detail in `bfs_cluster_summary.csv`
- [x] Train/val/test split frozen and documented — 35/7/8 files (seed=42)
- [x] Graph construction achieves 100% pair recall and 100% cluster connectivity (r_max=210mm)
- [x] Debug event displays show correct truth/BFS/graph topology — `scripts/plot_event_display.py`

### Milestone B — SimpleEdgeNet on Mixed v2 ✓
- [x] SimpleEdgeNet implemented and trains without NaN (CPU dry run verified)
- [x] Build graphs with MC truth labels (all 50 files) — 41,656 graphs
- [x] Full training on GPU — val F1 = 0.925, 18 epochs, 6s/epoch on A100 MIG
- [x] Head-to-head evaluation: GNN vs BFS against MC truth — GNN wins on purity + truth match rate + fewer merges
- [x] GNN cluster purity (0.977) exceeds BFS (0.974)

### Milestone C — SimpleEdgeNet Inference & Final Evaluation ✓
- [x] Inference pipeline implemented (cluster_reco.py + postprocess.py) with unit tests (22 tests, 55 total)
- [x] τ_edge threshold swept on val set; optimal τ_edge=0.34 frozen in `configs/default.yaml` (pairwise F1=0.9326)
- [x] Test-set evaluation run once — GNN competitive with BFS; 46% fewer splits, higher completeness
- [x] Debug event displays: 110/200 scanned events had GNN failures (mostly merges); no pathological modes
- [x] All results saved to `outputs/test_eval/` and `outputs/threshold_sweep/`

### Milestone D — Full model (CaloClusterNet-v1) ✓
- [x] CaloClusterNet implemented (676K params, 3.1× SimpleEdgeNet) — `src/models/calo_cluster_net.py`
- [x] Multi-task training infrastructure (losses, trainer, train script) updated and backward-compatible
- [x] 18 unit tests passing (73 total), inference pipeline compatible with new model
- [x] Shared `build_model()` factory + `--resume` for staged training + Stage 2/3 configs
- [x] Stage 1–3 training on GPU (A100 MIG): Stage 1 F1=0.9252, Stage 2 F1=0.9154, Stage 3 F1=0.9156
- [x] Evaluation scripts updated for CaloClusterNet (tune_threshold, evaluate_test, plot_gnn_clusters)
- [x] Threshold tuning: τ_edge=0.30 frozen (pairwise F1=0.9337); τ_node provided marginal benefit
- [x] Test set: CaloClusterNet purity 0.9731, truth match 88.1%, **fewest merges (2,808)** — marginal gain over SimpleEdgeNet
- [x] Event displays: 6 normal + 6 failure cases, no pathological modes beyond expected merges

### Milestone E — Calo-Entrant Truth & Re-Evaluation ✓
- [x] SimParticle ancestry investigated; root cause identified: EventNtuple drops parent info for calo (11a)
- [x] EventNtuple C++ modified, built, and tested — `ancestorSimIds` populated with full parent chains (11a2)
- [x] 20/50 MCS files reprocessed (18 valid); remaining 30 need lower parallelism or build node (11a3)
- [x] Ancestry validated: zero broken chains, 57% ambiguity reduction, 14% singleton reduction (11a4)
- [x] Calo-entrant truth implemented with 15 unit tests — per-disk root ancestor grouping (11b)
- [x] Truth stats compared: singletons 53% → 48%, clusters reduced 5% (11c via validate_ancestry.py)
- [x] Models re-evaluated: **truth match +6.2%, merges halved, purity +0.015** — no retraining (11d)
- [x] Singleton origin analysis: 66% gamma, 90% at 10-50 MeV, 99% have no parent in calo — irreducible pileup (11e)
- [x] Decision: full v2 rebuild justified — calo-entrant truth is strictly superior

### Milestone F — v2 Full Rebuild (calo-entrant truth) ✓

**2026-04-05:** v1 outputs archived to `~/gnn_v1_results.tar.gz`, working tree cleaned. All new work uses v2 ROOT files + calo-entrant truth exclusively.

- [x] **Step 1:** v2 ROOT file reprocessing — 50/50 files produced via FermiGrid (cluster `90854576`)
- [x] **Step 2:** Validate all 50 v2 files (ancestry populated, no corruption)
- [x] **Step 3:** Data pipeline setup (splits, graph builder → `truth_labels_primary`, configs)
- [x] **Step 4:** Build graphs + normalization + pack — 41,656 graphs
- [x] **Step 5:** Train SimpleEdgeNet — val F1 0.9662, epoch 9, early stop 24
- [x] **Step 6:** Train CaloClusterNet Stage 1 — val F1 0.9609, epoch 13, early stop 28
- [x] **Step 7:** Threshold tuning — SimpleEdgeNet τ=0.26 (F1=0.9734), CaloClusterNet τ=0.20 (F1=0.9748)
- [x] **Step 8:** Test set evaluation — BFS 94.3% TMR, GNN 94.0-94.2% TMR, merges halved vs v1
- [x] **Step 9:** Event displays (normal, failure, success) + failure audit (1,512 merges, 237 splits)
- [x] **Step 10:** Documentation, freeze checkpoints, commit (`eaad801`)

---

### Task 12: Run1B No-Field Evaluation

**Goal:** Evaluate trained GNN models on the Run1B (no magnetic field) dataset to test generalization to different physics scenarios. In Run1B, electrons travel straight (no B-field curving), which changes shower profiles in the calorimeter.

**Dataset:** `FlateMinus-KL / Run1B-005` — 20 NTS ROOT files (~40K events each) on tape. MCS art files available at `/pnfs/mu2e/tape/phy-sim/mcs/mu2e/FlateMinus-KL/Run1Bah_best_v1_4-001/art/` (20 files, 24 GB, run 001450).

**Approach:** Reprocess MCS art files through modified EventNtuple (same tarball as v2) to get `ancestorSimIds`, then evaluate with calo-entrant truth for apples-to-apples comparison with MDC2025 v2 results. No retraining — tests how well models trained on with-field data generalize.

**Space management:** MDC2025 v2 ROOT files deleted 2026-04-13 (97 GB freed, now 9 GB / 150 GB quota). Reproducible via FermiGrid.

#### 12a: Reprocess Run1B Art Files via FermiGrid

- [x] Located MCS art files: `/pnfs/mu2e/tape/phy-sim/mcs/mu2e/FlateMinus-KL/Run1Bah_best_v1_4-001/art/` (20 files, 1.2 GB each, 24 GB total)
- [x] Verified muse tarball still exists: `/exp/mu2e/data/users/wzhou2/museTarball/tmp.Vq0XbRhzu6/Code.tar.bz2` (736 MB)
- [x] Created output dir: `/exp/mu2e/data/users/wzhou2/GNN/root_files_run1b/`
- [x] Created input file list: `root_files_run1b/input_art_files.txt` (20 art files)
- [x] Created grid submission script: `root_files_run1b/grid_submit.sh` (workflow project `eventntuple_run1b`, dsconf `run1b_ancestry`)
  - Fixed multiple issues during debugging: `setup mu2etools` needed for `generate_fcl`; `htgettoken` needs `--vaultserver htvaultprod.fnal.gov`; `generate_fcl` needs `--merge-factor=1`; output `000/` dir must be tarred to `.tgz` for `--fcllist`; `--disk=10GB` (5GB caused hold errors in v2 campaign); `mu2eprodsys` flags must be on one line. See Notion guide: https://www.notion.so/3390da4f6e8c81cfa488cba50e48829a
  - Used dedicated scratch dir `/pnfs/mu2e/scratch/users/wzhou2/fcl_run1b/` to avoid collision with v2 `000/`
- [x] Grid jobs submitted: cluster `27355798` (20 jobs, 2026-04-13)
- [x] **All 20 jobs failed** — `PrescaleFilterFraction` dictionary missing. Run1B art files (`Run1Bah_best_v1_4`) were produced with Offline ≥v13_02_00; our tarball was built against an older Offline that lacks `DataProducts/inc/PrescaleFilterFraction.hh`. Output files were 519 bytes (empty ROOT).
- [x] **Fix:** Pulled latest Offline (`git pull origin main`, 628 commits) in working_dir. Need to rebuild + new tarball.
- [x] Rebuilt on build node: fixed CRV API (`GetInfoFromCrvRecoPulse` now takes `ewmh` as 9th arg) and trigger API (`getTrigPathName` → `getTrigPathNameByIndex` again). Build: `al9-prof-e29-p092`.
- [x] New tarball: `/exp/mu2e/data/users/wzhou2/museTarball/tmp.iTQSPdWZBd/Code.tar.bz2`
- [x] Resubmitted: cluster `27583402` (20 jobs, 2026-04-13)
- [x] All 20 jobs completed successfully. Output: 20 files, ~20 MB each (405 MB total, smaller than original 440 MB NTS because calo-only FCL skips track branches). 40,148 events per file.
- [x] Copied to `/exp/mu2e/data/users/wzhou2/GNN/root_files_run1b/`. `ancestorSimIds` branch verified present.

**Expected output size:** Existing Run1B NTS files (without ancestry) are ~440 MB each (8.6 GB total for 20 files). With `ancestorSimIds` added, expect ~500 MB each, ~10 GB total. Well within 141 GB free quota.

#### 12b: Validate Run1B v2 Files ✓

- [x] All 20 files have `calomcsim.ancestorSimIds` branch (verified via uproot)
- [x] 40,148 events per file; files are ~20 MB (calo-only, no track branches)

#### 12c: Evaluate Models on Run1B ✓

- [x] Wrote `scripts/evaluate_run1b.py` — evaluates BFS + both GNNs in one pass
- [x] Run on 10,000 events (500/file × 20 files), 8,641 disk-graphs, 58.7s on CPU
- [x] Results saved to `outputs/run1b_eval/`

Run1B results table and energy/multiplicity binning: see `docs/findings.md` §5.

#### 12d: Run1B vs MDC2025 Comparison ✓

Full comparison and conclusions (no pileup → no singletons → trivially easy, GNNs generalize perfectly, MDC2025 remains the meaningful benchmark): see `docs/findings.md` §5.

---

### Milestone G — Run1B No-Field Generalization Test

- [x] Run1B MCS art files located (20 files, 24 GB)
- [x] FermiGrid reprocessing set up (grid_submit.sh, input list, same tarball)
- [x] MDC2025 v2 ROOT files deleted to free quota (97 GB → 9 GB used)
- [x] Grid jobs submitted and completed — cluster `27583402`, 20 files (405 MB total)
- [x] Run1B files validated — `ancestorSimIds` present, 40K events/file
- [x] Evaluation complete — 10,000 events, 8,641 disk-graphs: all methods >99.6% match rate
- [x] Compared with MDC2025: Run1B is trivially easy (no pileup), GNNs generalize perfectly

---

### Task 13: Cluster-Level Physics Evaluation (Energy, Centroid, Time)

**Goal:** Evaluate BFS and GNN clustering not just on match rates, but on the downstream-relevant quantities: total energy, centroid position, and time. These are the three fields that the reconstruction chain actually consumes.

**Motivation:** Match rate and purity measure hit grouping quality, but two algorithms with similar match rates could produce different energy/position/time accuracy for the matched clusters. A merge adds extra energy and pulls the centroid; a split loses energy. These residuals quantify the downstream physics impact directly.

**Background — how Offline computes cluster quantities:**
- **Energy:** Sum of `eDep` over all hits in the cluster.
- **Centroid (cog3Vector):** Energy-weighted center of gravity of hit positions, in the disk front-face (DiskFF) coordinate frame. Linear energy weighting by default.
- **Time:** Time of the most energetic hit (seed hit), NOT energy-weighted average.

**Approach:** For each matched reco↔truth cluster pair (using existing energy-weighted greedy matching), compute:

1. **Energy residual:** `ΔE = E_reco - E_truth` and `E_reco / E_truth`
   - Perfect clustering → ΔE ≈ 0. Merges → ΔE > 0 (extra hits). Splits → ΔE < 0 (missing hits).
2. **Centroid displacement:** `Δr = |centroid_reco - centroid_truth|` (Euclidean distance in x-y)
   - Compute both energy-weighted centroids from the hit positions in each cluster.
3. **Time residual:** `Δt = t_reco - t_truth`
   - Use time of the most energetic hit in each cluster (matching Offline convention).

**Metrics to report (for BFS, SimpleEdgeNet, CaloClusterNet):**
- Mean, median, std, and 90th percentile of |ΔE|, Δr, |Δt|
- 1D histograms of ΔE, Δr, Δt for each method (overlay BFS vs GNN)
- Stratify by cluster energy bin (<50 MeV, 50–200 MeV, >200 MeV) and multiplicity (1-hit, 2–3 hits, 4+ hits)
- Fraction of clusters with |ΔE| > 10 MeV, Δr > 10 mm, |Δt| > 1 ns (quality cut fractions)

**Data:** Use val split ROOT files (v2 with ancestry, calo-entrant truth). Same events as threshold tuning — no test set contamination.

#### 13a: Implement evaluation script ✓
- [x] Write `scripts/evaluate_cluster_physics.py`
  - [x] For each event: build truth clusters (calo-entrant), run BFS reco, run GNN inference for both models
  - [x] Match reco↔truth using existing greedy energy-weighted matching
  - [x] For each matched pair: compute ΔE, centroid displacement, Δt
  - [x] Output per-cluster CSV + summary statistics + histograms
  - [x] Results to `outputs/cluster_physics_eval/`

#### 13b: Generate comparison plots ✓
- [x] Overlay histograms: BFS vs SimpleEdgeNet vs CaloClusterNet for ΔE, Δr, Δt
- [x] Energy-binned and multiplicity-binned breakdowns (|dE| and dr vs truth energy)
- [x] Summary comparison bar chart

**Note on data:** MDC2025 v2 ROOT files were deleted 2026-04-13 to free quota. Val split (7 files) recopied from FermiGrid scratch (`/pnfs/mu2e/scratch/users/wzhou2/workflow/eventntuple_ancestry_v2/outstage/90854576/`). Grid output still available for all 50 files.

All-cluster val-set results, energy- and multiplicity-binned breakdowns: see `docs/findings.md` §6.2. Headline: CaloClusterNet wins every metric on all-cluster physics.

#### 13c: Downstream-relevant evaluation (E_reco >= 50 MeV) ✓

- [x] Re-evaluated with E_reco >= 50 MeV cut (clusters that actually enter track finding)
- [x] Results in `outputs/cluster_physics_eval/residual_plots_downstream.png`

Downstream val-set table and root cause (BFS wins on downstream because its ExpandCut is a natural firewall against stray pileup absorption): see `docs/findings.md` §6.3–6.4. Motivated Task 14 fringe-hit cleanup.

---

### Milestone H — Cluster-Level Physics Evaluation ✓

- [x] Evaluation script implemented and tested on val split
- [x] Energy, centroid, and time residuals computed for BFS + both GNNs
- [x] Comparison plots and summary table produced
- [x] Results documented in CLAUDE.md
- [x] Downstream-relevant evaluation reveals BFS wins on E_reco >= 50 MeV clusters
- [x] Root cause identified: GNNs lack BFS's ExpandCut firewall against stray hit absorption

---

### Task 14: Post-Clustering Fringe Hit Removal

**Goal:** Add a cleanup step after GNN cluster reconstruction that removes low-energy fringe hits from clusters, analogous to BFS's `ExpandCut` threshold. This should improve GNN performance on downstream-relevant clusters (E_reco >= 50 MeV).

**Motivation:** The GNN edge classifier sometimes absorbs stray low-energy hits (~10-30 MeV singletons) into adjacent clusters. BFS avoids this because its `ExpandCut` stops expansion through low-energy hits. These absorbed strays add ~20 MeV energy bias and ~33 mm centroid displacement to clusters that enter track finding. A post-hoc cleanup should recover GNN performance.

**Evaluation:** Re-run `evaluate_cluster_physics.py` with downstream cut (E_reco >= 50 MeV) and compare against BFS baseline.

#### 14a: Energy-based expand_cut ✓

**Approach:** Analogous to BFS's `ExpandCut` (1 MeV in Offline). Before building the adjacency for connected components, suppress edges where BOTH endpoints have energy below a threshold. This prevents low-energy pileup hits from bridging between clusters, while still allowing them to join a cluster through an edge to a high-energy hit. Implemented as `expand_cut` parameter in `reconstruct_clusters()`. No retraining needed.

**Key insight from Offline code:** BFS's ExpandCut doesn't remove low-energy hits from clusters — it prevents them from acting as bridges. A 0.5 MeV hit is still added to the cluster, but BFS won't expand through it to neighbors. Our expand_cut mirrors this: edges between two low-energy hits are suppressed before connected components.

- [x] Implement `expand_cut` parameter in `reconstruct_clusters()` in `src/inference/cluster_reco.py`
- [x] Sweep expand_cut on val set: [None, 0.5, 1, 2, 5, 10, 15, 20, 30] MeV
- [x] Evaluate with E_reco >= 50 MeV cut

Full sweep results, both methods' optimal thresholds, and the trade-off analysis (downstream wins but splits explode 20–33×): see `docs/findings.md` §7.1. Verdict: abandoned in favor of §7.2/§7.3.

Results in `outputs/cluster_physics_eval/` and `outputs/eval_with_expandcut/`.

#### 14b: Learned node saliency for bridge-hit identification

**Goal:** Train the node saliency head to identify stray pileup hits that would contaminate downstream clusters, then use it during inference to prevent those hits from bridging between clusters. Unlike expand_cut (which uses a hard energy threshold), this is a learned approach that sees the full local context from message passing.

**Why the old saliency label was useless:** The current `y_node` label is `1` for any non-ambiguous hit, `0` for ambiguous. Result: 98.5% salient, 1.5% non-salient. The model trivially predicts all-1, node F1=1.000, and learns nothing about stray hits. Singletons (46.6% of truth clusters) are all labeled salient — the exact hits we want to identify are invisible to the loss.

**New label: "multi-hit cluster member"**
- `y_node = 1` if the hit belongs to a truth cluster with >= 2 hits (multi-hit)
- `y_node = 0` if the hit is a singleton truth cluster OR ambiguous/unassigned
- This directly labels the stray pileup hits (10-30 MeV singletons) as non-salient
- Expected distribution: ~52% salient (multi-hit) vs ~48% non-salient (singletons + ambiguous) — much better class balance than 98.5/1.5

**Inference strategy — saliency-reweighted cluster physics (not clustering changes):**
Bridge suppression (zeroing edges) and hit pruning (removing hits) both fail: bridge suppression misses the actual failure mode (non-salient→salient edges), and pruning destroys TMR by removing all singleton matches. The solution: **leave the clustering unchanged, but recompute cluster energy/centroid using only salient hits.** Standard clustering metrics (TMR, purity, completeness, splits, merges) are identical to the baseline. Only the downstream physics quantities (E_reco, centroid, time) improve because stray pileup contamination is excluded from the computation.

**Steps:**

##### 14b-1: Change y_node label to "multi-hit cluster member" ✓
- [x] Modify `src/data/dataset.py`: `y_node = 1` if `hit_truth_cluster[i]` has >= 2 hits in the graph, else 0
- [x] Rebuild packed graphs with new labels: `bash scripts/build_all_graphs.sh && python3 scripts/pack_graphs.py`
- [x] Verify new label distribution: expect ~52% salient, ~48% non-salient

##### 14b-2: Create Stage 2 config for new saliency training
- [x] Create `configs/calo_cluster_net_saliency.yaml` — lambda_node=0.3, resume from Stage 1, lr=5e-4
- [x] Add class-balanced pos_weight for node loss (since ~52/48 split, pos_weight ≈ 0.92)

##### 14b-3: Implement saliency-based bridge suppression in inference
- [x] Add `saliency_bridge_cut` option to `reconstruct_clusters()`: suppress edges where both endpoints have saliency below τ_node (learned analog of expand_cut)
- [x] This differs from current τ_node behavior which zeros edges when EITHER endpoint is non-salient

##### 14b-4: Train on GPU node ✓
- [x] Train: `python3 scripts/train_gnn.py --config configs/calo_cluster_net_saliency.yaml --device cuda --run-name calo_cluster_net_v2_saliency --resume outputs/runs/calo_cluster_net_v2_stage1/checkpoints/best_model.pt`
- [x] Checkpoint: `outputs/runs/calo_cluster_net_v2_saliency/checkpoints/best_model.pt`
- [x] Node saliency label distribution (val set): 75.9% salient (multi-hit), 24.1% non-salient (singleton/ambiguous)
- [x] CUDA warmup: 70s first epoch, then ~15s/epoch on A100 MIG (batch_size=32)

Saliency training numbers (edge F1 0.962, node F1 0.888 with P=1.000, R=0.800), plus the critical role of perfect node precision: see `docs/findings.md` §4.1 and §7.2.

##### 14b-5: Sweep and evaluate ✓
- [x] Sweep τ_edge on val set (no τ_node): frozen **τ_edge=0.14**. Results: `outputs/threshold_sweep_saliency/`
- [x] Frozen τ_edge=0.14 in `configs/calo_cluster_net_saliency.yaml`
- [x] Bridge suppression (pre-clustering): marginal merge reduction, does not improve downstream physics
- [x] Hit pruning (post-clustering): improves downstream |dE| but destroys TMR (94%→55%)
- [x] **Saliency-reweighted physics:** clustering unchanged, cluster energy/centroid recomputed from salient hits only. CCN-Saliency beats BFS on all downstream metrics while standard clustering metrics are identical to CaloClusterNet.

Saliency reweighting details, full all-cluster and downstream tables, and the modest-but-zero-trade-off verdict: see `docs/findings.md` §7.2.

Results in `outputs/cluster_physics_eval_saliency/`.

#### 14c: BFS-style traversal on GNN edges (breakthrough) ✓

**Goal:** Replace connected components with BFS traversal that mirrors Offline's ClusterFinder — combining GNN edge classification (better at deciding which hits belong together) with BFS's ExpandCut traversal (better at controlling how clusters grow).

**Key insight:** Connected components has no concept of traversal order — a hit is either connected or not. BFS's ExpandCut says "you can join the cluster, but if your energy is below threshold, you can't expand to your neighbors." This prevents low-energy stray hits from bridging between clusters while keeping them as cluster members. The previous expand_cut approach (deleting edges) was a crude approximation that fragmented clusters.

**Implementation:** `_bfs_expand_cut()` in `src/inference/cluster_reco.py`:
1. Build adjacency list from thresholded GNN edges
2. Seed from highest-energy hits (like BFS's EminSeed)
3. BFS expansion: add neighbors to cluster, but only continue expanding from hits with E >= bfs_expand_cut
4. Low-energy hits join but don't recruit — they're leaves in the traversal

- [x] Implement `_bfs_expand_cut()` and `bfs_expand_cut` parameter in `reconstruct_clusters()`
- [x] Sweep bfs_expand_cut on val set for both SEN and CCN: optimal EC=10 MeV for both
- [x] Full cluster physics evaluation: all clusters + downstream (E_reco >= 50 MeV)

Val sweep table, EC=10 sweet-spot analysis (29% drop in downstream abs(dE) with negligible standard-metric cost), test-set results (4,000 events, standard/all-cluster/downstream/signal-region tables), and the CCN+BFS10 verdict: see `docs/findings.md` §7.3–7.4.

Results in `outputs/cluster_physics_eval_bfs_test/`.

---

### Milestone I — Fringe Hit Cleanup ✓

- [x] 14a: Energy-based expand_cut (edge deletion) — beats BFS on downstream but destroys standard metrics (splits 7-20x, TMR -4-11%)
- [x] 14b: Learned node saliency — saliency-reweighted physics beats BFS on downstream (|dE| 0.825 vs 0.848) with identical standard metrics, but improvement is modest
- [x] **14c: BFS-style traversal on GNN edges — both GNNs beat BFS on every metric.** Test set: CCN+BFS10 DS |dE| 0.616 (-27%), SEN+BFS10 DS |dE| 0.651 (-22%). Signal region (95-110 MeV, 47,279 clusters): CCN+BFS10 mean |dE| 0.210 vs BFS 0.368 (-43%), mean dr 0.460 vs 0.559 (-18%). No retraining needed.
- [x] Test set evaluation run once (276,688 events, 481,543 disk-graphs). Results in `outputs/cluster_physics_eval_test_full/` (CSV redirected to `/exp/mu2e/data/users/wzhou2/GNN/cluster_physics_eval_test_full/` via symlink due to 25 GB `/exp/mu2e/app` quota).

---

### Task 15: ONNX Export for Offline Integration

**Goal:** Prepare CaloClusterNet for deployment in Mu2e Offline (C++) via ONNX Runtime, following the pattern in Andy Edmonds's draft PR `Mu2e/ArtAnalysis#4` (TrackQuality ONNX integration).

**Scope:** Python-side only — produce a validated `.onnx` file and document its tensor interface. The downstream Offline C++ module (new art producer, SConscript entry, FHiCL config) waits on Andy's PR landing and the central `onnxruntime` muse install.

**Design decision — Option A (deploy-only wrapper):** leave the research code untouched (keep `NodeSaliencyHead`, multi-task loss, `tau_node` plumbing in `src/` and scripts). Add a thin `CaloClusterNetDeploy` wrapper that loads a `CaloClusterNet` checkpoint, bypasses the node head at inference, and returns edge logits as a plain tensor suitable for ONNX export. Rationale: preserves reproducibility of §7.2 saliency experiments; keeps the exported ONNX graph lean (single output `edge_logits`, ~1% fewer FLOPs, no untrained head weights leaking to C++ callers); node-saliency removal is a trivial git-revert away if needed.

#### 15a: Deploy wrapper — edge-only inference ✓

- [x] Add `src/models/calo_cluster_net_deploy.py` — `CaloClusterNetDeploy`
  - Composition over inheritance: wrapper holds references to `node_encoder`, `edge_encoder`, `mp_blocks`, `edge_head` of a loaded `CaloClusterNet` (no weight copy).
  - Forward takes `(x, edge_index, edge_attr)` tensors directly (not a PyG `Data` object — `torch.onnx.export` can't trace `Data`)
  - Forward returns `edge_logits` tensor of shape `(E,)` — no dict, no `node_logits`
  - Classmethod `from_checkpoint(path)` reads the run-dir `config.yaml`, rebuilds the full `CaloClusterNet`, loads the state_dict, and returns the wrapper in eval mode.
- [x] Write unit tests `tests/test_calo_cluster_net_deploy.py` — 9 tests, all passing:
  - Output is a plain tensor (not a dict), dtype float32, shape `(E,)`
  - Wrapper has no `node_head` attribute
  - Eval-mode determinism
  - Dynamic graph sizes (N, E from 1 to 300)
  - Edge-logit parity with full `CaloClusterNet` on synthetic graphs (`atol=1e-6`)
  - Submodules are shared by reference, not copied
  - Loads from the real `v2_stage1` checkpoint and produces finite output
  - Parity against the full model on real packed val graphs (`data/processed/val.pt`)
- [x] Ancillary cleanup: the `CaloClusterNetV1` → `CaloClusterNet` rename (commit `78af52e`) had missed the saved run-dir `config.yaml`, five output directories (`outputs/{debug,success,gnn_cluster_display,threshold_sweep,test_eval}_caloclusternetv1`), `outputs/failure_audit/audit_summary.json`, and `outputs/run1b_eval/run1b_results.csv`. All swept — no `CaloClusterNetV1`/`caloclusternetv1` anywhere in the project. Total test count: 97 (was 88 before 15a).

#### 15b: ONNX export script ✓

- [x] Add `scripts/export_onnx.py`
  - Loads `CaloClusterNetDeploy.from_checkpoint(...)`
  - Dummy input: first non-trivial val disk-graph from `data/processed/val.pt` (already z-score normalised)
  - `torch.onnx.export` with `dynamic_axes = {"x": {0: "N"}, "edge_index": {1: "E"}, "edge_attr": {0: "E"}, "edge_logits": {0: "E"}}`
  - Opset 17, `do_constant_folding=True`
  - Writes `outputs/onnx/calo_cluster_net_v2_stage1.onnx` (2.60 MB). `outputs/onnx/` gitignored.
- [x] New dependencies: `onnx 1.21.0` (required by `torch.onnx.export`) and `onnxruntime 1.25.0` (for 15c). Installed via `pip install --user` into `/nashome/w/wzhou2/.local/...`, same convention as `torch_geometric`.
- [x] PyG `scatter` op in `EdgeAwareResBlock` **traces cleanly at opset 17** with PyTorch 2.5.1 — no `torch.index_add_` / `torch.jit.script` workarounds needed. Two cosmetic `TracerWarning`s from `src, dst = edge_index` unpacking in `layers.py:74` and `heads.py:74`; verified by parity test below that they don't affect output correctness.
- [x] Ad-hoc parity spot check (5 val graphs with (N, E) spanning (3, 6) to (23, 56)): max abs diff PyTorch vs ONNX Runtime = **5.25e-06**. Dynamic axes confirmed working — the same exported graph handled every shape tested. Full parity sweep moves to 15c.

#### 15c: Parity validation (PyTorch ↔ ONNX Runtime Python) ✓

- [x] Add `scripts/validate_onnx.py`
  - Loads packed val graphs (`data/processed/val.pt`), runs both runtimes on every non-trivial graph, reports max/mean/p50/p90/p99/p99.9 edge-logit abs diffs.
  - Also counts per-edge threshold-decision flips at `tau_edge=0.20` (proxy for cluster-assembly parity — if logits match and no thresholds flip, the deterministic downstream post-processing agrees by construction, so a separate `evaluate_test.py`-style rerun is not needed to prove equivalence).
  - Exits non-zero if max abs diff > `--tol` (default 1e-5) or any flip is observed.
- [x] Ran over full val split: **5,793 graphs, 166,342 edges, CPU, tolerance 1e-5**
  - Max abs diff: **9.060e-06** (under 1e-5)
  - Mean abs diff per edge: 1.333e-06; p99.9 = 5.722e-06
  - **Zero threshold flips at τ=0.20** across all 166K edges → cluster reconstruction (symmetrize → BFS traversal → cleanup) is provably byte-identical
  - Dynamic axes verified across N ∈ [2, 65] and E ∈ [2, 180] (same exported graph)
  - CPU timing: PyTorch median 11.00 ms/graph, ONNX Runtime median 0.88 ms/graph → **ONNX is ~12.5× faster** on this CPU. (Side result useful for the integration meeting — hint at what the C++ `art::EDProducer` will cost per event.)

#### 15d: Document deployment tensor spec ✓

- [x] Add `docs/onnx_deployment.md` — the interface contract for the C++ caller. Seven sections:
  1. Model artifact (path, opset 17, regenerate/validate commands)
  2. Tensor interface (exact input/output names, shapes, dtypes)
  3. Normalisation — the six node and eight edge z-score stats with their exact mean/std values from the train split, so the C++ side doesn't need to open the `.pt` blob
  4. Graph construction (upstream) — one graph per disk, r_max=210mm, dt_max=25ns, k_min=3, k_max=20, directed edges; notes on whether to reuse Offline `cal.neighbors()`/`nextNeighbors()` vs. porting the cKDTree path
  5. Cluster assembly (downstream) — CCN+BFS10 recipe as pseudocode: sigmoid → symmetrise → threshold `τ=0.20` → BFS traversal seeded from highest-E hits with `bfs_expand_cut=10 MeV` → cleanup `min_hits=2`, `min_energy_mev=10`
  6. Parity proof — the 15c numbers (9.06e-06 max diff, zero threshold flips on 166K edges, 12.5× CPU speedup)
  7. Open items for the integration meeting — muse onnxruntime status, module boundary decision, neighbour-list strategy, stats sidecar format, version-string policy

---

### Milestone J — ONNX Deployment Prep (gate before Offline integration)

- [x] `CaloClusterNetDeploy` wrapper implemented with unit tests passing (15a)
- [x] ONNX export works on CPU; any PyG workarounds documented (15b) — no workarounds needed at opset 17
- [x] PyTorch ↔ ONNX Runtime parity verified to 1e-5 on val set (15c) — max abs diff 9.06e-06 over 166K edges, zero threshold flips at τ=0.20
- [x] Deployment tensor spec documented in `docs/onnx_deployment.md` (15d)
- [ ] Reviewed with Sophie + Andy Edmonds in the ONNX integration meeting (blocked on scheduling); Andy's `Mu2e/ArtAnalysis#4` PR landed and central `onnxruntime` muse install available

---

### Task 16: C++ Offline Integration

**Goal:** Land the trained CaloClusterNet model in Mu2e Offline as an `art::EDProducer` so the GNN clustering can run inside production reconstruction **alongside** the existing BFS `CaloClusterMaker` — both producers run, downstream consumers pick by `(module_label, instance_name)`. The GNN does not replace BFS.

**Pre-meeting design note:** `docs/offline_integration.md` — the integration picture, the design decisions for the meeting, the open questions, and the pre-meeting state. Read alongside this checklist.

**Decisions taken (pre- and at-meeting; meeting held 2026-04-29 with Sophie + Andy):**
- Repo location: `Offline/CaloCluster/` (same package as BFS). [pre-meeting]
- Graph-construction implementation: brute-force pairwise distance loop in C++ (faithful port of `scipy.spatial.cKDTree` behaviour; N≤~65 per disk-event makes O(N²) trivially cheap). [pre-meeting]
- Coexistence with BFS: GNN producer alongside BFS, distinct instance name `"GNN"`, BFS untouched. [pre-meeting]
- Artifact runtime location: `art::ConfigFileLookupPolicy` (Mu2e standard pattern; matches reviewer feedback on `Mu2e/ArtAnalysis#4`). FHiCL gives a relative path; framework resolves it. [pre-meeting]
- **Module boundary: split.** `CaloHitGraphMaker` emits a `CaloHitGraphCollection` per event; `CaloClusterMakerGNN` consumes it and emits `CaloClusterCollection` under instance `"GNN"`. Sub-decisions still open in `docs/offline_integration.md` §2.2 (graph data-product schema, persistence, helper-class home, naming). [meeting]
- **onnxruntime / build:** link against the central muse onnxruntime via the `u092` qualifier. Sophie shared the `u092` muse manifest 2026-04-30 (Slack DM); install under a `muse/` directory in the working environment, then `mu2einit && muse setup -q u092`. The "stub the ONNX call" interim option is retired. [meeting]
- **Version-string carrier:** ONNX `metadata_props`. Drives 16b. [meeting]

**Patterns to mirror / pre-empt from `Mu2e/ArtAnalysis#4`** (see `docs/offline_integration.md` §2.9 for full list): use `ConfigFileLookupPolicy` for artifact paths; document constructor member-init order; don't mutate ONNX output; no dead helpers; standard SConscript indentation; one `'onnxruntime'` dependency line.

**Scope:** C++ side — the new art module (graph construction, ONNX inference, cluster assembly), build/FHiCL wiring, and parity validation against the Python reference. Module boundary, target repo, and graph-construction strategy land at the ONNX integration meeting (Milestone J open item) and refine 16d–16f below.

**Repo target:** Offline (`Offline/CaloCluster/`) is the architectural home — GNN clustering is a *reconstruction* step like BFS, not an analysis step. ArtAnalysis is where Andy's `Mu2e/ArtAnalysis#4` first wires up `onnxruntime`, so the build pattern is more mature there in the short term; the meeting decides whether to mirror that pattern directly into Offline or take a temporary detour through ArtAnalysis.

**Hard dependencies (others' work):**
- Andy Edmonds's `Mu2e/ArtAnalysis#4` PR (TrackQuality ONNX integration) lands — the pattern this module follows.
- Central `onnxruntime` install in muse (Andy + Ray) — so the module builds against a stock muse setup rather than a local install.

#### 16a: Normalisation sidecar (pre-meeting, no dependencies) ✓

**Goal:** Export the train-split z-score statistics to a plain-text/JSON file alongside the `.onnx`, so the C++ module reads 28 floats without needing a LibTorch dependency to open `data/normalization_stats.pt`.

- [x] Wrote `scripts/export_norm_stats.py` — reads `data/normalization_stats.pt`, emits `outputs/onnx/calo_cluster_net_v2_stage1.norm.json` with `schema_version`, canonical `node_features`/`edge_features` lists, and `{node_mean, node_std, edge_mean, edge_std, node_count, edge_count}`.
- [x] Updated `docs/onnx_deployment.md` §3 — added sidecar schema, regen command, and a `JSON key` column in the feature tables so C++ implementers can index by name.
- [x] Added `tests/test_export_norm_stats.py` — 8 tests covering payload shape, canonical name match, bit-exact tensor↔list round-trip, count types, dimension-mismatch errors, sidecar↔torch-blob equivalence, and end-to-end save/load. Full suite now 105 tests (was 97).

#### 16b: Version-string guard ✓

**Goal:** Embed a short version identifier in the `.onnx` (decided 2026-04-29: ONNX `metadata_props` — travels with the artifact, no risk of desync with a separate sidecar) so the C++ module rejects mismatched models at load time and prevents silent tensor-layout drift.

- [x] Decide carrier — `metadata_props` (decided at the 2026-04-29 meeting).
- [x] `scripts/export_onnx.py` writes `metadata_props["model_version"] = "calo-cluster-net-v2-stage1"` after `torch.onnx.export` via `stamp_metadata_props()`. Re-export landed: `outputs/onnx/calo_cluster_net_v2_stage1.onnx` 2.60 MB.
- [x] Documented in `docs/onnx_deployment.md` §7 — C++ asserts at session load against FHiCL `expected_model_version`.
- [x] Tests in `tests/test_export_onnx.py` (5 tests; idempotent overwrite, foreign-key preservation, canonical-name match with sidecar). Full suite now 110 tests passing; ONNX↔PyTorch parity preserved (max diff 9.06e-06, zero threshold flips at τ=0.20).

#### 16c: ONNX integration meeting — held 2026-04-29 with Sophie + Andy

**Goal:** Lock down the remaining open design decisions before C++ implementation starts.

- [x] Module boundary → **split** (graph producer + cluster producer with an intermediate `CaloHitGraphCollection`).
- [x] Interim build stance → **link the central muse onnxruntime via the `u092` qualifier**. Sophie shared the `u092` muse manifest 2026-04-30; install under `muse/` and activate with `muse setup -q u092`. Stub option retired.
- [x] Artifact runtime location → `ConfigFileLookupPolicy` confirmed; in-tree subdirectory still TBD (likely `Offline/CaloCluster/data/` or shared `Offline/Mu2eData/`).
- [x] Version-string carrier → **ONNX `metadata_props`** (drives 16b).
- [x] Ownership: Sam writes 16d–16g (EDProducers, graph builder, cluster assembler, FHiCL, parity tests); Andy reviews against the `ArtAnalysis#4` pattern; Sophie reviews calorimeter-group-facing pieces.
- [x] Pre-decisions confirmed (repo location `Offline/CaloCluster/`, brute-force pairwise graph construction, BFS coexistence with distinct `"GNN"` instance name).
- [x] Capture decisions back into `docs/offline_integration.md` and refine 16d–16h sub-tasks.

**Newly opened post-meeting (track in `docs/offline_integration.md` §3):**
- Sophie is writing **generic ONNX utils** (translating Leo's code). Decide whether `CaloClusterMakerGNN` consumes those utils for session loading / sidecar reading / version checking, or rolls its own. Ask Sophie before starting 16d-cluster.
- **PR sequencing:** EventNtuple `ancestorSimIds` patch must land in `Mu2e/EventNtuple` before any Offline PR depending on the branch (Sophie asked for the PR 2026-04-30).
- **Training repo:** Sophie pointed at `Mu2e/MLTrain` for training code. Not a deployment blocker but tracks alongside.

#### 16d: art::EDProducer skeletons (split design) ✓

**Goal:** Stand up two new EDProducers and one new data product following Andy's `ArtAnalysis#4` pattern. The graph producer builds the per-disk graph; the cluster producer loads the ONNX session at construction and consumes the graph product.

##### 16d-product: `CaloHitGraph` data product ✓

- [x] Schema: flat tensors `x` (`std::vector<float>`, length 6N), `edge_index` (`std::vector<int64_t>`, length 2E), `edge_attr` (`std::vector<float>`, length 8E), plus `(N, E)` headers and per-node `std::vector<art::Ptr<CaloHit>>` back-references.
- [x] Persistence: transient — declared via `produces<CaloHitGraphCollection>()` in `CaloHitGraphMaker`, no `classes_def.xml` registration.
- [x] Host directory: `Offline/RecoDataProducts/inc/CaloHitGraph.hh` (header-only).

##### 16d-graph: `CaloHitGraphMaker_module.cc` ✓

- [x] Added at `Offline/CaloCluster/src/CaloHitGraphMaker_module.cc` — `produce()` partitions the input `CaloHitCollection` by disk and runs `GnnGraphBuilder` once per disk; emits a `CaloHitGraphCollection` (one entry per disk that has hits).
- [x] Constructor reads the norm sidecar (`Offline/ConfigTools/inc/ConfigFileLookupPolicy.hh` → resolved path → `GnnGraphBuilder::loadStatsFromJson`) so the emitted tensors are already z-score normalised.
- [x] FHiCL parameter set: `caloHitCollection`, `normSidecar` (resolved by `ConfigFileLookupPolicy`), graph hyperparameters (`rMax=210`, `dtMax=25`, `kMin=3`, `kMax=20`).
- [x] SConscript: auto-picked up by `helper.make_plugins`. Builds `libmu2e_CaloCluster_CaloHitGraphMaker_module.so`.
- [x] `produce()` is fully implemented (graph construction is in 16e); not stubbed.

##### 16d-cluster: `CaloClusterMakerGNN_module.cc` (model-agnostic, instanced via FHiCL) ✓

Single C++ class. Each FHiCL instance specifies which model to load, what version to expect, and the recipe values for that model. Production declares one instance; A/B studies declare multiple. See `docs/offline_integration.md` §2.2.

- [x] Added at `Offline/CaloCluster/src/CaloClusterMakerGNN_module.cc` — constructor loads ONNX session via the C++ API (`Ort::Env`, `Ort::SessionOptions`, `Ort::Session`, `Ort::AllocatorWithDefaultOptions`), reads `metadata_props["model_version"]` and asserts equal to FHiCL `expectedModelVersion` (16b); reads `metadata_props["node_features"]`/`["edge_features"]`, splits on commas, and asserts equal to the FHiCL `expectedNodeFeatures`/`expectedEdgeFeatures` lists (16j). Member-declaration order (`env_` → `sessionOptions_` → `session_` → `allocator_`) documented in the header.
- [x] FHiCL parameter set:
  - `modelPath` (resolved by `ConfigFileLookupPolicy`)
  - `expectedModelVersion` (string; `metadata_props` assertion)
  - `expectedNodeFeatures`, `expectedEdgeFeatures` (string lists; `metadata_props` assertion)
  - `outputInstance` (string; default `"GNN"`)
  - recipe values: `tauEdge` (no default — must be set per model), `bfsExpandCut=10.0`, `minHits=2`, `minEnergyMeV=10.0`
- [x] SConscript: added `'onnxruntime'` to plugin deps. `ldd` confirms the built `.so` links against `libonnxruntime.so.1`.
- [x] `produce()` consumes the `CaloHitGraphCollection` and emits an empty `CaloClusterCollection` under `outputInstance` — full inference + CCN+BFS10 assembly lands with 16f.
- [ ] Decide whether to consume Sophie's generic ONNX utils (session loader, sidecar reader, version check) once those exist, or keep the rolled-our-own version.

#### 16e: Graph construction in C++ (called from 16d-graph) ✓

**Goal:** Port `src/data/graph_builder.py` to C++ as `Offline/CaloCluster/{src,inc}/GnnGraphBuilder.{cc,hh}`; output `(x, edge_index, edge_attr)` tensors matching the ONNX interface. Lives behind `CaloHitGraphMaker` (16d-graph).

- [x] Per-disk hit collection from `CaloHitCollection` (looked up via `Calorimeter::crystal(crystalID).diskID()` and `localPosition()`).
- [x] Node-feature computation (6 features per `docs/onnx_deployment.md` §3 tables).
- [x] Adjacency: brute-force pairwise distance loop with `r_max=210 mm` cut (faithful to `scipy.spatial.cKDTree` behaviour at N ≤ ~65).
- [x] Time filter (`|dt|<25 ns`), kNN fallback (`k_min=3`, mirrors Python's `k_query = min(k_min*3, n)` semantics including the self-skip behaviour), degree cap (`k_max=20` nearest-by-distance).
- [x] Edge-feature computation (8 features).
- [x] Z-score normalisation using sidecar stats from 16a — `loadStatsFromJson` parses the JSON via nlohmann/json, validates `schema_version=1`, asserts canonical `node_features`/`edge_features` names, and asserts positive `std` entries.
- [x] Edges sorted lexicographically by (src, dst) after dedup, matching Python's `np.unique`-on-encoded-value ordering — needed for byte-exact parity in 16g.

#### 16f: Cluster assembly in C++ (called from 16d-cluster) ✓

**Goal:** Port `src/inference/cluster_reco.py` (`reconstruct_clusters` / `_bfs_expand_cut`) to C++; consume ONNX edge logits + the `CaloHitGraph` from 16d-graph, emit `CaloClusterCollection`. Lives behind `CaloClusterMakerGNN` (16d-cluster).

- [x] Sigmoid + symmetrise directed scores — running mean over each unordered `{i, j}` pair via `std::map<std::pair<int,int>, ...>`.
- [x] Threshold + adjacency-list build at FHiCL `tauEdge` (model-specific; `0.20` for CCN).
- [x] BFS traversal with `bfsExpandCut=10 MeV`, seeded by descending hit energy (`std::iota` + lambda sort). Hits below cut join the cluster but cannot recruit (Python `_bfs_expand_cut` semantics).
- [x] Cleanup: `minHits=2`, `minEnergyMeV=10` + contiguous re-labelling in first-appearance order.
- [x] Per-cluster postprocessing in `CaloClusterMakerGNN::produce()`: energy = Σ eDep, energyErr = sqrt(Σ eDepErr²), seed = highest-energy hit (sets time/timeErr), 3D centroid via the existing `ClusterUtils(cal, hits, Linear)` helper (matches BFS module's convention). Cluster→hit refs resolved through `graph.caloHitPtrs`.
- [x] ONNX inference loop in `produce()`: per-disk `Ort::Session::Run` with `Ort::Value::CreateTensor` views (no copy of graph tensors), input/output names cached in the constructor.

`Offline/CaloCluster/{inc,src}/GnnClusterAssembler.{hh,cc}` — port of `src/inference/cluster_reco.py`. `CaloClusterMakerGNN_module.cc::produce()` is fully implemented.

#### 16g: C++↔Python parity validation (two-stage + end-to-end)

**Goal:** Prove the C++ pipeline reproduces Python edge logits and cluster labels to documented tolerance on the val set. Split design adds a graph-product parity check at the module boundary.

- [x] Dump val graph payloads (`edge_index`, `edge_logits` from PyTorch, `energies`, Python `cluster_labels`) to a portable JSON via `scripts/dump_parity_payloads.py`. Default output `tests/parity/calo_cluster_net_v2_stage1.parity.json`. 100 graphs, 1,147 nodes, 2,768 edges.
- [x] Stage 2 — cluster-assembler parity: standalone `testGnnClusterAssembler` binary (built via `make_bin` in `Offline/CaloCluster/src/SConscript`) loads the JSON, replays `GnnClusterAssembler::assemble`, asserts byte-exact match against Python `cluster_labels`. **Result: 100 / 100 graphs match byte-exactly** (zero mismatched nodes). Run binary at `build/al9-prof-e29-u092/Offline/bin/testGnnClusterAssembler`.
- [x] Stage 1 + Stage 3 — end-to-end parity: real `mu2e` art job reads MCS art files (`/pnfs/mu2e/persistent/.../FlateMinusMix1BBTriggered/MDC2025af_best_v1_1/art/...`), runs `CaloHitGraphMaker` + `CaloClusterMakerGNN`, and the new `CaloHitGraphParityDump` analyzer dumps the GNN cluster labels per event-disk to a TTree. `scripts/compare_parity_dump.py` re-runs the same `CaloHit`s through the Python pipeline and asserts the partition matches. **Result on 50 events / 100 disk-graphs / 8,502 hits: zero mismatches.** This implicitly covers Stage 1 — any divergence in graph construction would propagate to mismatched labels.
- [x] Fail loudly if any bound is exceeded.

Reference run:
```
mu2e -c Offline/CaloCluster/fcl/from_mcs-gnn-test.fcl \
     -s <mcs.art> -n 50 -T parity_dump.root
python3 scripts/compare_parity_dump.py parity_dump.root
```

#### 16h: Production FHiCL wiring ✓

**Goal:** Hook both new producers into the production sequence alongside the existing BFS `CaloClusterMaker` -- both clustering chains run, neither is removed.

- [x] `Offline/CaloCluster/fcl/prolog.fcl` extended with a `CaloClusterGNN` block that defines `caloHitGraphMakerGNN` + `caloClusterMakerGNN` ready to drop into any reco sequence (`@local::CaloClusterGNN.caloHitGraphMakerGNN`, `@local::CaloClusterGNN.caloClusterMakerGNN`, or the `@sequence::CaloClusterGNN.Reco` bundle).
- [x] `Offline/CaloCluster/fcl/from_mcs-gnn-prod.fcl` -- production-style FCL that runs the GNN chain on MCS art-format input and emits an output art file with the new `CaloClusterCollection` under instance `"GNN"`. `CaloHitGraphCollection` is dropped from the output via `outputCommands` (transient by design).
- [x] Smoke-tested end-to-end on a real MCS file: 5 events produced an art file containing both `mu2e::CaloClusters_CaloClusterMaker__Reconstruct` (existing BFS, carried over) and `mu2e::CaloClusters_caloClusterMakerGNN_GNN_GnnProd` (new GNN). Existing BFS-reading analyses keep working unchanged; GNN consumers select via `(module_label="caloClusterMakerGNN", instance_name="GNN")`.
- [x] Frozen recipe values baked into the prolog defaults (`tauEdge=0.20`, `bfsExpandCut=10.0`, `minHits=2`, `minEnergyMeV=10.0`); each is FHiCL-overridable for studies and per-instance for swappable models.
- [x] CPU overhead noted in the prolog comment: ONNX inference at ~0.9 ms/disk-graph, BFS still runs. Not a default-chain change beyond adding the two GNN producers.

#### 16i: Export SimpleEdgeNet to ONNX ✓

**Goal:** Produce a second deployable `.onnx` artifact so production FCLs can swap models without code changes (see `docs/offline_integration.md` §2.2). Mirrors 15a-15c for the `SimpleEdgeNet` checkpoint.

- [x] `src/models/simple_edge_net_deploy.py` -- `SimpleEdgeNetDeploy` wrapper. Composition over inheritance; re-implements the message-passing loop with explicit tensor arguments (no PyG `Data` object needed). Classmethod `from_checkpoint(path)` rebuilds the full model from run-dir `config.yaml` and loads weights.
- [x] `scripts/export_onnx.py` extended with `--model {ccn,sen}` and per-model presets (checkpoint path, output path, `model_version`). `python3 scripts/export_onnx.py --model sen` writes `outputs/onnx/simple_edge_net_v2.onnx` (0.84 MB) with `metadata_props["model_version"] = "simple-edge-net-v2"` plus the canonical feature-name strings.
- [x] `scripts/validate_onnx.py` extended with the same `--model` flag and per-model presets (incl. tau_edge and tol). SEN parity over 5,793 val graphs / 166,342 edges: max abs diff **2.44e-03** vs the 5e-3 tol (SEN's raw logits run ~1000x larger in magnitude than CCN's, so the same fp32 precision shows up as a larger absolute number). **Zero threshold flips at `tau_edge=0.26`** -- cluster reconstruction will be byte-identical to PyTorch.
- [x] `Offline/CaloCluster/data/simple_edge_net_v2.onnx` staged for `ConfigFileLookupPolicy`.

#### 16i: Export SimpleEdgeNet to ONNX

**Goal:** Produce a second deployable `.onnx` artifact so production FCLs can swap models without code changes (see `docs/offline_integration.md` §2.2). Mirrors 15a–15c for the `SimpleEdgeNet` checkpoint.

- [ ] Add `src/models/simple_edge_net_deploy.py` — `SimpleEdgeNetDeploy` wrapper. SEN has no node head, so the wrapper is a thin pass-through that returns `edge_logits` as a plain tensor (not a dict). Classmethod `from_checkpoint(path)` rebuilds the full model from run-dir `config.yaml` and loads weights.
- [ ] Unit tests in `tests/test_simple_edge_net_deploy.py` — output type/shape/dtype, eval-mode determinism, dynamic graph sizes, parity against the original `SimpleEdgeNet` on synthetic + real packed val graphs.
- [ ] Update `scripts/export_onnx.py` (or fork into `scripts/export_onnx_sen.py`) to support `--model {ccn,sen}` selection. Default model_version `simple-edge-net-v2`. Output `outputs/onnx/simple_edge_net_v2.onnx`.
- [ ] Run parity: `scripts/validate_onnx.py --model sen --checkpoint ...` over the val split, max abs diff target 1e-5, zero threshold flips at `tau_edge=0.26`.
- [ ] Document the SEN artifact in `docs/onnx_deployment.md` §1 (model artifact table).

#### 16j: Feature-spec `metadata_props` + graph-maker handshake

**Goal:** Extend 16b's `metadata_props` with `node_features` and `edge_features` strings, so the cluster module can validate at job start that the loaded model expects exactly the features the graph maker is wired to emit. Catches silent wiring drift if anyone later trains with a different feature set.

- [x] `scripts/export_onnx.py` writes `metadata_props["node_features"] = "log_e,t,x,y,r,e_rel"` and `["edge_features"] = "dx,dy,d,dt,dlog_e,asym_e,logsum_e,dr"` alongside `model_version` (one combined `stamp_metadata_props()` call).
- [x] Documented in `docs/onnx_deployment.md` §7 (the deployment-contract section), with the validation table linking FHiCL ↔ ONNX ↔ graph-maker.
- [x] Tests cover the canonical-name agreement between `scripts/export_onnx.py` and `scripts/export_norm_stats.py` (one source of truth — drift between them would now break a test).
- [ ] In `CaloClusterMakerGNN` (16d-cluster) constructor: read `metadata_props["node_features"]` / `["edge_features"]` from the loaded session; assert string-equal to the FHiCL `expected_node_features` / `expected_edge_features`. Mismatch → abort job with a clear error before the first event.
- [ ] In `CaloHitGraphMaker` (16d-graph) FHiCL: expose `node_features` / `edge_features` parameters whose values are the canonical names; emit tensors in that order; default values match the train-time spec from `data/normalization_stats.pt`.

---

### Milestone K — C++ Offline Integration

- [x] 16a: Normalisation sidecar exported and documented
- [x] 16b: Version-string guard wired into `metadata_props` + spec (Python side complete; C++ session-load assertion lands with 16d-cluster)
- [x] 16c: ONNX integration meeting held 2026-04-29 — split design, central muse onnxruntime via `u092`, `metadata_props` version carrier, repo `Offline/CaloCluster/` confirmed; Sam owns 16d–16g, Andy + Sophie review
- [x] 16d: Graph + cluster EDProducers + `CaloHitGraph` data product scaffolded (cluster module is model-agnostic, instanced via FHiCL); both `.so` plugins build under u092
- [x] 16e: Graph construction implemented inside `CaloHitGraphMaker` (full Python port; metadata_props handshake wired through to 16d-cluster)
- [x] 16f: Cluster assembly implemented inside `CaloClusterMakerGNN` (`GnnClusterAssembler` helper + ONNX inference + CaloCluster construction in `produce()`)
- [x] 16g: All stages validated — Stage 2 (assembler-only, 100 val graphs) byte-exact; Stages 1 + 3 (real `mu2e` art job on MCS files, 50 events / 100 disk-graphs / 8,502 hits) byte-exact.
- [x] 16h: Both new producers wired into production FHiCL (`prolog.fcl` + `from_mcs-gnn-prod.fcl`) and smoke-tested -- output art file carries both BFS and GNN `CaloClusterCollection`s
- [x] 16i: SimpleEdgeNet exported (`outputs/onnx/simple_edge_net_v2.onnx`, 0.84 MB) and parity-validated (max abs diff 2.44e-03, zero threshold flips at tau_edge=0.26 over 166K val edges). Deployable alongside CCN via FHiCL.
- [ ] 16j: `metadata_props` carries `node_features` / `edge_features` (Python side complete); graph maker ↔ cluster module handshake at job start (C++ side, with 16d)

---

### Task 17: Calo-Entrant Pair-Recall Regate

**Goal:** Verify that the §2 graph construction gate (`r_max=210mm` → 100% pair recall) still holds under calo-entrant truth. The §2 result (34 graphs, 95 multi-hit clusters) was established before the truth redefinition in §1; calo-entrant truth merges hits from different SimParticles in the same shower, so the maximum same-cluster pair distance can only grow. If any pair now exceeds 210 mm, the §4 v2 metrics (TMR/RMR) are capped below 100% by graph topology rather than by the model — and Task 18's higher-pileup study would inherit the same cap.

**Approach (cheap, fast — packed graphs):** Operate on `data/processed/{train,val,test}.pt`. The graph builder preserves all hits as nodes (only edges are filtered at `r_max`), and `hit_truth_cluster` is stored per node under calo-entrant truth. Node positions live in `x[:, 2]` (x) and `x[:, 3]` (y) — packed graphs store *raw physical* features (mm), normalization is applied at training time inside `CaloGraphDataset`.

#### 17a: Same-cluster pair-distance scan ✓

- [x] `scripts/regate_pair_recall.py` written and run on all three v2 splits
- [x] Result (2026-05-07): **gate fails materially** — 1.88–1.90% of multi-hit clusters have at least one pair > 210 mm (consistent across train/val/test). Pair-rate over 210 mm: 0.87–0.90%. Max distances reach ~1.1 m (far-flung deposits from same shower, e.g. escaped-photon reconversion at distant crystal). 99th percentile of max-pair-dist ≈ 245 mm; 95th percentile ≈ 172 mm.
- [x] Stratified by cluster multiplicity: 2-hit 0.63% pair-rate, 3–4 hit 0.77%, 5+ hit 1.02% — bigger clusters fail slightly more, as expected.
- [x] Results written to `docs/findings.md` §2.1.

#### 17b: Decision ✓ (accept cap; do not rebuild)

- [x] §4 v2 TMR is 94.3%; topology cap is ~98% → **topology is not the dominant bottleneck**, model behavior is. Rebuilding graphs + retraining purely for this would gain at most ~1.9% TMR even if topology were fully removed.
- [x] Bumping `r_max` to cover 99.9th percentile (~700 mm) would exceed the disk diameter — all-pairs connected, destroys locality structure. Not viable.
- [x] Bumping `r_max` to cover 99th percentile (~245 mm) is cheap (~24 edges/graph more, like the original 150 → 210 bump) but only moves the cap from 98% → ~99%; defer unless §4 metrics improve enough to make topology the new bottleneck.
- [x] **Decision: accept the cap.** Document in §2.1, flag for Task 18 re-evaluation under denser pileup. No graph rebuild, no retrain.

---

### Milestone L — Graph Gate Re-verification ✓

- [x] 17a: Pair-distance scan complete on existing v2 packed graphs
- [x] 17b: Decision recorded — gate fails at ~1.9% of multi-hit clusters, but topology cap (98%) is well above current model TMR (94.3%); accepted, no rebuild.

---

### Task 18: Run1B + Pileup Generalization & Retraining Decision

**Goal:** Determine whether models trained on MDC2025 (with-field, with-pileup) generalize to the no-field-with-pileup regime, or whether retraining is required. Without the magnetic field, low-momentum charged backgrounds are no longer curled away from the calorimeter, so pileup density per disk is expected to be substantially higher than MDC2025. §5 already showed zero-shot generalization from with-field-with-pileup to no-field-no-pileup; this task closes the missing quadrant.

**Prerequisite:** Task 17 must complete first — the answer to "what is the right `r_max`?" feeds directly into graph construction here.

**Dataset:** `FlateMinusMixLow-KL / Run1Baf_best_v1_4-001` — 192 art MCS files on tape at `/pnfs/mu2e/tape/phy-sim/mcs/mu2e/FlateMinusMixLow-KL/Run1Baf_best_v1_4-001/art/`. Standard NTS without ancestry available at `/pnfs/mu2e/tape/phy-nts/nts/mu2e/FlateMinusMixLow-KL/Run1B-004/root/` (192 files, ~9.5 GB). See `docs/findings.md` §5.1 for the full candidate-dataset table and naming-convention decoder.

**Strategy:** Stage gates A → B → C → D, cheapest first. Decide on retraining (Stage D) based on Stage C truth-aware results — not preemptively.

#### 18a: Stage A — Regime characterization ✓

- [x] Wrote `scripts/characterize_run1b_pileup.py`: streams MixLow standard NTS from `/pnfs/`, builds graphs at `r_max=210 mm`, computes hit/edge/energy distributions; reads MDC2025 train baseline from packed `data/processed/train.pt`. Standard Run1B-004 NTS lacks `crystalPos_`; falls back to `crystal_map`.
- [x] Ran on 3 files × 500 events → 2,810 disk-graphs.
- [x] Result: hit density **2.4× median, 3.8× at p95**; BFS clusters/disk **~2×**; per-hit energies slightly lower (~0.93×). Edges scale linearly with hits — graph builder not pathological. Full table in `docs/findings.md` §5.2.
- [x] Verdict: regime is materially harder than MDC2025; existing model unlikely to be optimal. Proceed to Stage B (inference comparison) and Stage C (truth-aware eval).

#### 18b: Stage B — Inference vs BFS reference ✓

- [x] `scripts/stageB_inference_diagnostics.py` written; runs SEN (`simple_edge_net_v2`, τ=0.26) + CCN-saliency (`calo_cluster_net_v2_saliency`, τ=0.14) on the same 3-file × 500-event sample, using MDC2025 normalization stats.
- [x] **No catastrophic failure.** Edge sigmoid scores are decisive (medians 0.80 SEN / 0.89 CCN, not collapsed to 0.5); ~80% of in-graph edges classified positive (consistent with the graph builder already filtering edges to time-coincident close-by hits where most pairs are same-cluster).
- [x] **GNN clusters mirror BFS at the count/size level** — same count distribution (med 10, p95 56), same size distribution (med 2 hits). This is consistent with §4.3 on MDC2025 where bare GNN tracked BFS within 0.3 pp on TMR; the GNN's wins were in split count, completeness, and §7 post-processing — none of which Stage B can probe without truth.
- [x] Edge feature distributions mostly in MDC2025 z-score range; mild tails on dt (p99=4.67) and dist (p99=2.57) reflect the wider no-field timing window. Not OOD-catastrophic.
- [x] Diagnostics + raw arrays saved to `outputs/run1b_mixlow_stageB/stageB_diagnostics.npz`. Full table in `docs/findings.md` §5.3.

#### 18c: Stage C — Local ancestry reprocessing + truth-aware eval ✓

- [x] First reprocess attempt failed: `illegal production target version specified = 4` — Offline source (HEAD `a58c90ea5`, today) had v4 stickman registered, but May-3 build was stale (source updated since then). Triggered incremental rebuild via `run_u092_build.sh`; completed in 3 min on `mu2ebuild02.fnal.gov`.
- [x] Reprocessed 3 MCS files of `FlateMinusMixLow-KL/Run1Baf_best_v1_4-001` (~22s each post-rebuild, parallelized 2 at a time). Output: `/exp/mu2e/data/users/wzhou2/GNN/root_files_run1b_mixlow/` (3 files × ~30 MB, total 7,764 events with `calomcsim.ancestorSimIds`).
- [x] Ran `scripts/evaluate_run1b.py --root-dir <new dir> --output-dir outputs/run1b_mixlow_eval --n-events 1000` → 5,617 disk-graphs, 98,590 truth clusters in 88s.
- [x] First pass (bare connected-components): GNNs lose to BFS on splits/merges/TMR. Re-ran with `--bfs-expand-cut 10` to add §7-style post-processing for fair comparison. Modified `evaluate_run1b.py` to accept `--bfs-expand-cut` flag.
- [x] **Decision gate triggered → Stage D required.** With CCN+BFS10 the picture is more nuanced: TMR ties BFS (96.4%/96.4%) and CCN+BFS10 *beats* BFS in the 1-hit pileup-cluster bin (92.9 vs 92.7%, where 39% of truth clusters live). Merges near-parity (CCN+BFS10 = 2,465 vs BFS = 2,412). But splits regress sharply: BFS=647 vs **CCN+BFS10=1,403** — a 2.2× BFS win, opposite §4.3 where GNN halved BFS splits. The MDC2025 splits-reduction story doesn't transfer to this regime.
- [x] Outputs persisted: `outputs/run1b_mixlow_eval/` (bare) and `outputs/run1b_mixlow_eval_bfs10/` (with BFS10). Full per-method table + per-bin breakdown in `docs/findings.md` §5.4.
- [x] **Cluster-physics evaluation** (§6-style, MixLow val 29 files × 500 events, 27K disk-graphs, ~454K matched pairs per method): all 4 GNN variants (SEN, SEN+BFS10, CCN, CCN+BFS10) **lose to BFS** on mean &#124;dE&#124;, mean dr, and outlier rate. Best GNN (CCN+BFS10) gives mean &#124;dE&#124; = 0.487 MeV vs BFS = 0.444 MeV (10% worse) — clean inversion of §7.4's GNN −19% advantage on MDC2025. Outputs: `outputs/run1b_mixlow_phys_baseline/`. Decisive evidence for retraining.

#### 18d: Stage D — Local batch reprocess (no FermiGrid needed)

The MixLow MCS files are small (130–330 MB vs Run1Bah's 1.2 GB) and the Offline rebuild done in 18c on `mu2ebuild02.fnal.gov` reprocesses one file in ~22 s. So the full 192-file batch fits comfortably in a local `xargs -P 20`-style parallel run (~4–5 min) — no grid, no tarball staging, no `mu2eprodsys`. Driver: `/exp/mu2e/data/users/wzhou2/GNN/root_files_run1b_mixlow/run_batch.sh`.

- [x] Local batch: 189 files × ~22 s with `PARALLEL=20` finished in a few minutes; 192/192 produced (all 3 stage-C files + 189 fresh)
- [x] Validation: all 192 files have `calomcsim.ancestorSimIds`; total **619,268 events** across all files (~3,225 events/file average)
- [x] Splits: `splits/run1b_mixlow_{train,val,test}_files.txt` — 134 / 29 / 29 (70 / 15 / 15), seed=42 shuffle of the 192 files
- [x] Config: `configs/run1b_mixlow_default.yaml` — same schema as `default.yaml`, but points at `data/processed_run1b_mixlow/`, `data/normalization_stats_run1b_mixlow.pt`, and the new splits files (so MDC2025 artifacts are not overwritten)
- [x] Built graphs (134/29/29 files × 500 events/file) via `scripts/build_all_mixlow.sh` (in tmux session `mixlow_build`):
  - train: **123,390 graphs** (897 MB) — 4.2× the v2 MDC2025 train (29,143 graphs)
  - val: **26,703 graphs** (192 MB)
  - test: **26,738 graphs** (195 MB)
- [x] Norm stats: `data/normalization_stats_run1b_mixlow.pt` (computed from MixLow train, 2.3 KB)
- [x] Bundles packed at `data/processed_run1b_mixlow/{train,val,test}.pt`
- [x] CCN/SEN configs ready: `configs/run1b_mixlow_default.yaml`, `configs/calo_cluster_net_run1b_mixlow.yaml`, `configs/calo_cluster_net_saliency_run1b_mixlow.yaml`

#### 18e: Stage E — Retrain on MixLow (gated on 18d)

- [x] Train SEN on MixLow train split (`configs/run1b_mixlow_default.yaml`, GPU node) — best val F1 0.9708, epoch 2, 17 epochs, run dir `outputs/runs/simple_edge_net_run1b_mixlow/`
- [x] Train CCN-saliency with the same staged recipe — Stage 1 best val F1 0.9718 (ep 2, 17 epochs); saliency Stage 2 best val F1 0.9720 (ep 11, 26 epochs), node F1 0.902 (P=1.000, R=0.821). Run dirs `outputs/runs/calo_cluster_net_run1b_mixlow_stage1/`, `..._saliency/`.
- [x] Threshold tune τ_edge on MixLow val (`scripts/tune_threshold.py --config configs/run1b_mixlow_default.yaml`) — SEN **τ=0.34** (was 0.26 on MDC2025), CCN-saliency **τ=0.32** (was 0.14 on MDC2025). Frozen in respective MixLow configs.
- [x] **Sweep `bfs_expand_cut` at frozen τ_edge** (new step — EC=10 was tuned on MDC2025 §7.3, MixLow has different pileup density / per-hit E so the optimum can shift). Implemented `scripts/sweep_bfs_expand_cut.py` — sweeps EC over {None,3,5,7,10,12,15,20} on val, reports standard match metrics + downstream |dE|/dr (E_reco≥50 MeV cut on val.pt), recommends EC by best DS |dE| within 0.5pp of bare-CC TMR. **MixLow result: EC=10 sweet spot for both SEN and CCN-saliency** (matches §7.3 finding) — TMR cliff at EC=12+. Frozen `bfs_expand_cut: 10.0` in each MixLow config.
- [x] Test-set evaluation: ran §5.4-style eval (`evaluate_run1b.py --bfs-expand-cut 10 --ignore-tau-node`) on MixLow test (29 files × 500 events, 27,161 disk-graphs). **All §5.4 regressions inverted** — both retrained GNNs beat BFS on every metric: TMR 96.5/96.9/97.1% (BFS/SEN/CCN), splits −19% (SEN), merges −22% (CCN). Outputs `outputs/run1b_mixlow_eval_retrained_bfs10/`.
- [x] Cluster-physics evaluation (§6-style) on MixLow test (`evaluate_cluster_physics.py --ignore-tau-node`). **CCN+BFS10 mean |dE| = 0.281 MeV vs BFS = 0.446** (−37%, beats plan target of ≤0.40); mean dr = 1.038 mm vs BFS = 1.328 (beats target ≤1.30). vs §5.4 with MDC2025 weights: 0.487 → 0.281 (−42%). Outputs `outputs/cluster_physics_eval_mixlow_retrained/`.
- [x] Cross-evaluate retrained model on MDC2025 test (`data/processed/test.pt` with MDC2025 norm stats). **Retraining specializes** — MixLow CCN-saliency on MDC2025 has 8.6× more splits and +85% worse downstream |dE| than v2 MDC2025-trained baseline. EC=10 *hurts* on MDC2025 for the MixLow-trained model (DS |dE| 1.47 → 2.81). Production deployment needs regime-specific model selection or joint training on both. Outputs `outputs/cross_eval_*_on_mdc2025/`.
- [x] Recorded final metrics in `docs/findings.md` §5.5.

#### 18f: Augmented features (gated on 18e — only if a gap remains)

If retraining-alone closes the splits regression and MixLow physics meets MDC2025 picture, skip this task. Otherwise — and only on calo-only fields:

- [ ] Add `calohits.nSiPMs_` to graph builder as an extra node feature (1 channel) — pure quality flag, single-SiPM hits skew toward pileup splinters
- [ ] Optionally add `calohits.timeErr_` and `calohits.eDepErr_` as 2 more node-feature channels and as denominators for normalized-residual edge features
- [ ] Rebuild graphs, compute fresh norm stats, retrain SEN + CCN with augmented input dims
- [ ] Ablation table: existing 6+8 vs augmented 9+10 on MixLow test set; retain only if ≥1 pp TMR or ≥10% splits improvement over the 6+8 baseline
- [ ] If retained: update `metadata_props` in ONNX export (Task 16j) so the C++ graph maker emits matching tensors

---

### Milestone M — No-Field-With-Pileup Closure

- [x] 18a: Regime characterized; hit/edge density vs MDC2025 quantified
- [x] 18b: Existing model behavior on MixLow probed via BFS comparison
- [x] 18c: Truth-aware eval on small sample done; retraining decision made with data
- [x] 18d: Local batch reprocess complete; v2-style MixLow ROOT files in hand; splits + config in place; graphs packed (123K train / 26K val / 26K test)
- [x] 18e: Retrained models evaluated on MixLow test (TMR, RMR, splits/merges, cluster-physics); cross-eval on MDC2025 confirms regime specialization
- [ ] 18f: (gated) Augmented features attempted only if 18e leaves a measurable gap — **not gated open**: 18e closes the §5.4 regression decisively (CCN+BFS10 |dE| 0.281 vs BFS 0.446, mean dr 1.038 vs 1.328). Skip 18f.
- [x] Sections §5.2, §5.3, §5.4, §5.5 in `docs/findings.md`

---

## Notes

- **PyG environment:** `pyenv` is a shell function; scripts must `source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh` before calling it. The `mu2einit` alias only works in interactive shells. All required packages confirmed in `ana 2.6.1`.
- **torch-cluster NOT available:** Use `scipy.spatial.cKDTree` for all spatial graph construction.
- **GPU:** CUDA not available on login node. Use GPU node or condor batch for training.
- **Graph unit:** One graph per disk, not per event.
- **MC truth:** MDC2025-002 has `calohitsmc` and `caloclustersmc` branches. All 50 files copied locally.
- **Crystal positions:** MDC2025-002 provides `calohits.crystalPos_.fCoordinates.{fX,fY,fZ}` directly, reducing dependence on Task 0's C++ dump.
- **Branch format differences:** MDC2025-002 uses `cog_.fCoordinates.{fX,fY,fZ}` instead of `cog_.data[3]`.
- **Comparison policy:** Keep the existing cluster matching rule, split policy, and test-once discipline unchanged for any new models.
- **Muse build (2026-04-04):** On al9, `setupmu2e-art.sh` handles spack + muse automatically — no separate `$SPACK_ROOT` sourcing. Working dir needs EventNtuple + Offline + Production (cloned for `epilog.fcl`). Build: `al9-prof-e29-p087`.
- **Art `-T` flag:** Use `-T` (short form), NOT `--TFileName`, to override TFileService output path. Long form silently fails.
- **`from_mcs-calo-only.fcl`:** Custom FCL that skips TrkQual/TrkPID to avoid ArtAnalysis dependency. Lives in `EventNtuple/fcl/`. Suitable for producing calo-only ntuples from MCS art files.
- **MCS art files:** 50 files (not 141 as originally estimated) in `/pnfs/.../MDC2025af_best_v1_1/art/`.
- **Batch reprocessing:** `run_all.sh` uses xargs -P 20 for local parallel processing. Must run inside muse environment. Use tmux for persistence. ~70 min per file, ~3-4 hours total for 50 files.
- **FermiGrid submission gotchas (2026-04-13):** `generate_fcl` requires `setup mu2etools` (not just `mu2egrid`); needs `--merge-factor=1` for RootInput FCL; outputs to `000/` dir which must be `tar czf`'d into `.tgz` for `--fcllist`. `mu2eprodsys` flags must be on one unbroken line (line breaks silently drop flags). `htgettoken` needs `--vaultserver htvaultprod.fnal.gov`. Use `--disk=10GB` for EventNtuple jobs (5GB caused hold errors). Full guide in Notion: "How to Submit Jobs to Fermilab's Grid".
- **Run1B (no-field) dataset:** `FlateMinus-KL/Run1B-005`, 20 NTS files (~40K events each, ~440 MB each). MCS art files at `/pnfs/mu2e/tape/phy-sim/mcs/mu2e/FlateMinus-KL/Run1Bah_best_v1_4-001/art/` (20 files, 1.2 GB each). No `ancestorSimIds` in standard NTS — requires reprocessing through modified EventNtuple.
- **Run1B + pileup dataset (Task 18):** `FlateMinusMixLow-KL/Run1Baf_best_v1_4-001`, 192 MCS art files at `/pnfs/mu2e/tape/phy-sim/mcs/mu2e/FlateMinusMixLow-KL/Run1Baf_best_v1_4-001/art/`. Standard NTS at `/pnfs/mu2e/tape/phy-nts/nts/mu2e/FlateMinusMixLow-KL/Run1B-004/root/` (~9.5 GB, no ancestry). `-KL` = KalmanLine (no-field) reco. Naming-convention decoder + alternative candidates (`MixLowTriggerable`, `OnSpillTriggerable`) in `docs/findings.md` §5.1.
- **MDC2025 v2 ROOT files deleted (2026-04-13):** Freed 97 GB from `/exp/mu2e/data/`. Reproducible via FermiGrid (cluster `90854576`, tarball + `grid_submit.sh` still exist).
