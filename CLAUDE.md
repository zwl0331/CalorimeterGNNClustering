# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Attribution rule

**Never include Claude authorship or co-authorship in git commits.** Do not add `Co-Authored-By`, `Generated-By`, or any similar trailer referencing Claude, AI, or Anthropic. Commit messages should read as if written by the user. This applies to all commits — new, amended, or rebased.

# GNN Calorimeter Clustering

## Project overview

Graph Neural Network-based clustering algorithm for the Mu2e electromagnetic calorimeter.
Replaces the existing BFS seed-based algorithm (`CaloClusterMaker` in Offline).
Approach: **edge classification** (SimpleEdgeNet, CaloClusterNet).
Built with PyTorch Geometric.

Working directory: `/exp/mu2e/app/users/wzhou2/projects/calorimeter/GNN/`
Plan: `/nashome/w/wzhou2/.claude/plans/wild-leaping-micali.md`

---

## Environment

**Always activate before running any Python:**
```bash
source setup_env.sh   # works in both interactive and non-interactive shells
```

`setup_env.sh` handles the full activation: sources `setupmu2e-art.sh`, activates `ana 2.6.1` (with fallback for non-interactive shells), and adds user site-packages + project src to `PYTHONPATH`.

`mu2einit` and `pyenv` are interactive shell functions — they do NOT work in scripts or `bash -c`. The `setup_env.sh` fallback sources the conda activate script directly.

**`set -e` caveat:** Mu2e env scripts (`setupmu2e-art.sh`, conda `activate`) return non-zero exit codes internally. Any script using `set -e` must place it **after** `source setup_env.sh`, not before. See `scripts/build_all_graphs.sh` for the correct pattern.

**Large file globs:** `data/processed/` can contain 50K+ `.pt` files. Shell globs (`rm *.pt`) will hit "Argument list too long". Use `find -delete` instead.

**Confirmed package versions (ana 2.6.1):**
- Python 3.12.13, torch 2.5.1, torch_geometric 2.7.0, uproot 5.7.2, scipy 1.17.1
- `torch_geometric` is installed in user site-packages (`/nashome/w/wzhou2/.local/lib/python3.12/site-packages/`)
- `torch-cluster` is NOT installed — `radius_graph` and `knn_graph` from PyG are unavailable.
- Use `scipy.spatial.cKDTree` for all spatial graph construction (see `src/data/graph_builder.py`).

**GPU:** Not available on the login node. Training must run on a GPU node (A100 MIG ~20 GB available).

**Long-running jobs:** Use `tmux` to persist jobs across disconnects. Convention:
- `tmux new -s <name>` to create, `tmux attach -t <name>` to reconnect, `tmux list-sessions` to check
- Active session `reprocess` may be running batch EventNtuple reprocessing (check with `tmux list-sessions`)
- Batch scripts should `tee` output to a log file for monitoring without attaching

**Mu2e build environment (muse):** For building C++ packages (EventNtuple, Offline, Production), `mu2einit` and `muse` are interactive shell functions. On AlmaLinux 9, `setupmu2e-art.sh` handles spack + muse setup automatically — no separate `$SPACK_ROOT` sourcing needed:
```bash
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh
cd /exp/mu2e/app/users/wzhou2/working_dir
muse setup
muse build -j64   # use many cores on build nodes
```

**Muse build gotchas (discovered 2026-04-04):**
- `--TFileName` does NOT work with `mu2e` command — use `-T` (short flag) instead.
- EventNtuple `getTrigPathNameByIndex` → `getTrigPathName`: API renamed in current CVMFS Offline. Already fixed in local checkout.
- `from_mcs-mockdata.fcl` requires `ArtAnalysis` package (for TrkQual/TrkPID). Use `from_mcs-calo-only.fcl` to skip track analysis and avoid this dependency.
- `Production` repo must be cloned into working dir for `epilog.fcl` dependency.
- `epilog.fcl` sets `TFileService.fileName: "/dev/null"` by default — FCL lines after the include override this.

**EventNtuple working directory:** `/exp/mu2e/app/users/wzhou2/working_dir/` — contains local checkouts of `EventNtuple/`, `Offline/`, and `Production/`. Build with `muse` on a dedicated build node (Offline rebuild is slow). Modified EventNtuple adds `calomcsim.ancestorSimIds` branch for SimParticle ancestry.

**MCS art files (input to EventNtuple):**
- **MDC2025 (with field):** `/pnfs/mu2e/persistent/datasets/phy-sim/mcs/mu2e/FlateMinusMix1BBTriggered/MDC2025af_best_v1_1/art/` (50 files, run 001430). Reco+MC art files used by Sophie for MDC2025-002 NTS ROOT files.
- **Run1B (no field):** `/pnfs/mu2e/tape/phy-sim/mcs/mu2e/FlateMinus-KL/Run1Bah_best_v1_4-001/art/` (20 files, run 001450, 24 GB). No magnetic field scenario — electrons travel straight. Corresponding NTS files (without ancestry) at `/pnfs/mu2e/tape/phy-nts/nts/mu2e/FlateMinus-KL/Run1B-005/root/` (20 files, ~40K events each).

**Reprocessed ROOT files (v2, with ancestry):**
- **MDC2025:** `root_files_v2/` — **deleted 2026-04-13** to free quota (97 GB). Reproducible from MCS art files via FermiGrid using `from_mcs-calo-only.fcl` + modified EventNtuple. Grid submission script: `root_files_v2/grid_submit.sh`. Original cluster `90854576`.
- **Run1B:** `root_files_run1b/` — reprocessing via FermiGrid (pending). Grid submission script: `root_files_run1b/grid_submit.sh`, workflow project `eventntuple_run1b`.

**Batch reprocessing notes:**
- `run_all.sh` uses `MAX_PARALLEL=10`. Originally 20, but login node killed all processes after ~10 min (resource limits). 10 parallel is safe.
- Each file takes ~8-10 min. Full 50-file batch: ~40-50 min with 10 parallel.
- Script skips files that already exist — safe to rerun after partial failures. Delete partial outputs first (check file sizes; complete files are ~1.9-2.0 GB).
- Must run inside tmux with muse environment: `tmux new -s reprocess`, then `source setupmu2e-art.sh && cd working_dir && muse setup && bash run_all.sh`.

---

## Running things

```bash
source setup_env.sh

# Smoke test (confirm environment)
python3 scripts/smoke_test_env.py

# Run all unit tests (unittest, NOT pytest) — 73 tests
python3 -m unittest discover -s tests -p "test_*.py" -v

# Build graphs from local ROOT files (CPU node, ~10 min with 500 events/file)
bash scripts/build_all_graphs.sh

# Pack graphs into single files per split
python3 scripts/pack_graphs.py

# Train SimpleEdgeNet (GPU node)
python3 scripts/train_gnn.py --config configs/default.yaml --device cuda --epochs 100 --batch-size 64

# Train CaloClusterNet (GPU node) — staged training with --resume
python3 scripts/train_gnn.py --config configs/calo_cluster_net.yaml --device cuda --run-name calo_cluster_net_stage1
python3 scripts/train_gnn.py --config configs/calo_cluster_net_stage2.yaml --device cuda --run-name calo_cluster_net_stage2 --resume outputs/runs/calo_cluster_net_stage1/checkpoints/best_model.pt

# Threshold tuning on val set (model-agnostic)
python3 scripts/tune_threshold.py
python3 scripts/tune_threshold.py --config configs/calo_cluster_net.yaml --checkpoint outputs/runs/calo_cluster_net_v2_stage1/checkpoints/best_model.pt

# Test set evaluation (run ONCE)
OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 python3 -u scripts/evaluate_test.py --n-events 500

# Failure audit (root-cause analysis of merge/split errors)
python3 scripts/failure_audit.py

# 3-panel event displays: MC Truth | BFS | GNN
python3 scripts/plot_gnn_clusters.py --n-events 6
python3 scripts/plot_gnn_clusters.py --find-failures --n-scan 200
python3 scripts/plot_gnn_clusters.py --find-successes --n-scan 200

# Validate ancestry data in v2 ROOT files
OMP_NUM_THREADS=4 python3 scripts/validate_ancestry.py --max-events 500

# Evaluate models against old vs new (calo-entrant) truth definitions
OMP_NUM_THREADS=4 python3 scripts/evaluate_new_truth.py --split val --max-events 500

# Run1B (no-field) evaluation: BFS + both GNNs
OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 python3 -u scripts/evaluate_run1b.py --n-events 500
```

---

## Architecture and data flow

```
ROOT files v2 (EventNtuple/ntuple TTree, with ancestorSimIds)
  |
  +-> src/geometry/crystal_geometry.py    crystalId -> (diskId, x, y) lookup
  +-> src/data/graph_builder.py           radius+kNN graph, node/edge features
  +-> src/data/truth_labels_primary.py    calo-entrant truth (EXCLUSIVE — from ancestorSimIds)
  |
  +-> src/data/dataset.py
        extract_events_from_file()        yields PyG Data per disk per event
        CaloGraphDataset                  loads saved .pt files (split-aware)
              |
              +-> src/data/normalization.py   z-score stats (node/edge features)
              |
              +-> src/models/simple_edge_net.py      SimpleEdgeNet (215K params)
              +-> src/models/calo_cluster_net.py     CaloClusterNet (676K params)
              |     +-> src/models/layers.py         EdgeAwareResBlock (residual MP + gated agg)
              |     +-> src/models/heads.py          NodeSaliencyHead + EdgeClusteringHead
              +-> src/training/trainer.py            train loop, multi-task loss, early stopping
              +-> src/inference/
                    cluster_reco.py                 symmetrize, threshold, connected components, cleanup
                    postprocess.py                  per-cluster features (energy, centroid, time, RMS)
```

### PyG Data object fields

**Core:** `x` (n,6), `edge_index` (2,E), `edge_attr` (E,8), `y_edge` (E,), `edge_mask` (E,), `y_node` (n,), `hit_truth_cluster` (n,), `n_hits`, `disk_id`.

**Node features (6-dim):** log energy, time, x, y, radial distance, relative energy (per-graph).

**Edge features (8-dim):** dx, dy, distance, dt, d log energy, energy asymmetry, log summed energy, dr.

**Edge masking:** `edge_mask` filters out edges where either hit is unassigned (-1) or ambiguous. Loss is computed only on `edge_mask=True` edges.

---

## Key data paths

| What | Path |
|------|------|
| MDC2025 v2 ROOT files | `root_files_v2/` — **deleted 2026-04-13** (97 GB freed). Reproducible via FermiGrid (cluster `90854576`). |
| Run1B v2 ROOT files | `root_files_run1b/` — pending FermiGrid reprocessing (20 files, no-field scenario) |
| Run1B MCS art files | `/pnfs/mu2e/tape/phy-sim/mcs/mu2e/FlateMinus-KL/Run1Bah_best_v1_4-001/art/` (20 files, 24 GB) |
| Run1B NTS files (no ancestry) | `/pnfs/mu2e/tape/phy-nts/nts/mu2e/FlateMinus-KL/Run1B-005/root/` (20 files, ~40K events each) |
| Packed graphs (MDC2025) | `data/processed/{train,val,test}.pt` (119 + 23 + 27 MB, calo-entrant truth) |
| Node/edge norm stats | `data/normalization_stats.pt` |
| Crystal geometry | `data/crystal_geometry.csv`, `data/crystal_neighbors.csv` |
| Splits (frozen) | `splits/{train,val,test}_files.txt` (35/7/8 v2 files) |
| EventNtuple build | `/exp/mu2e/app/users/wzhou2/working_dir/` (EventNtuple + Offline + Production) |
| Muse tarball | `/exp/mu2e/data/users/wzhou2/museTarball/tmp.iTQSPdWZBd/Code.tar.bz2` (rebuilt 2026-04-13, `al9-prof-e29-p092`, with `ancestorSimIds` + latest Offline) |
| v1 results archive | `~/gnn_v1_results.tar.gz` (35 MB — old outputs, plots, training runs) |
| v1 ROOT files | **Deleted** (were at `root_files/`, 97 GB — originals on tape at `/pnfs/mu2e/tape/phy-nts/...`) |

---

## Truth labeling

**RULE: Use calo-entrant truth exclusively.** All graph building, training, evaluation, and analysis must use `src/data/truth_labels_primary.py` (calo-entrant truth). The old SimParticle-level truth (`src/data/truth_labels.py`) is retained only for reference — never use it for new work. v2 ROOT files (with `calomcsim.ancestorSimIds`) are required.

**Calo-entrant truth** (`src/data/truth_labels_primary.py`):
- Groups by **calo-entrant ancestor** — the highest ancestor in the Geant4 parent chain that also deposited in the same disk
- Energy deposits from same-shower SimParticles sum before purity check
- Requires v2 ROOT files with `calomcsim.ancestorSimIds` branch
- Ambiguity: 1.7% (vs 4.1% under old truth)
- Singletons: 48% (vs 53% under old truth, irreducible — see below)

**Remaining singletons (48%) are irreducible pileup:**
- 66% photons (eBrem from upstream), 25% electrons, 4% neutrons
- 90.6% deposit 10-50 MeV — single CsI crystal absorbs full shower at this energy
- 99.1% of eBrem singletons have no parent in the calorimeter (emitted in tracker/transport)
- Cross-disk correlation not useful; track matching won't help (photons leave no tracks)
- **Real concern:** pileup singletons bias reconstructed cluster energy by ~10-50 MeV when merged into adjacent showers. BFS has no per-hit energy filter during expansion.

---

## Model results summary

### v1 campaign (old SimParticle truth) — historical reference

**Test set (4,000 events, 6,996 disk-graphs, old truth):**

| Metric | BFS | SimpleEdgeNet (t=0.34) | CaloClusterNet (t=0.30) |
|--------|-----|------------------------|---------------------------|
| Reco match rate | 94.8% | **95.3%** | 95.2% |
| Truth match rate | **88.1%** | 87.7% | **88.1%** |
| Mean purity | 0.9727 | 0.9724 | **0.9731** |
| Mean completeness | 0.9958 | **0.9983** | 0.9982 |
| Splits | 385 | **208** | 235 |
| Merges | 2,940 | 2,878 | **2,808** |

**Calo-entrant truth re-evaluation (val set, 1,500 events, no retraining):**

| Method | Truth | Truth MR | Purity | Merges |
|--------|-------|----------|--------|--------|
| BFS | old | 88.4% | 0.9731 | 1,028 |
| BFS | **new** | **94.7%** | **0.9879** | **527** |
| CaloClusterNet | old | 88.5% | 0.9734 | 977 |
| CaloClusterNet | **new** | **94.7%** | **0.9882** | **480** |

**Key finding:** ~50% of old merge errors were artificial (same shower, different SimParticle IDs). Switching to calo-entrant truth halved merges and boosted truth match rate by +6.2% without retraining. This motivated the full v2 rebuild.

v1 outputs archived in `~/gnn_v1_results.tar.gz` (2026-04-05).

### v2 campaign (calo-entrant truth, retrained)

**Data:** 50 v2 ROOT files via FermiGrid, 41,656 graphs (29,143 train / 5,793 val / 6,720 test).

**Training (2026-04-05, A100 GPU):**

| Model | Best Val F1 | Best Epoch | Epochs |
|-------|-----------|------------|--------|
| SimpleEdgeNet | 0.966 | 9 | 24 (early stop) |
| CaloClusterNet (Stage 1) | 0.961 | 13 | 28 (early stop) |

**Threshold tuning (val set):** SimpleEdgeNet τ_edge=0.26 (F1=0.9734), CaloClusterNet τ_edge=0.20 (F1=0.9748). Frozen in configs.

**Test set (4,000 events, 6,996 disk-graphs, calo-entrant truth):**

| Metric | BFS | SimpleEdgeNet (τ=0.26) | CaloClusterNet (τ=0.20) |
|--------|-----|------------------------|---------------------------|
| Reco match rate | 96.5% | **97.1%** | **97.1%** |
| Truth match rate | **94.3%** | 94.0% | 94.2% |
| Mean purity | 0.9877 | 0.9875 | **0.9877** |
| Mean completeness | 0.9951 | 0.9981 | **0.9982** |
| Splits | 467 | 238 | **214** |
| Merges | 1,533 | 1,480 | **1,454** |

**v2 vs v1 improvement:** Truth match +6.2% (88→94%), purity +0.015, merges halved (2,940→1,454). Both GNNs beat BFS on reco match rate, completeness, splits, and merges.

**Checkpoints:** `outputs/runs/simple_edge_net_v2/`, `outputs/runs/calo_cluster_net_v2_stage1/`.

### Run1B campaign (no magnetic field, calo-entrant truth)

**Goal:** Test generalization of MDC2025-trained models on the no-field scenario (electrons travel straight).

**Data:** 20 Run1B v2 ROOT files (reprocessed via FermiGrid, cluster `27583402`), 405 MB total, ~40K events/file. `FlateMinus-KL` dataset — no pileup, no B-field.

**Evaluation (10,000 events, 8,641 disk-graphs, calo-entrant truth):**

| Metric | BFS | SimpleEdgeNet (τ=0.26) | CaloClusterNet (τ=0.20) |
|--------|-----|------------------------|---------------------------|
| Reco match rate | 99.6% | **99.7%** | **99.7%** |
| Truth match rate | **99.9%** | **99.9%** | **99.9%** |
| Mean purity | 0.9997 | 0.9997 | 0.9997 |
| Mean completeness | 0.9991 | **0.9993** | **0.9993** |
| Splits | 38 | 29 | **28** |
| Merges | 5 | 5 | 5 |

**Key findings:**
- Near-perfect performance from all methods — no-field, no-pileup scenario is trivially easy.
- Only 12 singletons (0.1%) vs 48% in MDC2025 — confirms singletons are pileup-driven.
- Only 5 merges vs 1,454 in MDC2025. ~1 truth cluster per disk vs ~5 in MDC2025.
- GNNs generalize perfectly to no-field data despite being trained on with-field data.
- MDC2025 with pileup remains the meaningful benchmark.

**Script:** `scripts/evaluate_run1b.py`. **Results:** `outputs/run1b_eval/`.

---

## Failure audit findings (v2 campaign)

**v2 audit (CaloClusterNet, τ=0.20, val set, 5,793 graphs):**
- Total merges: 1,512 (down from 2,289 in v1)
- 75% of merges caused by ≤2 bridge edges; median bridge score 0.59 (threshold 0.20)
- 93.8% of merges involve at least one singleton truth cluster
- Precision 0.957 (up from 0.882 in v1), Recall 0.993
- Remaining merges are irreducible: physically indistinguishable singletons adjacent to clusters

Full audit in `outputs/failure_audit/audit_summary.json`.

---

## Critical invariants — never violate these

1. **Calo-entrant truth only:** All truth labels via `src/data/truth_labels_primary.py` using `calomcsim.ancestorSimIds` from v2 ROOT files. Never use old SimParticle-level truth (`truth_labels.py`) for training, evaluation, or graph building. Never use BFS reco (`clusterIdx_`) as truth.
2. **v2 ROOT files only:** All data pipelines must use v2 files (`root_files_v2/`) with ancestry branches. v1 files (`root_files/`) are legacy — do not use for new work.
3. **Split discipline:** `splits/` files are frozen (35/7/8 v2 files, same file IDs as v1). Never re-split.
4. **Normalization:** `data/normalization_stats.pt` computed from train split only.
5. **Threshold tuning:** Always on validation set. Never on test set. Report final metrics once on test.
6. **Graph unit:** One graph per disk per event. Disk 0 and Disk 1 are separate graphs.
7. **Graph r_max:** 210mm (tuned for 100% pair recall).

---

## Coding conventions

- All scripts must be runnable after `source setup_env.sh` with no other setup.
- Use `uproot` + `awkward`/`numpy` for ROOT file reading — not PyROOT.
- Graph objects are PyG `Data(x, edge_index, edge_attr, y_node, y_edge)`.
- Configs in YAML; load with `yaml.safe_load`. No hardcoded hyperparameters in model code.
- Tests use `unittest` (not pytest). Run with `python3 -m unittest discover -s tests -v`.
- Log experiment details (git hash, config, dataset manifest) to `outputs/runs/<run_name>/`.
- Model factory: `src/models/__init__.py` exports `build_model()` — shared by all scripts.
- `OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1` for scripts reading ROOT files (limits scipy thread contention).
