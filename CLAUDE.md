# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Git conventions

Commit-message format, granularity, branching, and what-not-to-commit rules live in `docs/git_conventions.md`. Follow them for every commit: imperative subject ≤72 chars with a type prefix (`feat`/`fix`/`docs`/`refactor`/`perf`/`chore`/`data`/`wip`), one logical change per commit, update `docs/findings.md` in the same commit when overturning a prior finding.

**Attribution (overrides all):** never include Claude authorship or co-authorship in git commits. Do not add `Co-Authored-By`, `Generated-By`, or any similar trailer referencing Claude, AI, or Anthropic. Commit messages should read as if written by the user. Applies to all commits — new, amended, or rebased.

# GNN Calorimeter Clustering

## Project overview

Graph Neural Network-based clustering algorithm for the Mu2e electromagnetic calorimeter.
Replaces the existing BFS seed-based algorithm (`CaloClusterMaker` in Offline).
Approach: **edge classification** (SimpleEdgeNet, CaloClusterNet).
Built with PyTorch Geometric.

Working directory: `/exp/mu2e/app/users/wzhou2/projects/calorimeter/GNN/`
- Task history and progress: `docs/plan.md`
- Research findings (results tables, physics insights, conclusions): `docs/findings.md`

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
- **MDC2025:** `root_files_v2/` — **train split deleted 2026-04-13** to free quota (~68 GB freed). **Val + test still present** (15 files, 29 GB) — needed for any future evaluation. Train is reproducible from MCS art files via FermiGrid using `from_mcs-calo-only.fcl` + modified EventNtuple. Grid submission script: `root_files_v2/grid_submit.sh`. Original cluster `90854576`.
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

# Run all unit tests (unittest, NOT pytest) — 97 tests
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

# Train CaloClusterNet with learned saliency (GPU node) — resumes from Stage 1
python3 scripts/train_gnn.py --config configs/calo_cluster_net_saliency.yaml --device cuda --run-name calo_cluster_net_v2_saliency --resume outputs/runs/calo_cluster_net_v2_stage1/checkpoints/best_model.pt

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

# Cluster-level physics evaluation (energy, centroid, time residuals)
# Default: glob all *.root in --root-dir. For val-only, pass --file-list explicitly.
OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 python3 -u scripts/evaluate_cluster_physics.py --n-events 500 --file-list splits/val_files.txt
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
              +-> src/models/calo_cluster_net_deploy.py
              |                                      CaloClusterNetDeploy — edge-only inference wrapper
              |                                      around a trained CaloClusterNet for ONNX export
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
| MDC2025 v2 ROOT files | `root_files_v2/` — **train split deleted 2026-04-13** (~68 GB freed); val + test still present (15 files, 29 GB). Train reproducible via FermiGrid (cluster `90854576`). |
| Run1B v2 ROOT files | `root_files_run1b/` — 20 files via FermiGrid (cluster `27583402`, 405 MB total, no-field scenario) |
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

Truth-definition rationale, ambiguity/singleton statistics, and re-evaluation numbers: `docs/findings.md` §1.

---

## Research findings

Experimental results, physics insights, and conclusions (v1 campaign, v2 campaign, Run1B generalization, cluster-level physics evaluation, failure audits, fringe-hit cleanup including the CCN+BFS10 result): `docs/findings.md`.

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
