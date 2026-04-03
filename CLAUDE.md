# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Attribution rule

**Never include Claude authorship or co-authorship in git commits.** Do not add `Co-Authored-By`, `Generated-By`, or any similar trailer referencing Claude, AI, or Anthropic. Commit messages should read as if written by the user. This applies to all commits — new, amended, or rebased.

# GNN Calorimeter Clustering

## Project overview

Graph Neural Network-based clustering algorithm for the Mu2e electromagnetic calorimeter.
Replaces the existing BFS seed-based algorithm (`CaloClusterMaker` in Offline).
Approach: **edge classification** (SimpleEdgeNet, CaloClusterNetV1).
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

# Train CaloClusterNetV1 (GPU node) — staged training with --resume
python3 scripts/train_gnn.py --config configs/calo_cluster_net_v1.yaml --device cuda --run-name calo_cluster_net_v1_stage1
python3 scripts/train_gnn.py --config configs/calo_cluster_net_v1_stage2.yaml --device cuda --run-name calo_cluster_net_v1_stage2 --resume outputs/runs/calo_cluster_net_v1_stage1/checkpoints/best_model.pt

# Threshold tuning on val set (model-agnostic)
python3 scripts/tune_threshold.py
python3 scripts/tune_threshold.py --config configs/calo_cluster_net_v1.yaml --checkpoint outputs/runs/calo_cluster_net_v1_stage1/checkpoints/best_model.pt

# Test set evaluation (run ONCE)
OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 python3 -u scripts/evaluate_test.py --n-events 500

# Failure audit (root-cause analysis of merge/split errors)
python3 scripts/failure_audit.py

# 3-panel event displays: MC Truth | BFS | GNN
python3 scripts/plot_gnn_clusters.py --n-events 6
python3 scripts/plot_gnn_clusters.py --find-failures --n-scan 200
```

---

## Architecture and data flow

```
ROOT files (EventNtuple/ntuple TTree)
  |
  +-> src/geometry/crystal_geometry.py    crystalId -> (diskId, x, y) lookup
  +-> src/data/graph_builder.py           radius+kNN graph, node/edge features
  +-> src/data/truth_labels.py            MC truth edge labels (from SimParticle IDs)
  |
  +-> src/data/dataset.py
        extract_events_from_file()        yields PyG Data per disk per event
        CaloGraphDataset                  loads saved .pt files (split-aware)
              |
              +-> src/data/normalization.py   z-score stats (node/edge features)
              |
              +-> src/models/simple_edge_net.py      SimpleEdgeNet (215K params)
              +-> src/models/calo_cluster_net.py     CaloClusterNetV1 (676K params)
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
| ROOT files (local) | `/exp/mu2e/data/users/wzhou2/GNN/root_files/` (50 files, 97 GB) |
| Packed graphs | `data/processed/{train,val,test}.pt` |
| Node/edge norm stats | `data/normalization_stats.pt` |
| Crystal geometry | `data/crystal_geometry.csv`, `data/crystal_neighbors.csv` |
| Splits (frozen) | `splits/{train,val,test}_files.txt` |

ROOT files are MDC2025-002 format (`EventNtuple/ntuple` TTree). MC truth via `calohitsmc` branches.

---

## Truth labeling

Implemented in `src/data/truth_labels.py`:
- **MC truth:** `calohitsmc.simParticleIds` + `calohitsmc.eDeps` from MDC2025-002
- Dominant SimParticle per hit; ambiguous if purity < 0.7
- Truth cluster = (dominant SimParticle, disk) pair

**Known issue (Task 11):** Current truth groups by individual SimParticle ID, not by primary particle. Secondary shower products (bremsstrahlung photons, etc.) get their own truth clusters even though they are physically part of the parent shower. This creates artificial singleton truth clusters (52% of all truth clusters) and inflates merge error counts. Fix: trace SimParticle ancestry back to the primary and group all shower products together. See `calomcsim` branches for ancestry info.

---

## Model results summary

**Test set (4,000 events, 6,996 disk-graphs, run once):**

| Metric | BFS | SimpleEdgeNet (t=0.34) | CaloClusterNetV1 (t=0.30) |
|--------|-----|------------------------|---------------------------|
| Reco match rate | 94.8% | **95.3%** | 95.2% |
| Truth match rate | **88.1%** | 87.7% | **88.1%** |
| Mean purity | 0.9727 | 0.9724 | **0.9731** |
| Mean completeness | 0.9958 | **0.9983** | 0.9982 |
| Splits | 385 | **208** | 235 |
| Merges | 2,940 | 2,878 | **2,808** |

**Frozen checkpoints & thresholds (tuned on val, do not change):**
- SimpleEdgeNet: `outputs/runs/simple_edge_net_v1/checkpoints/best_model.pt`, t_edge=0.34
- CaloClusterNetV1: `outputs/runs/calo_cluster_net_v1_stage1/checkpoints/best_model.pt`, t_edge=0.30
- Production cleanup: `min_hits=2`, `min_energy_mev=10.0`

---

## Failure audit findings (Task 10)

**Root cause of remaining errors:** The dominant failure is a single confident bridge edge (median score 0.65, threshold 0.30) merging a singleton truth cluster into a nearby cluster. 93% of merges involve at least one single-hit truth cluster. The model is not borderline wrong — it genuinely believes these cross-cluster edges belong together.

**Key insight:** This is a truth-target / observability mismatch, not a model weakness. The current truth is defined by SimParticle ancestry (MC genealogy), but a single-hit truth object has no shape, spread, or internal structure that distinguishes it from a neighboring cluster at the hit level. The model is being penalized for failing to recover distinctions that exist in MC bookkeeping but not in the detector observables.

**Implication:** The project is less limited by model design than by whether SimParticle-based truth labels are the right reconstruction target, especially for singleton truth clusters.

**Resolution (in progress):** Most singleton truth clusters are secondary shower products (bremsstrahlung photons from the primary particle) that deposit small energy in one crystal. They arrive at the same time (~0.2 ns) as the primary shower and are physically indistinguishable. The fix is to redefine truth at the **primary particle level** — trace each SimParticle back to its primary and group the entire shower as one truth cluster. See plan Task 11.

Full audit: `outputs/failure_audit/audit_summary.json`, analysis script: `scripts/failure_audit.py`.

---

## Critical invariants — never violate these

1. **MC truth only:** All truth labels from `calohitsmc.simParticleIds` + `calohitsmc.eDeps`. Never use BFS reco (`clusterIdx_`) as truth. Only MDC2025-002 files.
2. **Split discipline:** `splits/` files are frozen. Never re-split.
3. **Normalization:** `data/normalization_stats.pt` computed from train split only.
4. **Threshold tuning:** Always on validation set. Never on test set. Report final metrics once on test.
5. **Graph unit:** One graph per disk per event. Disk 0 and Disk 1 are separate graphs.
6. **Graph r_max:** 210mm (tuned for 100% pair recall).

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
