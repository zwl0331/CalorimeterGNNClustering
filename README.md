# GNN Calorimeter Clustering for Mu2e

Graph Neural Network-based clustering for the Mu2e electromagnetic calorimeter.
Replaces the existing BFS seed-based algorithm (`CaloClusterMaker` in Offline)
with an edge-classification GNN built on PyTorch Geometric.

## Approach

Each calorimeter event is converted into a graph (one per disk per event) where
nodes are crystal hits and edges connect spatially and temporally nearby hits.
The GNN predicts which edges connect hits belonging to the same physics cluster.
Connected components of the thresholded edge graph yield the final clusters.

Two models are implemented:

| Model | Parameters | Description |
|-------|-----------|-------------|
| **SimpleEdgeNet** | 215K | Lightweight baseline: MLP encoders + sum message passing + edge MLP head |
| **CaloClusterNet** | 676K | Residual MP blocks with gated aggregation, global context, node saliency + edge clustering heads |

## Scenarios

The project covers two Mu2e physics scenarios distinguished by the solenoid magnetic-field configuration:

| Scenario | Field | Dataset | Path |
|----------|-------|---------|------|
| **run1a** | with field | MDC2025 | `root_files_v2/` |
| **run1b** | no field | Run1B | `root_files_run1b/` |

Models are trained per scenario; in the no-field run1b case, electrons travel straight, so cluster geometry and timing distributions differ from run1a. Run-output and config names tagged with `_run1b_` are explicit run1b; the older `_v2_` naming (e.g., `calo_cluster_net_v2_stage1`) refers to run1a / MDC2025 data implicitly.

## Current recommended path

For follow-up analysis or deployment work, use the **CCN+BFS10** recipe:
CaloClusterNet edge logits with BFS-style traversal at `bfs_expand_cut=10 MeV`.
The corresponding config is `configs/calo_cluster_net.yaml`, with the frozen
checkpoint at `outputs/runs/calo_cluster_net_v2_stage1/checkpoints/best_model.pt`
and `inference.tau_edge: 0.20`. The ONNX deployment contract and C++ interface
notes are in `docs/onnx_deployment.md`.

## Quick start

```bash
# Activate environment (required before any Python command)
source setup_env.sh

# Verify environment
python3 scripts/smoke_test_env.py

# Run unit tests (97 tests)
python3 -m unittest discover -s tests -p "test_*.py" -v
```

## Pipeline

The full pipeline from ROOT files to evaluated clusters:

```
1. Build graphs    bash scripts/build_all_graphs.sh
2. Pack graphs     python3 scripts/pack_graphs.py
3. Train model     python3 scripts/train_gnn.py --config configs/default.yaml --device cuda
4. Tune threshold  python3 scripts/tune_threshold.py
5. Evaluate        python3 scripts/evaluate_test.py --n-events 500
6. Visualize       python3 scripts/plot_gnn_clusters.py --n-events 6
```

Steps 3-6 require a GPU node (A100 MIG). Steps 1-2 run on CPU.

### Staged training (CaloClusterNet)

```bash
# Stage 1: edge-only loss
python3 scripts/train_gnn.py --config configs/calo_cluster_net.yaml \
    --device cuda --run-name calo_cluster_net_stage1

# Stage 2: add node saliency loss, resume from stage 1
python3 scripts/train_gnn.py --config configs/calo_cluster_net_stage2.yaml \
    --device cuda --run-name calo_cluster_net_stage2 \
    --resume outputs/runs/calo_cluster_net_stage1/checkpoints/best_model.pt
```

## Data flow

```
ROOT files (EventNtuple/ntuple TTree, MDC2025-002)
  |
  +-> src/geometry/crystal_geometry.py    crystalId -> (diskId, x, y) lookup
  +-> src/data/graph_builder.py           radius + kNN graph, node/edge features
  +-> src/data/truth_labels_primary.py    calo-entrant truth (per-disk ancestor grouping)
  |
  +-> src/data/dataset.py
        extract_events_from_file()        yields PyG Data per disk per event
        CaloGraphDataset                  loads saved .pt files (split-aware)
              |
              +-> src/data/normalization.py    z-score stats (node/edge features)
              |
              +-> src/models/
              |     simple_edge_net.py         SimpleEdgeNet (baseline)
              |     calo_cluster_net.py        CaloClusterNet (primary)
              |       layers.py                EdgeAwareResBlock (residual MP)
              |       heads.py                 NodeSaliencyHead + EdgeClusteringHead
              |     calo_cluster_net_deploy.py CaloClusterNetDeploy (edge-only ONNX wrapper)
              |
              +-> src/training/
              |     trainer.py                 train/val loop, early stopping
              |     losses.py                  masked BCE, multi-task loss
              |     metrics.py                 edge F1, AUC, cluster purity
              |
              +-> src/inference/
                    cluster_reco.py            symmetrize, threshold, BFS traversal (ExpandCut)
                    postprocess.py             per-cluster energy, centroid, time, RMS
```

## Features

### Node features (6-dim)

| Index | Feature | Units |
|-------|---------|-------|
| 0 | log(1 + energy) | log MeV |
| 1 | hit time | ns |
| 2 | x position | mm |
| 3 | y position | mm |
| 4 | radial distance | mm |
| 5 | relative energy (E / E_max) | dimensionless |

### Edge features (8-dim)

| Index | Feature | Units |
|-------|---------|-------|
| 0 | dx | mm |
| 1 | dy | mm |
| 2 | distance | mm |
| 3 | dt | ns |
| 4 | d(log energy) | log MeV |
| 5 | energy asymmetry | dimensionless |
| 6 | log(summed energy) | log MeV |
| 7 | dr (radial) | mm |

## Graph construction

Hybrid radius + kNN strategy (see `src/data/graph_builder.py`):

1. **Radius graph:** connect hits within `r_max` mm (default 210 mm)
2. **Time filter:** drop edges with |dt| > `dt_max` ns (default 25 ns)
3. **kNN fallback:** isolated nodes get edges to `k_min` nearest time-compatible neighbors
4. **Degree cap:** keep at most `k_max` nearest neighbors per node

## Truth labeling

Calo-entrant truth from `calohitsmc.simParticleIds` + `calohitsmc.eDeps` + `calomcsim.ancestorSimIds` (v2 ROOT files):

- Each SimParticle is mapped to its **calo-entrant ancestor** — the highest ancestor in the Geant4 parent chain that also deposited in the same disk
- Hits sharing the same calo-entrant ancestor form one truth cluster (per disk)
- Hit is ambiguous if the dominant calo-entrant's energy purity < 0.7
- Edge label: 1 if both endpoints share the same calo-entrant truth cluster, 0 otherwise
- Edges involving ambiguous or unassigned hits are masked out of the loss

## Results

Test set (4,000 events, 6,996 disk-graphs, calo-entrant truth):

| Metric | BFS | SimpleEdgeNet (τ=0.26) | CaloClusterNet (τ=0.20) | **CCN+BFS10** |
|--------|-----|------------------------|--------------------------|----------------|
| Reco match rate | 96.5% | **97.1%** | **97.1%** | 96.9% |
| Truth match rate | 94.3% | 93.9% | 94.1% | **94.3%** |
| Mean purity | 0.9877 | 0.9872 | 0.9875 | **0.9879** |
| Mean completeness | 0.9951 | 0.9980 | **0.9982** | 0.9975 |
| Splits | 467 | 238 | **214** | 290 |
| Merges | 1,533 | 1,480 | 1,454 | **1,404** |

CCN+BFS10 is the recommended deployment recipe: CaloClusterNet edge logits + BFS-style traversal (`bfs_expand_cut = 10 MeV`). On downstream-relevant clusters (`E_reco ≥ 50 MeV`) it improves mean |ΔE| by ~20% vs BFS while matching or beating it on every standard clustering metric. Full numbers in `docs/findings.md`.

## Project layout

```
configs/                  YAML model/training configurations
data/
  crystal_geometry.csv    static detector geometry (committed)
  crystal_neighbors.csv   crystal adjacency map (committed)
  processed/              built graph .pt files (gitignored)
  normalization_stats.pt  z-score stats from train split (gitignored)
scripts/                  runnable entry points (all via setup_env.sh)
splits/                   frozen train/val/test file lists
src/
  data/                   dataset, graph building, truth labels, normalization
  geometry/               crystal geometry loader
  inference/              cluster reconstruction and postprocessing
  models/                 SimpleEdgeNet, CaloClusterNet, layers, heads
  training/               trainer, losses, metrics
tests/                    unit tests (unittest, not pytest)
outputs/                  run logs, checkpoints, plots (gitignored)
```

## Environment

Requires the Mu2e `ana 2.6.1` conda environment:

- Python 3.12, PyTorch 2.5.1, PyTorch Geometric 2.7.0
- `torch-cluster` is NOT available; graph construction uses `scipy.spatial.cKDTree`
- GPU (A100 MIG) required for training; login node is CPU-only

See `setup_env.sh` for full activation sequence.

## Tests

```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
```

97 tests covering graph construction, truth labeling, model forward passes,
inference pipeline, and postprocessing.

## Documentation

Detailed reference material lives in `docs/`:

| Doc | What it covers |
|-----|----------------|
| [`docs/plan.md`](docs/plan.md) | Chronological task history and progress checklist |
| [`docs/findings.md`](docs/findings.md) | Experimental results, physics insights, and conclusions across the v1/v2/Run1B campaigns |
| [`docs/onnx_deployment.md`](docs/onnx_deployment.md) | ONNX deployment contract and C++ interface for the Offline integration |
| [`docs/offline_integration.md`](docs/offline_integration.md) | Mu2e Offline integration plan (pre-meeting design notes) |
| [`docs/presentation.md`](docs/presentation.md) | Beamer slide style spec — used by `scripts/make_slide_plots.py` |
| [`docs/git_conventions.md`](docs/git_conventions.md) | Commit-message format and granularity rules |
| [`docs/build_runbook.md`](docs/build_runbook.md) | muse builds, batch reprocessing, FermiGrid submission |
| [`CLAUDE.md`](CLAUDE.md) | Hidden gotchas, critical invariants, and workflow guidance for Claude Code |
