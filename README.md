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

## Quick start

```bash
# Activate environment (required before any Python command)
source setup_env.sh

# Verify environment
python3 scripts/smoke_test_env.py

# Run unit tests (73 tests)
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
  +-> src/data/truth_labels.py            MC truth edge labels (SimParticle IDs)
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
              |
              +-> src/training/
              |     trainer.py                 train/val loop, early stopping
              |     losses.py                  masked BCE, multi-task loss
              |     metrics.py                 edge F1, AUC, cluster purity
              |
              +-> src/inference/
                    cluster_reco.py            symmetrize, threshold, connected components
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

MC truth from `calohitsmc.simParticleIds` + `calohitsmc.eDeps` (MDC2025-002 format):

- Dominant SimParticle per hit; ambiguous if energy purity < 0.7
- Truth cluster = (dominant SimParticle, disk) pair
- Edge label: 1 if both endpoints share the same truth cluster, 0 otherwise
- Edges involving ambiguous hits are masked out of the loss

## Results

Test set (4,000 events, 6,996 disk-graphs):

| Metric | BFS | SimpleEdgeNet (t=0.34) | CaloClusterNet (t=0.30) |
|--------|-----|------------------------|---------------------------|
| Reco match rate | 94.8% | **95.3%** | 95.2% |
| Truth match rate | **88.1%** | 87.7% | **88.1%** |
| Mean purity | 0.9727 | 0.9724 | **0.9731** |
| Mean completeness | 0.9958 | **0.9983** | 0.9982 |
| Splits | 385 | **208** | 235 |
| Merges | 2,940 | 2,878 | **2,808** |

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

73 tests covering graph construction, truth labeling, model forward passes,
inference pipeline, and postprocessing.
