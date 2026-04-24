# ONNX Deployment Spec — CaloClusterNet

Interface contract for deploying the GNN calorimeter-clustering model
(CaloClusterNet, `v2_stage1` + CCN+BFS10 recipe) in Mu2e Offline C++
via ONNX Runtime. Target consumer: an `art::EDProducer` following the
pattern in Andy Edmonds's `Mu2e/ArtAnalysis#4`.

The exported `.onnx` covers **only the forward pass** (node/edge
encoders + 4 message-passing blocks + edge head). Graph construction
(Offline `CaloHit` → PyG tensors) and cluster assembly (edge logits →
cluster labels) are **not** in the ONNX graph and must be implemented
on the C++ side; pseudocode is below.

---

## 1. Model artifact

| Property | Value |
|---|---|
| File | `outputs/onnx/calo_cluster_net_v2_stage1.onnx` (gitignored, ~2.6 MB) |
| Opset | 17 |
| IR version | 8 |
| Producer | PyTorch 2.5.1 |
| Source checkpoint | `outputs/runs/calo_cluster_net_v2_stage1/checkpoints/best_model.pt` |
| Regenerate | `python3 scripts/export_onnx.py` |
| Validate | `python3 scripts/validate_onnx.py` |

All inference is CPU. No GPU needed in deployment.

---

## 2. Tensor interface

### Inputs

One graph per calorimeter disk per event. `N` = hit count; `E` = directed edge count. Both are dynamic axes.

| Name | Shape | Dtype | Semantics |
|---|---|---|---|
| `x` | `(N, 6)` | `float32` | Per-hit node features, **z-score normalised** (§3) |
| `edge_index` | `(2, E)` | `int64` | Directed adjacency: `edge_index[0]` = src, `edge_index[1]` = dst |
| `edge_attr` | `(E, 8)` | `float32` | Per-edge features, **z-score normalised** (§3) |

### Output

| Name | Shape | Dtype | Semantics |
|---|---|---|---|
| `edge_logits` | `(E,)` | `float32` | Raw pre-sigmoid logits, one per directed edge |

`edge_logits[k]` is the logit for `edge_index[:, k]`. Apply `sigmoid` externally to get a probability.

---

## 3. Normalisation

Both `x` and `edge_attr` must be z-scored using the train-split
statistics in `data/normalization_stats.pt` **before** feeding to the
ONNX model. The trained weights assume this normalisation.

```
x_norm[:, i]        = (x_raw[:, i]        - node_mean[i]) / node_std[i]
edge_attr_norm[:, i] = (edge_attr_raw[:, i] - edge_mean[i]) / edge_std[i]
```

### Node feature stats (train split, 348,548 hits)

| Idx | Feature | Unit | Mean | Std |
|---|---|---|---|---|
| 0 | `log(1 + E/MeV)` | log-MeV | 2.4069 | 0.7528 |
| 1 | hit time `t` | ns | 834.6544 | 390.6363 |
| 2 | disk-local `x` | mm | −23.6271 | 325.2221 |
| 3 | disk-local `y` | mm | 70.9853 | 315.6965 |
| 4 | radial `r = sqrt(x² + y²)` | mm | 455.1254 | 62.3836 |
| 5 | `E / E_max` (per graph) | — | 0.3843 | 0.2934 |

### Edge feature stats (train split, 831,668 directed edges)

| Idx | Feature | Unit | Mean | Std |
|---|---|---|---|---|
| 0 | `dx = x_i − x_j` | mm | 0.0 | 108.0510 |
| 1 | `dy = y_i − y_j` | mm | 0.0 | 107.5170 |
| 2 | `d = sqrt(dx² + dy²)` | mm | 95.8032 | 118.5608 |
| 3 | `dt = t_i − t_j` | ns | 0.0 | 4.8740 |
| 4 | `d log E = log(1+E_i) − log(1+E_j)` | log-MeV | 0.0 | 1.2146 |
| 5 | `(E_i − E_j) / (E_i + E_j)` | — | 0.0 | 0.5318 |
| 6 | `log(1 + E_i + E_j)` | log-MeV | 3.1024 | 0.6095 |
| 7 | `dr = r_i − r_j` | mm | 0.0 | 47.5153 |

Stats are frozen — never recompute on test or deployment data.

---

## 4. Graph construction (upstream of the ONNX model)

**One graph per disk per event.** Node = `CaloHit` on that disk. Not on the ONNX graph; must be implemented in C++.

1. For each event, for each disk `d ∈ {0, 1}`:
2. Collect all `CaloHit`s with `diskID == d`.
3. For each hit, compute the 6 node features above using
   `calohits.crystalPos_` for `(x, y)` (or the Offline Calorimeter geometry service).
4. **Radius graph** at `r_max = 210 mm` — pairs with `sqrt(dx² + dy²) ≤ r_max`.
5. **Time filter:** drop pairs with `|dt| > dt_max = 25 ns`.
6. **kNN fallback:** isolated hits get edges to their `k_min = 3` nearest time-compatible neighbours.
7. **Degree cap:** keep at most `k_max = 20` nearest neighbours per hit.
8. Edges are **directed** — each undirected pair appears twice (i→j and j→i).
9. Compute the 8 edge features.
10. Normalise `x` and `edge_attr` (§3); `edge_index` is not normalised.

Reference Python implementation: `src/data/graph_builder.py`.
Crystal geometry table: `data/crystal_geometry.csv` (crystalId → disk, x, y).

The graph-construction gate is already met at these parameters: 100%
pair recall and 100% cluster connectivity on the train set (plan.md §2
/ findings.md §2). Changing any of the thresholds requires re-running
that gate and almost certainly re-training.

---

## 5. Cluster assembly (CCN+BFS10 — post-processing of ONNX output)

Not in the ONNX graph; must be implemented in C++. The winning recipe
(plan.md §7.4 / findings.md §7.4).

```
# Inputs: edge_index (2, E), edge_logits (E,) from ONNX;
#         raw hit energies E_hit[N] in MeV (not normalised).

# Hyperparameters (frozen):
tau_edge        = 0.20      # v2_stage1 frozen threshold
bfs_expand_cut  = 10.0 MeV  # BFS-style ExpandCut
min_hits        = 2         # drop clusters smaller than this
min_energy_mev  = 10.0      # drop clusters below this total energy

# 1. Sigmoid + symmetrise directed scores.
#    For each unordered pair {i, j} that appears in both directions,
#    replace p_ij and p_ji with their mean.
p = sigmoid(edge_logits)
p_sym = symmetrise(edge_index, p)        # average of p_ij and p_ji

# 2. Threshold.
keep = p_sym >= tau_edge
adj  = build_adjacency_list(edge_index[:, keep])   # size N

# 3. BFS traversal (Offline-style ExpandCut), seeded from highest-E hits.
labels[0..N] = -1
cid = 0
for seed in argsort(E_hit, descending):
    if labels[seed] != -1: continue
    queue = [seed]; labels[seed] = cid
    while queue not empty:
        u = queue.pop_front()
        if E_hit[u] < bfs_expand_cut: continue   # joins, does not recruit
        for v in adj[u]:
            if labels[v] == -1:
                labels[v] = cid
                queue.push_back(v)
    cid += 1

# 4. Cleanup: drop clusters with < min_hits or total E < min_energy_mev,
#    re-map remaining labels to 0..K-1 contiguously. Dropped hits → -1.
```

Reference Python implementation: `src/inference/cluster_reco.py`
(`reconstruct_clusters`, `symmetrize_edge_scores`, `_bfs_expand_cut`).

**Cluster-level quantities** (consumed downstream by track finding): total
energy = Σ E_hit; centroid = energy-weighted (x, y); time = time of the
most energetic hit (matching Offline `CaloCluster` convention). Reference
in `src/inference/postprocess.py`.

---

## 6. Parity with PyTorch (proof that the ONNX graph is faithful)

Full val set (5,793 disk-graphs, 166,342 directed edges, CPU):

| Check | Result |
|---|---|
| Max abs diff in `edge_logits` | **9.06e-06** (tol 1e-5) |
| p99.9 abs diff | 5.72e-06 |
| Threshold flips at τ=0.20 | **0 / 166,342** |
| Dynamic-axes range exercised | N ∈ [2, 65], E ∈ [2, 180] |

Zero threshold flips means the downstream cluster assembly (symmetrise
→ threshold → BFS traversal → cleanup) is provably byte-identical to
the PyTorch pipeline — the scientific results in findings.md carry
over to the C++ deployment without needing a re-run.

Side result on CPU: ONNX Runtime Python median latency 0.88 ms/graph
vs PyTorch 11.00 ms/graph (12.5× speedup). The C++ latency should be
at worst comparable, likely better.

Reproduce: `python3 scripts/validate_onnx.py`.

---

## 7. Open items / coordination

- **Central `onnxruntime` in muse.** Andy is working with Ray on this.
  Until it lands we can build against a local install; not a blocker
  for drafting the art module.
- **Module boundaries.** Does the `art::EDProducer` wrap graph
  construction + ONNX inference + cluster assembly all in one module,
  or do we split construction into a separate producer that emits a PyG-
  style art data product? Worth a decision with Sophie/Andy.
- **Graph construction in C++.** The Python builder uses
  `scipy.spatial.cKDTree` for the radius graph. The Offline Calorimeter
  service already exposes neighbour lists (`cal.neighbors`,
  `cal.nextNeighbors`) — faster than a cKDTree at runtime but only covers
  the nearest two rings. Need to confirm whether a two-ring neighbour
  set + explicit long-range candidates reproduces the 210 mm radius graph
  faithfully, or whether we port the cKDTree approach.
- **Normalisation storage.** `data/normalization_stats.pt` is a PyTorch
  blob. For C++ deployment we should export it once to a plain text or
  JSON sidecar next to the `.onnx` so the module doesn't need a
  LibTorch dependency just to read six floats.
- **Versioning.** Any change to the `.onnx` — new training run, new
  feature set, opset bump — must bump a version string that the C++
  module checks at load time so silent tensor-layout drift is caught.
