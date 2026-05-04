# ONNX Deployment Spec — CaloClusterNet

Interface contract for deploying the GNN calorimeter-clustering model
(CaloClusterNet, `v2_stage1` + CCN+BFS10 recipe) in Mu2e Offline C++
via ONNX Runtime. Target consumer: an `art::EDProducer` following the
pattern in Andy Edmonds's `Mu2e/ArtAnalysis#4`.

For the surrounding integration plan (where the producer lives, how it
coexists with BFS, design decisions for the integration meeting), see
`docs/offline_integration.md`.

The exported `.onnx` covers **only the forward pass** (node/edge
encoders + 4 message-passing blocks + edge head). Graph construction
(Offline `CaloHit` → PyG tensors) and cluster assembly (edge logits →
cluster labels) are **not** in the ONNX graph and must be implemented
on the C++ side; pseudocode is below.

---

## 1. Model artifacts

The C++ deployment supports running any model that conforms to the
tensor interface (§2) and embeds the right `metadata_props` (§7).
Production today ships **two** artifacts; new models drop in via
config without code changes (see `docs/offline_integration.md` §2.2).

| Artifact | Source checkpoint | `model_version` (§7) | Frozen `tau_edge` |
|---|---|---|---|
| `outputs/onnx/calo_cluster_net_v2_stage1.onnx` (default, ~2.6 MB) | `outputs/runs/calo_cluster_net_v2_stage1/checkpoints/best_model.pt` | `calo-cluster-net-v2-stage1` | 0.20 |
| `outputs/onnx/simple_edge_net_v2.onnx` (Task 16i, planned) | `outputs/runs/simple_edge_net_v2/checkpoints/best_model.pt` | `simple-edge-net-v2` | 0.26 |

| Property | Value |
|---|---|
| Opset | 17 |
| IR version | 8 |
| Producer | PyTorch 2.5.1 |
| Regenerate | `python3 scripts/export_onnx.py` (CCN default; `--model sen` once 16i lands) |
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
statistics **before** feeding to the ONNX model. The trained weights
assume this normalisation.

```
x_norm[:, i]        = (x_raw[:, i]        - node_mean[i]) / node_std[i]
edge_attr_norm[:, i] = (edge_attr_raw[:, i] - edge_mean[i]) / edge_std[i]
```

### Canonical source for C++

The C++ deployment reads stats from a JSON sidecar that ships next to
the `.onnx`:

```
outputs/onnx/calo_cluster_net_v2_stage1.norm.json
```

Schema (flat, one entry per field):

| Field | Type | Notes |
|---|---|---|
| `schema_version` | int | Bump on layout-breaking changes (currently `1`) |
| `node_features` | string[6] | Canonical feature names in index order |
| `edge_features` | string[8] | Canonical feature names in index order |
| `node_mean`, `node_std` | float[6] | Z-score stats (float32 values, JSON-encoded) |
| `edge_mean`, `edge_std` | float[8] | Z-score stats |
| `node_count`, `edge_count` | int | Sample counts the stats were computed from (audit only) |

The sidecar is bit-exact with `data/normalization_stats.pt` (the torch
blob produced by training). Regenerate with:

```
python3 scripts/export_norm_stats.py
```

The C++ side should assert `node_features` and `edge_features` match
the names it expects, then index `_mean` / `_std` by feature index.
Round-trip parity with the torch blob is verified by
`tests/test_export_norm_stats.py`.

The same canonical names live a second time in the ONNX file's
`metadata_props` map (§7), so the cluster module can independently
verify the loaded model expects the features the graph maker is
emitting. Sidecar names ↔ metadata_props names ↔ FHiCL names must all
agree for the job to start.

### Node feature stats (train split, 348,548 hits)

`JSON key` is the entry in `node_features` of the sidecar at the same index.

| Idx | JSON key | Feature | Unit | Mean | Std |
|---|---|---|---|---|---|
| 0 | `log_e` | `log(1 + E/MeV)` | log-MeV | 2.4069 | 0.7528 |
| 1 | `t` | hit time | ns | 834.6544 | 390.6363 |
| 2 | `x` | disk-local `x` | mm | −23.6271 | 325.2221 |
| 3 | `y` | disk-local `y` | mm | 70.9853 | 315.6965 |
| 4 | `r` | radial `r = sqrt(x² + y²)` | mm | 455.1254 | 62.3836 |
| 5 | `e_rel` | `E / E_max` (per graph) | — | 0.3843 | 0.2934 |

### Edge feature stats (train split, 831,668 directed edges)

`JSON key` is the entry in `edge_features` of the sidecar at the same index.

| Idx | JSON key | Feature | Unit | Mean | Std |
|---|---|---|---|---|---|
| 0 | `dx` | `x_i − x_j` | mm | 0.0 | 108.0510 |
| 1 | `dy` | `y_i − y_j` | mm | 0.0 | 107.5170 |
| 2 | `d` | `sqrt(dx² + dy²)` | mm | 95.8032 | 118.5608 |
| 3 | `dt` | `t_i − t_j` | ns | 0.0 | 4.8740 |
| 4 | `dlog_e` | `log(1+E_i) − log(1+E_j)` | log-MeV | 0.0 | 1.2146 |
| 5 | `asym_e` | `(E_i − E_j) / (E_i + E_j)` | — | 0.0 | 0.5318 |
| 6 | `logsum_e` | `log(1 + E_i + E_j)` | log-MeV | 3.1024 | 0.6095 |
| 7 | `dr` | `r_i − r_j` | mm | 0.0 | 47.5153 |

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

## 7. Version + feature contract (`metadata_props`)

Each `.onnx` carries three keys in its `metadata_props` map. The C++
session loader (`CaloClusterMakerGNN` constructor) reads them and
asserts each one against the corresponding FHiCL parameter passed by
the production config. Any mismatch aborts the job before the first
event runs — so silent tensor-layout drift after a retraining cannot
go undetected.

| Key | Example value | FHiCL parameter to compare against | Bump when… |
|---|---|---|---|
| `model_version` | `calo-cluster-net-v2-stage1` | `expected_model_version` | weights or feature set or opset changes |
| `node_features` | `log_e,t,x,y,r,e_rel` | `expected_node_features` | per-node feature columns change order or meaning |
| `edge_features` | `dx,dy,d,dt,dlog_e,asym_e,logsum_e,dr` | `expected_edge_features` | per-edge feature columns change order or meaning |

Values are written by `scripts/export_onnx.py` after `torch.onnx.export`
returns. Keys are stable; values evolve with the model. The two
feature-name keys are comma-separated strings (no whitespace), in
canonical column order — they must match the names in
`calo_cluster_net_v2_stage1.norm.json` (§3) and the columns the
`CaloHitGraphMaker` is wired to emit. The graph-maker side of the
handshake lives in its own FHiCL (`node_features` / `edge_features`
parameters), so the chain
`graph-maker FHiCL → cluster-maker FHiCL → ONNX metadata_props` is
asserted end-to-end at job start.

Inspect manually with:

```python
import onnx
m = onnx.load("outputs/onnx/calo_cluster_net_v2_stage1.onnx")
print({p.key: p.value for p in m.metadata_props})
```

### Validation table at the session boundary

| Side | Reads | Asserts |
|---|---|---|
| `CaloHitGraphMaker` | FHiCL `node_features`/`edge_features` (the names it emits in column order) | (none — it's the source of truth on the graph side) |
| `CaloClusterMakerGNN` | FHiCL `expected_*_features`, ONNX `metadata_props["*_features"]` | both equal each other; both equal what the graph maker emits (verified by passing the same FHiCL list into both modules in the production config) |

### Deployment status (resolved at the 2026-04-29 meeting)

Earlier open items in this section have moved to `docs/offline_integration.md`:

- Central `onnxruntime` in muse (§2.5) — link via the `u092` qualifier.
- Module boundaries (§2.2) — split into graph-maker + cluster-maker EDProducers.
- Graph construction strategy (§2.4) — brute-force pairwise distance loop.
- Normalisation storage (§3) — JSON sidecar shipped alongside the `.onnx` (Task 16a, done).
- Version-string carrier (§2.7) — ONNX `metadata_props` (this section).
