# GNN Calorimeter Clustering — Offline Integration Notes

Pre-meeting notes for the integration meeting between Sam (Caltech),
Sophie Middleton, and Andy Edmonds. The goal of the meeting is to lock
the design of the C++ module that runs the trained CaloClusterNet
inside Mu2e Offline — i.e., to figure out *how the GNN gets called
during normal Offline reconstruction*.

Companion docs:
- Tensor / artifact contract: `docs/onnx_deployment.md`
- Task and milestone tracking: `docs/plan.md` (Task 16, Milestone K)

---

## A note on terminology

The integration uses Mu2e's standard reconstruction framework (**art**),
so a few framework terms recur throughout. In plain language:

- **art** is the C++ event-processing framework Mu2e shares with most
  Fermilab experiments. A reconstruction job is a chain of art *modules*
  configured by a *FHiCL* file.
- A **module** is a plugin the framework loads and runs once per event.
  An **`EDProducer`** ("Event Data Producer") is the module type that
  *creates new outputs* and adds them to the event for downstream modules
  to read. The existing BFS clustering is the `CaloClusterMaker`
  EDProducer; the new GNN clustering will be `CaloClusterMakerGNN`,
  also an EDProducer.
- A **data product** is a typed output a producer puts into the event
  (`CaloHitCollection`, `CaloClusterCollection`, …). Later modules pull
  it back out by type plus a label.
- An **instance name** is a short string tag that distinguishes multiple
  outputs of the same type from the same module. Two
  `CaloClusterCollection`s in the same event are kept separate by giving
  them different instance names — this is the mechanism that lets BFS
  clusters and GNN clusters coexist without ambiguity in downstream code.
- **FHiCL** is the configuration language art uses to wire modules into
  a job and pass parameters to them. Frozen recipe values
  (`tau_edge=0.20`, `bfs_expand_cut=10`, …) live there.
- **muse** is Mu2e's build/setup tool — it knows how to compile the
  Offline source tree and set up the matching runtime environment.

Skip this section if you already work in art day-to-day.

---

## 1. Integration picture envisioned

The new GNN clustering runs **alongside** the existing BFS
`CaloClusterMaker` — not as a replacement — and is itself **split into
two art modules** so the per-disk graph is a first-class data product
that other analyses can consume. `CaloHitGraphMaker` reads the same
`CaloHitCollection` BFS reads and emits a `CaloHitGraphCollection`;
`CaloClusterMakerGNN` consumes the graph collection, runs ONNX
inference + cluster assembly, and emits a `CaloClusterCollection` under
instance `"GNN"`. Existing analyses keep reading BFS exactly as today;
new analyses opt into the GNN clustering by reading the `"GNN"`
instance, and study modules wanting the raw graph can read the graph
collection directly.

```
event ─▶ ... ─▶ CaloHitMaker ─▶ CaloHitCollection ──┬──▶ CaloClusterMaker             (existing, BFS)
                                                    │       └─▶ CaloClusterCollection
                                                    │           [instance: default]
                                                    │
                                                    └──▶ CaloHitGraphMaker             (NEW — step ①)
                                                            └─▶ CaloHitGraphCollection
                                                                    │
                                                                    ▼
                                                         CaloClusterMakerGNN           (NEW — steps ②③)
                                                            ├─ ② ONNX inference (CaloClusterNet)
                                                            └─ ③ cluster assembly (BFS-with-ExpandCut)
                                                            └─▶ CaloClusterCollection
                                                                [instance: "GNN"]
```

### 1.1 What lives where

The new code follows the existing `CaloCluster` package layout: the art
module is a thin wrapper, and the heavier algorithm logic lives in
plain C++ helper classes (mirroring the way BFS splits its module from
`ClusterFinder.cc` / `ClusterAssociator.cc`).

| Piece | Location | Role |
|---|---|---|
| `CaloHitGraphMaker_module.cc` | `Offline/CaloCluster/src/` | First EDProducer (step ①). Calls `GnnGraphBuilder` once per disk, normalises features, emits a `CaloHitGraphCollection`. |
| `CaloClusterMakerGNN_module.cc` | `Offline/CaloCluster/src/` | Second EDProducer (steps ②③). Loads the ONNX session at construction, consumes the graph collection, runs inference + assembly, emits a `CaloClusterCollection` under instance `"GNN"`. |
| `CaloHitGraph` data product | `Offline/RecoDataProducts/{src,inc}/` (TBD) | New data product carrying per-disk tensors (`x`, `edge_index`, `edge_attr`) plus per-node `art::Ptr<CaloHit>` back-references. Schema, persistence, and exact directory still open (§2.2). |
| `GnnGraphBuilder.{cc,hh}` | `Offline/CaloCluster/{src,inc}/` | Step ① helper used by `CaloHitGraphMaker`. C++ port of `src/data/graph_builder.py`. |
| `GnnClusterAssembler.{cc,hh}` | `Offline/CaloCluster/{src,inc}/` | Step ③ helper used by `CaloClusterMakerGNN`. C++ port of `src/inference/cluster_reco.py`. |
| FHiCL configs | `Offline/CaloCluster/fcl/` | Per-module parameter sets (artifact paths, frozen recipe values as defaults). |
| ONNX artifact (`.onnx`) | `outputs/onnx/calo_cluster_net_v2_stage1.onnx` | The trained model, 2.6 MB, opset 17. `metadata_props` carries the version string the C++ module asserts at load. |
| Norm stats sidecar | `outputs/onnx/calo_cluster_net_v2_stage1.norm.json` | 28 floats (z-score means/stds) the graph maker reads to normalise features. |

The two artifacts (ONNX file + sidecar) need to live somewhere both
version-controlled and accessible at job runtime — a separate decision
(see §2.6).

### 1.2 What each module does, per event, per disk

The same ①→②③ pipeline used by the Python prototype, now split across
two C++ producers. Each is called once per event, with each calorimeter
disk producing one `CaloHitGraph` and one or more `CaloCluster`s.

`CaloHitGraphMaker::produce()` (step ①):

```
1. Collect CaloHits on each disk.
2. Graph construction (helper class GnnGraphBuilder):
   - compute 6 node features per hit, 8 edge features per pair
   - radius graph at r_max=210mm, time filter |dt|<25ns,
     kNN fallback k_min=3, degree cap k_max=20
   - z-score features using the sidecar stats
3. Pack tensors + per-node art::Ptr<CaloHit> back-references into a
   CaloHitGraph; emit one entry per disk into CaloHitGraphCollection.
```

`CaloClusterMakerGNN::produce()` (steps ②③):

```
1. Read CaloHitGraphCollection; for each disk's graph:
2. ② ONNX inference
   - one call: session.Run({x, edge_index, edge_attr}) -> edge_logits
3. ③ cluster assembly (helper class GnnClusterAssembler)
   - sigmoid + symmetrise directed scores
   - threshold at tau_edge = 0.20
   - BFS traversal seeded from highest-energy hits; only hits with
     E >= bfs_expand_cut = 10 MeV continue recruiting neighbours
   - cleanup: drop clusters with <2 hits or <10 MeV total
4. Resolve cluster→hit references through the graph product's
   art::Ptr<CaloHit>; build CaloClusters with energy = Σ eDep,
   centroid = energy-weighted, time = seed-hit time (matching BFS).
5. Append to a CaloClusterCollection; emit under instance name "GNN".
```

Reference Python (the spec for the C++ port):
- ① `src/data/graph_builder.py`
- ② `src/models/calo_cluster_net_deploy.py`
- ③ `src/inference/cluster_reco.py` (`reconstruct_clusters`, `_bfs_expand_cut`)

Parity bar: `edge_logits` agree to ~1e-5; cluster labels byte-identical
to Python on the val split. PyTorch ↔ ONNX Runtime parity is already
proven (`docs/onnx_deployment.md` §6); the split adds an additional
graph-product parity check at the module boundary (16g Stage 1).

---

## 2. Design decisions

Status as of the 2026-04-29 meeting with Sophie + Andy. Decided
subsections are marked **DECIDED** in the heading: §2.1, §2.2, §2.4,
§2.5, §2.6, §2.7, §2.8. §2.3 (BFS coexistence) was uncontested. All
subsections are now closed. Each records the final decision and the
reasoning that led there.

### 2.1 Repo location — DECIDED: `Offline/CaloCluster/`

| Option | Pros | Cons |
|---|---|---|
| **`Offline/CaloCluster/` (chosen)** | Same package as BFS `CaloClusterMaker`; reconstruction lives in Offline by convention | Need to mirror Andy's ArtAnalysis onnxruntime build pattern into Offline once it's mature |
| `ArtAnalysis/` | Andy's `Mu2e/ArtAnalysis#4` lands the build pattern here first | Architectural mismatch — this is reconstruction, not analysis |

**Decision (Sam, pre-meeting):** `Offline/CaloCluster/`. The GNN does
the same job as the BFS `CaloClusterMaker` — clustering — so it
belongs in the same package. We follow Andy's ArtAnalysis pattern
once it's settled and mirror it into Offline.

### 2.2 Module boundary — DECIDED: split (graph producer + cluster producer)

The pipeline has three logical pieces — graph construction (①), ONNX
inference (②), cluster assembly (③). They could all live in *one*
art module, or each piece could be its own module passing intermediate
data products to the next one in the chain.

| Option | Pros | Cons |
|---|---|---|
| Single EDProducer | Simpler; no new intermediate data product to define and persist; matches the shape of the BFS module | The graph isn't reusable as a separate object — once-per-event work but no other modules can consume it |
| **Split (graph producer + cluster producer) — chosen** | Graph is a first-class data product other modules (study analyzers, alternative clusterers, future combination modules) can consume; each piece is testable in isolation against its own parity dump; cleaner mapping onto framework idioms | More files; one new data product to declare and (optionally) serialise |

**Decision (Sam, 2026-04-29 meeting):** split. The graph collection
becomes a re-usable intermediate, useful for any downstream study that
wants to re-cluster at different thresholds or compare alternative
edge-classifier heads without re-running graph construction.

**Sub-decisions resolved (Sam, 2026-05-03):**

- **Schema** — flat tensors `x` (`std::vector<float>`, length 6N), `edge_index` (`std::vector<int64_t>`, length 2E), `edge_attr` (`std::vector<float>`, length 8E), plus `(N, E)` shape headers and a per-node `std::vector<art::Ptr<CaloHit>>` of length N for cluster-assembly back-references. **No** per-event feature-name metadata: feature names live in the ONNX `metadata_props` (§2.7 / 16j) and are validated once at job start, not per-event.
- **Persistence** — **transient.** `produces<CaloHitGraphCollection>()` without `classes_def.xml` registration. Building one graph for one disk is sub-millisecond, so disk caching adds I/O for no analysis that needs it. A future study can rerun the chain to recover the graph.
- **Helper-class home** — both `GnnGraphBuilder` and `GnnClusterAssembler` under `Offline/CaloCluster/{src,inc}/`, each called from its respective module — same package as BFS `ClusterFinder` / `ClusterAssociator`.
- **Module class + swappability** — a single C++ class `CaloClusterMakerGNN`, instanced as many times as needed via FHiCL. Each instance specifies `model_path`, `expected_model_version`, recipe values (`tau_edge`, `bfs_expand_cut`, `min_hits`, `min_energy_mev`), and an `output_instance` name. The graph maker is `CaloHitGraphMaker` (no instance qualifier — there is no other graph producer to disambiguate against).

  - **Single-model production** (CCN+BFS10 default): one `CaloClusterMakerGNN` instance, module label `caloClusterMakerGNN`, output instance `"GNN"`, `tau_edge=0.20`, `expected_model_version="calo-cluster-net-v2-stage1"`.
  - **A/B comparison job**: two instances of the same class — e.g., `caloClusterMakerCCN` (output instance `"CCN"`) and `caloClusterMakerSEN` (output instance `"SEN"`, `tau_edge=0.26`, `expected_model_version="simple-edge-net-v2"`). Both feed off the same `CaloHitGraphCollection`. Downstream picks via `(module_label, instance_name)`.

  Adding a third model later is config-only: train, export with appropriate `metadata_props`, drop in a new `.onnx`, declare a new instance. No C++ change.

### 2.3 Coexistence with BFS

**Position:** Coexist, do not replace. Both modules run in the standard
reco chain; they emit `CaloClusterCollection`s under different instance
names (`"GNN"` vs the existing default). No existing analysis sees any
change unless it explicitly asks for the `"GNN"` instance.

The CPU cost of running both is small: ONNX inference is ~0.9 ms/graph
on CPU (per `docs/onnx_deployment.md` §6), well under the rest of reco.
Confirm this is acceptable for the production sequence.

### 2.4 Graph-construction implementation — DECIDED: brute-force pairwise

The Python code finds edge candidates with `scipy.spatial.cKDTree` —
for each pair of hits on the disk, keep the pair if their crystal
positions are within `r_max = 210 mm`. The C++ port has to reproduce
the same edge set; the only question is *how* to find those pairs.

| Option | Pros | Cons |
|---|---|---|
| **Brute-force pairwise distance loop (chosen)** | Faithful port of the Python algorithm; trivially fast at our hit counts (mean ~12 hits/disk, max ~65 — an O(N²) loop is ~2K ops worst case); easy to parity-check | None at this scale |
| Use Offline `cal.neighbors()` + `cal.nextNeighbors()` + long-range augmentation | Reuses the geometry service's precomputed crystal-adjacency tables | Those tables only cover the 2nd ring (~70 mm reach); `r_max = 210 mm` is roughly 6 crystal-widths, so we'd silently drop the long-range edges that the 100% pair-recall gate needs. Would have to augment with explicit long-range search and re-run the gate. |

**Decision (Sam, pre-meeting):** Brute-force pairwise. With at most
~65 hits per disk per event, the inner loop is already cheap. The
neighbor-list path is premature optimization and risks breaking the
pair-recall guarantee.

### 2.5 onnxruntime availability — DECIDED: link central muse onnxruntime via `u092`

To run inference inside Offline, the build needs to link against the
ONNX Runtime C++ library. The central muse onnxruntime is now reachable
through the `u092` qualifier — Sophie shared a `u092` muse manifest on
2026-04-30 (Slack DM, file `u092`, 6.2 KB, text/plain).

**Decision (Sam, 2026-04-29 meeting):** link the central muse
onnxruntime via the `u092` qualifier. The "stub the ONNX call" interim
option is retired — there's no need to fake inference now that the
runtime is available. The "local install" path is also off the table.

**Setup recipe (one-time):**

```bash
# install Sophie's u092 manifest under a `muse/` directory in the working environment,
# then activate it before building Offline:
mu2einit
muse setup -q u092
```

The `SConscript` change for both new modules (16d-graph and 16d-cluster)
is a single `'onnxruntime'` dependency line, mirroring the entry in
`Mu2e/ArtAnalysis#4`.

### 2.6 ONNX artifact + sidecar location at runtime — DECIDED: `ConfigFileLookupPolicy`

Two files need to be readable from the muse job environment:
- `calo_cluster_net_v2_stage1.onnx` (2.6 MB)
- `calo_cluster_net_v2_stage1.norm.json` (~1 KB)

Mu2e's standard pattern for locating runtime artifacts (geometry,
calibration, ML weights) is **`art::ConfigFileLookupPolicy`**: FHiCL
specifies a relative path; the framework resolves it against the build
environment's search path at job start.

**Decision (Sam, post-review of `Mu2e/ArtAnalysis#4`):** use
`ConfigFileLookupPolicy`. This is what reviewers asked Andy to do in
his PR; we follow the same pattern from the start. The two artifacts
go into a versioned location under `Offline/` (exact subdirectory TBD —
likely `Offline/CaloCluster/data/` or a shared `Offline/Mu2eData/`),
and the FHiCL specifies the relative names.

### 2.7 Version-string carrier — DECIDED: ONNX `metadata_props`

If the model is retrained with different feature shapes (or anything
else that changes the tensor layout) and re-exported, the C++ module
must reject it loudly rather than running silently with stale weights.
We embed a short version string somewhere the module can check at load.

| Option | Pros | Cons |
|---|---|---|
| **ONNX `metadata_props` (chosen)** | Travels with the artifact; one file, no chance of desync | Reading it requires an onnxruntime API call |
| `.version` sidecar | Trivial to read (plain text) | Separate file — can drift out of sync with the `.onnx` |

**Decision (Sam, 2026-04-29 meeting):** `metadata_props`. Concrete
plan: `scripts/export_onnx.py` writes a `model_version` entry into the
ONNX `metadata_props` map after export; the C++ session loader reads
it back and compares against an expected value passed in via FHiCL,
aborting on mismatch. Wired up in 16b.

### 2.8 Ownership split — DECIDED: Sam writes 16d–16g; Andy + Sophie review

**Sam writes** the EDProducer, the graph builder, the cluster
assembler, the FHiCL, and the parity tests (Tasks 16d–16g). **Andy
reviews** following his ArtAnalysis#4 pattern. **Sophie reviews** the
calorimeter-group-facing pieces (FHiCL exposure, instance names,
downstream consumer documentation).

---

## 2.9 Patterns to mirror / pitfalls to avoid (from `ArtAnalysis#4`)

Andy's `Mu2e/ArtAnalysis#4` (TrackQuality + ONNX Runtime) is the
template. Reading the open review comments on that PR, we should
pre-empt the following so reviewers don't have to re-flag them on ours:

- **Use `ConfigFileLookupPolicy`** for the `.onnx` (and the norm
  sidecar) path. Don't hardcode (§2.6).
- **`SConscript`** adds `'onnxruntime'` as a dependency line in the
  standard indentation. No tabs, no extra blank lines.
- **Constructor member initialisation order** (env → session options →
  session → IO metadata) is fragile — document it with a clear comment
  block above the declarations explaining why this order matters.
- **Don't mutate the output tensor after inference.** Read the values
  out into local variables; don't write into the buffer the runtime
  returned.
- **Avoid unnecessary type casts.** ONNX Runtime returns `float`;
  consume it as `float` end-to-end unless there's a specific reason to
  promote.
- **No dead helpers.** Andy left a `print_shape` debug function that
  reviewers asked be removed; don't carry the same baggage.

These aren't decisions — they're known-good (or known-bad) C++ shapes
to use (or avoid) when writing 16d–16f.

---

## 3. Open questions (not strong positions)

**Newly opened post-meeting**

- **Sophie's generic ONNX utils.** Sophie is translating Leo's code
  into ONNX in parallel and explicitly said *"some generic utils might
  help"* (DM, 2026-04-29). Decision: does `CaloClusterMakerGNN`
  consume those utils for session loading / sidecar reading / version
  checking, or roll its own? Ask Sophie before starting 16d-cluster.
- **PR sequencing.** EventNtuple `ancestorSimIds` patch needs to land
  in `Mu2e/EventNtuple` before any Offline PR depending on the branch.
  Sophie asked for this PR on 2026-04-30.
- **Training repo (`Mu2e/MLTrain`).** Sophie pointed at this repo for
  training code on 2026-04-29. Move now or after the deployment PR
  ships? Position-leaning: don't block deployment on it.

**Standing**

- Should the GNN module also emit a *match map* (GNN cluster → BFS
  cluster) for downstream comparison studies, or is that better done by
  a separate analyzer that reads both `CaloClusterCollection`s?
- Diagnostics: do we want a parallel monitoring producer/analyzer that
  computes per-event GNN-vs-BFS residuals (energy / centroid / time)
  during the rollout? Probably yes; not necessarily a deliverable for
  the first PR.

(Recipe-parameter overridability — previously open — is now **required**:
swappability between SimpleEdgeNet (`tau_edge=0.26`) and CCN
(`tau_edge=0.20`) makes per-instance FHiCL parameters non-negotiable.
Captured in §2.2.)

---

## 4. Status (post-meeting)

Meeting held 2026-04-29 with Sophie + Andy. Status as of 2026-05-03.

| Item | Status |
|---|---|
| CaloClusterNet trained, frozen, CCN+BFS10 recipe locked | done |
| ONNX export (`scripts/export_onnx.py`, opset 17) | done |
| PyTorch ↔ ONNX Runtime parity validated (`scripts/validate_onnx.py`) | done — 9.06e-06 max diff, zero threshold flips on 166K edges |
| Tensor / artifact contract documented | `docs/onnx_deployment.md` |
| Norm stats sidecar (16a) | done |
| Module boundary (§2.2) | DECIDED — split (graph producer + cluster producer) |
| onnxruntime availability (§2.5) | DECIDED — link central muse onnxruntime via `u092`; manifest in hand |
| Version-string carrier (§2.7) | DECIDED — `metadata_props` |
| Version-string guard (16b) | done — `metadata_props` stamped on export; C++ assertion lands with 16d-cluster |
| Feature-spec metadata_props (16j Python side) | done — `node_features` / `edge_features` stamped; C++ handshake lands with 16d |
| `CaloHitGraph` data product schema (§2.2 sub) | DECIDED — flat tensors + `art::Ptr<CaloHit>` back-references |
| Persistence of graph product (§2.2 sub) | DECIDED — transient |
| Module-class swappability (§2.2 sub) | DECIDED — single model-agnostic `CaloClusterMakerGNN`, instanced via FHiCL |
| Ownership split (§2.8) | DECIDED — Sam writes 16d–16g; Andy + Sophie review |
| u092 envset installed; Offline + EventNtuple built under u092 | done |
| EDProducer skeletons (16d-graph, 16d-cluster) | unblocked, ready to start |
| Graph construction C++ port (16e) | planned |
| Cluster assembly C++ port (16f) | planned |
| C++↔Python parity harness (16g) | planned (now two-stage + end-to-end) |
| SimpleEdgeNet ONNX export (16i) | planned |
| Generic ONNX utils (Sophie / Leo) | watching — may inform 16d-cluster |
| EventNtuple PR for `ancestorSimIds` | requested by Sophie 2026-04-30; pending |

---

## 5. What unblocked after the meeting

§2.1–§2.8 are all decided; §2.3 (BFS coexistence) confirmed. The §2.2
graph-product schema/persistence sub-decisions are also now locked
(see §2.2). Nothing in §2 is blocking the C++ work.

- §2.2 split confirmed ⇒ 16d expands to graph + cluster modules + the
  `CaloHitGraph` data product (16d-graph / 16d-cluster / 16d-product).
- §2.5 muse onnxruntime via `u092` ⇒ 16d-cluster can link the runtime
  directly, no stub stage needed; install `u092` first.
- §2.7 `metadata_props` ⇒ 16b is concrete (write into `metadata_props`,
  document the key/value, C++ asserts at load).
- §2.3 BFS coexistence confirmed ⇒ 16h FHiCL wiring straightforward
  (two new producers, no removal of BFS).
- After 16d-product, 16d-graph, 16d-cluster, 16e, 16f land:
  run 16g (two-stage parity + end-to-end) on val data, then write
  the PR against `Mu2e/Offline:main`.

---

## 6. References

- `docs/onnx_deployment.md` — full tensor / artifact / parity contract
- `docs/plan.md` Task 16 — itemized work plan with checkboxes
- `docs/findings.md` §7 — physics motivation for the CCN+BFS10 recipe
- Reference Python: `src/data/graph_builder.py`, `src/inference/cluster_reco.py`,
  `src/models/calo_cluster_net_deploy.py`
- Andy Edmonds's pattern PR: `Mu2e/ArtAnalysis#4` (TrackQuality ONNX integration)
