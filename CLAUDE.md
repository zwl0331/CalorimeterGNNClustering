# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Git conventions

Commit-message format, granularity, branching, and what-not-to-commit rules live in `docs/git_conventions.md`. Follow them for every commit: imperative subject ≤72 chars with a type prefix (`feat`/`fix`/`docs`/`refactor`/`perf`/`chore`/`data`/`wip`), one logical change per commit, update `docs/findings.md` in the same commit when overturning a prior finding.

**Attribution (overrides all):** never include Claude authorship or co-authorship in git commits. Do not add `Co-Authored-By`, `Generated-By`, or any similar trailer referencing Claude, AI, or Anthropic. Commit messages should read as if written by the user. Applies to all commits — new, amended, or rebased.

# GNN Calorimeter Clustering

## Project overview

GNN edge-classification clusterer for the Mu2e electromagnetic calorimeter — replaces `CaloClusterMaker` (Offline). Built with PyTorch Geometric. Full architecture, features, and results: `README.md`. Task history: `docs/plan.md`. Research findings: `docs/findings.md`.

---

## Environment

**Always activate before running any Python:**
```bash
source setup_env.sh   # works in both interactive and non-interactive shells
```

`setup_env.sh` sources `setupmu2e-art.sh`, activates `ana 2.6.1` (with non-interactive fallback), and adds user site-packages + project src to `PYTHONPATH`. `mu2einit` and `pyenv` are interactive-shell functions — they don't work in scripts or `bash -c`.

**Hidden gotchas:**
- **`set -e` ordering:** Mu2e env scripts return non-zero exit codes internally. Place `set -e` **after** `source setup_env.sh`, not before. See `scripts/build_all_graphs.sh`.
- **Large file globs:** `data/processed/` can hold 50K+ `.pt` files. `rm *.pt` hits "Argument list too long" — use `find -delete`.
- **No torch-cluster:** Not installed in `ana 2.6.1`. `radius_graph` / `knn_graph` are unavailable; use `scipy.spatial.cKDTree` (see `src/data/graph_builder.py`).
- **No GPU on login node:** Training and GPU eval must run on a GPU node (A100 MIG ~20 GB).
- **`OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1`** for any script reading ROOT files (limits scipy thread contention).

**Confirmed package versions (ana 2.6.1):** Python 3.12.13, torch 2.5.1, torch_geometric 2.7.0 (user site-packages), uproot 5.7.2, scipy 1.17.1.

**Long-running jobs:** Use `tmux` (`tmux new -s <name>`, `tmux attach -t <name>`, `tmux list-sessions`). Batch scripts should `tee` to a log file for monitoring without attaching.

**Build & reprocessing:** muse builds, batch reprocessing, FermiGrid submission — see `docs/build_runbook.md`. The local EventNtuple checkout (`/exp/mu2e/app/users/wzhou2/working_dir/`) adds the `calomcsim.ancestorSimIds` branch — this is what makes v2 ROOT files distinct from upstream.

---

## Running things

Pipeline commands (build → train → tune → evaluate → plot) are in `README.md`. CLAUDE-specific entry points:

```bash
source setup_env.sh

# Smoke test (confirm environment)
python3 scripts/smoke_test_env.py

# Run all unit tests (unittest, NOT pytest)
python3 -m unittest discover -s tests -p "test_*.py" -v

# Run a single test file or method (faster iteration)
python3 -m unittest tests.test_graph_builder
python3 -m unittest tests.test_graph_builder.<TestClass>.<test_method>
```

For training, evaluation, threshold tuning, plotting, and Run1B/MixLow pipeline commands: README + `scripts/`. Prefix any ROOT-reading script with `OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1`.

---

## Default model recipe

For deployment, downstream physics analysis, or any task that needs *the* recommended GNN clusterer, use **CCN+BFS10**: CaloClusterNet edge logits + BFS-style traversal at `bfs_expand_cut = 10 MeV`.

| Item | Value |
|------|-------|
| Config | `configs/calo_cluster_net.yaml` |
| Frozen checkpoint | `outputs/runs/calo_cluster_net_v2_stage1/checkpoints/best_model.pt` |
| Edge threshold | `inference.tau_edge: 0.20` |
| BFS expand cut | `inference.bfs_expand_cut: 10` (MeV) |

CCN+BFS10 improves mean |ΔE| by ~20% on `E_reco ≥ 50 MeV` clusters vs legacy BFS while matching or beating BFS on every standard clustering metric. Don't switch recipes without explicit reason — full numbers in `docs/findings.md`.

---

## ONNX deployment

Deployment to Mu2e Offline goes through `scripts/export_onnx.py` + `scripts/export_norm_stats.py` + `scripts/validate_onnx.py`. Pipeline, contract, C++ interface: `docs/onnx_deployment.md`. Offline integration plan: `docs/offline_integration.md`.

**Parity contract:** `tests/parity/calo_cluster_net_v2_stage1.parity.json` is a committed snapshot of edge-logit outputs on a fixed input. Any change to the model, pre/post-processing, or ONNX export must regenerate this file and confirm bit-for-bit consistency, or document the diff in the commit body.

---

## Key data paths

| What | Path |
|------|------|
| MDC2025 v2 ROOT files | `root_files_v2/` — train split deleted 2026-04-13 (~68 GB freed); val + test still present (15 files, 29 GB). Train reproducible via FermiGrid (cluster `90854576`). |
| Run1B v2 ROOT files | `root_files_run1b/` — 20 files via FermiGrid (cluster `27583402`, no-field scenario) |
| MDC2025 MCS art files | `/pnfs/mu2e/persistent/datasets/phy-sim/mcs/mu2e/FlateMinusMix1BBTriggered/MDC2025af_best_v1_1/art/` (50 files, run 001430) |
| Run1B MCS art files | `/pnfs/mu2e/tape/phy-sim/mcs/mu2e/FlateMinus-KL/Run1Bah_best_v1_4-001/art/` (20 files, 24 GB) |
| Run1B NTS files (no ancestry) | `/pnfs/mu2e/tape/phy-nts/nts/mu2e/FlateMinus-KL/Run1B-005/root/` (20 files, ~40K events each) |
| Packed graphs (MDC2025) | `data/processed/{train,val,test}.pt` (calo-entrant truth) |
| Norm stats | `data/normalization_stats.pt` |
| Crystal geometry | `data/crystal_geometry.csv`, `data/crystal_neighbors.csv` |
| Splits (frozen) | `splits/{train,val,test}_files.txt` (35/7/8 v2 files) |
| Training run outputs | `outputs/runs/<run_name>/` — naming: `<model>_v2_<stage>` for MDC2025, `<model>_run1b_mixlow_<stage>` for Run1B. Each dir has `checkpoints/best_model.pt`, training log, config snapshot. |
| EventNtuple build | `/exp/mu2e/app/users/wzhou2/working_dir/` (see `docs/build_runbook.md`) |
| v1 ROOT files | **Deleted** — originals on tape at `/pnfs/mu2e/tape/phy-nts/...` |

---

## Truth labeling

**Use calo-entrant truth exclusively.** All graph building, training, and evaluation must use `src/data/truth_labels_primary.py` against v2 ROOT files (with `calomcsim.ancestorSimIds`). Old SimParticle-level truth (`truth_labels.py`) is retained for reference only — never use for new work. Rationale, ambiguity statistics, re-evaluation numbers: `docs/findings.md` §1.

---

## Documentation map

| Doc | Purpose |
|-----|---------|
| `docs/plan.md` | Task history and progress |
| `docs/findings.md` | Experimental results, physics insights, conclusions (v1, v2, Run1B, cluster-physics, failure audits, CCN+BFS10) |
| `docs/onnx_deployment.md` | ONNX deployment contract + C++ interface |
| `docs/offline_integration.md` | Mu2e Offline integration plan |
| `docs/presentation.md` | Beamer slide style spec (used by `scripts/make_slide_plots.py`) |
| `docs/git_conventions.md` | Commit-message rules (see top of this file) |
| `docs/build_runbook.md` | Mu2e build env (muse), batch reprocessing, FermiGrid submission |

---

## Critical invariants — never violate these

1. **Calo-entrant truth only:** All truth labels via `src/data/truth_labels_primary.py` using `calomcsim.ancestorSimIds` from v2 ROOT files. Never use old SimParticle-level truth (`truth_labels.py`) for training, evaluation, or graph building. Never use BFS reco (`clusterIdx_`) as truth.
2. **v2 ROOT files only:** All data pipelines must use v2 files (`root_files_v2/`) with ancestry branches. v1 files (`root_files/`) are legacy — do not use for new work.
3. **Split discipline:** `splits/` files are frozen (35/7/8 v2 files, same file IDs as v1). Never re-split.
4. **Normalization:** `data/normalization_stats.pt` computed from train split only.
5. **Threshold tuning:** Always on validation set. Never on test set. Report final metrics once on test.
6. **Graph unit:** One graph per disk per event. Disk 0 and Disk 1 are separate graphs.
7. **Graph r_max:** 210mm (tuned for 100% pair recall).
8. **CCN+BFS10 is the default deployment recipe.** Don't switch model or BFS_EC value without explicit reason — see "Default model recipe".
9. **ONNX parity is a contract.** Any change to the model, pre/post-processing, or ONNX export pipeline must regenerate `tests/parity/calo_cluster_net_v2_stage1.parity.json` and confirm consistency, or document the diff.

---

## Coding conventions

- All scripts must be runnable after `source setup_env.sh` with no other setup.
- Use `uproot` + `awkward`/`numpy` for ROOT file reading — not PyROOT.
- Graph objects are PyG `Data(x, edge_index, edge_attr, y_node, y_edge)`.
- Configs in YAML; load with `yaml.safe_load`. No hardcoded hyperparameters in model code. `configs/` is flat — no subdirectories.
- Tests use `unittest` (not pytest). Run with `python3 -m unittest discover -s tests -v`.
- Log experiment details (git hash, config, dataset manifest) to `outputs/runs/<run_name>/`.
- Model factory: `src/models/__init__.py` exports `build_model()` — shared by all scripts.
- `OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1` for scripts reading ROOT files.
