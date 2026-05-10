"""Stage A regime characterization (Tasks 18a, 19a).

Read a sample of standard NTS files for a target regime (no ancestry,
no calo-entrant truth needed) and compare per-disk-graph distributions
against a chosen training-set baseline. Quantifies how far a regime sits
from the model's training distribution before paying any reprocessing cost.

Distributions reported (per disk-graph):
  - hits per graph
  - edges per graph (built at r_max=210 mm, dt_max=25 ns)
  - mean hit energy
  - sum disk energy
  - BFS cluster multiplicity per disk (from caloclusters.diskID_)

Baseline is read from a packed train.pt (already-built graphs).
Target is read from raw NTS; graphs are built in-process with the same builder.

Examples:
  # Task 18a (legacy): MixLow vs MDC2025 train
  python scripts/characterize_run1b_pileup.py \
    --target-dir /pnfs/mu2e/tape/phy-nts/nts/mu2e/FlateMinusMixLow-KL/Run1B-004/root \
    --target-label MixLow --baseline mdc2025

  # Task 19a: MLT or OST vs MixLow train
  python scripts/characterize_run1b_pileup.py \
    --target-dir /pnfs/mu2e/tape/phy-nts/nts/mu2e/FlateMinusMixLowTriggerable-KL/Run1B-003/root \
    --target-label MLT --baseline mixlow \
    --output-dir outputs/task19a_mlt_stageA
"""

from pathlib import Path
import argparse
import sys

import numpy as np
import torch
import uproot

import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.graph_builder import build_graph
from src.geometry.crystal_geometry import load_crystal_map


R_MAX_MM = 210.0
DT_MAX_NS = 25.0
K_MIN = 3
K_MAX = 20

NTS_BRANCHES = [
    "calohits.crystalId_",
    "calohits.eDep_",
    "calohits.time_",
    "caloclusters.diskID_",
    "caloclusters.size_",
    "caloclusters.energyDep_",
]

BASELINE_DEFAULTS = {
    "mdc2025": (Path("data/processed/train.pt"),
                "MDC2025 train (with-field, with-pileup)"),
    "mixlow":  (Path("data/processed_run1b_mixlow/train.pt"),
                "MixLow train (no-field, low pileup)"),
}


def percentiles(arr, qs=(0.5, 0.75, 0.9, 0.95, 0.99)):
    if len(arr) == 0:
        return {q: 0 for q in qs}
    return {q: float(np.quantile(arr, q)) for q in qs}


def fmt_dist(arr, name):
    if len(arr) == 0:
        return f"  {name}: (empty)"
    p = percentiles(arr)
    return (f"  {name}:  n={len(arr):>7}  mean={np.mean(arr):>7.2f}  "
            f"med={p[0.5]:>6.1f}  p75={p[0.75]:>6.1f}  p90={p[0.9]:>6.1f}  "
            f"p95={p[0.95]:>6.1f}  p99={p[0.99]:>7.1f}  max={np.max(arr):>7.1f}")


def scan_target(files, crystal_map, max_events_per_file=None):
    n_hits_per_disk = []
    n_edges_per_disk = []
    mean_e_per_disk = []
    sum_e_per_disk = []
    bfs_clusters_per_disk = []
    bfs_cluster_size = []
    bfs_cluster_energy = []

    for fp in files:
        print(f"  reading {fp.name} ...", flush=True)
        tree = uproot.open(str(fp) + ":EventNtuple/ntuple")
        arrays = tree.arrays(NTS_BRANCHES, entry_stop=max_events_per_file)
        n_events = len(arrays)

        for ev in range(n_events):
            cryids = np.array(arrays["calohits.crystalId_"][ev], dtype=np.int64)
            n_total = len(cryids)
            if n_total == 0:
                continue
            energies = np.array(arrays["calohits.eDep_"][ev], dtype=np.float64)
            times = np.array(arrays["calohits.time_"][ev], dtype=np.float64)
            xs = np.zeros(n_total, dtype=np.float64)
            ys = np.zeros(n_total, dtype=np.float64)
            disks = np.full(n_total, -1, dtype=np.int64)
            for i, c in enumerate(cryids):
                c = int(c)
                if c in crystal_map:
                    disks[i], xs[i], ys[i] = crystal_map[c]

            bfs_disks = np.array(arrays["caloclusters.diskID_"][ev], dtype=np.int64)
            bfs_sizes = np.array(arrays["caloclusters.size_"][ev], dtype=np.int64)
            bfs_energies = np.array(arrays["caloclusters.energyDep_"][ev], dtype=np.float64)

            for disk_id in (0, 1):
                m = disks == disk_id
                n = int(m.sum())
                if n < 2:
                    continue
                d_pos = np.stack([xs[m], ys[m]], axis=1)
                d_t = times[m]
                d_e = energies[m]

                edge_index, _ = build_graph(
                    d_pos, d_t,
                    r_max=R_MAX_MM, dt_max=DT_MAX_NS, k_min=K_MIN, k_max=K_MAX,
                )

                n_hits_per_disk.append(n)
                n_edges_per_disk.append(int(edge_index.shape[1]) // 2)
                mean_e_per_disk.append(float(np.mean(d_e)))
                sum_e_per_disk.append(float(np.sum(d_e)))

                bfs_m = bfs_disks == disk_id
                bfs_clusters_per_disk.append(int(bfs_m.sum()))
                for sz, en in zip(bfs_sizes[bfs_m], bfs_energies[bfs_m]):
                    bfs_cluster_size.append(int(sz))
                    bfs_cluster_energy.append(float(en))

    return {
        "n_hits": np.array(n_hits_per_disk),
        "n_edges": np.array(n_edges_per_disk),
        "mean_e": np.array(mean_e_per_disk),
        "sum_e": np.array(sum_e_per_disk),
        "bfs_clusters_per_disk": np.array(bfs_clusters_per_disk),
        "bfs_cluster_size": np.array(bfs_cluster_size),
        "bfs_cluster_energy": np.array(bfs_cluster_energy),
    }


def scan_packed(path):
    """Pull n_hits / n_edges / energies straight from packed v2 graphs."""
    graphs = torch.load(path, weights_only=False)
    n_hits = []
    n_edges = []
    mean_e = []
    sum_e = []
    for g in graphs:
        n = int(g.x.shape[0])
        # x[:, 0] is log_E (per CLAUDE.md). exp() recovers E in MeV-ish units.
        e = np.exp(g.x[:, 0].numpy())
        n_hits.append(n)
        n_edges.append(int(g.edge_index.shape[1]) // 2)
        mean_e.append(float(np.mean(e)))
        sum_e.append(float(np.sum(e)))
    return {
        "n_hits": np.array(n_hits),
        "n_edges": np.array(n_edges),
        "mean_e": np.array(mean_e),
        "sum_e": np.array(sum_e),
    }


def emit(out_lines, line):
    print(line)
    out_lines.append(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-dir", type=Path, required=True,
                        help="Standard NTS dir to characterize (raw .root files)")
    parser.add_argument("--target-label", type=str, required=True,
                        help="Display label for target (e.g. 'MLT', 'OST', 'MixLow')")
    parser.add_argument("--baseline", choices=list(BASELINE_DEFAULTS.keys()),
                        default="mdc2025",
                        help="Baseline packed-train selection")
    parser.add_argument("--baseline-path", type=Path, default=None,
                        help="Optional override for baseline packed train.pt")
    parser.add_argument("--baseline-label", type=str, default=None,
                        help="Optional override for baseline display label")
    parser.add_argument("--n-files", type=int, default=3)
    parser.add_argument("--max-events-per-file", type=int, default=500)
    parser.add_argument("--crystal-csv", type=Path,
                        default=Path("data/crystal_geometry.csv"))
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="If set, save diagnostics.npz + summary.txt here")
    args = parser.parse_args()

    base_path, base_label = BASELINE_DEFAULTS[args.baseline]
    if args.baseline_path is not None:
        base_path = args.baseline_path
    if args.baseline_label is not None:
        base_label = args.baseline_label

    out_lines = []
    emit(out_lines, f"r_max={R_MAX_MM} mm, dt_max={DT_MAX_NS} ns, "
                    f"k_min={K_MIN}, k_max={K_MAX}")
    emit(out_lines, f"max_events_per_file={args.max_events_per_file}")
    emit(out_lines, f"target: {args.target_label} ({args.target_dir})")
    emit(out_lines, f"baseline: {base_label} ({base_path})\n")

    crystal_map = load_crystal_map(args.crystal_csv)
    emit(out_lines, f"loaded crystal map: {len(crystal_map)} crystals\n")

    all_files = sorted(args.target_dir.rglob("*.root"))
    files = all_files[: args.n_files]
    emit(out_lines,
         f"{args.target_label}: reading {len(files)} files of {len(all_files)} available")
    tgt = scan_target(files, crystal_map, args.max_events_per_file)

    emit(out_lines, f"\n{base_label}: loading {base_path}")
    base = scan_packed(base_path)

    emit(out_lines, "\n=== Per-disk-graph distributions ===")
    emit(out_lines, f"{base_label}:")
    emit(out_lines, fmt_dist(base["n_hits"], "hits/disk    "))
    emit(out_lines, fmt_dist(base["n_edges"], "edges/disk   "))
    emit(out_lines, fmt_dist(base["mean_e"], "mean E (MeV) "))
    emit(out_lines, fmt_dist(base["sum_e"], "sum  E (MeV) "))

    emit(out_lines, f"\n{args.target_label}:")
    emit(out_lines, fmt_dist(tgt["n_hits"], "hits/disk    "))
    emit(out_lines, fmt_dist(tgt["n_edges"], "edges/disk   "))
    emit(out_lines, fmt_dist(tgt["mean_e"], "mean E (MeV) "))
    emit(out_lines, fmt_dist(tgt["sum_e"], "sum  E (MeV) "))

    emit(out_lines, f"\n{args.target_label} BFS reco (per disk):")
    emit(out_lines, fmt_dist(tgt["bfs_clusters_per_disk"], "BFS clusters/disk "))
    emit(out_lines, fmt_dist(tgt["bfs_cluster_size"], "BFS cluster size  "))
    emit(out_lines, fmt_dist(tgt["bfs_cluster_energy"], "BFS cluster E (MeV)"))

    emit(out_lines, f"\n=== Ratios ({args.target_label} / {args.baseline}) ===")
    for key, label in [
        ("n_hits", "hits"),
        ("n_edges", "edges"),
        ("mean_e", "mean E"),
        ("sum_e", "sum E"),
    ]:
        a = tgt[key]
        b = base[key]
        if len(a) == 0 or len(b) == 0:
            continue
        emit(out_lines,
             f"  {label:<8}: median {np.median(a)/max(np.median(b),1e-9):>5.2f}x   "
             f"mean {np.mean(a)/max(np.mean(b),1e-9):>5.2f}x   "
             f"p95 {np.quantile(a,0.95)/max(np.quantile(b,0.95),1e-9):>5.2f}x")

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        np.savez(args.output_dir / "diagnostics.npz",
                 target_label=args.target_label,
                 baseline_label=base_label,
                 baseline_kind=args.baseline,
                 **{f"target_{k}": v for k, v in tgt.items()},
                 **{f"baseline_{k}": v for k, v in base.items()})
        with open(args.output_dir / "summary.txt", "w") as f:
            f.write("\n".join(out_lines) + "\n")
        print(f"\nSaved diagnostics + summary to {args.output_dir}/")


if __name__ == "__main__":
    main()
