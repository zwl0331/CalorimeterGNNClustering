"""Task 18a — Stage A regime characterization.

Read a sample of FlateMinusMixLow-KL/Run1B-004 standard NTS files (no ancestry,
no calo-entrant truth) and compare per-disk-graph distributions against
MDC2025 train. Quantifies how much harder no-field-with-pileup is for graph
construction without needing any model inference.

Distributions reported (per disk-graph):
  - hits per graph
  - edges per graph (built at r_max=210 mm, dt_max=25 ns)
  - mean hit energy
  - BFS cluster multiplicity per disk (from caloclusters.diskID_)

For MDC2025 baseline, read existing packed train.pt (already-built graphs).
For MixLow, read raw NTS, build graphs in-process with the same builder.
"""

from collections import defaultdict
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


def scan_mixlow(files, crystal_map, max_events_per_file=None):
    n_hits_per_disk = []
    n_edges_per_disk = []
    mean_e_per_disk = []
    sum_e_per_disk = []
    bfs_clusters_per_disk = []   # number of BFS clusters per (event, disk)
    bfs_cluster_size = []         # size_ of each individual BFS cluster
    bfs_cluster_energy = []       # energyDep_ of each BFS cluster

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
            # Standard Run1B-004 NTS lacks crystalPos_; use crystal_map only.
            xs = np.zeros(n_total, dtype=np.float64)
            ys = np.zeros(n_total, dtype=np.float64)
            disks = np.full(n_total, -1, dtype=np.int64)
            for i, c in enumerate(cryids):
                c = int(c)
                if c in crystal_map:
                    disks[i], xs[i], ys[i] = crystal_map[c]

            # BFS clusters from reco (already built per event)
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

                edge_index, diag = build_graph(
                    d_pos, d_t,
                    r_max=R_MAX_MM, dt_max=DT_MAX_NS, k_min=K_MIN, k_max=K_MAX,
                )

                n_hits_per_disk.append(n)
                # Undirected edge count
                n_edges_per_disk.append(int(edge_index.shape[1]) // 2)
                mean_e_per_disk.append(float(np.mean(d_e)))
                sum_e_per_disk.append(float(np.sum(d_e)))

                # BFS clusters in this disk for this event
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
        # x[:, 0] is log_E (per CLAUDE.md). Recover E in MeV by exp() — but we
        # don't strictly need physical units here, only relative shape. Still,
        # report exp(log_E) so units match MixLow side.
        e = np.exp(g.x[:, 0].numpy())
        n_hits.append(n)
        # Packed graphs store directed edge_index (both directions); undirected
        # count is len/2.
        n_edges.append(int(g.edge_index.shape[1]) // 2)
        mean_e.append(float(np.mean(e)))
        sum_e.append(float(np.sum(e)))
    return {
        "n_hits": np.array(n_hits),
        "n_edges": np.array(n_edges),
        "mean_e": np.array(mean_e),
        "sum_e": np.array(sum_e),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mixlow-dir",
        type=Path,
        default=Path("/pnfs/mu2e/tape/phy-nts/nts/mu2e/FlateMinusMixLow-KL/Run1B-004/root"),
    )
    parser.add_argument("--n-files", type=int, default=3)
    parser.add_argument("--max-events-per-file", type=int, default=500)
    parser.add_argument(
        "--mdc2025-train", type=Path, default=Path("data/processed/train.pt")
    )
    parser.add_argument(
        "--crystal-csv", type=Path, default=Path("data/crystal_geometry.csv")
    )
    args = parser.parse_args()

    print(f"r_max={R_MAX_MM} mm, dt_max={DT_MAX_NS} ns, k_min={K_MIN}, k_max={K_MAX}")
    print(f"max_events_per_file={args.max_events_per_file}")

    crystal_map = load_crystal_map(args.crystal_csv)
    print(f"loaded crystal map: {len(crystal_map)} crystals\n")

    # Pick first n_files files
    all_files = sorted(args.mixlow_dir.rglob("*.root"))
    files = all_files[: args.n_files]
    print(f"MixLow: reading {len(files)} files of {len(all_files)} available")
    mix = scan_mixlow(files, crystal_map, args.max_events_per_file)

    print(f"\nMDC2025 train: loading {args.mdc2025_train}")
    mdc = scan_packed(args.mdc2025_train)

    print("\n=== Per-disk-graph distributions ===")
    print("MDC2025 train (with-field, with-pileup):")
    print(fmt_dist(mdc["n_hits"], "hits/disk    "))
    print(fmt_dist(mdc["n_edges"], "edges/disk   "))
    print(fmt_dist(mdc["mean_e"], "mean E (MeV) "))
    print(fmt_dist(mdc["sum_e"], "sum  E (MeV) "))

    print("\nFlateMinusMixLow-KL Run1B-004 (no-field, low pileup):")
    print(fmt_dist(mix["n_hits"], "hits/disk    "))
    print(fmt_dist(mix["n_edges"], "edges/disk   "))
    print(fmt_dist(mix["mean_e"], "mean E (MeV) "))
    print(fmt_dist(mix["sum_e"], "sum  E (MeV) "))

    print("\nMixLow BFS reco (per disk):")
    print(fmt_dist(mix["bfs_clusters_per_disk"], "BFS clusters/disk "))
    print(fmt_dist(mix["bfs_cluster_size"], "BFS cluster size  "))
    print(fmt_dist(mix["bfs_cluster_energy"], "BFS cluster E (MeV)"))

    print("\n=== Ratios (MixLow / MDC2025) ===")
    for key, label in [
        ("n_hits", "hits"),
        ("n_edges", "edges"),
        ("mean_e", "mean E"),
        ("sum_e", "sum E"),
    ]:
        a = mix[key]
        b = mdc[key]
        if len(a) == 0 or len(b) == 0:
            continue
        print(f"  {label:<8}: median {np.median(a)/max(np.median(b),1e-9):>5.2f}x   "
              f"mean {np.mean(a)/max(np.mean(b),1e-9):>5.2f}x   "
              f"p95 {np.quantile(a,0.95)/max(np.quantile(b,0.95),1e-9):>5.2f}x")


if __name__ == "__main__":
    main()
