"""Verify the §2 graph construction gate (r_max=210mm → 100% pair recall) under
calo-entrant truth (§1).

The §2 result was established before the truth redefinition; calo-entrant truth
merges hits from different SimParticles in the same shower, so max same-cluster
pair distance can only grow. If any pair now exceeds r_max, §4 v2 metrics are
capped below 100% by graph topology, not by the model.

This script operates on packed v2 graphs `data/processed/{train,val,test}.pt`,
which preserve all hits as nodes (only edges are filtered at r_max). Packed
graphs store *raw physical* node features in mm (normalization is applied at
training time inside `CaloGraphDataset`, not at packing time), so distances
can be computed directly from `g.x[:, 2:4]`.

Output: per-split tables showing distribution of same-cluster pair distances
broken out by cluster multiplicity, plus pass/fail against r_max=210mm.
"""

from collections import defaultdict
from pathlib import Path
import argparse

import numpy as np
import torch


R_MAX = 210.0  # mm — current graph construction gate
SPLITS = ("train", "val", "test")


def physical_xy(x_raw):
    """Pull physical (x, y) in mm from raw node features.

    Node features are (log_E, time, x, y, r, rel_E); positions live at idx 2, 3
    and are stored unnormalized.
    """
    return x_raw[:, 2].numpy(), x_raw[:, 3].numpy()


def size_bucket(n):
    if n == 2:
        return "2-hit"
    if n <= 4:
        return "3-4 hit"
    return "5+ hit"


def scan_split(graphs, name):
    cluster_max_dist = []          # max pair-dist per multi-hit cluster
    cluster_max_dist_by_size = defaultdict(list)
    pair_total = 0
    pair_over = 0
    pair_total_by_size = defaultdict(int)
    pair_over_by_size = defaultdict(int)
    cluster_over = 0
    cluster_total = 0

    for g in graphs:
        x_phys, y_phys = physical_xy(g.x)
        labels = g.hit_truth_cluster.numpy()

        # Group node indices by truth cluster (drop unassigned/ambiguous = -1)
        cluster_to_idxs = defaultdict(list)
        for i, lab in enumerate(labels):
            if lab >= 0:
                cluster_to_idxs[int(lab)].append(i)

        for idxs in cluster_to_idxs.values():
            n = len(idxs)
            if n < 2:
                continue
            xs = x_phys[idxs]
            ys = y_phys[idxs]
            # Vectorized pairwise distances, upper triangle only
            dx = xs[:, None] - xs[None, :]
            dy = ys[:, None] - ys[None, :]
            d = np.sqrt(dx * dx + dy * dy)
            iu, ju = np.triu_indices(n, k=1)
            pair_d = d[iu, ju]

            bucket = size_bucket(n)
            n_pairs = pair_d.size
            n_over = int((pair_d > R_MAX).sum())
            cmax = float(pair_d.max())

            cluster_total += 1
            pair_total += n_pairs
            pair_over += n_over
            pair_total_by_size[bucket] += n_pairs
            pair_over_by_size[bucket] += n_over
            cluster_max_dist.append(cmax)
            cluster_max_dist_by_size[bucket].append(cmax)
            if n_over > 0:
                cluster_over += 1

    arr = np.asarray(cluster_max_dist) if cluster_max_dist else np.zeros(0)
    print(f"\n=== {name} (multi-hit clusters: {cluster_total}, pairs: {pair_total}) ===")
    if arr.size > 0:
        print(f"  max cluster pair-dist: {arr.max():.2f} mm")
        print(f"  99.9 percentile:       {np.quantile(arr, 0.999):.2f} mm")
        print(f"  99   percentile:       {np.quantile(arr, 0.99):.2f} mm")
        print(f"  95   percentile:       {np.quantile(arr, 0.95):.2f} mm")
        print(f"  median:                {np.quantile(arr, 0.5):.2f} mm")
    pct_clusters = 100.0 * cluster_over / max(cluster_total, 1)
    pct_pairs = 100.0 * pair_over / max(pair_total, 1)
    print(f"  clusters with any pair > {R_MAX:.0f}mm: {cluster_over} / {cluster_total} ({pct_clusters:.4f}%)")
    print(f"  pairs > {R_MAX:.0f}mm:                    {pair_over} / {pair_total} ({pct_pairs:.4f}%)")

    print("\n  by cluster multiplicity:")
    print(f"  {'bucket':<10} {'clusters':>10} {'pairs':>10} {'pairs>R':>10} {'%':>8} {'max-dist':>10} {'99.9%':>8}")
    for k in ("2-hit", "3-4 hit", "5+ hit"):
        ks = cluster_max_dist_by_size.get(k, [])
        if not ks:
            continue
        ka = np.asarray(ks)
        npairs = pair_total_by_size[k]
        nover = pair_over_by_size[k]
        pct = 100.0 * nover / max(npairs, 1)
        print(f"  {k:<10} {len(ks):>10} {npairs:>10} {nover:>10} {pct:>7.3f}% {ka.max():>9.2f} {np.quantile(ka, 0.999):>7.2f}")

    return {
        "n_clusters": cluster_total,
        "n_pairs": pair_total,
        "max_dist": float(arr.max()) if arr.size else 0.0,
        "p999": float(np.quantile(arr, 0.999)) if arr.size else 0.0,
        "clusters_over": cluster_over,
        "pairs_over": pair_over,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--splits", nargs="+", default=list(SPLITS))
    args = parser.parse_args()

    summaries = {}
    for split in args.splits:
        path = args.processed_dir / f"{split}.pt"
        print(f"\nLoading {path} ...")
        graphs = torch.load(path, weights_only=False)
        print(f"  {len(graphs)} graphs")
        summaries[split] = scan_split(graphs, split)

    print("\n=== Summary ===")
    print(f"  {'split':<8} {'clusters':>10} {'max':>10} {'99.9%':>8} {'cl>R':>8} {'pairs>R':>8}")
    for split, s in summaries.items():
        print(f"  {split:<8} {s['n_clusters']:>10} {s['max_dist']:>9.2f} {s['p999']:>7.2f} "
              f"{s['clusters_over']:>8} {s['pairs_over']:>8}")

    any_over = any(s["pairs_over"] > 0 for s in summaries.values())
    if any_over:
        print(f"\nGATE FAILURE: pairs > {R_MAX:.0f}mm exist under calo-entrant truth.")
    else:
        print(f"\nGATE HOLDS: all same-cluster pairs are within {R_MAX:.0f}mm under calo-entrant truth.")


if __name__ == "__main__":
    main()
