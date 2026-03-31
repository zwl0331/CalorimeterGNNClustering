"""
Generate summary histograms from BFS baseline benchmark results.

Reads data/baseline/bfs_benchmark.csv (produced by baseline_existing.py)
and saves plots to data/baseline/plots/.

Usage:
    source setup_env.sh
    python3 scripts/baseline_plots.py
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_benchmark(path="data/baseline/bfs_benchmark.csv"):
    """Load per-event benchmark CSV into dict of arrays."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    data = {}
    for key in rows[0]:
        vals = []
        for r in rows:
            v = r[key]
            try:
                v = float(v)
                if v == int(v):
                    v = int(v)
            except ValueError:
                pass
            vals.append(v)
        data[key] = np.array(vals)
    return data


def main():
    outdir = Path("data/baseline/plots")
    outdir.mkdir(parents=True, exist_ok=True)

    d = load_benchmark()

    # Filter to events with at least one reco cluster
    has_clusters = d["n_reco"] > 0
    purity = d["mean_purity"][has_clusters]
    completeness = d["mean_completeness"][has_clusters]
    n_reco = d["n_reco"][has_clusters]
    n_truth = d["n_truth"][has_clusters]
    n_matched = d["n_matched"][has_clusters]
    n_hits = d["n_hits"]
    n_ambiguous = d["n_ambiguous"]
    n_merged = d["n_merged"][has_clusters]

    n_events = len(d["n_reco"])
    n_with = int(has_clusters.sum())

    # --- 1. Purity distribution ---
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(purity, bins=50, range=(0.5, 1.02), color="steelblue", edgecolor="white")
    ax.set_xlabel("Mean Purity (per event)")
    ax.set_ylabel("Events")
    ax.set_title(f"BFS Cluster Purity (N={n_with} events)")
    ax.axvline(np.mean(purity), color="red", ls="--", label=f"mean={np.mean(purity):.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "purity_distribution.png", dpi=150)
    plt.close(fig)

    # --- 2. Completeness distribution ---
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(completeness, bins=50, range=(0.5, 1.02), color="darkorange", edgecolor="white")
    ax.set_xlabel("Mean Completeness (per event)")
    ax.set_ylabel("Events")
    ax.set_title(f"BFS Cluster Completeness (N={n_with} events)")
    ax.axvline(np.mean(completeness), color="red", ls="--", label=f"mean={np.mean(completeness):.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "completeness_distribution.png", dpi=150)
    plt.close(fig)

    # --- 3. Reco vs truth cluster count scatter ---
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(n_truth, n_reco, alpha=0.15, s=8, color="teal")
    max_n = max(n_truth.max(), n_reco.max())
    ax.plot([0, max_n], [0, max_n], "k--", lw=0.8, label="y=x")
    ax.set_xlabel("Truth clusters per event")
    ax.set_ylabel("Reco clusters per event")
    ax.set_title("Reco vs Truth Cluster Multiplicity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "reco_vs_truth_clusters.png", dpi=150)
    plt.close(fig)

    # --- 4. Match rate per event ---
    reco_match_rate = np.where(n_reco > 0, n_matched / n_reco, 0)
    truth_match_rate = np.where(n_truth > 0, n_matched / n_truth, 0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(reco_match_rate, bins=50, range=(0, 1.02), color="steelblue", edgecolor="white")
    axes[0].set_xlabel("Reco Match Rate")
    axes[0].set_ylabel("Events")
    axes[0].set_title(f"Reco Match Rate (mean={reco_match_rate.mean():.3f})")

    axes[1].hist(truth_match_rate, bins=50, range=(0, 1.02), color="darkorange", edgecolor="white")
    axes[1].set_xlabel("Truth Match Rate")
    axes[1].set_ylabel("Events")
    axes[1].set_title(f"Truth Match Rate (mean={truth_match_rate.mean():.3f})")
    fig.tight_layout()
    fig.savefig(outdir / "match_rates.png", dpi=150)
    plt.close(fig)

    # --- 5. Hits per event and ambiguous fraction ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(n_hits, bins=80, color="gray", edgecolor="white")
    axes[0].set_xlabel("Hits per event")
    axes[0].set_ylabel("Events")
    axes[0].set_title(f"Hit Multiplicity (N={n_events})")

    amb_frac = np.where(n_hits > 0, n_ambiguous / n_hits, 0)
    axes[1].hist(amb_frac, bins=50, range=(0, 0.5), color="crimson", edgecolor="white")
    axes[1].set_xlabel("Ambiguous Hit Fraction")
    axes[1].set_ylabel("Events")
    axes[1].set_title(f"Ambiguous Fraction (mean={amb_frac.mean():.3f})")
    fig.tight_layout()
    fig.savefig(outdir / "hit_multiplicity_and_ambiguous.png", dpi=150)
    plt.close(fig)

    # --- 6. Merged clusters per event ---
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(n_merged, bins=range(0, int(n_merged.max()) + 2), color="purple",
            edgecolor="white", align="left")
    ax.set_xlabel("Merged Reco Clusters per Event")
    ax.set_ylabel("Events")
    ax.set_title(f"BFS Merging (total merged={int(n_merged.sum())})")
    fig.tight_layout()
    fig.savefig(outdir / "merged_clusters.png", dpi=150)
    plt.close(fig)

    # --- Summary text file ---
    total_reco = int(d["n_reco"].sum())
    total_truth = int(d["n_truth"].sum())
    total_matched = int(d["n_matched"].sum())
    total_hits = int(d["n_hits"].sum())
    total_ambiguous = int(d["n_ambiguous"].sum())
    total_merged = int(d["n_merged"].sum())
    total_split = int(d["n_split"].sum())

    summary = f"""\
BFS Baseline Benchmark Summary
==============================
Events processed:       {n_events}
Events with clusters:   {n_with}

Hits:
  Total hits:           {total_hits}
  Ambiguous hits:       {total_ambiguous} ({100*total_ambiguous/total_hits:.1f}%)

Clusters:
  Total reco:           {total_reco}
  Total truth:          {total_truth}
  Matched:              {total_matched}
  Reco match rate:      {100*total_matched/total_reco:.1f}%
  Truth match rate:     {100*total_matched/total_truth:.1f}%

Quality:
  Mean purity:          {np.mean(purity):.4f}
  Mean completeness:    {np.mean(completeness):.4f}
  Median purity:        {np.median(purity):.4f}
  Median completeness:  {np.median(completeness):.4f}

Failure modes:
  Split truth clusters: {total_split}
  Merged reco clusters: {total_merged}
"""
    with open(outdir / "summary.txt", "w") as f:
        f.write(summary)

    print(summary)
    print(f"Plots saved to {outdir}/")


if __name__ == "__main__":
    main()
