"""Regenerate presentation-quality plots for the slide deck.

Reads existing run histories and evaluation CSVs and writes large-font,
clean figures to outputs/slide_plots/. One-off, no GPU required.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "outputs" / "slide_plots"
OUT.mkdir(parents=True, exist_ok=True)

# Presentation-friendly defaults
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "figure.dpi": 130,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 2.2,
})

METHOD_COLORS = {
    "BFS":            "#4C4C4C",
    "SimpleEdgeNet":  "#1f77b4",
    "SEN+BFS10":      "#5fbcd3",
    "CaloClusterNet": "#d95f02",
    "CCN+BFS10":      "#1b7837",
}
METHOD_LABEL = {
    "BFS":            "BFS (baseline)",
    "SimpleEdgeNet":  "SimpleEdgeNet",
    "SEN+BFS10":      "SEN+BFS10",
    "CaloClusterNet": "CaloClusterNet",
    "CCN+BFS10":      "CCN+BFS10 (mine)",
}


def training_curves() -> None:
    """Single clean F1-vs-epoch plot for both models."""
    sen = json.loads((PROJECT / "outputs/runs/simple_edge_net_v2/history.json").read_text())
    ccn = json.loads((PROJECT / "outputs/runs/calo_cluster_net_v2_stage1/history.json").read_text())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    for ax, (h, label, color) in zip(
        axes,
        [
            (sen, "SimpleEdgeNet", METHOD_COLORS["SimpleEdgeNet"]),
            (ccn, "CaloClusterNet (Stage 1)", METHOD_COLORS["CaloClusterNet"]),
        ],
    ):
        ep = [r["epoch"] for r in h]
        f1_train = [r["train"]["f1"] for r in h]
        f1_val = [r["val"]["f1"] for r in h]
        best_idx = int(np.argmax(f1_val))
        ax.plot(ep, f1_train, color=color, alpha=0.55, label="train")
        ax.plot(ep, f1_val, color=color, linewidth=2.6, label="val")
        ax.axvline(ep[best_idx], color="k", linestyle=":", alpha=0.5)
        ax.scatter([ep[best_idx]], [f1_val[best_idx]], color="red", zorder=5, s=80,
                   label=f"best val: {f1_val[best_idx]:.3f}")
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Edge F1")
        ax.set_ylim(0.85, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", framealpha=0.95)

    fig.suptitle("Training: validation edge F1 saturates within 10–15 epochs", y=1.02, fontsize=15)
    fig.savefig(OUT / "training_curves_clean.png")
    plt.close(fig)
    print("wrote", OUT / "training_curves_clean.png")


def load_residuals() -> pd.DataFrame:
    full = PROJECT / "outputs/cluster_physics_eval_test_full/cluster_residuals.csv"
    if full.exists():
        return pd.read_csv(full)
    return pd.read_csv(PROJECT / "outputs/cluster_physics_eval_bfs_test/cluster_residuals.csv")


def energy_residual_hist(df: pd.DataFrame) -> None:
    """dE distribution: BFS vs CCN+BFS10, excluding exact-match cluster pairs.

    Exact matches (dE == 0) account for >90% of matched clusters and hide
    the actual error distribution. Drop them so the tail comparison reads
    directly on a linear scale.
    """
    methods = ["BFS", "CCN+BFS10"]
    bins = np.linspace(-12, 12, 97)
    nonzero = df["dE"].abs() > 1e-9

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for title, outer_mask, ax in [
        ("All non-trivial matched clusters",
         np.ones(len(df), bool), axes[0]),
        (r"Track-seeding ($E_\mathrm{reco}\geq50$ MeV, non-trivial)",
         df["reco_energy"] >= 50, axes[1]),
    ]:
        for m in methods:
            # Overall mean (over ALL matched clusters in the regime) --- matches
            # what's reported in the summary tables.
            overall = df[(df["method"] == m) & outer_mask]
            shown = df[(df["method"] == m) & outer_mask & nonzero]
            ax.hist(
                shown["dE"], bins=bins, histtype="step", linewidth=2.5,
                color=METHOD_COLORS[m],
                label=f"{METHOD_LABEL[m]}: mean |dE|={overall['dE'].abs().mean():.3f} MeV",
            )
        ax.set_xlabel(r"$\Delta E = E_\mathrm{reco} - E_\mathrm{truth}$ (MeV)")
        ax.set_ylabel("Clusters / 0.25 MeV")
        ax.set_title(title)
        ax.set_xlim(-12, 12)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.06), ncol=1,
                  framealpha=0.0, fontsize=11.5, handlelength=1.8)
        ymax = ax.get_ylim()[1]
        ax.set_ylim(0, ymax * 1.05)
    fig.suptitle("Energy residual distributions (exact matches excluded from plot)",
                 fontsize=16, y=1.07)
    fig.subplots_adjust(top=0.78)
    fig.savefig(OUT / "energy_residual_hist.png")
    plt.close(fig)
    print("wrote", OUT / "energy_residual_hist.png")


def centroid_residual_hist(df: pd.DataFrame) -> None:
    """dr distribution: BFS vs CCN+BFS10, excluding exact-match (dr=0) pairs.

    Same treatment as the energy plot --- exact matches drown out the
    shape of the actual error distribution.
    """
    methods = ["BFS", "CCN+BFS10"]
    bins = np.linspace(0, 15, 76)
    nonzero = df["dr"] > 1e-6

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for title, outer_mask, ax in [
        ("All non-trivial matched clusters",
         np.ones(len(df), bool), axes[0]),
        (r"Track-seeding ($E_\mathrm{reco}\geq50$ MeV, non-trivial)",
         df["reco_energy"] >= 50, axes[1]),
    ]:
        for m in methods:
            overall = df[(df["method"] == m) & outer_mask]
            shown = df[(df["method"] == m) & outer_mask & nonzero]
            ax.hist(
                shown["dr"], bins=bins, histtype="step", linewidth=2.5,
                color=METHOD_COLORS[m],
                label=f"{METHOD_LABEL[m]}: mean dr={overall['dr'].mean():.3f} mm",
            )
        ax.set_xlabel(r"$\Delta r$ (mm)")
        ax.set_ylabel("Clusters / 0.2 mm")
        ax.set_title(title)
        ax.set_xlim(0, 15)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.06), ncol=1,
                  framealpha=0.0, fontsize=11.5, handlelength=1.8)
        ymax = ax.get_ylim()[1]
        ax.set_ylim(0, ymax * 1.05)
    fig.suptitle("Centroid displacement distributions (exact matches excluded from plot)",
                 fontsize=16, y=1.07)
    fig.subplots_adjust(top=0.78)
    fig.savefig(OUT / "centroid_residual_hist.png")
    plt.close(fig)
    print("wrote", OUT / "centroid_residual_hist.png")


def energy_binned_dE(df: pd.DataFrame) -> None:
    """Grouped bar chart: mean |dE| by truth energy bin, all 5 methods."""
    methods = ["BFS", "SimpleEdgeNet", "SEN+BFS10", "CaloClusterNet", "CCN+BFS10"]
    edges = [0, 25, 50, 75, 100, 125]
    centers = [(a + b) / 2 for a, b in zip(edges[:-1], edges[1:])]
    width = 4.0  # MeV per bar

    fig, ax = plt.subplots(figsize=(11, 4.8))
    n_methods = len(methods)
    offsets = (np.arange(n_methods) - (n_methods - 1) / 2) * width

    for i, m in enumerate(methods):
        sub = df[df["method"] == m]
        means = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = sub[(sub["truth_energy"] >= lo) & (sub["truth_energy"] < hi)]
            means.append(sel["dE"].abs().mean() if len(sel) else 0.0)
        ax.bar(np.array(centers) + offsets[i], means, width=width,
               color=METHOD_COLORS[m], label=METHOD_LABEL[m], edgecolor="white", linewidth=0.5)

    ax.set_xticks(centers)
    ax.set_xticklabels([f"{a}–{b}" for a, b in zip(edges[:-1], edges[1:])])
    ax.set_xlabel("Truth cluster energy (MeV)")
    ax.set_ylabel(r"Mean $|\Delta E|$ (MeV)")
    # Legend above the bars, title above the legend
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.04),
              ncol=5, framealpha=0.0, fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    ymax = ax.get_ylim()[1]
    ax.set_ylim(0, ymax * 1.05)
    fig.suptitle("Mean energy error vs truth cluster energy (test set)",
                 fontsize=15, y=1.02)
    fig.subplots_adjust(top=0.82)
    fig.savefig(OUT / "energy_binned_dE.png")
    plt.close(fig)
    print("wrote", OUT / "energy_binned_dE.png")


def improvement_bars() -> None:
    """Horizontal bar chart of CCN+BFS10 % improvement vs BFS."""
    metrics = [
        ("Splits",                            38.0, "Standard"),
        ("Merges",                             8.4, "Standard"),
        (r"Mean $|\Delta E|$ (all)",          19.0, "All clusters"),
        (r"Std $\Delta E$ (all)",             15.3, "All clusters"),
        (r"Mean $\Delta r$ (all)",            17.2, "All clusters"),
        (r"Mean $|\Delta E|$ ($E\geq$50 MeV)", 26.5, "Track-seeding"),
        (r"95th $|\Delta E|$ tail",            33.6, "Track-seeding"),
        (r"95th $\Delta r$ tail",              36.4, "Track-seeding"),
        (r"Frac $|\Delta E|>10$ MeV",          30.1, "Track-seeding"),
        (r"Mean $|\Delta E|$ (signal 95–110)", 43.0, "Signal region"),
    ]
    cat_color = {
        "Standard":      "#4C4C4C",
        "All clusters":  "#1f77b4",
        "Track-seeding": "#1b7837",
        "Signal region": "#d62728",
    }

    labels = [m[0] for m in metrics]
    vals = [m[1] for m in metrics]
    colors = [cat_color[m[2]] for m in metrics]
    y = np.arange(len(metrics))[::-1]  # top-to-bottom order

    fig, ax = plt.subplots(figsize=(10.5, 6))
    ax.barh(y, vals, color=colors, edgecolor="white", linewidth=0.6)
    for yi, v in zip(y, vals):
        ax.text(v + 0.6, yi, f"-{v:g}%", va="center", ha="left", fontsize=12.5, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Reduction vs BFS (%)")
    ax.set_xlim(0, max(vals) + 10)
    ax.grid(True, axis="x", alpha=0.3)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in cat_color.values()]
    ax.legend(handles, list(cat_color.keys()),
              loc="lower center", bbox_to_anchor=(0.5, 1.04),
              ncol=len(cat_color), framealpha=0.0, fontsize=11.5)
    fig.suptitle("CCN+BFS10 improvement over BFS (test set, 276,688 events)",
                 fontsize=15, y=1.02)
    fig.subplots_adjust(top=0.84)

    fig.savefig(OUT / "improvement_bars.png")
    plt.close(fig)
    print("wrote", OUT / "improvement_bars.png")


def signal_region_focus(df: pd.DataFrame) -> None:
    """Mean |dE| in the conversion-electron signal region: BFS vs methods."""
    sub = df[(df["truth_energy"] >= 95) & (df["truth_energy"] <= 110)]
    methods = ["BFS", "SimpleEdgeNet", "SEN+BFS10", "CaloClusterNet", "CCN+BFS10"]
    means = [sub[sub["method"] == m]["dE"].abs().mean() for m in methods]
    drs = [sub[sub["method"] == m]["dr"].mean() for m in methods]
    counts = [(sub["method"] == m).sum() for m in methods]
    bfs_mean = means[0]
    bfs_dr = drs[0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    # |dE| bars
    ax = axes[0]
    bars = ax.bar(range(len(methods)), means,
                  color=[METHOD_COLORS[m] for m in methods],
                  edgecolor="white", linewidth=0.6)
    ax.axhline(bfs_mean, color="k", linestyle="--", alpha=0.5, label=f"BFS = {bfs_mean:.3f}")
    for i, (b, v) in enumerate(zip(bars, means)):
        delta = (v - bfs_mean) / bfs_mean * 100 if i > 0 else 0
        label = f"{v:.3f}"
        if i > 0:
            label += f"\n({delta:+.0f}%)"
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, label,
                ha="center", va="bottom", fontsize=11.5, fontweight="bold")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylabel(r"Mean $|\Delta E|$ (MeV)")
    ax.set_title(f"Energy error in signal region (95–110 MeV, N={counts[0]})")
    ax.set_ylim(0, max(means) * 1.25)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower right", fontsize=11, framealpha=0.95)

    # Δr bars
    ax = axes[1]
    bars = ax.bar(range(len(methods)), drs,
                  color=[METHOD_COLORS[m] for m in methods],
                  edgecolor="white", linewidth=0.6)
    ax.axhline(bfs_dr, color="k", linestyle="--", alpha=0.5, label=f"BFS = {bfs_dr:.3f}")
    for i, (b, v) in enumerate(zip(bars, drs)):
        delta = (v - bfs_dr) / bfs_dr * 100 if i > 0 else 0
        label = f"{v:.3f}"
        if i > 0:
            label += f"\n({delta:+.0f}%)"
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, label,
                ha="center", va="bottom", fontsize=11.5, fontweight="bold")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylabel(r"Mean $\Delta r$ (mm)")
    ax.set_title("Centroid displacement in signal region")
    ax.set_ylim(0, max(drs) * 1.25)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower right", fontsize=11, framealpha=0.95)

    fig.savefig(OUT / "signal_region_focus.png")
    plt.close(fig)
    print("wrote", OUT / "signal_region_focus.png")


def cluster_size_distribution(df: pd.DataFrame) -> None:
    """Histogram of truth cluster size (singleton problem visualization)."""
    sub = df[df["method"] == "BFS"]  # truth cluster sizes are method-independent at top-K
    sizes = sub["truth_nhits"].values

    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    bins = np.arange(0.5, 15.5, 1)
    ax.hist(sizes, bins=bins, color="#888888", edgecolor="white")
    n_singleton = int((sizes == 1).sum())
    pct = n_singleton / len(sizes) * 100
    ax.axvline(1, color="red", linestyle="--", linewidth=2)
    # Annotate far from the tall singleton bar
    ax.text(5.5, ax.get_ylim()[1] * 0.72,
            f"singletons: {n_singleton:,}\n({pct:.1f}% of all clusters)",
            color="red", fontsize=13, fontweight="bold",
            ha="left", va="center",
            bbox=dict(facecolor="white", edgecolor="red", boxstyle="round,pad=0.4"))
    ax.set_xlabel("Truth cluster size (hits)")
    ax.set_ylabel("Number of clusters")
    ax.set_title("Truth cluster size distribution (calo-entrant truth, test set)")
    ax.set_xlim(0.5, 14.5)
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(OUT / "cluster_size_distribution.png")
    plt.close(fig)
    print("wrote", OUT / "cluster_size_distribution.png")


def main() -> None:
    training_curves()
    df = load_residuals()
    print(f"loaded {len(df):,} cluster residuals across {df['method'].nunique()} methods")
    energy_residual_hist(df)
    centroid_residual_hist(df)
    energy_binned_dE(df)
    improvement_bars()
    signal_region_focus(df)
    cluster_size_distribution(df)
    print("\nAll plots written to", OUT)


if __name__ == "__main__":
    main()
