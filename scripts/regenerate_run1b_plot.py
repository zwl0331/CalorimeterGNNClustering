"""Regenerate run1b_evaluation.png from cached CSVs.

The original plot was produced by `evaluate_run1b.py`, which historically
hardcoded "SEN (0.26) / CCN (0.20)" into the table column header and the
suptitle (a vestige from the §5.4 cross-evaluation). When the same script
was reused to evaluate retrained run1b models with different tau_edge
values (0.34 / 0.32), the bar heights and CSV outputs were correct but
the printed labels stayed wrong. This regenerator rebuilds the PNG using
the actual tau_edge values stored in `run1b_results.csv`.

The per-cluster purity-distribution histogram (panel 0,2 in the original)
cannot be reconstructed from the CSV — only `mean_purity` is persisted —
so that panel is replaced with a mean-purity / mean-completeness bar
view derived from the same CSV.

Usage:
    python scripts/regenerate_run1b_plot.py --input-dir outputs/task19c_ost_eval
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


_BFS = "BFS"
_SEN = "SimpleEdgeNet"
# CCN method name varies by era: "CaloClusterNet" (v2) vs "CaloClusterNetV1" (v1).
# Resolved per-input by matching whatever third method is present in the summary CSV.

_BASE_SHORT = {_BFS: "BFS", _SEN: "SEN"}
_BASE_COLORS = {_BFS: "coral", _SEN: "steelblue"}
_CCN_COLOR = "seagreen"


def resolve_ccn_method(summary: dict[str, dict]) -> str:
    """Return the method name in `summary` that isn't BFS or SimpleEdgeNet."""
    for m in summary:
        if m not in (_BFS, _SEN):
            return m
    raise KeyError(f"No CCN-like method found in summary: {list(summary)}")


def resolve_match_column(detail_header: list[str], method: str) -> str:
    """Return the *_matched column corresponding to `method`."""
    short = "ccnv1" if "V1" in method else "ccn"
    target = f"{short}_matched"
    if target in detail_header:
        return target
    # Fall back: any *_matched not already taken.
    for c in detail_header:
        if c.endswith("_matched") and c not in ("bfs_matched", "sen_matched"):
            return c
    raise KeyError(f"No CCN match column in {detail_header}")

ENERGY_BINS = [0, 50, 200, float("inf")]
ENERGY_LABELS = ["<50 MeV", "50-200 MeV", ">200 MeV"]
MULT_BINS = [1, 2, 4, float("inf")]
MULT_LABELS = ["1 hit", "2-3 hits", "4+ hits"]


def read_summary(path: Path) -> dict[str, dict]:
    out = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            out[row["method"]] = row
    return out


def read_truth_detail(path: Path, ccn_method: str
                      ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    with path.open() as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        ccn_col = resolve_match_column(header, ccn_method)
        key_for = {_BFS: "bfs_matched", _SEN: "sen_matched", ccn_method: ccn_col}
        energies, n_hits = [], []
        matched = {m: [] for m in key_for}
        for row in reader:
            energies.append(float(row["energy"]))
            n_hits.append(int(row["n_hits"]))
            for m, k in key_for.items():
                matched[m].append(int(row[k]))
    return (np.array(energies),
            np.array(n_hits),
            {m: np.array(v) for m, v in matched.items()})


def binned_match_rate(matched: np.ndarray, key_vals: np.ndarray,
                      bins: list[float]) -> tuple[list[float], list[int]]:
    rates, counts = [], []
    for i in range(len(bins) - 1):
        sel = (key_vals >= bins[i]) & (key_vals < bins[i + 1])
        n = int(sel.sum())
        rates.append(float(matched[sel].mean()) * 100 if n > 0 else 0.0)
        counts.append(n)
    return rates, counts


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", type=Path, required=True,
                   help="Directory containing run1b_results.csv and "
                        "truth_cluster_detail.csv.")
    p.add_argument("--output-name", default="run1b_evaluation.png",
                   help="PNG filename inside --input-dir. "
                        "Default overwrites the existing run1b_evaluation.png.")
    args = p.parse_args()

    summary_path = args.input_dir / "run1b_results.csv"
    detail_path = args.input_dir / "truth_cluster_detail.csv"
    if not summary_path.exists() or not detail_path.exists():
        print(f"ERROR: missing CSV(s) in {args.input_dir}")
        return 2

    summary = read_summary(summary_path)
    ccn_method = resolve_ccn_method(summary)
    energies, n_hits, matched = read_truth_detail(detail_path, ccn_method)

    methods = [_BFS, _SEN, ccn_method]
    short = dict(_BASE_SHORT, **{ccn_method: "CCNv1" if "V1" in ccn_method else "CCN"})
    colors = dict(_BASE_COLORS, **{ccn_method: _CCN_COLOR})

    sen_tau = summary[_SEN]["tau_edge"]
    ccn_tau = summary[ccn_method]["tau_edge"]
    n_disk_graphs = int(summary[_BFS]["n_disk_graphs"])
    n_events = int(summary[_BFS]["n_events"])

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(
        f"Run1B (No Field) Evaluation -- {n_disk_graphs:,} disk-graphs, "
        f"{n_events:,} events\n"
        f"SEN (τ={float(sen_tau):.2f}), CCN (τ={float(ccn_tau):.2f}) "
        f"vs BFS baseline   [regenerated from CSV]",
        fontsize=13, fontweight="bold")

    w = 0.25

    # 1. Match rates
    ax = axes[0, 0]
    metrics = ["Reco MR", "Truth MR"]
    keys = ["reco_match_rate", "truth_match_rate"]
    x = np.arange(len(metrics))
    for i, m in enumerate(methods):
        vals = [float(summary[m][k]) * 100 for k in keys]
        ax.bar(x + (i - 1) * w, vals, w, label=short[m],
               color=colors[m], alpha=0.8)
        for j, v in enumerate(vals):
            ax.text(x[j] + (i - 1) * w, v + 0.5, f"{v:.1f}%",
                    ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylabel("%"); ax.set_title("Match Rates")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y"); ax.set_ylim(0, 105)

    # 2. Splits & Merges
    ax = axes[0, 1]
    metrics = ["Splits", "Merges"]; keys = ["n_split", "n_merged"]
    x = np.arange(len(metrics))
    all_vals = [int(summary[m][k]) for m in methods for k in keys]
    ymax = max(all_vals) if all_vals else 1
    for i, m in enumerate(methods):
        vals = [int(summary[m][k]) for k in keys]
        ax.bar(x + (i - 1) * w, vals, w, label=short[m],
               color=colors[m], alpha=0.8)
        for j, v in enumerate(vals):
            ax.text(x[j] + (i - 1) * w, v + ymax * 0.02, f"{v:,}",
                    ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_title("Splits & Merges")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")

    # 3. Mean purity / completeness bars (replaces the histogram, which
    #    requires raw inference output not stored in the CSV).
    ax = axes[0, 2]
    metrics = ["Mean purity", "Mean compl."]
    keys = ["mean_purity", "mean_completeness"]
    x = np.arange(len(metrics))
    for i, m in enumerate(methods):
        vals = [float(summary[m][k]) for k in keys]
        ax.bar(x + (i - 1) * w, vals, w, label=short[m],
               color=colors[m], alpha=0.8)
        for j, v in enumerate(vals):
            ax.text(x[j] + (i - 1) * w, v + 0.001, f"{v:.4f}",
                    ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_title("Cluster Purity / Completeness")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(min(0.95, min(float(summary[m][k]) for m in methods for k in keys)) - 0.005, 1.005)

    # 4. Truth match rate by energy
    ax = axes[1, 0]
    x_e = np.arange(len(ENERGY_LABELS))
    for i, m in enumerate(methods):
        rates, _ = binned_match_rate(matched[m], energies, ENERGY_BINS)
        ax.bar(x_e + (i - 1) * w, rates, w, label=short[m],
               color=colors[m], alpha=0.8)
        for j, v in enumerate(rates):
            ax.text(x_e[j] + (i - 1) * w, v + 1, f"{v:.1f}%",
                    ha="center", fontsize=7)
    ax.set_xticks(x_e); ax.set_xticklabels(ENERGY_LABELS)
    ax.set_ylabel("Truth match rate (%)")
    ax.set_title("Truth Match Rate by Energy")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y"); ax.set_ylim(0, 105)

    # 5. Truth match rate by hit count
    ax = axes[1, 1]
    x_m = np.arange(len(MULT_LABELS))
    for i, m in enumerate(methods):
        rates, _ = binned_match_rate(matched[m], n_hits, MULT_BINS)
        ax.bar(x_m + (i - 1) * w, rates, w, label=short[m],
               color=colors[m], alpha=0.8)
        for j, v in enumerate(rates):
            ax.text(x_m[j] + (i - 1) * w, v + 1, f"{v:.1f}%",
                    ha="center", fontsize=7)
    ax.set_xticks(x_m); ax.set_xticklabels(MULT_LABELS)
    ax.set_ylabel("Truth match rate (%)")
    ax.set_title("Truth Match Rate by Hit Count")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y"); ax.set_ylim(0, 105)

    # 6. Summary table
    ax = axes[1, 2]; ax.axis("off")
    rows = []
    for label, key, fmt in [
        ("Reco match rate", "reco_match_rate", "{:.1%}"),
        ("Truth match rate", "truth_match_rate", "{:.1%}"),
        ("Mean purity", "mean_purity", "{:.4f}"),
        ("Mean completeness", "mean_completeness", "{:.4f}"),
        ("Splits", "n_split", "{:,}"),
        ("Merges", "n_merged", "{:,}"),
    ]:
        row = [label]
        for m in methods:
            v = summary[m][key]
            row.append(fmt.format(float(v)) if "." in fmt or "%" in fmt
                       else fmt.format(int(v)))
        rows.append(row)
    ccn_short = short[ccn_method]
    table = ax.table(
        cellText=rows,
        colLabels=["Metric", "BFS",
                   f"SEN ({float(sen_tau):.2f})",
                   f"{ccn_short} ({float(ccn_tau):.2f})"],
        cellLoc="center", loc="center")
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 1.5)
    ax.set_title("Run1B Summary", pad=20)

    plt.tight_layout()
    out_path = args.input_dir / args.output_name
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
