#!/usr/bin/env python3
"""
Plot training curves from a completed run's history.json.

Usage:
    python3 scripts/plot_training.py --run-dir outputs/runs/simple_edge_net_v1
    python3 scripts/plot_training.py  # auto-finds latest run
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def find_latest_run(base="outputs/runs"):
    """Find the most recently modified run directory."""
    base = Path(base)
    if not base.exists():
        return None
    runs = [d for d in base.iterdir() if d.is_dir() and (d / "history.json").exists()]
    if not runs:
        return None
    return max(runs, key=lambda d: (d / "history.json").stat().st_mtime)


def load_history(run_dir):
    """Load history.json from a run directory."""
    path = Path(run_dir) / "history.json"
    if not path.exists():
        print(f"ERROR: {path} not found.")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def plot_loss(history, ax):
    """Train vs val loss."""
    epochs = [h["epoch"] for h in history]
    ax.plot(epochs, [h["train"]["loss"] for h in history], label="train", linewidth=1.5)
    ax.plot(epochs, [h["val"]["loss"] for h in history], label="val", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (weighted BCE)")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_f1(history, ax):
    """Train vs val edge F1."""
    epochs = [h["epoch"] for h in history]
    ax.plot(epochs, [h["train"]["f1"] for h in history], label="train", linewidth=1.5)
    ax.plot(epochs, [h["val"]["f1"] for h in history], label="val", linewidth=1.5)

    # Mark best val F1
    val_f1 = [h["val"]["f1"] for h in history]
    best_idx = int(np.argmax(val_f1))
    ax.axvline(epochs[best_idx], color="gray", linestyle="--", alpha=0.5)
    ax.annotate(f"best={val_f1[best_idx]:.3f}\nepoch {epochs[best_idx]}",
                xy=(epochs[best_idx], val_f1[best_idx]),
                xytext=(10, -20), textcoords="offset points", fontsize=8,
                arrowprops=dict(arrowstyle="->", color="gray"))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1")
    ax.set_title("Edge F1 (positive class)")
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_precision_recall(history, ax):
    """Val precision and recall over epochs."""
    epochs = [h["epoch"] for h in history]
    ax.plot(epochs, [h["val"]["precision"] for h in history],
            label="precision", linewidth=1.5)
    ax.plot(epochs, [h["val"]["recall"] for h in history],
            label="recall", linewidth=1.5)
    ax.plot(epochs, [h["train"]["precision"] for h in history],
            label="train P", linewidth=1, linestyle="--", alpha=0.5)
    ax.plot(epochs, [h["train"]["recall"] for h in history],
            label="train R", linewidth=1, linestyle="--", alpha=0.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_auc(history, ax):
    """Val ROC-AUC and PR-AUC over epochs."""
    epochs = [h["epoch"] for h in history]
    roc = [h["val"].get("roc_auc", 0) for h in history]
    pr = [h["val"].get("pr_auc", 0) for h in history]

    if max(roc) == 0 and max(pr) == 0:
        ax.text(0.5, 0.5, "No AUC data", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="gray")
        ax.set_title("Val AUC")
        return

    ax.plot(epochs, roc, label="ROC AUC", linewidth=1.5)
    ax.plot(epochs, pr, label="PR AUC", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC")
    ax.set_title("Val AUC")
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_lr(history, ax):
    """Learning rate schedule."""
    epochs = [h["epoch"] for h in history]
    lrs = [h["lr"] for h in history]
    ax.plot(epochs, lrs, color="tab:orange", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("LR Schedule")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)


def plot_overview(history, out_path):
    """Generate the 5-panel overview figure."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    plot_loss(history, axes[0, 0])
    plot_f1(history, axes[0, 1])
    plot_precision_recall(history, axes[0, 2])
    plot_auc(history, axes[1, 0])
    plot_lr(history, axes[1, 1])

    # Summary text in bottom-right panel
    ax_text = axes[1, 2]
    ax_text.axis("off")

    val_f1 = [h["val"]["f1"] for h in history]
    best_idx = int(np.argmax(val_f1))
    best = history[best_idx]

    lines = [
        f"Total epochs: {len(history)}",
        f"Best epoch: {best['epoch']}",
        f"",
        f"Best val F1:        {best['val']['f1']:.4f}",
        f"Best val precision: {best['val']['precision']:.4f}",
        f"Best val recall:    {best['val']['recall']:.4f}",
        f"Best val ROC AUC:   {best['val'].get('roc_auc', 0):.4f}",
        f"Best val PR AUC:    {best['val'].get('pr_auc', 0):.4f}",
        f"",
        f"Final train loss: {history[-1]['train']['loss']:.4f}",
        f"Final val loss:   {history[-1]['val']['loss']:.4f}",
        f"Final LR: {history[-1]['lr']:.1e}",
    ]
    ax_text.text(0.1, 0.95, "\n".join(lines), transform=ax_text.transAxes,
                 fontsize=10, verticalalignment="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    ax_text.set_title("Summary")

    fig.suptitle("SimpleEdgeNet Training", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot training curves")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Run directory (auto-finds latest if omitted)")
    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = find_latest_run()
        if run_dir is None:
            print("ERROR: No completed runs found in outputs/runs/")
            sys.exit(1)
        print(f"Using latest run: {run_dir}")

    history = load_history(run_dir)
    print(f"Loaded {len(history)} epochs from {run_dir / 'history.json'}")

    out_path = run_dir / "training_curves.png"
    plot_overview(history, out_path)


if __name__ == "__main__":
    main()
