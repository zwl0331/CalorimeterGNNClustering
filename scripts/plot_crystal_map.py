"""
Plot crystal map for both calorimeter disks with crystal IDs labeled.

Draws each crystal as a square patch (pitch ~34.3 mm) at its true
position, colored by type (CsI vs CAPHRI).

Usage:
    source setup_env.sh
    python3 scripts/plot_crystal_map.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors
import numpy as np
import csv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CRYSTAL_PITCH = 34.3  # mm — measured nearest-neighbor distance
CRYSTAL_SIZE = CRYSTAL_PITCH * 0.92  # slight gap between crystals
CAPHRI_IDS = {582, 609, 610, 637}


def load_geometry(csv_path):
    """Load crystal geometry CSV into arrays."""
    ids, disks, xs, ys = [], [], [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.append(int(row["crystalId"]))
            disks.append(int(row["diskId"]))
            xs.append(float(row["x_mm"]))
            ys.append(float(row["y_mm"]))
    return np.array(ids), np.array(disks), np.array(xs), np.array(ys)


def plot_disk(ax, crystal_ids, xs, ys, disk_id, label_fontsize=3.0,
              xlim=None, ylim=None):
    """Draw one calorimeter disk: crystal patches colored by type (CsI/CAPHRI), labeled with IDs."""
    half = CRYSTAL_SIZE / 2

    csi_patches = []
    caphri_patches = []

    for cid, x, y in zip(crystal_ids, xs, ys):
        if xlim and (x < xlim[0] - CRYSTAL_PITCH or x > xlim[1] + CRYSTAL_PITCH):
            continue
        if ylim and (y < ylim[0] - CRYSTAL_PITCH or y > ylim[1] + CRYSTAL_PITCH):
            continue

        rect = Rectangle((x - half, y - half), CRYSTAL_SIZE, CRYSTAL_SIZE)
        if cid in CAPHRI_IDS:
            caphri_patches.append(rect)
        else:
            csi_patches.append(rect)

        ax.text(x, y, str(cid), fontsize=label_fontsize, ha="center",
                va="center", color="black", zorder=3, fontweight="bold",
                clip_on=True)

    if csi_patches:
        pc = PatchCollection(csi_patches, facecolor="#6baed6", edgecolor="white",
                             linewidth=0.4, zorder=2)
        ax.add_collection(pc)
    if caphri_patches:
        pc = PatchCollection(caphri_patches, facecolor="#fc8d59", edgecolor="white",
                             linewidth=0.4, zorder=2)
        ax.add_collection(pc)

    ax.set_aspect("equal")
    ax.autoscale_view()
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    ax.set_xlabel("x (mm)", fontsize=11)
    ax.set_ylabel("y (mm)", fontsize=11)
    ax.grid(True, linewidth=0.3, alpha=0.3, zorder=0)
    ax.set_facecolor("#f7f7f7")


def make_legend():
    """Create legend handles for CsI and CAPHRI."""
    from matplotlib.patches import Patch
    return [
        Patch(facecolor="#6baed6", edgecolor="white", label="CsI crystal"),
        Patch(facecolor="#fc8d59", edgecolor="white", label="CAPHRI crystal"),
    ]


def main():
    csv_path = Path("data/crystal_geometry.csv")
    out_dir = Path("outputs/crystal_map/")
    out_dir.mkdir(parents=True, exist_ok=True)

    ids, disks, xs, ys = load_geometry(csv_path)
    legend_elements = make_legend()

    # ── Combined figure (both disks side by side) ────────────────
    fig, axes = plt.subplots(1, 2, figsize=(28, 14))
    for di, ax in enumerate(axes):
        mask = disks == di
        plot_disk(ax, ids[mask], xs[mask], ys[mask], di, label_fontsize=3.5)
        ax.set_title(f"Disk {di}  ({mask.sum()} crystals)", fontsize=13)

    fig.legend(handles=legend_elements, loc="upper center", ncol=2,
               fontsize=11, frameon=True)
    fig.suptitle("Mu2e Calorimeter Crystal Map", fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    combined_path = out_dir / "crystal_map_both_disks.png"
    fig.savefig(combined_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {combined_path}")

    # ── Individual full-disk figures ─────────────────────────────
    for di in [0, 1]:
        mask = disks == di
        fig, ax = plt.subplots(figsize=(22, 22))
        plot_disk(ax, ids[mask], xs[mask], ys[mask], di, label_fontsize=6.0)
        ax.set_title(f"Disk {di}  ({mask.sum()} crystals)", fontsize=14)
        fig.legend(handles=legend_elements, loc="upper right", fontsize=11,
                   frameon=True)
        fig.suptitle(f"Mu2e Calorimeter — Disk {di}", fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        disk_path = out_dir / f"crystal_map_disk{di}.png"
        fig.savefig(disk_path, dpi=250, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {disk_path}")

    # ── Zoomed quadrant views (IDs clearly legible) ──────────────
    x_lo, x_hi = xs.min(), xs.max()
    y_lo, y_hi = ys.min(), ys.max()
    x_mid, y_mid = (x_lo + x_hi) / 2, (y_lo + y_hi) / 2
    pad = 30

    quadrants = [
        ("top_left",     (x_lo - pad, x_mid + pad), (y_mid - pad, y_hi + pad)),
        ("top_right",    (x_mid - pad, x_hi + pad), (y_mid - pad, y_hi + pad)),
        ("bottom_left",  (x_lo - pad, x_mid + pad), (y_lo - pad, y_mid + pad)),
        ("bottom_right", (x_mid - pad, x_hi + pad), (y_lo - pad, y_mid + pad)),
    ]

    for di in [0, 1]:
        mask = disks == di
        for qname, xlim, ylim in quadrants:
            fig, ax = plt.subplots(figsize=(18, 18))
            plot_disk(ax, ids[mask], xs[mask], ys[mask], di,
                      label_fontsize=9.0, xlim=xlim, ylim=ylim)
            ax.set_title(f"Disk {di} — {qname.replace('_', ' ')}",
                         fontsize=14)
            fig.legend(handles=legend_elements, loc="upper right",
                       fontsize=11, frameon=True)
            fig.tight_layout()
            qpath = out_dir / f"crystal_map_disk{di}_{qname}.png"
            fig.savefig(qpath, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {qpath}")


if __name__ == "__main__":
    main()
