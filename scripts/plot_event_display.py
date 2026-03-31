"""
Event display: show hit clusters on the crystal map.

For each selected event-disk graph, draws:
  - All crystals as light gray background squares
  - Hit crystals colored by truth cluster assignment
  - Graph edges between hits (thin lines)
  - Crystal IDs on hit crystals
  - Energy indicated by color intensity

Usage:
    source setup_env.sh
    python3 scripts/plot_event_display.py                     # first 6 events
    python3 scripts/plot_event_display.py --n-events 10       # first 10
    python3 scripts/plot_event_display.py --graph-idx 0 3 7   # specific graphs
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection, LineCollection
import matplotlib.colors as mcolors
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CRYSTAL_PITCH = 34.3
CRYSTAL_SIZE = CRYSTAL_PITCH * 0.92
CAPHRI_IDS = {582, 609, 610, 637}

# Distinct colors for clusters
CLUSTER_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
    "#a65628", "#f781bf", "#999999", "#66c2a5", "#fc8d62",
    "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f", "#e5c494",
    "#b3b3b3", "#1b9e77", "#d95f02", "#7570b3", "#e7298a",
]


def load_geometry(csv_path):
    """Load crystal geometry into dict: crystalId -> (diskId, x, y)."""
    crystals = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            cid = int(row["crystalId"])
            crystals[cid] = (int(row["diskId"]), float(row["x_mm"]),
                             float(row["y_mm"]))
    return crystals


def find_crystal_id(x, y, disk_id, crystal_map):
    """Find crystal ID closest to (x, y) on the given disk."""
    best_id, best_dist = -1, float("inf")
    for cid, (did, cx, cy) in crystal_map.items():
        if did != disk_id:
            continue
        d = (cx - x)**2 + (cy - y)**2
        if d < best_dist:
            best_dist = d
            best_id = cid
    return best_id


def plot_event(data, crystal_map, out_path, graph_label=""):
    """Plot one event-disk graph on the crystal map."""
    import torch

    x = data.x.numpy()
    edge_index = data.edge_index.numpy()
    truth = data.hit_truth_cluster.numpy()
    y_edge = data.y_edge.numpy()
    edge_mask = data.edge_mask.numpy()
    disk_id = int(data.disk_id)
    n_hits = x.shape[0]

    hit_x = x[:, 2]
    hit_y = x[:, 3]
    hit_log_e = x[:, 0]
    hit_energy = np.expm1(hit_log_e)  # undo log1p

    # Match hits to crystal IDs
    hit_cids = []
    for i in range(n_hits):
        hit_cids.append(find_crystal_id(hit_x[i], hit_y[i], disk_id, crystal_map))

    # Get unique truth clusters
    unique_tc = sorted(set(truth[truth >= 0].tolist()))
    tc_to_color = {}
    for i, tc in enumerate(unique_tc):
        tc_to_color[tc] = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]

    # ── Figure setup ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(18, 18))

    # Draw all crystals on this disk as background
    bg_patches = []
    half = CRYSTAL_SIZE / 2
    for cid, (did, cx, cy) in crystal_map.items():
        if did != disk_id:
            continue
        bg_patches.append(Rectangle((cx - half, cy - half),
                                     CRYSTAL_SIZE, CRYSTAL_SIZE))
    pc_bg = PatchCollection(bg_patches, facecolor="#e8e8e8",
                            edgecolor="#cccccc", linewidth=0.3, zorder=1)
    ax.add_collection(pc_bg)

    # Draw graph edges
    edge_lines_pos = []  # same-cluster edges
    edge_lines_neg = []  # cross-cluster edges
    for ei in range(edge_index.shape[1]):
        s, d = edge_index[0, ei], edge_index[1, ei]
        if s >= d:  # draw each undirected edge once
            continue
        line = [(hit_x[s], hit_y[s]), (hit_x[d], hit_y[d])]
        if edge_mask[ei] and y_edge[ei] == 1:
            edge_lines_pos.append(line)
        else:
            edge_lines_neg.append(line)

    if edge_lines_neg:
        lc_neg = LineCollection(edge_lines_neg, colors="#bbbbbb",
                                linewidths=0.6, zorder=2, alpha=0.5)
        ax.add_collection(lc_neg)
    if edge_lines_pos:
        lc_pos = LineCollection(edge_lines_pos, colors="#333333",
                                linewidths=1.2, zorder=3, alpha=0.7)
        ax.add_collection(lc_pos)

    # Draw hit crystals colored by truth cluster
    for i in range(n_hits):
        tc = truth[i]
        if tc >= 0:
            color = tc_to_color[tc]
            alpha = 0.5 + 0.5 * (hit_energy[i] / hit_energy.max())
        else:
            color = "#aaaaaa"
            alpha = 0.6

        rect = Rectangle((hit_x[i] - half, hit_y[i] - half),
                          CRYSTAL_SIZE, CRYSTAL_SIZE,
                          facecolor=color, edgecolor="black",
                          linewidth=1.0, alpha=alpha, zorder=4)
        ax.add_patch(rect)

        # Label with crystal ID and energy
        cid_str = str(hit_cids[i]) if hit_cids[i] >= 0 else "?"
        ax.text(hit_x[i], hit_y[i] + 4, cid_str,
                fontsize=7, ha="center", va="bottom", color="black",
                fontweight="bold", zorder=5)
        ax.text(hit_x[i], hit_y[i] - 4, f"{hit_energy[i]:.1f}",
                fontsize=5.5, ha="center", va="top", color="black",
                zorder=5, style="italic")

    # Legend for clusters
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_handles = []
    for tc in unique_tc:
        nhits = (truth == tc).sum()
        legend_handles.append(Patch(facecolor=tc_to_color[tc], edgecolor="black",
                                     linewidth=0.5,
                                     label=f"Cluster {tc} ({nhits} hits)"))
    n_amb = (truth == -1).sum()
    if n_amb > 0:
        legend_handles.append(Patch(facecolor="#aaaaaa", edgecolor="black",
                                     linewidth=0.5,
                                     label=f"Ambiguous ({n_amb} hits)"))
    legend_handles.append(Line2D([0], [0], color="#333333", linewidth=1.2,
                                  label="Same-cluster edge"))
    legend_handles.append(Line2D([0], [0], color="#bbbbbb", linewidth=0.6,
                                  alpha=0.5, label="Cross-cluster edge"))

    ax.legend(handles=legend_handles, loc="upper right", fontsize=9,
              frameon=True, framealpha=0.9)

    ax.set_aspect("equal")
    ax.autoscale_view()
    # Zoom to hit region with padding
    pad = 120
    ax.set_xlim(hit_x.min() - pad, hit_x.max() + pad)
    ax.set_ylim(hit_y.min() - pad, hit_y.max() + pad)
    ax.set_xlabel("x (mm)", fontsize=12)
    ax.set_ylabel("y (mm)", fontsize=12)
    ax.set_facecolor("#f7f7f7")
    ax.grid(True, linewidth=0.2, alpha=0.3, zorder=0)

    total_e = hit_energy.sum()
    ax.set_title(
        f"{graph_label}\n"
        f"Disk {disk_id} — {n_hits} hits, {len(unique_tc)} truth clusters, "
        f"E_total = {total_e:.1f} MeV",
        fontsize=13,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Event display on crystal map")
    parser.add_argument("--processed-dir", type=str, default="data/processed/")
    parser.add_argument("--geometry", type=str, default="data/crystal_geometry.csv")
    parser.add_argument("--n-events", type=int, default=6,
                        help="Number of graphs to plot")
    parser.add_argument("--graph-idx", type=int, nargs="*", default=None,
                        help="Specific graph indices to plot")
    parser.add_argument("--out-dir", type=str, default="outputs/event_display/")
    args = parser.parse_args()

    import torch

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    crystal_map = load_geometry(args.geometry)

    pt_files = sorted(Path(args.processed_dir).glob("*.pt"))
    pt_files = [f for f in pt_files if not f.name.startswith("diagnostics")]

    if args.graph_idx:
        indices = args.graph_idx
    else:
        indices = list(range(min(args.n_events, len(pt_files))))

    print(f"Plotting {len(indices)} event displays...")

    for gi in indices:
        if gi >= len(pt_files):
            print(f"  Skipping index {gi} (only {len(pt_files)} graphs)")
            continue
        f = pt_files[gi]
        data = torch.load(f, weights_only=False)
        label = f.stem.replace("nts.mu2e.FlateMinusMix1BBTriggered.MDC2025-002.", "")
        out_path = out_dir / f"event_display_{gi:03d}.png"
        plot_event(data, crystal_map, out_path, graph_label=label)
        print(f"  [{gi}] {label} -> {out_path.name}")

    print(f"Done. Plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
