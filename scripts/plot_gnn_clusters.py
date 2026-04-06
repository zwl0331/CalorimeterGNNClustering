#!/usr/bin/env python3
"""
GNN cluster visualization: 3-panel crystal-map event displays.

For each selected event/disk, plots side-by-side:
  Panel 1: MC truth clusters
  Panel 2: BFS reco clusters (from EventNtuple)
  Panel 3: GNN predicted clusters (with edge probability gradient)

Covers plan tasks 7e (debug visualization) and 7f (GNN cluster display).

Usage:
    source setup_env.sh

    # Plot first 6 events from val split
    python3 scripts/plot_gnn_clusters.py

    # Specific events from test split
    python3 scripts/plot_gnn_clusters.py --split test --event-indices 0 5 10

    # Auto-find failure cases (merges/splits)
    python3 scripts/plot_gnn_clusters.py --find-failures --n-scan 200
"""

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors

from src.data.graph_builder import build_graph, compute_edge_features, compute_node_features
from src.data.normalization import load_stats, normalize_graph
from src.data.truth_labels_primary import build_calo_root_map
from src.geometry.crystal_geometry import load_crystal_map
from src.inference.cluster_reco import reconstruct_clusters
from src.models import build_model
from torch_geometric.data import Data

# ── Plot style ──────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
    "figure.titlesize": 15,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
})

CRYSTAL_PITCH = 34.3
CRYSTAL_SIZE = CRYSTAL_PITCH * 0.92

# Perceptually distinct, colorblind-friendly palette (tab20 inspired)
CLUSTER_COLORS = [
    "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
    "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f",
    "#aec7e8", "#ffbb78", "#98df8a", "#c5b0d5", "#ff9896",
    "#c49c94", "#f7b6d2", "#dbdb8d", "#9edae5", "#c7c7c7",
]
UNCLUSTERED_COLOR = "#d9d9d9"
BG_CRYSTAL_FACE = "#f0f0f0"
BG_CRYSTAL_EDGE = "#d5d5d5"
PANEL_BG = "#fafafa"

# Red-to-green colormap for edge probabilities
EDGE_CMAP = plt.cm.RdYlGn


def build_mc_truth_clusters(simids, edeps, disks, nhits,
                            calo_root_map, purity_thresh=0.7):
    """Build MC truth cluster labels per hit using calo-entrant truth."""
    truth_labels = np.full(nhits, -1, dtype=np.int64)
    cluster_map = {}
    next_label = 0
    for i in range(nhits):
        sids = np.array(simids[i])
        deps = np.array(edeps[i], dtype=np.float64)
        if len(sids) == 0 or deps.sum() <= 0:
            continue
        disk = int(disks[i])
        root_edep = {}
        for pid, dep in zip(sids, deps):
            root = calo_root_map.get((int(pid), disk), int(pid))
            root_edep[root] = root_edep.get(root, 0.0) + float(dep)
        best_root = max(root_edep, key=root_edep.get)
        purity = root_edep[best_root] / deps.sum()
        if purity < purity_thresh:
            continue
        key = (disk, best_root)
        if key not in cluster_map:
            cluster_map[key] = next_label
            next_label += 1
        truth_labels[i] = cluster_map[key]
    return truth_labels


def detect_failures(pred_labels, truth_labels, energies):
    """Detect merges and splits between pred and truth clusters."""
    pred_ids = sorted(set(pred_labels[pred_labels >= 0].tolist()))
    truth_ids = sorted(set(truth_labels[truth_labels >= 0].tolist()))
    if not pred_ids or not truth_ids:
        return [], []

    overlap = defaultdict(lambda: defaultdict(float))
    pred_energy = defaultdict(float)
    for i in range(len(energies)):
        p, t, e = pred_labels[i], truth_labels[i], energies[i]
        if p >= 0:
            pred_energy[p] += e
        if p >= 0 and t >= 0:
            overlap[p][t] += e

    # Merges: pred cluster overlaps >1 truth cluster significantly
    merged_preds = []
    for p in pred_ids:
        if p not in overlap:
            continue
        sig = [t for t, e in overlap[p].items()
               if pred_energy[p] > 0 and e / pred_energy[p] > 0.1]
        if len(sig) > 1:
            merged_preds.append(p)

    # Splits: truth cluster covered by >1 pred cluster
    truth_to_pred = defaultdict(list)
    for p in pred_ids:
        if p not in overlap:
            continue
        for t, e in overlap[p].items():
            if pred_energy[p] > 0 and e / pred_energy[p] > 0.5:
                truth_to_pred[t].append(p)
    split_truths = [t for t, ps in truth_to_pred.items() if len(ps) > 1]

    return merged_preds, split_truths


def assign_colors(labels):
    """Map cluster labels to colors. -1 gets UNCLUSTERED_COLOR."""
    unique = sorted(set(labels[labels >= 0].tolist()))
    cmap = {cid: CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
            for i, cid in enumerate(unique)}
    return cmap


def draw_panel(ax, hit_x, hit_y, hit_energy, labels, disk_id, crystal_map,
               title, edge_index=None, edge_probs=None,
               merged_clusters=None, split_clusters=None, truth_labels=None,
               focus_hits=None):
    """Draw one panel of the 3-panel display.

    If *focus_hits* is provided (set of hit indices), non-focus hits are
    drawn very faintly so the failing clusters stand out.
    """
    half = CRYSTAL_SIZE / 2
    n_hits = len(hit_x)
    e_max = hit_energy.max() if hit_energy.max() > 0 else 1.0

    # Background crystals
    bg_patches = []
    for cid, (did, cx, cy) in crystal_map.items():
        if did != disk_id:
            continue
        bg_patches.append(Rectangle((cx - half, cy - half),
                                     CRYSTAL_SIZE, CRYSTAL_SIZE))
    pc_bg = PatchCollection(bg_patches, facecolor=BG_CRYSTAL_FACE,
                            edgecolor=BG_CRYSTAL_EDGE, linewidth=0.3, zorder=1)
    ax.add_collection(pc_bg)

    # Draw edges (only on GNN panel, colored by probability)
    if edge_index is not None and edge_probs is not None:
        lines = []
        colors = []
        seen = set()
        for ei in range(edge_index.shape[1]):
            s, d = edge_index[0, ei], edge_index[1, ei]
            key = (min(s, d), max(s, d))
            if key in seen:
                continue
            seen.add(key)
            # Dim edges not involving focus hits
            ea = 0.6
            if focus_hits is not None and s not in focus_hits and d not in focus_hits:
                ea = 0.08
            lines.append([(hit_x[s], hit_y[s]), (hit_x[d], hit_y[d])])
            c = list(EDGE_CMAP(edge_probs[ei]))
            c[3] = ea
            colors.append(c)
        if lines:
            lc = LineCollection(lines, colors=colors, linewidths=1.2,
                                zorder=2)
            ax.add_collection(lc)

    # Color map for this panel's clusters
    cmap = assign_colors(labels)

    # Group hits by crystal position for multi-hit handling
    pos_hits = defaultdict(list)  # (rounded x, y) -> [hit indices]
    for i in range(n_hits):
        pos_key = (round(hit_x[i], 1), round(hit_y[i], 1))
        pos_hits[pos_key].append(i)

    # Draw hit crystals
    for pos_key, hit_indices in pos_hits.items():
        n_at_pos = len(hit_indices)

        for slot, i in enumerate(hit_indices):
            in_focus = (focus_hits is None) or (i in focus_hits)
            cid = labels[i]
            if cid >= 0:
                color = cmap[cid]
                alpha = 0.5 + 0.5 * (hit_energy[i] / e_max)
            else:
                color = UNCLUSTERED_COLOR
                alpha = 0.5

            # Dim non-focus hits
            if not in_focus:
                alpha = 0.1

            # Mark merged/split clusters with thick border
            lw = 0.8
            ec = "#333333"
            if merged_clusters and cid in merged_clusters:
                lw = 3.0
                ec = "#cc0000"
            if split_clusters and truth_labels is not None:
                tc = truth_labels[i]
                if tc in split_clusters:
                    lw = 3.0
                    ec = "#e67300"

            cx, cy = hit_x[i], hit_y[i]

            if n_at_pos == 1:
                # Single hit — full crystal
                rect = Rectangle((cx - half, cy - half),
                                  CRYSTAL_SIZE, CRYSTAL_SIZE,
                                  facecolor=color, edgecolor=ec,
                                  linewidth=lw, alpha=alpha, zorder=4)
                ax.add_patch(rect)
            else:
                # Multi-hit — split crystal into horizontal bands
                band_h = CRYSTAL_SIZE / n_at_pos
                band_y = cy - half + slot * band_h
                rect = Rectangle((cx - half, band_y),
                                  CRYSTAL_SIZE, band_h,
                                  facecolor=color, edgecolor=ec,
                                  linewidth=lw, alpha=alpha,
                                  zorder=4 + slot)
                ax.add_patch(rect)

            # Energy label
            text_alpha = 1.0 if in_focus else 0.12
            fs = 8 if in_focus else 5

            if n_at_pos == 1:
                ty = cy
            else:
                ty = cy - half + (slot + 0.5) * (CRYSTAL_SIZE / n_at_pos)
                fs = max(5, fs - 1)  # slightly smaller for stacked labels

            ax.text(cx, ty, f"{hit_energy[i]:.1f}",
                    fontsize=fs, ha="center", va="center", color="#1a1a1a",
                    fontweight="semibold" if in_focus else "normal",
                    zorder=5 + slot, alpha=text_alpha)

    # Count clusters
    unique_ids = sorted(set(labels[labels >= 0].tolist()))
    n_clust = len(unique_ids)
    n_unclust = (labels == -1).sum()
    total_e = hit_energy.sum()

    subtitle = f"{n_clust} clusters, {n_hits} hits, E = {total_e:.0f} MeV"
    if n_unclust > 0:
        subtitle += f", {n_unclust} unclustered"
    ax.set_title(f"{title}\n{subtitle}", fontsize=13, fontweight="bold",
                 pad=10)

    ax.set_aspect("equal")
    ax.set_facecolor(PANEL_BG)
    ax.set_xlabel("x (mm)", fontsize=11)
    ax.set_ylabel("y (mm)", fontsize=11)
    ax.tick_params(axis="both", which="major", labelsize=9)
    ax.grid(False)


def _get_failure_focus(gnn_labels, truth_labels, merged_preds, split_truths):
    """Return set of hit indices involved in any merge or split."""
    focus = set()
    for i in range(len(gnn_labels)):
        if gnn_labels[i] in merged_preds:
            focus.add(i)
        if truth_labels[i] in split_truths:
            focus.add(i)
    return focus


def plot_event_3panel(hit_x, hit_y, hit_energy, disk_id,
                      truth_labels, bfs_labels, gnn_labels,
                      edge_index, edge_probs,
                      crystal_map, out_path, event_label="",
                      zoomed=False, focus_override=None):
    """Plot 3-panel display: Truth | BFS | GNN.

    If *zoomed* is True, zoom into the failing clusters, dim irrelevant
    hits, and label panels as "zoomed".

    *focus_override* — explicit set of hit indices to zoom into (e.g.
    BFS failure hits for success displays).
    """
    fig, axes = plt.subplots(1, 3, figsize=(36, 12))

    # Detect GNN failure modes
    merged_preds, split_truths = detect_failures(
        gnn_labels, truth_labels, hit_energy)

    merged_set = set(merged_preds)
    split_set = set(split_truths)

    # Focus hits and zoom limits
    focus_hits = None
    if focus_override:
        focus_hits = focus_override
    elif zoomed and (merged_preds or split_truths):
        focus_hits = _get_failure_focus(
            gnn_labels, truth_labels, merged_set, split_set)

    if focus_hits:
        fx = hit_x[list(focus_hits)]
        fy = hit_y[list(focus_hits)]
        pad = 120
        xlim = (fx.min() - pad, fx.max() + pad)
        ylim = (fy.min() - pad, fy.max() + pad)
    else:
        pad = 80
        xlim = (hit_x.min() - pad, hit_x.max() + pad)
        ylim = (hit_y.min() - pad, hit_y.max() + pad)

    zoom_tag = "  (zoomed)" if zoomed else ""

    # Panel 1: MC Truth
    draw_panel(axes[0], hit_x, hit_y, hit_energy, truth_labels,
               disk_id, crystal_map, f"MC Truth{zoom_tag}",
               focus_hits=focus_hits)

    # Panel 2: BFS
    draw_panel(axes[1], hit_x, hit_y, hit_energy, bfs_labels,
               disk_id, crystal_map, f"BFS Reco{zoom_tag}",
               focus_hits=focus_hits)

    # Panel 3: GNN
    gnn_title = "GNN Predicted"
    annotations = []
    if merged_preds:
        annotations.append(f"{len(merged_preds)} merge")
    if split_truths:
        annotations.append(f"{len(split_truths)} split")
    if annotations:
        gnn_title += f"  ({', '.join(annotations)})"
    gnn_title += zoom_tag

    draw_panel(axes[2], hit_x, hit_y, hit_energy, gnn_labels,
               disk_id, crystal_map, gnn_title,
               edge_index=edge_index, edge_probs=edge_probs,
               merged_clusters=merged_set,
               split_clusters=split_set,
               truth_labels=truth_labels,
               focus_hits=focus_hits)

    # Set consistent limits
    for ax in axes:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    # Legend — clean, compact
    legend_handles = [
        Line2D([0], [0], color=EDGE_CMAP(0.0), linewidth=2.5,
               label="Edge prob \u2248 0"),
        Line2D([0], [0], color=EDGE_CMAP(1.0), linewidth=2.5,
               label="Edge prob \u2248 1"),
        Patch(facecolor=UNCLUSTERED_COLOR, edgecolor="#666666", linewidth=0.5,
              label="Unclustered"),
        Patch(facecolor="white", edgecolor="#cc0000", linewidth=2.5,
              label="Merged cluster"),
        Patch(facecolor="white", edgecolor="#e67300", linewidth=2.5,
              label="Split cluster"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=5,
               fontsize=11, frameon=True, framealpha=0.9,
               edgecolor="#cccccc", bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(event_label, fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout(w_pad=3)
    fig.savefig(out_path, dpi=180, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)


def process_event_disk(arrays, ev, disk_id, crystal_map, graph_cfg,
                       model, stats, device, tau_edge, tau_node=None,
                       calo_root_map=None):
    """Extract one event-disk, run GNN, return all data for plotting.

    Returns None if the disk has < 2 hits.
    """
    nhits = len(arrays["calohits.crystalId_"][ev])
    if nhits == 0:
        return None

    cryids = np.array(arrays["calohits.crystalId_"][ev], dtype=np.int64)
    energies = np.array(arrays["calohits.eDep_"][ev], dtype=np.float64)
    times = np.array(arrays["calohits.time_"][ev], dtype=np.float64)
    cluster_idx = np.array(arrays["calohits.clusterIdx_"][ev], dtype=np.int64)
    xs = np.array(arrays["calohits.crystalPos_.fCoordinates.fX"][ev],
                  dtype=np.float64)
    ys = np.array(arrays["calohits.crystalPos_.fCoordinates.fY"][ev],
                  dtype=np.float64)
    simids = arrays["calohitsmc.simParticleIds"][ev]
    edeps_mc = arrays["calohitsmc.eDeps"][ev]

    disks = np.array([crystal_map[int(c)][0] if int(c) in crystal_map
                      else -1 for c in cryids], dtype=np.int64)

    if np.all(xs == 0) and np.all(ys == 0):
        for i, c in enumerate(cryids):
            if int(c) in crystal_map:
                _, xs[i], ys[i] = crystal_map[int(c)]

    dm = disks == disk_id
    n_disk = dm.sum()
    if n_disk < 2:
        return None

    d_e = energies[dm]
    d_t = times[dm]
    d_x = xs[dm]
    d_y = ys[dm]
    d_pos = np.stack([d_x, d_y], axis=1)
    d_cidx = cluster_idx[dm]
    d_disks = np.full(n_disk, disk_id, dtype=np.int64)

    disk_indices = np.where(dm)[0]
    d_simids = [list(simids[i]) for i in disk_indices]
    d_edeps = [list(edeps_mc[i]) for i in disk_indices]

    mc_truth = build_mc_truth_clusters(d_simids, d_edeps, d_disks, n_disk,
                                       calo_root_map)

    # Build graph and run GNN
    edge_index, _ = build_graph(
        d_pos, d_t,
        r_max=graph_cfg["r_max_mm"], dt_max=graph_cfg["dt_max_ns"],
        k_min=graph_cfg["k_min"], k_max=graph_cfg["k_max"])

    if edge_index.shape[1] == 0:
        gnn_labels = np.arange(n_disk)
        edge_probs = np.array([])
        return {
            "hit_x": d_x, "hit_y": d_y, "energies": d_e,
            "truth_labels": mc_truth, "bfs_labels": d_cidx,
            "gnn_labels": gnn_labels, "edge_index": edge_index,
            "edge_probs": edge_probs, "disk_id": disk_id,
        }

    node_feat = compute_node_features(d_pos, d_t, d_e)
    edge_feat = compute_edge_features(d_pos, d_t, d_e, edge_index)

    data = Data(
        x=torch.from_numpy(node_feat),
        edge_index=torch.from_numpy(edge_index),
        edge_attr=torch.from_numpy(edge_feat),
    )
    normalize_graph(data, stats)

    with torch.no_grad():
        output = model(data.to(device))

    # Handle both dict (CaloClusterNetV1) and tensor (SimpleEdgeNet)
    if isinstance(output, dict):
        logits_np = output["edge_logits"].cpu().numpy()
        nl = output.get("node_logits")
        node_logits_np = nl.cpu().numpy() if nl is not None else None
    else:
        logits_np = output.cpu().numpy()
        node_logits_np = None

    edge_probs = 1.0 / (1.0 + np.exp(-logits_np.astype(np.float64)))

    gnn_labels, _ = reconstruct_clusters(
        edge_index=edge_index, edge_logits=logits_np,
        n_nodes=n_disk, energies=d_e,
        tau_edge=tau_edge, min_hits=1, min_energy_mev=0.0,
        node_logits=node_logits_np, tau_node=tau_node)

    return {
        "hit_x": d_x, "hit_y": d_y, "energies": d_e,
        "truth_labels": mc_truth, "bfs_labels": d_cidx,
        "gnn_labels": gnn_labels, "edge_index": edge_index,
        "edge_probs": edge_probs, "disk_id": disk_id,
    }


def main():
    parser = argparse.ArgumentParser(
        description="3-panel event display: MC Truth | BFS | GNN")
    parser.add_argument("--root-dir", type=str,
                        default="/exp/mu2e/data/users/wzhou2/GNN/root_files_v2")
    parser.add_argument("--checkpoint", type=str,
                        default="outputs/runs/simple_edge_net_v2/checkpoints/best_model.pt")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--split", type=str, default="val",
                        choices=["val", "test"],
                        help="Which split to draw events from")
    parser.add_argument("--n-events", type=int, default=6,
                        help="Number of event-disk graphs to plot")
    parser.add_argument("--event-indices", type=int, nargs="*", default=None,
                        help="Specific event indices within the first file")
    parser.add_argument("--tau-edge", type=float, default=None,
                        help="Override tau_edge (default: from config)")
    parser.add_argument("--find-failures", action="store_true",
                        help="Scan events to find and plot failure cases "
                             "(merges/splits)")
    parser.add_argument("--find-successes", action="store_true",
                        help="Scan events to find and plot cases where "
                             "GNN clustering was very good (no failures, "
                             "many clusters)")
    parser.add_argument("--n-scan", type=int, default=200,
                        help="Events to scan in --find-failures/successes mode")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: outputs/gnn_cluster_display_<model>)")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    tau_edge = args.tau_edge or cfg["inference"]["tau_edge"]
    graph_cfg = cfg["graph"]
    model_name = cfg["model"].get("name", "SimpleEdgeNet")
    has_node_head = model_name == "CaloClusterNetV1"
    # Only apply tau_node if the node head was actually trained (lambda_node > 0)
    lambda_node = cfg.get("train", {}).get("lambda_node", 0.0)
    tau_node = cfg["inference"].get("tau_node") if (has_node_head and lambda_node > 0) else None

    # Load model
    model = build_model(cfg)
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"Model: {model_name}, epoch {ckpt['epoch']}, val F1={ckpt['val_f1']:.4f}")
    print(f"tau_edge = {tau_edge}, tau_node = {tau_node}, device = {device}")

    stats = load_stats(cfg["data"]["normalization_stats"])
    crystal_map = load_crystal_map("data/crystal_geometry.csv")

    # Load split file list
    import uproot
    split_key = args.split
    with open(cfg["data"]["splits"][split_key]) as f:
        file_list = [l.strip() for l in f if l.strip()]
    print(f"Split '{split_key}': {len(file_list)} files")

    if args.output_dir is None:
        args.output_dir = f"outputs/gnn_cluster_display_{model_name.lower()}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    branches = [
        "calohits.crystalId_", "calohits.eDep_", "calohits.time_",
        "calohits.clusterIdx_",
        "calohits.crystalPos_.fCoordinates.fX",
        "calohits.crystalPos_.fCoordinates.fY",
        "calohitsmc.simParticleIds", "calohitsmc.eDeps",
        "calomcsim.id", "calomcsim.ancestorSimIds",
    ]

    crystal_disk_map = {cid: disk for cid, (disk, _, _) in crystal_map.items()}

    if args.find_failures:
        # Scan events to find interesting failure cases
        print(f"Scanning {args.n_scan} events for failure cases...")
        failure_events = []

        fname = Path(file_list[0]).name
        local_path = str(Path(args.root_dir) / fname)
        tree = uproot.open(local_path + ":EventNtuple/ntuple")
        arrays = tree.arrays(branches, entry_stop=args.n_scan)

        for ev in range(len(arrays)):
            # Build calo-entrant root map for this event
            crm = build_calo_root_map(
                arrays["calomcsim.id"][ev],
                arrays["calomcsim.ancestorSimIds"][ev],
                arrays["calohitsmc.simParticleIds"][ev],
                arrays["calohits.crystalId_"][ev],
                crystal_disk_map)
            for disk_id in [0, 1]:
                result = process_event_disk(
                    arrays, ev, disk_id, crystal_map, graph_cfg,
                    model, stats, device, tau_edge, tau_node=tau_node,
                    calo_root_map=crm)
                if result is None:
                    continue
                merged, split = detect_failures(
                    result["gnn_labels"], result["truth_labels"],
                    result["energies"])
                n_fail = len(merged) + len(split)
                if n_fail > 0:
                    failure_events.append((ev, disk_id, n_fail, result))

        failure_events.sort(key=lambda x: -x[2])
        print(f"Found {len(failure_events)} event-disks with failures")

        n_plot = min(args.n_events, len(failure_events))
        for idx in range(n_plot):
            ev, disk_id, n_fail, result = failure_events[idx]
            label = (f"Event {ev}, Disk {disk_id} — "
                     f"{n_fail} failure(s) [file: {Path(file_list[0]).name}]")
            out_path = out_dir / f"debug_{idx:03d}_evt{ev}_disk{disk_id}.png"
            plot_event_3panel(
                result["hit_x"], result["hit_y"], result["energies"],
                result["disk_id"], result["truth_labels"],
                result["bfs_labels"], result["gnn_labels"],
                result["edge_index"], result["edge_probs"],
                crystal_map, out_path, event_label=label,
                zoomed=True)
            print(f"  [{idx+1}/{n_plot}] {label} -> {out_path.name}")

    elif args.find_successes:
        # Scan events to find GNN success cases (no failures, many clusters)
        print(f"Scanning {args.n_scan} events for success cases...")
        success_events = []

        fname = Path(file_list[0]).name
        local_path = str(Path(args.root_dir) / fname)
        tree = uproot.open(local_path + ":EventNtuple/ntuple")
        arrays = tree.arrays(branches, entry_stop=args.n_scan)

        for ev in range(len(arrays)):
            crm = build_calo_root_map(
                arrays["calomcsim.id"][ev],
                arrays["calomcsim.ancestorSimIds"][ev],
                arrays["calohitsmc.simParticleIds"][ev],
                arrays["calohits.crystalId_"][ev],
                crystal_disk_map)
            for disk_id in [0, 1]:
                result = process_event_disk(
                    arrays, ev, disk_id, crystal_map, graph_cfg,
                    model, stats, device, tau_edge, tau_node=tau_node,
                    calo_root_map=crm)
                if result is None:
                    continue
                gnn_merged, gnn_split = detect_failures(
                    result["gnn_labels"], result["truth_labels"],
                    result["energies"])
                bfs_merged, bfs_split = detect_failures(
                    result["bfs_labels"], result["truth_labels"],
                    result["energies"])
                n_gnn_fail = len(gnn_merged) + len(gnn_split)
                n_bfs_fail = len(bfs_merged) + len(bfs_split)
                # GNN succeeds where BFS fails
                if n_gnn_fail == 0 and n_bfs_fail > 0:
                    # Store BFS failure hit indices for zooming
                    bfs_focus = _get_failure_focus(
                        result["bfs_labels"], result["truth_labels"],
                        set(bfs_merged), set(bfs_split))
                    success_events.append(
                        (ev, disk_id, n_bfs_fail, result, bfs_focus))

        # Sort by most BFS failures (biggest GNN advantage)
        success_events.sort(key=lambda x: -x[2])
        print(f"Found {len(success_events)} event-disks where "
              f"GNN perfect but BFS has failures")

        if args.output_dir is None:
            args.output_dir = f"outputs/success_{model_name.lower()}"
            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

        n_plot = min(args.n_events, len(success_events))
        for idx in range(n_plot):
            ev, disk_id, n_bfs_fail, result, bfs_focus = success_events[idx]
            label = (f"Event {ev}, Disk {disk_id} — "
                     f"GNN: 0 failures, BFS: {n_bfs_fail} failure(s) "
                     f"[file: {Path(file_list[0]).name}]")
            out_path = out_dir / f"success_{idx:03d}_evt{ev}_disk{disk_id}.png"
            plot_event_3panel(
                result["hit_x"], result["hit_y"], result["energies"],
                result["disk_id"], result["truth_labels"],
                result["bfs_labels"], result["gnn_labels"],
                result["edge_index"], result["edge_probs"],
                crystal_map, out_path, event_label=label,
                zoomed=True, focus_override=bfs_focus)
            print(f"  [{idx+1}/{n_plot}] {label} -> {out_path.name}")

    else:
        # Plot specific events or first N
        fname = Path(file_list[0]).name
        local_path = str(Path(args.root_dir) / fname)
        tree = uproot.open(local_path + ":EventNtuple/ntuple")

        # Determine how many events to read
        if args.event_indices:
            n_read = max(args.event_indices) + 1
        else:
            n_read = args.n_events * 3  # read extra since some disks are skipped
        arrays = tree.arrays(branches, entry_stop=n_read)

        plotted = 0
        target = args.n_events
        ev_iter = args.event_indices if args.event_indices else range(len(arrays))

        for ev in ev_iter:
            if plotted >= target:
                break
            if ev >= len(arrays):
                continue
            # Build calo-entrant root map for this event
            crm = build_calo_root_map(
                arrays["calomcsim.id"][ev],
                arrays["calomcsim.ancestorSimIds"][ev],
                arrays["calohitsmc.simParticleIds"][ev],
                arrays["calohits.crystalId_"][ev],
                crystal_disk_map)
            for disk_id in [0, 1]:
                if plotted >= target:
                    break
                result = process_event_disk(
                    arrays, ev, disk_id, crystal_map, graph_cfg,
                    model, stats, device, tau_edge, tau_node=tau_node,
                    calo_root_map=crm)
                if result is None:
                    continue

                label = (f"Event {ev}, Disk {disk_id} "
                         f"[{split_key} split, {Path(file_list[0]).name}]")
                out_path = out_dir / f"display_{plotted:03d}_evt{ev}_disk{disk_id}.png"
                plot_event_3panel(
                    result["hit_x"], result["hit_y"], result["energies"],
                    result["disk_id"], result["truth_labels"],
                    result["bfs_labels"], result["gnn_labels"],
                    result["edge_index"], result["edge_probs"],
                    crystal_map, out_path, event_label=label)
                print(f"  [{plotted+1}/{target}] {label} -> {out_path.name}")
                plotted += 1

    print(f"\nDone. Plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
