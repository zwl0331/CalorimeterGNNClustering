#!/usr/bin/env python3
"""3-panel event display using calo-entrant truth from v2 ROOT files.

For each selected event/disk, plots side-by-side:
  Panel 1: Calo-entrant MC truth clusters (new definition)
  Panel 2: BFS reco clusters (from EventNtuple)
  Panel 3: GNN predicted clusters (with edge probability gradient)

Usage:
    source setup_env.sh

    # Plot first 6 events from val split
    OMP_NUM_THREADS=4 python3 scripts/plot_new_truth_clusters.py

    # Auto-find failure cases under new truth
    OMP_NUM_THREADS=4 python3 scripts/plot_new_truth_clusters.py --find-failures --n-scan 200

    # Use CaloClusterNetV1
    OMP_NUM_THREADS=4 python3 scripts/plot_new_truth_clusters.py \
        --config configs/calo_cluster_net_v1.yaml \
        --checkpoint outputs/runs/calo_cluster_net_v1_stage1/checkpoints/best_model.pt
"""

import argparse
import csv
import sys
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

from src.data.graph_builder import build_graph, compute_edge_features, compute_node_features
from src.data.normalization import load_stats, normalize_graph
from src.data.truth_labels_primary import build_calo_root_map
from src.geometry.crystal_geometry import load_crystal_map
from src.inference.cluster_reco import reconstruct_clusters
from src.models import build_model
from torch_geometric.data import Data

CRYSTAL_PITCH = 34.3
CRYSTAL_SIZE = CRYSTAL_PITCH * 0.92

CLUSTER_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
    "#a65628", "#f781bf", "#999999", "#66c2a5", "#fc8d62",
    "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f", "#e5c494",
    "#b3b3b3", "#1b9e77", "#d95f02", "#7570b3", "#e7298a",
]
UNCLUSTERED_COLOR = "#cccccc"
EDGE_CMAP = plt.cm.RdYlGn


def build_truth_clusters_new(simids, edeps, disks, nhits,
                              calo_root_map, purity_thresh=0.7):
    """Build calo-entrant truth clusters."""
    truth_labels = np.full(nhits, -1, dtype=np.int64)
    cluster_map = {}
    next_label = 0
    for i in range(nhits):
        pids = list(simids[i])
        deps = list(edeps[i])
        if len(pids) == 0 or sum(deps) <= 0:
            continue
        total_e = sum(deps)
        disk = int(disks[i])
        root_edep = {}
        for pid, dep in zip(pids, deps):
            root = calo_root_map.get((int(pid), disk), int(pid))
            root_edep[root] = root_edep.get(root, 0.0) + float(dep)
        best_root = max(root_edep, key=root_edep.get)
        if root_edep[best_root] / total_e < purity_thresh:
            continue
        key = (disk, best_root)
        if key not in cluster_map:
            cluster_map[key] = next_label
            next_label += 1
        truth_labels[i] = cluster_map[key]
    return truth_labels


def detect_failures(pred_labels, truth_labels, energies):
    """Detect merges and splits between pred and truth clusters.

    Returns (merged_pred_ids, split_truth_ids, involved_truth_ids, involved_pred_ids).
    The last two sets contain all truth/pred cluster IDs participating in any failure.
    """
    pred_ids = sorted(set(pred_labels[pred_labels >= 0].tolist()))
    truth_ids = sorted(set(truth_labels[truth_labels >= 0].tolist()))
    if not pred_ids or not truth_ids:
        return [], [], set(), set()

    overlap = defaultdict(lambda: defaultdict(float))
    pred_energy = defaultdict(float)
    for i in range(len(energies)):
        p, t, e = pred_labels[i], truth_labels[i], energies[i]
        if p >= 0:
            pred_energy[p] += e
        if p >= 0 and t >= 0:
            overlap[p][t] += e

    merged_preds = []
    involved_truth = set()
    involved_pred = set()
    for p in pred_ids:
        if p not in overlap:
            continue
        sig = [t for t, e in overlap[p].items()
               if pred_energy[p] > 0 and e / pred_energy[p] > 0.1]
        if len(sig) > 1:
            merged_preds.append(p)
            involved_pred.add(p)
            involved_truth.update(sig)

    truth_to_pred = defaultdict(list)
    for p in pred_ids:
        if p not in overlap:
            continue
        for t, e in overlap[p].items():
            if pred_energy[p] > 0 and e / pred_energy[p] > 0.5:
                truth_to_pred[t].append(p)
    split_truths = [t for t, ps in truth_to_pred.items() if len(ps) > 1]
    for t in split_truths:
        involved_truth.add(t)
        involved_pred.update(truth_to_pred[t])

    return merged_preds, split_truths, involved_truth, involved_pred


def assign_colors(labels):
    unique = sorted(set(labels[labels >= 0].tolist()))
    return {cid: CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
            for i, cid in enumerate(unique)}


def draw_panel(ax, hit_x, hit_y, hit_energy, labels, disk_id, crystal_map,
               title, edge_index=None, edge_probs=None,
               merged_clusters=None, split_clusters=None, truth_labels=None,
               focus_labels=None):
    """Draw one panel.

    Parameters
    ----------
    focus_labels : set or None
        If given, only these cluster IDs (in *labels*) are colored.
        Everything else is dimmed to light gray.  Used for debug zoom.
    """
    half = CRYSTAL_SIZE / 2
    n_hits = len(hit_x)
    e_max = hit_energy.max() if hit_energy.max() > 0 else 1.0

    # Jitter overlapping hits (same crystal, multiple time windows)
    draw_x = hit_x.copy()
    draw_y = hit_y.copy()
    pos_count = {}
    jitter = CRYSTAL_SIZE * 0.25
    for i in range(n_hits):
        key = (round(hit_x[i], 1), round(hit_y[i], 1))
        idx = pos_count.get(key, 0)
        pos_count[key] = idx + 1
        if idx > 0:
            # Offset: first duplicate goes up-right, second down-left, etc.
            dx = jitter * (1 if idx % 2 == 1 else -1)
            dy = jitter * (1 if idx <= 2 else -1)
            draw_x[i] += dx
            draw_y[i] += dy

    # Background crystals
    bg_patches = []
    for cid, (did, cx, cy) in crystal_map.items():
        if did != disk_id:
            continue
        bg_patches.append(Rectangle((cx - half, cy - half),
                                     CRYSTAL_SIZE, CRYSTAL_SIZE))
    pc_bg = PatchCollection(bg_patches, facecolor="#e8e8e8",
                            edgecolor="#cccccc", linewidth=0.3, zorder=1)
    ax.add_collection(pc_bg)

    # Edges — only draw edges touching focused hits in focus mode
    if edge_index is not None and edge_probs is not None:
        lines, colors = [], []
        seen = set()
        for ei in range(edge_index.shape[1]):
            s, d = edge_index[0, ei], edge_index[1, ei]
            key = (min(s, d), max(s, d))
            if key in seen:
                continue
            seen.add(key)
            # In focus mode, only draw edges where at least one endpoint
            # is in a focused cluster
            if focus_labels is not None:
                s_in = labels[s] in focus_labels
                d_in = labels[d] in focus_labels
                if not (s_in or d_in):
                    continue
            lines.append([(draw_x[s], draw_y[s]), (draw_x[d], draw_y[d])])
            colors.append(EDGE_CMAP(edge_probs[ei]))
        if lines:
            lw = 1.5 if focus_labels else 0.8
            lc = LineCollection(lines, colors=colors, linewidths=lw,
                                zorder=2, alpha=0.7)
            ax.add_collection(lc)

    # Only color focused clusters; dim the rest
    if focus_labels is not None:
        focused_ids = sorted(focus_labels)
        cmap = {cid: CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
                for i, cid in enumerate(focused_ids)}
    else:
        cmap = assign_colors(labels)

    for i in range(n_hits):
        cid = labels[i]
        is_focused = focus_labels is None or cid in focus_labels

        if not is_focused:
            # Dim non-focused hits
            color = "#e0e0e0"
            alpha = 0.3
            lw, ec = 0.3, "#d0d0d0"
            fontsize = 0  # hide energy text
        elif cid >= 0:
            color = cmap.get(cid, UNCLUSTERED_COLOR)
            alpha = 0.6 + 0.4 * (hit_energy[i] / e_max)
            lw, ec = 1.5, "black"
            fontsize = 7 if focus_labels else 5
        else:
            color = UNCLUSTERED_COLOR
            alpha = 0.5
            lw, ec = 1.0, "black"
            fontsize = 5

        if is_focused:
            if merged_clusters and cid in merged_clusters:
                lw, ec = 3.0, "red"
            if split_clusters and truth_labels is not None:
                tc = truth_labels[i]
                if tc in split_clusters:
                    lw, ec = 3.0, "darkorange"

        rect = Rectangle((draw_x[i] - half, draw_y[i] - half),
                          CRYSTAL_SIZE, CRYSTAL_SIZE,
                          facecolor=color, edgecolor=ec,
                          linewidth=lw, alpha=alpha, zorder=4 if is_focused else 2)
        ax.add_patch(rect)
        if fontsize > 0:
            ax.text(draw_x[i], draw_y[i], f"{hit_energy[i]:.1f}",
                    fontsize=fontsize, ha="center", va="center",
                    color="black", fontweight="bold" if focus_labels else "normal",
                    zorder=5, style="italic")

    unique_ids = sorted(set(labels[labels >= 0].tolist()))
    n_clust = len(unique_ids)
    n_unclust = (labels == -1).sum()
    total_e = hit_energy.sum()

    subtitle = f"{n_clust} clusters, {n_hits} hits, E={total_e:.0f} MeV"
    if n_unclust > 0:
        subtitle += f", {n_unclust} unclustered"
    ax.set_title(f"{title}\n{subtitle}", fontsize=10)
    ax.set_aspect("equal")
    ax.set_facecolor("#f7f7f7")
    ax.grid(True, linewidth=0.2, alpha=0.3, zorder=0)


def plot_event_3panel(hit_x, hit_y, hit_energy, disk_id,
                      truth_labels, bfs_labels, gnn_labels,
                      edge_index, edge_probs,
                      crystal_map, out_path, event_label="",
                      debug=False):
    """Plot 3-panel display.

    If debug=True, identifies the failure region, zooms in, and dims
    all clusters not involved in the failure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(36, 12))

    merged_gnn, split_gnn, inv_truth_gnn, inv_pred_gnn = detect_failures(
        gnn_labels, truth_labels, hit_energy)
    merged_bfs, split_bfs, inv_truth_bfs, inv_pred_bfs = detect_failures(
        bfs_labels, truth_labels, hit_energy)

    # Combine all truth clusters involved in any failure (GNN or BFS)
    all_inv_truth = inv_truth_gnn | inv_truth_bfs

    # Focus sets per panel (which cluster IDs to highlight)
    focus_truth = all_inv_truth if debug else None
    focus_bfs = inv_pred_bfs if debug else None
    focus_gnn = inv_pred_gnn if debug else None

    if debug and all_inv_truth:
        # Zoom to the region containing involved hits (+ padding)
        inv_hits = np.array([i for i in range(len(truth_labels))
                             if truth_labels[i] in all_inv_truth])
        if len(inv_hits) > 0:
            pad = 120
            xlim = (hit_x[inv_hits].min() - pad, hit_x[inv_hits].max() + pad)
            ylim = (hit_y[inv_hits].min() - pad, hit_y[inv_hits].max() + pad)
        else:
            pad = 80
            xlim = (hit_x.min() - pad, hit_x.max() + pad)
            ylim = (hit_y.min() - pad, hit_y.max() + pad)
    else:
        pad = 80
        xlim = (hit_x.min() - pad, hit_x.max() + pad)
        ylim = (hit_y.min() - pad, hit_y.max() + pad)

    # Panel 1: Calo-entrant truth
    draw_panel(axes[0], hit_x, hit_y, hit_energy, truth_labels,
               disk_id, crystal_map, "MC Truth (calo-entrant)",
               focus_labels=focus_truth)

    # Panel 2: BFS
    bfs_title = "BFS Reco"
    bfs_ann = []
    if merged_bfs:
        bfs_ann.append(f"{len(merged_bfs)} merge(s)")
    if split_bfs:
        bfs_ann.append(f"{len(split_bfs)} split(s)")
    if bfs_ann:
        bfs_title += f" [{', '.join(bfs_ann)}]"
    draw_panel(axes[1], hit_x, hit_y, hit_energy, bfs_labels,
               disk_id, crystal_map, bfs_title,
               merged_clusters=set(merged_bfs),
               split_clusters=set(split_bfs),
               truth_labels=truth_labels,
               focus_labels=focus_bfs)

    # Panel 3: GNN
    gnn_title = "GNN Predicted"
    gnn_ann = []
    if merged_gnn:
        gnn_ann.append(f"{len(merged_gnn)} merge(s)")
    if split_gnn:
        gnn_ann.append(f"{len(split_gnn)} split(s)")
    if gnn_ann:
        gnn_title += f" [{', '.join(gnn_ann)}]"
    draw_panel(axes[2], hit_x, hit_y, hit_energy, gnn_labels,
               disk_id, crystal_map, gnn_title,
               edge_index=edge_index, edge_probs=edge_probs,
               merged_clusters=set(merged_gnn),
               split_clusters=set(split_gnn),
               truth_labels=truth_labels,
               focus_labels=focus_gnn)

    for ax in axes:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    legend_handles = [
        Line2D([0], [0], color=EDGE_CMAP(0.0), linewidth=2,
               label="Edge prob ~0 (different cluster)"),
        Line2D([0], [0], color=EDGE_CMAP(1.0), linewidth=2,
               label="Edge prob ~1 (same cluster)"),
        Patch(facecolor="#e0e0e0", edgecolor="#d0d0d0", linewidth=0.5,
              label="Not involved (dimmed)"),
        Patch(facecolor="white", edgecolor="red", linewidth=2.5,
              label="Merged cluster"),
        Patch(facecolor="white", edgecolor="darkorange", linewidth=2.5,
              label="Split cluster"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=5,
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(event_label, fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def v1_to_v2_path(v1_name, v2_dir):
    stem = Path(v1_name).stem
    seq = stem.split(".")[-1]
    matches = list(Path(v2_dir).glob(f"mcs.*{seq}.root"))
    return matches[0] if matches else None


def process_event_disk(arrays, ev, disk_id, crystal_map, crystal_disk_map,
                       graph_cfg, model, stats, device, tau_edge,
                       tau_node=None):
    """Extract one event-disk from v2 arrays, run GNN, return plot data."""
    import uproot

    nhits = len(arrays["calohits.crystalId_"][ev])
    if nhits == 0:
        return None

    cryids = np.array(arrays["calohits.crystalId_"][ev], dtype=np.int64)
    energies = np.array(arrays["calohits.eDep_"][ev], dtype=np.float64)
    times = np.array(arrays["calohits.time_"][ev], dtype=np.float64)
    cluster_idx = np.array(arrays["calohits.clusterIdx_"][ev], dtype=np.int64)
    xs = np.array(arrays["calohits.crystalPos_.fCoordinates.fX"][ev], dtype=np.float64)
    ys = np.array(arrays["calohits.crystalPos_.fCoordinates.fY"][ev], dtype=np.float64)
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

    # Build calo-root map for this event
    sim_ids_evt = arrays["calomcsim.id"][ev]
    anc_evt = arrays["calomcsim.ancestorSimIds"][ev]
    calo_root_map = build_calo_root_map(
        sim_ids_evt, anc_evt, simids, cryids, crystal_disk_map)

    mc_truth = build_truth_clusters_new(
        d_simids, d_edeps, d_disks, n_disk, calo_root_map)

    # Build graph and run GNN
    edge_index, _ = build_graph(
        d_pos, d_t,
        r_max=graph_cfg["r_max_mm"], dt_max=graph_cfg["dt_max_ns"],
        k_min=graph_cfg["k_min"], k_max=graph_cfg["k_max"])

    if edge_index.shape[1] == 0:
        return {
            "hit_x": d_x, "hit_y": d_y, "energies": d_e,
            "truth_labels": mc_truth, "bfs_labels": d_cidx,
            "gnn_labels": np.arange(n_disk),
            "edge_index": edge_index, "edge_probs": np.array([]),
            "disk_id": disk_id,
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
        description="3-panel display with calo-entrant truth (v2 ROOT files)")
    parser.add_argument("--v2-dir", default="/exp/mu2e/data/users/wzhou2/GNN/root_files_v2")
    parser.add_argument("--checkpoint", default="outputs/runs/simple_edge_net_v1/checkpoints/best_model.pt")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", default="val", choices=["val", "test", "train"])
    parser.add_argument("--n-events", type=int, default=6)
    parser.add_argument("--find-failures", action="store_true")
    parser.add_argument("--n-scan", type=int, default=200)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tau_edge = cfg["inference"]["tau_edge"]
    graph_cfg = cfg["graph"]
    model_name = cfg["model"].get("name", "SimpleEdgeNet")
    has_node_head = model_name == "CaloClusterNetV1"
    lambda_node = cfg.get("train", {}).get("lambda_node", 0.0)
    tau_node = cfg["inference"].get("tau_node") if (has_node_head and lambda_node > 0) else None

    model = build_model(cfg)
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"Model: {model_name}, epoch {ckpt['epoch']}, val F1={ckpt['val_f1']:.4f}")
    print(f"tau_edge={tau_edge}, tau_node={tau_node}, device={device}")

    stats = load_stats(cfg["data"]["normalization_stats"])
    crystal_map = load_crystal_map("data/crystal_geometry.csv")
    crystal_disk_map = {cid: info[0] for cid, info in crystal_map.items()}

    import uproot

    # Find available v2 files for this split
    with open(cfg["data"]["splits"][args.split]) as f:
        v1_files = [l.strip() for l in f if l.strip()]

    v2_files = []
    for v1 in v1_files:
        v2 = v1_to_v2_path(v1, args.v2_dir)
        if v2 and v2.stat().st_size >= 1800 * 1024 * 1024:
            try:
                uproot.open(f"{v2}:EventNtuple/ntuple")
                v2_files.append(v2)
            except Exception:
                pass
    print(f"Split '{args.split}': {len(v2_files)} v2 files available")
    if not v2_files:
        print("ERROR: No valid v2 files for this split")
        return

    if args.output_dir is None:
        suffix = "debug_newtruth" if args.find_failures else "display_newtruth"
        args.output_dir = f"outputs/{suffix}_{model_name.lower()}"
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

    if args.find_failures:
        print(f"Scanning {args.n_scan} events for failure cases (new truth)...")
        failure_events = []

        for v2_path in v2_files:
            tree = uproot.open(f"{v2_path}:EventNtuple/ntuple")
            arrays = tree.arrays(branches, entry_stop=args.n_scan)
            fname = v2_path.name

            for ev in range(len(arrays)):
                for disk_id in [0, 1]:
                    result = process_event_disk(
                        arrays, ev, disk_id, crystal_map, crystal_disk_map,
                        graph_cfg, model, stats, device, tau_edge, tau_node)
                    if result is None:
                        continue
                    merged_gnn, split_gnn, _, _ = detect_failures(
                        result["gnn_labels"], result["truth_labels"],
                        result["energies"])
                    merged_bfs, split_bfs, _, _ = detect_failures(
                        result["bfs_labels"], result["truth_labels"],
                        result["energies"])
                    n_fail = len(merged_gnn) + len(split_gnn)
                    n_fail_bfs = len(merged_bfs) + len(split_bfs)
                    if n_fail > 0 or n_fail_bfs > 0:
                        failure_events.append(
                            (ev, disk_id, n_fail, n_fail_bfs, result, fname))

            if len(failure_events) >= args.n_events * 3:
                break

        # Sort by GNN failures first, then BFS
        failure_events.sort(key=lambda x: -(x[2] + x[3]))
        print(f"Found {len(failure_events)} event-disks with failures")

        n_plot = min(args.n_events, len(failure_events))
        for idx in range(n_plot):
            ev, disk_id, nf_gnn, nf_bfs, result, fname = failure_events[idx]
            label = (f"Event {ev}, Disk {disk_id} — "
                     f"GNN: {nf_gnn} fail, BFS: {nf_bfs} fail "
                     f"[new truth, {fname}]")
            out_path = out_dir / f"debug_{idx:03d}_evt{ev}_disk{disk_id}.png"
            plot_event_3panel(
                result["hit_x"], result["hit_y"], result["energies"],
                result["disk_id"], result["truth_labels"],
                result["bfs_labels"], result["gnn_labels"],
                result["edge_index"], result["edge_probs"],
                crystal_map, out_path, event_label=label,
                debug=True)
            print(f"  [{idx+1}/{n_plot}] {label}")

    else:
        plotted = 0
        for v2_path in v2_files:
            if plotted >= args.n_events:
                break
            tree = uproot.open(f"{v2_path}:EventNtuple/ntuple")
            n_read = args.n_events * 3
            arrays = tree.arrays(branches, entry_stop=n_read)
            fname = v2_path.name

            for ev in range(len(arrays)):
                if plotted >= args.n_events:
                    break
                for disk_id in [0, 1]:
                    if plotted >= args.n_events:
                        break
                    result = process_event_disk(
                        arrays, ev, disk_id, crystal_map, crystal_disk_map,
                        graph_cfg, model, stats, device, tau_edge, tau_node)
                    if result is None:
                        continue

                    label = (f"Event {ev}, Disk {disk_id} "
                             f"[new truth, {args.split}, {fname}]")
                    out_path = out_dir / f"display_{plotted:03d}_evt{ev}_disk{disk_id}.png"
                    plot_event_3panel(
                        result["hit_x"], result["hit_y"], result["energies"],
                        result["disk_id"], result["truth_labels"],
                        result["bfs_labels"], result["gnn_labels"],
                        result["edge_index"], result["edge_probs"],
                        crystal_map, out_path, event_label=label)
                    print(f"  [{plotted+1}/{args.n_events}] {label}")
                    plotted += 1

    print(f"\nDone. Plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
