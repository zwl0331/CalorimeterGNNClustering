#!/usr/bin/env python3
"""
Head-to-head evaluation: GNN vs BFS, both against MC truth.

For each val-split event/disk:
  1. Extract MC truth clusters from calohitsmc branches
  2. Get BFS reco clusters from calohits.clusterIdx_
  3. Run GNN inference → predicted clusters from connected components
  4. Evaluate both BFS and GNN against MC truth with the same matching

Usage:
    source setup_env.sh
    python3 scripts/evaluate_head_to_head.py --root-dir /exp/mu2e/data/users/wzhou2/GNN/root_files
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
import uproot
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from src.data.graph_builder import build_graph, compute_edge_features, compute_node_features
from src.data.normalization import load_stats, normalize_graph
from src.geometry.crystal_geometry import load_crystal_map
from src.models.simple_edge_net import SimpleEdgeNet
from torch_geometric.data import Data


def build_mc_truth_clusters(simids, edeps, disks, nhits, purity_thresh=0.7):
    """Build MC truth cluster labels per hit.

    Returns:
        truth_labels: int array (nhits,), -1 = ambiguous/unassigned
    """
    truth_labels = np.full(nhits, -1, dtype=np.int64)
    cluster_map = {}
    next_label = 0

    for i in range(nhits):
        sids = np.array(simids[i])
        deps = np.array(edeps[i])
        if len(sids) == 0 or deps.sum() <= 0:
            continue
        best = np.argmax(deps)
        purity = deps[best] / deps.sum()
        if purity < purity_thresh:
            continue
        key = (int(disks[i]), int(sids[best]))
        if key not in cluster_map:
            cluster_map[key] = next_label
            next_label += 1
        truth_labels[i] = cluster_map[key]

    return truth_labels


def match_clusters_energy(pred_labels, truth_labels, energies):
    """Energy-weighted greedy matching. Returns per-cluster and aggregate stats."""
    valid = (pred_labels >= 0) & (truth_labels >= 0)
    pred_ids = set(pred_labels[pred_labels >= 0].tolist())
    truth_ids = set(truth_labels[truth_labels >= 0].tolist())

    if not pred_ids or not truth_ids:
        return {"n_pred": len(pred_ids), "n_truth": len(truth_ids),
                "n_matched_pred": 0, "n_matched_truth": 0,
                "purities": [], "completenesses": [],
                "n_split": 0, "n_merged": 0}

    # Build energy overlap
    overlap = defaultdict(lambda: defaultdict(float))
    pred_energy = defaultdict(float)
    truth_energy = defaultdict(float)

    for i in range(len(energies)):
        e = energies[i]
        p = pred_labels[i]
        t = truth_labels[i]
        if p >= 0:
            pred_energy[p] += e
        if t >= 0:
            truth_energy[t] += e
        if p >= 0 and t >= 0:
            overlap[p][t] += e

    # Greedy: for each pred cluster, find best truth match
    purities, completenesses = [], []
    matched_truth = set()
    for p in sorted(pred_ids):
        if p not in overlap:
            continue
        best_t = max(overlap[p], key=lambda t: overlap[p][t])
        shared = overlap[p][best_t]
        pur = shared / pred_energy[p] if pred_energy[p] > 0 else 0
        comp = shared / truth_energy[best_t] if truth_energy[best_t] > 0 else 0
        if pur > 0.5 and comp > 0.5:
            purities.append(pur)
            completenesses.append(comp)
            matched_truth.add(best_t)

    # Splits: truth cluster matched by >1 pred cluster
    truth_to_pred = defaultdict(list)
    for p in sorted(pred_ids):
        if p not in overlap:
            continue
        for t, e in overlap[p].items():
            if pred_energy[p] > 0 and e / pred_energy[p] > 0.5:
                truth_to_pred[t].append(p)
    n_split = sum(1 for t, ps in truth_to_pred.items() if len(ps) > 1)

    # Merges: pred cluster overlapping >1 truth cluster significantly
    n_merged = 0
    for p in sorted(pred_ids):
        if p not in overlap:
            continue
        sig = [t for t, e in overlap[p].items() if pred_energy[p] > 0 and e / pred_energy[p] > 0.1]
        if len(sig) > 1:
            n_merged += 1

    return {
        "n_pred": len(pred_ids),
        "n_truth": len(truth_ids),
        "n_matched_pred": len(purities),
        "n_matched_truth": len(matched_truth),
        "purities": purities,
        "completenesses": completenesses,
        "n_split": n_split,
        "n_merged": n_merged,
    }


def gnn_predict_clusters(model, data, device, threshold=0.5):
    """Run GNN inference, return predicted cluster labels per node."""
    data_gpu = data.clone().to(device)
    with torch.no_grad():
        logits = model(data_gpu)
    probs = torch.sigmoid(logits).cpu().numpy()
    ei = data.edge_index.numpy()
    mask = data.edge_mask.bool().numpy()

    # Positive predicted edges
    pred_pos = (probs >= threshold) & mask
    n_nodes = data.x.shape[0]

    if pred_pos.sum() == 0:
        return np.arange(n_nodes)  # each node is its own cluster

    src = ei[0, pred_pos]
    dst = ei[1, pred_pos]
    src_sym = np.concatenate([src, dst])
    dst_sym = np.concatenate([dst, src])
    vals = np.ones(len(src_sym))
    adj = coo_matrix((vals, (src_sym, dst_sym)), shape=(n_nodes, n_nodes))
    _, labels = connected_components(adj, directed=False)
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", type=str,
                        default="/exp/mu2e/data/users/wzhou2/GNN/root_files")
    parser.add_argument("--n-events", type=int, default=500)
    parser.add_argument("--checkpoint", type=str,
                        default="outputs/runs/simple_edge_net_v1/checkpoints/best_model.pt")
    parser.add_argument("--output-dir", type=str,
                        default="outputs/runs/simple_edge_net_v1")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = SimpleEdgeNet(node_dim=6, edge_dim=8, hidden_dim=64, n_mp_layers=3, dropout=0.1)
    ckpt = torch.load(args.checkpoint, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"Loaded model from epoch {ckpt['epoch']} (val F1={ckpt['val_f1']:.4f})")

    # Load normalization stats and crystal map
    stats = load_stats("data/normalization_stats.pt")
    crystal_map = load_crystal_map("data/crystal_geometry.csv")

    import yaml
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)
    graph_cfg = cfg["graph"]

    # Load val file list
    with open("splits/val_files.txt") as f:
        val_files = [l.strip() for l in f if l.strip()]

    branches = [
        "calohits.crystalId_", "calohits.eDep_", "calohits.time_",
        "calohits.clusterIdx_",
        "calohits.crystalPos_.fCoordinates.fX",
        "calohits.crystalPos_.fCoordinates.fY",
        "calohitsmc.simParticleIds", "calohitsmc.eDeps",
    ]

    bfs_results = []
    gnn_results = []
    t0 = time.time()

    for fi, fpath in enumerate(val_files):
        fname = Path(fpath).name
        local_path = str(Path(args.root_dir) / fname)
        print(f"  [{fi+1}/{len(val_files)}] {fname}...", end=" ", flush=True)

        tree = uproot.open(local_path + ":EventNtuple/ntuple")
        arrays = tree.arrays(branches, entry_stop=args.n_events)
        n_events = len(arrays)

        for ev in range(n_events):
            nhits = len(arrays["calohits.crystalId_"][ev])
            if nhits == 0:
                continue

            cryids = np.array(arrays["calohits.crystalId_"][ev], dtype=np.int64)
            energies = np.array(arrays["calohits.eDep_"][ev], dtype=np.float64)
            times = np.array(arrays["calohits.time_"][ev], dtype=np.float64)
            cluster_idx = np.array(arrays["calohits.clusterIdx_"][ev], dtype=np.int64)
            xs = np.array(arrays["calohits.crystalPos_.fCoordinates.fX"][ev], dtype=np.float64)
            ys = np.array(arrays["calohits.crystalPos_.fCoordinates.fY"][ev], dtype=np.float64)
            simids = arrays["calohitsmc.simParticleIds"][ev]
            edeps_mc = arrays["calohitsmc.eDeps"][ev]

            disks = np.array([crystal_map[int(c)][0] if int(c) in crystal_map else -1
                              for c in cryids], dtype=np.int64)

            # Fallback positions
            if np.all(xs == 0) and np.all(ys == 0):
                for i, c in enumerate(cryids):
                    if int(c) in crystal_map:
                        _, xs[i], ys[i] = crystal_map[int(c)]

            for disk_id in [0, 1]:
                dm = disks == disk_id
                n_disk = dm.sum()
                if n_disk < 2:
                    continue

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

                # MC truth clusters
                mc_truth = build_mc_truth_clusters(d_simids, d_edeps, d_disks, n_disk)

                # BFS clusters = clusterIdx_
                bfs_labels = d_cidx.copy()

                # Evaluate BFS vs MC truth
                bfs_match = match_clusters_energy(bfs_labels, mc_truth, d_e)
                bfs_results.append(bfs_match)

                # Build graph + run GNN
                edge_index, diag = build_graph(
                    d_pos, d_t,
                    r_max=graph_cfg["r_max_mm"], dt_max=graph_cfg["dt_max_ns"],
                    k_min=graph_cfg["k_min"], k_max=graph_cfg["k_max"])

                if edge_index.shape[1] == 0:
                    gnn_results.append(match_clusters_energy(
                        np.arange(n_disk), mc_truth, d_e))
                    continue

                node_feat = compute_node_features(d_pos, d_t, d_e)
                edge_feat = compute_edge_features(d_pos, d_t, d_e, edge_index)

                data = Data(
                    x=torch.from_numpy(node_feat),
                    edge_index=torch.from_numpy(edge_index),
                    edge_attr=torch.from_numpy(edge_feat),
                    y_edge=torch.zeros(edge_index.shape[1]),
                    edge_mask=torch.ones(edge_index.shape[1], dtype=torch.bool),
                )
                normalize_graph(data, stats)

                gnn_labels = gnn_predict_clusters(model, data, device)
                gnn_match = match_clusters_energy(gnn_labels, mc_truth, d_e)
                gnn_results.append(gnn_match)

        print(f"{n_events} events")

    elapsed = time.time() - t0
    print(f"\nProcessed {len(bfs_results)} disk-graphs in {elapsed:.1f}s")

    # Aggregate
    def aggregate(results, name):
        all_pur = [p for r in results for p in r["purities"]]
        all_comp = [c for r in results for c in r["completenesses"]]
        n_pred = sum(r["n_pred"] for r in results)
        n_truth = sum(r["n_truth"] for r in results)
        n_matched_pred = sum(r["n_matched_pred"] for r in results)
        n_matched_truth = sum(r["n_matched_truth"] for r in results)
        n_split = sum(r["n_split"] for r in results)
        n_merged = sum(r["n_merged"] for r in results)

        print(f"\n{'='*60}")
        print(f"  {name} vs MC Truth")
        print(f"{'='*60}")
        print(f"  Reco clusters:        {n_pred}")
        print(f"  Truth clusters:       {n_truth}")
        print(f"  Matched reco:         {n_matched_pred} ({100*n_matched_pred/n_pred:.1f}%)" if n_pred else "")
        print(f"  Matched truth:        {n_matched_truth} ({100*n_matched_truth/n_truth:.1f}%)" if n_truth else "")
        print(f"  Mean purity:          {np.mean(all_pur):.4f} +/- {np.std(all_pur):.4f}" if all_pur else "")
        print(f"  Median purity:        {np.median(all_pur):.4f}" if all_pur else "")
        print(f"  Mean completeness:    {np.mean(all_comp):.4f} +/- {np.std(all_comp):.4f}" if all_comp else "")
        print(f"  Median completeness:  {np.median(all_comp):.4f}" if all_comp else "")
        print(f"  Split truth clusters: {n_split}")
        print(f"  Merged reco clusters: {n_merged}")

        return {
            "purities": all_pur, "completenesses": all_comp,
            "n_pred": n_pred, "n_truth": n_truth,
            "n_matched_pred": n_matched_pred, "n_matched_truth": n_matched_truth,
            "reco_match_rate": n_matched_pred / n_pred if n_pred else 0,
            "truth_match_rate": n_matched_truth / n_truth if n_truth else 0,
            "n_split": n_split, "n_merged": n_merged,
        }

    bfs_agg = aggregate(bfs_results, "BFS")
    gnn_agg = aggregate(gnn_results, "GNN (SimpleEdgeNet)")

    # Save comparison CSV
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, agg in [("BFS", bfs_agg), ("GNN", gnn_agg)]:
        rows.append({
            "method": name,
            "reco_clusters": agg["n_pred"],
            "truth_clusters": agg["n_truth"],
            "reco_match_rate": f"{agg['reco_match_rate']:.4f}",
            "truth_match_rate": f"{agg['truth_match_rate']:.4f}",
            "mean_purity": f"{np.mean(agg['purities']):.4f}" if agg['purities'] else "0",
            "mean_completeness": f"{np.mean(agg['completenesses']):.4f}" if agg['completenesses'] else "0",
            "n_split": agg["n_split"],
            "n_merged": agg["n_merged"],
        })

    csv_path = out_dir / "head_to_head.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved to {csv_path}")

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Head-to-Head: GNN vs BFS (both evaluated against MC Truth)",
                 fontsize=14, fontweight="bold")

    # 1. Purity distributions
    ax = axes[0, 0]
    bins = np.linspace(0.5, 1.0, 60)
    ax.hist(bfs_agg["purities"], bins=bins, alpha=0.6, label="BFS", color="coral", edgecolor="white")
    ax.hist(gnn_agg["purities"], bins=bins, alpha=0.6, label="GNN", color="steelblue", edgecolor="white")
    ax.axvline(np.mean(bfs_agg["purities"]), color="red", linestyle="--", linewidth=2)
    ax.axvline(np.mean(gnn_agg["purities"]), color="blue", linestyle="--", linewidth=2)
    ax.set_xlabel("Purity (vs MC truth)")
    ax.set_ylabel("Count")
    ax.set_title("Cluster Purity")
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. Completeness distributions
    ax = axes[0, 1]
    ax.hist(bfs_agg["completenesses"], bins=bins, alpha=0.6, label="BFS", color="coral", edgecolor="white")
    ax.hist(gnn_agg["completenesses"], bins=bins, alpha=0.6, label="GNN", color="steelblue", edgecolor="white")
    ax.axvline(np.mean(bfs_agg["completenesses"]), color="red", linestyle="--", linewidth=2)
    ax.axvline(np.mean(gnn_agg["completenesses"]), color="blue", linestyle="--", linewidth=2)
    ax.set_xlabel("Completeness (vs MC truth)")
    ax.set_ylabel("Count")
    ax.set_title("Cluster Completeness")
    ax.legend()
    ax.grid(alpha=0.3)

    # 3. Match rates bar chart
    ax = axes[0, 2]
    metrics = ["Reco Match\nRate", "Truth Match\nRate"]
    bfs_vals = [bfs_agg["reco_match_rate"]*100, bfs_agg["truth_match_rate"]*100]
    gnn_vals = [gnn_agg["reco_match_rate"]*100, gnn_agg["truth_match_rate"]*100]
    x = np.arange(len(metrics))
    w = 0.35
    ax.bar(x - w/2, bfs_vals, w, label="BFS", color="coral", alpha=0.8)
    ax.bar(x + w/2, gnn_vals, w, label="GNN", color="steelblue", alpha=0.8)
    for i, (b, g) in enumerate(zip(bfs_vals, gnn_vals)):
        ax.text(i - w/2, b + 0.5, f"{b:.1f}%", ha="center", fontsize=9)
        ax.text(i + w/2, g + 0.5, f"{g:.1f}%", ha="center", fontsize=9)
    ax.set_ylabel("Percent (%)")
    ax.set_title("Cluster Match Rates")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(0, 105)

    # 4. Purity scatter: BFS vs GNN per graph
    ax = axes[1, 0]
    bfs_per_graph_pur = [np.mean(r["purities"]) if r["purities"] else 1.0 for r in bfs_results]
    gnn_per_graph_pur = [np.mean(r["purities"]) if r["purities"] else 1.0 for r in gnn_results]
    ax.scatter(bfs_per_graph_pur, gnn_per_graph_pur, alpha=0.15, s=8, color="navy")
    ax.plot([0.5, 1], [0.5, 1], "r--", alpha=0.5)
    ax.set_xlabel("BFS purity")
    ax.set_ylabel("GNN purity")
    ax.set_title("Per-Graph Purity: GNN vs BFS")
    ax.set_xlim(0.5, 1.02)
    ax.set_ylim(0.5, 1.02)
    ax.grid(alpha=0.3)

    # 5. Split/merge comparison
    ax = axes[1, 1]
    cats = ["Splits", "Merges"]
    bfs_sm = [bfs_agg["n_split"], bfs_agg["n_merged"]]
    gnn_sm = [gnn_agg["n_split"], gnn_agg["n_merged"]]
    x = np.arange(len(cats))
    ax.bar(x - w/2, bfs_sm, w, label="BFS", color="coral", alpha=0.8)
    ax.bar(x + w/2, gnn_sm, w, label="GNN", color="steelblue", alpha=0.8)
    for i, (b, g) in enumerate(zip(bfs_sm, gnn_sm)):
        ax.text(i - w/2, b + max(bfs_sm+gnn_sm)*0.02, str(b), ha="center", fontsize=9)
        ax.text(i + w/2, g + max(bfs_sm+gnn_sm)*0.02, str(g), ha="center", fontsize=9)
    ax.set_ylabel("Count")
    ax.set_title("Cluster Splits & Merges")
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    # 6. Summary table
    ax = axes[1, 2]
    ax.axis("off")
    table_data = [
        ["Reco clusters", str(bfs_agg["n_pred"]), str(gnn_agg["n_pred"])],
        ["Truth clusters", str(bfs_agg["n_truth"]), str(gnn_agg["n_truth"])],
        ["Reco match rate", f"{bfs_agg['reco_match_rate']:.1%}", f"{gnn_agg['reco_match_rate']:.1%}"],
        ["Truth match rate", f"{bfs_agg['truth_match_rate']:.1%}", f"{gnn_agg['truth_match_rate']:.1%}"],
        ["Mean purity", f"{np.mean(bfs_agg['purities']):.4f}", f"{np.mean(gnn_agg['purities']):.4f}"],
        ["Mean completeness", f"{np.mean(bfs_agg['completenesses']):.4f}", f"{np.mean(gnn_agg['completenesses']):.4f}"],
        ["Splits", str(bfs_agg["n_split"]), str(gnn_agg["n_split"])],
        ["Merges", str(bfs_agg["n_merged"]), str(gnn_agg["n_merged"])],
    ]
    table = ax.table(cellText=table_data, colLabels=["Metric", "BFS", "GNN"],
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)

    # Color winning cells
    for i, row in enumerate(table_data):
        try:
            bv = float(row[1].rstrip("%"))
            gv = float(row[2].rstrip("%"))
            if "split" in row[0].lower() or "merge" in row[0].lower():
                winner = 1 if bv <= gv else 2  # lower is better
            else:
                winner = 1 if bv >= gv else 2  # higher is better
            table[i+1, winner].set_facecolor("#d4edda")
        except ValueError:
            pass

    ax.set_title("Summary Comparison", pad=20)

    plt.tight_layout()
    plot_path = out_dir / "head_to_head.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
