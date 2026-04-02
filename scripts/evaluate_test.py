#!/usr/bin/env python3
"""
Test set evaluation: GNN vs BFS, both against MC truth.

This script is run ONCE to produce the final comparison. Do not iterate.

For each test-split event/disk:
  1. Extract MC truth clusters from calohitsmc branches
  2. Get BFS reco clusters from calohits.clusterIdx_
  3. Run GNN inference with frozen tau_edge -> predicted clusters
  4. Evaluate both against MC truth (energy-weighted matching)
  5. Record per-truth-cluster detail for binned metrics

Outputs:
  - outputs/test_eval/test_results.csv          (overall BFS vs GNN)
  - outputs/test_eval/truth_cluster_detail.csv   (per-truth-cluster, for binning)
  - outputs/test_eval/test_evaluation.png        (comparison plots)

Usage:
    source setup_env.sh
    python3 scripts/evaluate_test.py
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
import yaml

from src.data.graph_builder import build_graph, compute_edge_features, compute_node_features
from src.data.normalization import load_stats, normalize_graph
from src.geometry.crystal_geometry import load_crystal_map
from src.inference.cluster_reco import reconstruct_clusters
from src.models.simple_edge_net import SimpleEdgeNet
from torch_geometric.data import Data


def build_mc_truth_clusters(simids, edeps, disks, nhits, purity_thresh=0.7):
    """Build MC truth cluster labels per hit.

    Returns truth_labels (int array, -1 = ambiguous/unassigned) and
    per-truth-cluster metadata: {cluster_id: (energy, n_hits)}.
    """
    truth_labels = np.full(nhits, -1, dtype=np.int64)
    cluster_map = {}  # (disk, simparticle) -> label
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


def match_clusters_detail(pred_labels, truth_labels, energies):
    """Energy-weighted greedy matching with per-truth-cluster detail.

    Returns (aggregate_dict, truth_detail_list).
    truth_detail_list: one entry per truth cluster with energy, n_hits, matched.
    """
    pred_ids = sorted(set(pred_labels[pred_labels >= 0].tolist()))
    truth_ids = sorted(set(truth_labels[truth_labels >= 0].tolist()))

    # Per-truth-cluster properties
    truth_energy = {}
    truth_nhits = {}
    for tid in truth_ids:
        tmask = truth_labels == tid
        truth_energy[tid] = float(energies[tmask].sum())
        truth_nhits[tid] = int(tmask.sum())

    if not pred_ids or not truth_ids:
        truth_detail = [
            {"truth_id": tid, "energy": truth_energy[tid],
             "n_hits": truth_nhits[tid], "matched": False}
            for tid in truth_ids
        ]
        return {
            "n_pred": len(pred_ids), "n_truth": len(truth_ids),
            "n_matched_pred": 0, "n_matched_truth": 0,
            "purities": [], "completenesses": [],
            "n_split": 0, "n_merged": 0,
        }, truth_detail

    # Build energy overlap
    overlap = defaultdict(lambda: defaultdict(float))
    pred_energy = defaultdict(float)

    for i in range(len(energies)):
        e = energies[i]
        p = pred_labels[i]
        t = truth_labels[i]
        if p >= 0:
            pred_energy[p] += e
        if p >= 0 and t >= 0:
            overlap[p][t] += e

    # Greedy match: for each pred, find best truth
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

    # Splits
    truth_to_pred = defaultdict(list)
    for p in sorted(pred_ids):
        if p not in overlap:
            continue
        for t, e in overlap[p].items():
            if pred_energy[p] > 0 and e / pred_energy[p] > 0.5:
                truth_to_pred[t].append(p)
    n_split = sum(1 for ps in truth_to_pred.values() if len(ps) > 1)

    # Merges
    n_merged = 0
    for p in sorted(pred_ids):
        if p not in overlap:
            continue
        sig = [t for t, e in overlap[p].items()
               if pred_energy[p] > 0 and e / pred_energy[p] > 0.1]
        if len(sig) > 1:
            n_merged += 1

    # Per-truth detail
    truth_detail = [
        {"truth_id": tid, "energy": truth_energy[tid],
         "n_hits": truth_nhits[tid], "matched": tid in matched_truth}
        for tid in truth_ids
    ]

    return {
        "n_pred": len(pred_ids), "n_truth": len(truth_ids),
        "n_matched_pred": len(purities), "n_matched_truth": len(matched_truth),
        "purities": purities, "completenesses": completenesses,
        "n_split": n_split, "n_merged": n_merged,
    }, truth_detail


def aggregate_results(results_list):
    """Aggregate per-graph match results into summary stats."""
    all_pur = [p for r in results_list for p in r["purities"]]
    all_comp = [c for r in results_list for c in r["completenesses"]]
    n_pred = sum(r["n_pred"] for r in results_list)
    n_truth = sum(r["n_truth"] for r in results_list)
    n_matched_pred = sum(r["n_matched_pred"] for r in results_list)
    n_matched_truth = sum(r["n_matched_truth"] for r in results_list)
    n_split = sum(r["n_split"] for r in results_list)
    n_merged = sum(r["n_merged"] for r in results_list)

    return {
        "n_pred": n_pred, "n_truth": n_truth,
        "n_matched_pred": n_matched_pred, "n_matched_truth": n_matched_truth,
        "reco_match_rate": n_matched_pred / n_pred if n_pred > 0 else 0,
        "truth_match_rate": n_matched_truth / n_truth if n_truth > 0 else 0,
        "mean_purity": float(np.mean(all_pur)) if all_pur else 0,
        "median_purity": float(np.median(all_pur)) if all_pur else 0,
        "mean_completeness": float(np.mean(all_comp)) if all_comp else 0,
        "median_completeness": float(np.median(all_comp)) if all_comp else 0,
        "n_split": n_split, "n_merged": n_merged,
        "purities": all_pur, "completenesses": all_comp,
    }


def binned_match_rate(detail_records, key, bins, labels):
    """Compute truth match rate in bins of a given key.

    Returns list of dicts: [{label, n_total, n_matched, match_rate}, ...]
    """
    results = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        in_bin = [d for d in detail_records if lo <= d[key] < hi]
        n_total = len(in_bin)
        n_matched = sum(1 for d in in_bin if d["matched"])
        results.append({
            "label": labels[i],
            "n_total": n_total,
            "n_matched": n_matched,
            "match_rate": n_matched / n_total if n_total > 0 else 0,
        })
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Test set evaluation: GNN vs BFS (run once)")
    parser.add_argument("--root-dir", type=str,
                        default="/exp/mu2e/data/users/wzhou2/GNN/root_files")
    parser.add_argument("--checkpoint", type=str,
                        default="outputs/runs/simple_edge_net_v1/checkpoints/best_model.pt")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output-dir", type=str, default="outputs/test_eval")
    parser.add_argument("--n-events", type=int, default=None,
                        help="Max events per file (default: all). "
                             "500-1000 gives stable statistics.")
    parser.add_argument("--device", type=str, default=None,
                        help="Force device (cpu/cuda). Auto-detects if omitted.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Load model
    model_cfg = cfg["model"]
    model = SimpleEdgeNet(
        node_dim=6, edge_dim=8,
        hidden_dim=model_cfg.get("hidden_dim", 64),
        n_mp_layers=model_cfg.get("n_mp_layers", 3),
        dropout=model_cfg.get("dropout", 0.1),
    )
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"Loaded model from epoch {ckpt['epoch']} (val F1={ckpt['val_f1']:.4f})")

    # Frozen inference parameters
    inf_cfg = cfg["inference"]
    tau_edge = inf_cfg["tau_edge"]
    print(f"Frozen tau_edge = {tau_edge}")

    # Load normalization stats and crystal map
    stats = load_stats(cfg["data"]["normalization_stats"])
    crystal_map = load_crystal_map("data/crystal_geometry.csv")
    graph_cfg = cfg["graph"]

    # Load test file list
    with open(cfg["data"]["splits"]["test"]) as f:
        test_files = [l.strip() for l in f if l.strip()]
    print(f"Test files: {len(test_files)}")
    if args.n_events:
        print(f"Max events per file: {args.n_events}")

    branches = [
        "calohits.crystalId_", "calohits.eDep_", "calohits.time_",
        "calohits.clusterIdx_",
        "calohits.crystalPos_.fCoordinates.fX",
        "calohits.crystalPos_.fCoordinates.fY",
        "calohitsmc.simParticleIds", "calohitsmc.eDeps",
    ]

    bfs_results = []
    gnn_results = []
    # Per-truth-cluster detail for binned metrics
    bfs_truth_detail = []
    gnn_truth_detail = []
    n_disk_graphs = 0
    t0 = time.time()

    for fi, fpath in enumerate(test_files):
        fname = Path(fpath).name
        local_path = str(Path(args.root_dir) / fname)
        print(f"  [{fi+1}/{len(test_files)}] {fname}...", end=" ", flush=True)

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
            xs = np.array(arrays["calohits.crystalPos_.fCoordinates.fX"][ev],
                          dtype=np.float64)
            ys = np.array(arrays["calohits.crystalPos_.fCoordinates.fY"][ev],
                          dtype=np.float64)
            simids = arrays["calohitsmc.simParticleIds"][ev]
            edeps_mc = arrays["calohitsmc.eDeps"][ev]

            disks = np.array([crystal_map[int(c)][0] if int(c) in crystal_map
                              else -1 for c in cryids], dtype=np.int64)

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
                mc_truth = build_mc_truth_clusters(d_simids, d_edeps, d_disks,
                                                   n_disk)

                # ── BFS ──
                bfs_match, bfs_td = match_clusters_detail(d_cidx, mc_truth, d_e)
                bfs_results.append(bfs_match)
                bfs_truth_detail.extend(bfs_td)

                # ── GNN ──
                edge_index, _ = build_graph(
                    d_pos, d_t,
                    r_max=graph_cfg["r_max_mm"], dt_max=graph_cfg["dt_max_ns"],
                    k_min=graph_cfg["k_min"], k_max=graph_cfg["k_max"])

                if edge_index.shape[1] == 0:
                    # No edges — each node is its own cluster
                    gnn_labels = np.arange(n_disk)
                    gnn_match, gnn_td = match_clusters_detail(
                        gnn_labels, mc_truth, d_e)
                    gnn_results.append(gnn_match)
                    gnn_truth_detail.extend(gnn_td)
                    n_disk_graphs += 1
                    continue

                node_feat = compute_node_features(d_pos, d_t, d_e)
                edge_feat = compute_edge_features(d_pos, d_t, d_e, edge_index)

                data = Data(
                    x=torch.from_numpy(node_feat),
                    edge_index=torch.from_numpy(edge_index),
                    edge_attr=torch.from_numpy(edge_feat),
                )
                normalize_graph(data, stats)

                # Model inference
                with torch.no_grad():
                    logits = model(data.to(device))

                gnn_labels, _ = reconstruct_clusters(
                    edge_index=edge_index,
                    edge_logits=logits.cpu().numpy(),
                    n_nodes=n_disk,
                    energies=d_e,
                    tau_edge=tau_edge,
                    min_hits=1,
                    min_energy_mev=0.0,
                )

                gnn_match, gnn_td = match_clusters_detail(
                    gnn_labels, mc_truth, d_e)
                gnn_results.append(gnn_match)
                gnn_truth_detail.extend(gnn_td)
                n_disk_graphs += 1

        print(f"{n_events} events")

    elapsed = time.time() - t0
    print(f"\nProcessed {n_disk_graphs} disk-graphs from {len(test_files)} files "
          f"in {elapsed:.1f}s")

    # ── Aggregate overall metrics ──
    bfs_agg = aggregate_results(bfs_results)
    gnn_agg = aggregate_results(gnn_results)

    def print_summary(name, agg):
        print(f"\n{'='*60}")
        print(f"  {name} vs MC Truth")
        print(f"{'='*60}")
        print(f"  Reco clusters:        {agg['n_pred']:,}")
        print(f"  Truth clusters:       {agg['n_truth']:,}")
        print(f"  Reco match rate:      {agg['reco_match_rate']:.1%}")
        print(f"  Truth match rate:     {agg['truth_match_rate']:.1%}")
        print(f"  Mean purity:          {agg['mean_purity']:.4f}")
        print(f"  Median purity:        {agg['median_purity']:.4f}")
        print(f"  Mean completeness:    {agg['mean_completeness']:.4f}")
        print(f"  Median completeness:  {agg['median_completeness']:.4f}")
        print(f"  Splits:               {agg['n_split']:,}")
        print(f"  Merges:               {agg['n_merged']:,}")

    print_summary("BFS", bfs_agg)
    print_summary(f"GNN (tau_edge={tau_edge})", gnn_agg)

    # ── Binned metrics ──
    energy_bins = [0, 50, 200, float("inf")]
    energy_labels = ["<50 MeV", "50-200 MeV", ">200 MeV"]
    # Hit multiplicity bins (use n_hits directly)
    mult_bins = [1, 2, 4, float("inf")]
    mult_labels = ["1 hit", "2-3 hits", "4+ hits"]

    bfs_energy_bins = binned_match_rate(bfs_truth_detail, "energy",
                                        energy_bins, energy_labels)
    gnn_energy_bins = binned_match_rate(gnn_truth_detail, "energy",
                                        energy_bins, energy_labels)
    bfs_mult_bins = binned_match_rate(bfs_truth_detail, "n_hits",
                                      mult_bins, mult_labels)
    gnn_mult_bins = binned_match_rate(gnn_truth_detail, "n_hits",
                                      mult_bins, mult_labels)

    print(f"\n{'='*60}")
    print("  Energy-binned truth match rate")
    print(f"{'='*60}")
    print(f"  {'Bin':<15} {'BFS':>12} {'GNN':>12} {'N_truth':>10}")
    for b, g in zip(bfs_energy_bins, gnn_energy_bins):
        print(f"  {b['label']:<15} {b['match_rate']:>11.1%} "
              f"{g['match_rate']:>11.1%} {b['n_total']:>10,}")

    print(f"\n{'='*60}")
    print("  Multiplicity-binned truth match rate")
    print(f"{'='*60}")
    print(f"  {'Bin':<15} {'BFS':>12} {'GNN':>12} {'N_truth':>10}")
    for b, g in zip(bfs_mult_bins, gnn_mult_bins):
        print(f"  {b['label']:<15} {b['match_rate']:>11.1%} "
              f"{g['match_rate']:>11.1%} {b['n_total']:>10,}")

    # ── Save CSVs ──
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Overall results
    csv_path = out_dir / "test_results.csv"
    rows = []
    for name, agg in [("BFS", bfs_agg), ("GNN", gnn_agg)]:
        rows.append({
            "method": name,
            "tau_edge": "-" if name == "BFS" else f"{tau_edge}",
            "reco_clusters": agg["n_pred"],
            "truth_clusters": agg["n_truth"],
            "reco_match_rate": f"{agg['reco_match_rate']:.4f}",
            "truth_match_rate": f"{agg['truth_match_rate']:.4f}",
            "mean_purity": f"{agg['mean_purity']:.4f}",
            "median_purity": f"{agg['median_purity']:.4f}",
            "mean_completeness": f"{agg['mean_completeness']:.4f}",
            "median_completeness": f"{agg['median_completeness']:.4f}",
            "n_split": agg["n_split"],
            "n_merged": agg["n_merged"],
        })
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved overall results to {csv_path}")

    # Per-truth-cluster detail
    detail_csv = out_dir / "truth_cluster_detail.csv"
    with open(detail_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["energy", "n_hits", "bfs_matched", "gnn_matched"])
        writer.writeheader()
        for bd, gd in zip(bfs_truth_detail, gnn_truth_detail):
            writer.writerow({
                "energy": f"{bd['energy']:.4f}",
                "n_hits": bd["n_hits"],
                "bfs_matched": int(bd["matched"]),
                "gnn_matched": int(gd["matched"]),
            })
    print(f"Saved truth cluster detail to {detail_csv}")

    # ── Plot ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    fig.suptitle(f"Test Set Evaluation — GNN ($\\tau_{{edge}}$={tau_edge}) vs BFS\n"
                 f"{n_disk_graphs:,} disk-graphs from {len(test_files)} files",
                 fontsize=14, fontweight="bold")

    # ── Row 1: overall distributions ──

    # 1. Purity distribution
    ax = axes[0, 0]
    bins_hist = np.linspace(0.5, 1.0, 60)
    ax.hist(bfs_agg["purities"], bins=bins_hist, alpha=0.6, label="BFS",
            color="coral", edgecolor="white")
    ax.hist(gnn_agg["purities"], bins=bins_hist, alpha=0.6, label="GNN",
            color="steelblue", edgecolor="white")
    ax.axvline(bfs_agg["mean_purity"], color="red", linestyle="--", linewidth=2)
    ax.axvline(gnn_agg["mean_purity"], color="blue", linestyle="--", linewidth=2)
    ax.set_xlabel("Purity")
    ax.set_ylabel("Count")
    ax.set_title("Cluster Purity (vs MC truth)")
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. Completeness distribution
    ax = axes[0, 1]
    ax.hist(bfs_agg["completenesses"], bins=bins_hist, alpha=0.6, label="BFS",
            color="coral", edgecolor="white")
    ax.hist(gnn_agg["completenesses"], bins=bins_hist, alpha=0.6, label="GNN",
            color="steelblue", edgecolor="white")
    ax.axvline(bfs_agg["mean_completeness"], color="red", linestyle="--",
               linewidth=2)
    ax.axvline(gnn_agg["mean_completeness"], color="blue", linestyle="--",
               linewidth=2)
    ax.set_xlabel("Completeness")
    ax.set_ylabel("Count")
    ax.set_title("Cluster Completeness (vs MC truth)")
    ax.legend()
    ax.grid(alpha=0.3)

    # 3. Overall match rates + splits/merges
    ax = axes[0, 2]
    metrics = ["Reco\nmatch %", "Truth\nmatch %", "Splits", "Merges"]
    bfs_vals = [bfs_agg["reco_match_rate"] * 100,
                bfs_agg["truth_match_rate"] * 100,
                bfs_agg["n_split"], bfs_agg["n_merged"]]
    gnn_vals = [gnn_agg["reco_match_rate"] * 100,
                gnn_agg["truth_match_rate"] * 100,
                gnn_agg["n_split"], gnn_agg["n_merged"]]
    x = np.arange(len(metrics))
    w = 0.35
    bars_bfs = ax.bar(x - w / 2, bfs_vals, w, label="BFS", color="coral",
                      alpha=0.8)
    bars_gnn = ax.bar(x + w / 2, gnn_vals, w, label="GNN", color="steelblue",
                      alpha=0.8)
    for i, (b, g) in enumerate(zip(bfs_vals, gnn_vals)):
        fmt = f"{b:.1f}%" if i < 2 else f"{int(b):,}"
        ax.text(x[i] - w / 2, b + max(bfs_vals + gnn_vals) * 0.02, fmt,
                ha="center", fontsize=8)
        fmt = f"{g:.1f}%" if i < 2 else f"{int(g):,}"
        ax.text(x[i] + w / 2, g + max(bfs_vals + gnn_vals) * 0.02, fmt,
                ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_title("Match Rates & Error Modes")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    # ── Row 2: energy-binned metrics ──

    x_e = np.arange(len(energy_labels))

    # 4. Energy-binned truth match rate
    ax = axes[1, 0]
    bfs_mr = [b["match_rate"] * 100 for b in bfs_energy_bins]
    gnn_mr = [g["match_rate"] * 100 for g in gnn_energy_bins]
    ax.bar(x_e - w / 2, bfs_mr, w, label="BFS", color="coral", alpha=0.8)
    ax.bar(x_e + w / 2, gnn_mr, w, label="GNN", color="steelblue", alpha=0.8)
    for i, (b, g) in enumerate(zip(bfs_mr, gnn_mr)):
        ax.text(x_e[i] - w / 2, b + 1, f"{b:.1f}%", ha="center", fontsize=8)
        ax.text(x_e[i] + w / 2, g + 1, f"{g:.1f}%", ha="center", fontsize=8)
    ax.set_xticks(x_e)
    ax.set_xticklabels(energy_labels)
    ax.set_ylabel("Truth match rate (%)")
    ax.set_title("Truth Match Rate by Cluster Energy")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(0, 105)

    # 5. Energy-binned counts (stacked bar showing matched vs unmatched)
    ax = axes[1, 1]
    for method, detail, color, offset in [
            ("BFS", bfs_energy_bins, "coral", -w / 2),
            ("GNN", gnn_energy_bins, "steelblue", w / 2)]:
        matched = [b["n_matched"] for b in detail]
        unmatched = [b["n_total"] - b["n_matched"] for b in detail]
        ax.bar(x_e + offset, matched, w, label=f"{method} matched",
               color=color, alpha=0.8)
        ax.bar(x_e + offset, unmatched, w, bottom=matched,
               label=f"{method} unmatched", color=color, alpha=0.3,
               edgecolor=color, linewidth=0.5)
    ax.set_xticks(x_e)
    ax.set_xticklabels(energy_labels)
    ax.set_ylabel("Truth clusters")
    ax.set_title("Truth Cluster Counts by Energy")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3, axis="y")

    # 6. Energy bin truth cluster size distribution
    ax = axes[1, 2]
    for label, lo, hi, color in [
            ("<50 MeV", 0, 50, "#1b9e77"),
            ("50-200 MeV", 50, 200, "#d95f02"),
            (">200 MeV", 200, float("inf"), "#7570b3")]:
        nhits = [d["n_hits"] for d in gnn_truth_detail
                 if lo <= d["energy"] < hi]
        if nhits:
            ax.hist(nhits, bins=range(1, max(nhits) + 2), alpha=0.5,
                    label=f"{label} (n={len(nhits):,})", color=color,
                    edgecolor="white")
    ax.set_xlabel("Hits per truth cluster")
    ax.set_ylabel("Count")
    ax.set_title("Truth Cluster Size by Energy Bin")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0.5, 15.5)

    # ── Row 3: multiplicity-binned metrics ──

    x_m = np.arange(len(mult_labels))

    # 7. Multiplicity-binned truth match rate
    ax = axes[2, 0]
    bfs_mr_m = [b["match_rate"] * 100 for b in bfs_mult_bins]
    gnn_mr_m = [g["match_rate"] * 100 for g in gnn_mult_bins]
    ax.bar(x_m - w / 2, bfs_mr_m, w, label="BFS", color="coral", alpha=0.8)
    ax.bar(x_m + w / 2, gnn_mr_m, w, label="GNN", color="steelblue", alpha=0.8)
    for i, (b, g) in enumerate(zip(bfs_mr_m, gnn_mr_m)):
        ax.text(x_m[i] - w / 2, b + 1, f"{b:.1f}%", ha="center", fontsize=8)
        ax.text(x_m[i] + w / 2, g + 1, f"{g:.1f}%", ha="center", fontsize=8)
    ax.set_xticks(x_m)
    ax.set_xticklabels(mult_labels)
    ax.set_ylabel("Truth match rate (%)")
    ax.set_title("Truth Match Rate by Cluster Hit Count")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(0, 105)

    # 8. Per-graph purity scatter
    ax = axes[2, 1]
    bfs_pg = [np.mean(r["purities"]) if r["purities"] else 1.0
              for r in bfs_results]
    gnn_pg = [np.mean(r["purities"]) if r["purities"] else 1.0
              for r in gnn_results]
    ax.scatter(bfs_pg, gnn_pg, alpha=0.15, s=8, color="navy")
    ax.plot([0.5, 1], [0.5, 1], "r--", alpha=0.5)
    ax.set_xlabel("BFS purity")
    ax.set_ylabel("GNN purity")
    ax.set_title("Per-Graph Purity: GNN vs BFS")
    ax.set_xlim(0.5, 1.02)
    ax.set_ylim(0.5, 1.02)
    ax.grid(alpha=0.3)

    # 9. Summary table
    ax = axes[2, 2]
    ax.axis("off")
    table_data = [
        ["Reco clusters", f"{bfs_agg['n_pred']:,}", f"{gnn_agg['n_pred']:,}"],
        ["Truth clusters", f"{bfs_agg['n_truth']:,}", f"{gnn_agg['n_truth']:,}"],
        ["Reco match rate", f"{bfs_agg['reco_match_rate']:.1%}",
         f"{gnn_agg['reco_match_rate']:.1%}"],
        ["Truth match rate", f"{bfs_agg['truth_match_rate']:.1%}",
         f"{gnn_agg['truth_match_rate']:.1%}"],
        ["Mean purity", f"{bfs_agg['mean_purity']:.4f}",
         f"{gnn_agg['mean_purity']:.4f}"],
        ["Mean completeness", f"{bfs_agg['mean_completeness']:.4f}",
         f"{gnn_agg['mean_completeness']:.4f}"],
        ["Splits", f"{bfs_agg['n_split']:,}", f"{gnn_agg['n_split']:,}"],
        ["Merges", f"{bfs_agg['n_merged']:,}", f"{gnn_agg['n_merged']:,}"],
    ]
    table = ax.table(cellText=table_data, colLabels=["Metric", "BFS", "GNN"],
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)
    # Highlight winning cells
    for i, row in enumerate(table_data):
        try:
            bv = float(row[1].rstrip("%").replace(",", ""))
            gv = float(row[2].rstrip("%").replace(",", ""))
            metric = row[0].lower()
            if "split" in metric or "merge" in metric:
                winner = 1 if bv <= gv else 2
            else:
                winner = 1 if bv >= gv else 2
            table[i + 1, winner].set_facecolor("#d4edda")
        except ValueError:
            pass
    ax.set_title("Summary Comparison", pad=20)

    plt.tight_layout()
    plot_path = out_dir / "test_evaluation.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
