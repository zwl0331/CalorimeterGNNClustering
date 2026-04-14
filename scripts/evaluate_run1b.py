#!/usr/bin/env python3
"""
Run1B (no-field) evaluation: BFS + SimpleEdgeNet + CaloClusterNetV1 vs calo-entrant truth.

Evaluates models trained on MDC2025 (with-field) data on the Run1B (no-field)
dataset. Tests generalization to a different physics scenario where electrons
travel straight (no B-field curving).

Outputs:
  - outputs/run1b_eval/run1b_results.csv       (overall comparison)
  - outputs/run1b_eval/truth_cluster_detail.csv (per-truth-cluster for binning)
  - outputs/run1b_eval/run1b_evaluation.png     (comparison plots)

Usage:
    source setup_env.sh
    OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 python3 -u scripts/evaluate_run1b.py
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
from src.data.truth_labels_primary import build_calo_root_map
from src.geometry.crystal_geometry import load_crystal_map
from src.inference.cluster_reco import reconstruct_clusters
from src.models import build_model
from torch_geometric.data import Data


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


def match_clusters_detail(pred_labels, truth_labels, energies):
    """Energy-weighted greedy matching with per-truth-cluster detail."""
    pred_ids = sorted(set(pred_labels[pred_labels >= 0].tolist()))
    truth_ids = sorted(set(truth_labels[truth_labels >= 0].tolist()))

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

    truth_to_pred = defaultdict(list)
    for p in sorted(pred_ids):
        if p not in overlap:
            continue
        for t, e in overlap[p].items():
            if pred_energy[p] > 0 and e / pred_energy[p] > 0.5:
                truth_to_pred[t].append(p)
    n_split = sum(1 for ps in truth_to_pred.values() if len(ps) > 1)

    n_merged = 0
    for p in sorted(pred_ids):
        if p not in overlap:
            continue
        sig = [t for t, e in overlap[p].items()
               if pred_energy[p] > 0 and e / pred_energy[p] > 0.1]
        if len(sig) > 1:
            n_merged += 1

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
        "mean_completeness": float(np.mean(all_comp)) if all_comp else 0,
        "n_split": n_split, "n_merged": n_merged,
        "purities": all_pur, "completenesses": all_comp,
    }


def binned_match_rate(detail_records, key, bins, labels):
    """Compute truth match rate in bins of a given key."""
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


def load_model(config_path, checkpoint_path, device):
    """Load model from config and checkpoint."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    model = build_model(cfg)
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    inf_cfg = cfg["inference"]
    tau_edge = inf_cfg["tau_edge"]
    model_name = cfg["model"].get("name", "SimpleEdgeNet")
    has_node_head = model_name == "CaloClusterNetV1"
    tau_node_cfg = inf_cfg.get("tau_node")
    lambda_node = cfg.get("train", {}).get("lambda_node", 0.0)
    tau_node = tau_node_cfg if (has_node_head and lambda_node > 0) else None

    return model, cfg, model_name, tau_edge, tau_node


def run_gnn_inference(model, data, device, edge_index, n_disk, energies,
                      tau_edge, tau_node):
    """Run GNN inference and reconstruct clusters."""
    with torch.no_grad():
        output = model(data.to(device))

    if isinstance(output, dict):
        logits_np = output["edge_logits"].cpu().numpy()
        nl = output.get("node_logits")
        node_logits_np = nl.cpu().numpy() if nl is not None else None
    else:
        logits_np = output.cpu().numpy()
        node_logits_np = None

    labels, _ = reconstruct_clusters(
        edge_index=edge_index,
        edge_logits=logits_np,
        n_nodes=n_disk,
        energies=energies,
        tau_edge=tau_edge,
        min_hits=1,
        min_energy_mev=0.0,
        node_logits=node_logits_np,
        tau_node=tau_node,
    )
    return labels


def main():
    parser = argparse.ArgumentParser(
        description="Run1B evaluation: BFS + both GNNs vs calo-entrant truth")
    parser.add_argument("--root-dir", type=str,
                        default="/exp/mu2e/data/users/wzhou2/GNN/root_files_run1b")
    parser.add_argument("--n-events", type=int, default=500,
                        help="Max events per file (default 500)")
    parser.add_argument("--n-files", type=int, default=None,
                        help="Max number of files (default: all)")
    parser.add_argument("--output-dir", type=str, default="outputs/run1b_eval")
    args = parser.parse_args()

    device = torch.device("cpu")
    print(f"Device: {device}")

    # Load both models
    models = {}
    for name, cfg_path, ckpt_path in [
        ("SimpleEdgeNet",
         "configs/default.yaml",
         "outputs/runs/simple_edge_net_v2/checkpoints/best_model.pt"),
        ("CaloClusterNetV1",
         "configs/calo_cluster_net_v1.yaml",
         "outputs/runs/calo_cluster_net_v1_v2_stage1/checkpoints/best_model.pt"),
    ]:
        model, cfg, mname, tau_edge, tau_node = load_model(
            cfg_path, ckpt_path, device)
        models[name] = {
            "model": model, "cfg": cfg, "tau_edge": tau_edge,
            "tau_node": tau_node,
        }
        print(f"  {name}: tau_edge={tau_edge}, tau_node={tau_node}")

    # Use graph config from first model (same for both)
    graph_cfg = models["SimpleEdgeNet"]["cfg"]["graph"]
    stats = load_stats(
        models["SimpleEdgeNet"]["cfg"]["data"]["normalization_stats"])
    crystal_map = load_crystal_map("data/crystal_geometry.csv")
    crystal_disk_map = {cid: disk for cid, (disk, _, _) in crystal_map.items()}

    # Get file list
    root_dir = Path(args.root_dir)
    files = sorted(root_dir.glob("*.root"))
    if args.n_files:
        files = files[:args.n_files]
    print(f"\nRun1B files: {len(files)}")
    print(f"Max events per file: {args.n_events}")

    branches = [
        "calohits.crystalId_", "calohits.eDep_", "calohits.time_",
        "calohits.clusterIdx_",
        "calohits.crystalPos_.fCoordinates.fX",
        "calohits.crystalPos_.fCoordinates.fY",
        "calohitsmc.simParticleIds", "calohitsmc.eDeps",
        "calomcsim.id", "calomcsim.ancestorSimIds",
    ]

    # Results per method
    all_results = {name: [] for name in ["BFS", "SimpleEdgeNet", "CaloClusterNetV1"]}
    all_detail = {name: [] for name in ["BFS", "SimpleEdgeNet", "CaloClusterNetV1"]}
    n_disk_graphs = 0
    n_events_total = 0
    t0 = time.time()

    for fi, fpath in enumerate(files):
        print(f"  [{fi+1}/{len(files)}] {fpath.name}...", end=" ", flush=True)

        tree = uproot.open(str(fpath) + ":EventNtuple/ntuple")
        arrays = tree.arrays(branches, entry_stop=args.n_events)
        n_events = len(arrays)
        n_events_total += n_events

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

            sim_ids_evt = arrays["calomcsim.id"][ev]
            ancestor_ids_evt = arrays["calomcsim.ancestorSimIds"][ev]
            calo_root_map = build_calo_root_map(
                sim_ids_evt, ancestor_ids_evt,
                simids, cryids, crystal_disk_map)

            disks = np.array([crystal_map[int(c)][0] if int(c) in crystal_map
                              else -1 for c in cryids], dtype=np.int64)

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

                mc_truth = build_mc_truth_clusters(d_simids, d_edeps, d_disks,
                                                   n_disk, calo_root_map)

                # BFS
                bfs_match, bfs_td = match_clusters_detail(d_cidx, mc_truth, d_e)
                all_results["BFS"].append(bfs_match)
                all_detail["BFS"].extend(bfs_td)

                # Build graph (shared by both GNNs)
                edge_index, _ = build_graph(
                    d_pos, d_t,
                    r_max=graph_cfg["r_max_mm"], dt_max=graph_cfg["dt_max_ns"],
                    k_min=graph_cfg["k_min"], k_max=graph_cfg["k_max"])

                if edge_index.shape[1] == 0:
                    gnn_labels = np.arange(n_disk)
                    for gnn_name in ["SimpleEdgeNet", "CaloClusterNetV1"]:
                        gnn_match, gnn_td = match_clusters_detail(
                            gnn_labels, mc_truth, d_e)
                        all_results[gnn_name].append(gnn_match)
                        all_detail[gnn_name].extend(gnn_td)
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

                # Run both GNNs
                for gnn_name in ["SimpleEdgeNet", "CaloClusterNetV1"]:
                    m = models[gnn_name]
                    gnn_labels = run_gnn_inference(
                        m["model"], data, device, edge_index, n_disk, d_e,
                        m["tau_edge"], m["tau_node"])
                    gnn_match, gnn_td = match_clusters_detail(
                        gnn_labels, mc_truth, d_e)
                    all_results[gnn_name].append(gnn_match)
                    all_detail[gnn_name].extend(gnn_td)

                n_disk_graphs += 1

        print(f"{n_events} events")

    elapsed = time.time() - t0
    print(f"\nProcessed {n_disk_graphs:,} disk-graphs from {n_events_total:,} "
          f"events ({len(files)} files) in {elapsed:.1f}s")

    # Aggregate
    agg = {name: aggregate_results(results) for name, results in all_results.items()}

    def print_summary(name, a):
        tau = ""
        if name in models:
            tau = f" (tau_edge={models[name]['tau_edge']})"
        print(f"\n{'='*60}")
        print(f"  {name}{tau} vs MC Truth (calo-entrant)")
        print(f"{'='*60}")
        print(f"  Reco clusters:        {a['n_pred']:,}")
        print(f"  Truth clusters:       {a['n_truth']:,}")
        print(f"  Reco match rate:      {a['reco_match_rate']:.1%}")
        print(f"  Truth match rate:     {a['truth_match_rate']:.1%}")
        print(f"  Mean purity:          {a['mean_purity']:.4f}")
        print(f"  Mean completeness:    {a['mean_completeness']:.4f}")
        print(f"  Splits:               {a['n_split']:,}")
        print(f"  Merges:               {a['n_merged']:,}")

    for name in ["BFS", "SimpleEdgeNet", "CaloClusterNetV1"]:
        print_summary(name, agg[name])

    # Binned metrics
    energy_bins = [0, 50, 200, float("inf")]
    energy_labels = ["<50 MeV", "50-200 MeV", ">200 MeV"]
    mult_bins = [1, 2, 4, float("inf")]
    mult_labels = ["1 hit", "2-3 hits", "4+ hits"]

    print(f"\n{'='*60}")
    print("  Energy-binned truth match rate")
    print(f"{'='*60}")
    print(f"  {'Bin':<15} {'BFS':>10} {'SEN':>10} {'CCNv1':>10} {'N_truth':>10}")
    for i in range(len(energy_labels)):
        vals = {}
        for name in ["BFS", "SimpleEdgeNet", "CaloClusterNetV1"]:
            bm = binned_match_rate(all_detail[name], "energy",
                                   energy_bins, energy_labels)
            vals[name] = bm[i]
        print(f"  {energy_labels[i]:<15} "
              f"{vals['BFS']['match_rate']:>9.1%} "
              f"{vals['SimpleEdgeNet']['match_rate']:>9.1%} "
              f"{vals['CaloClusterNetV1']['match_rate']:>9.1%} "
              f"{vals['BFS']['n_total']:>10,}")

    print(f"\n{'='*60}")
    print("  Multiplicity-binned truth match rate")
    print(f"{'='*60}")
    print(f"  {'Bin':<15} {'BFS':>10} {'SEN':>10} {'CCNv1':>10} {'N_truth':>10}")
    for i in range(len(mult_labels)):
        vals = {}
        for name in ["BFS", "SimpleEdgeNet", "CaloClusterNetV1"]:
            bm = binned_match_rate(all_detail[name], "n_hits",
                                   mult_bins, mult_labels)
            vals[name] = bm[i]
        print(f"  {mult_labels[i]:<15} "
              f"{vals['BFS']['match_rate']:>9.1%} "
              f"{vals['SimpleEdgeNet']['match_rate']:>9.1%} "
              f"{vals['CaloClusterNetV1']['match_rate']:>9.1%} "
              f"{vals['BFS']['n_total']:>10,}")

    # Save CSVs
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "run1b_results.csv"
    rows = []
    for name in ["BFS", "SimpleEdgeNet", "CaloClusterNetV1"]:
        a = agg[name]
        tau = "-"
        if name in models:
            tau = str(models[name]["tau_edge"])
        rows.append({
            "method": name, "tau_edge": tau,
            "n_events": n_events_total, "n_disk_graphs": n_disk_graphs,
            "reco_clusters": a["n_pred"], "truth_clusters": a["n_truth"],
            "reco_match_rate": f"{a['reco_match_rate']:.4f}",
            "truth_match_rate": f"{a['truth_match_rate']:.4f}",
            "mean_purity": f"{a['mean_purity']:.4f}",
            "mean_completeness": f"{a['mean_completeness']:.4f}",
            "n_split": a["n_split"], "n_merged": a["n_merged"],
        })
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved results to {csv_path}")

    # Per-truth-cluster detail
    detail_csv = out_dir / "truth_cluster_detail.csv"
    with open(detail_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["energy", "n_hits",
                           "bfs_matched", "sen_matched", "ccnv1_matched"])
        writer.writeheader()
        for bd, sd, cd in zip(all_detail["BFS"], all_detail["SimpleEdgeNet"],
                              all_detail["CaloClusterNetV1"]):
            writer.writerow({
                "energy": f"{bd['energy']:.4f}",
                "n_hits": bd["n_hits"],
                "bfs_matched": int(bd["matched"]),
                "sen_matched": int(sd["matched"]),
                "ccnv1_matched": int(cd["matched"]),
            })
    print(f"Saved truth cluster detail to {detail_csv}")

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    method_colors = {"BFS": "coral", "SimpleEdgeNet": "steelblue",
                     "CaloClusterNetV1": "seagreen"}
    method_short = {"BFS": "BFS", "SimpleEdgeNet": "SEN", "CaloClusterNetV1": "CCNv1"}

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(
        f"Run1B (No Field) Evaluation -- {n_disk_graphs:,} disk-graphs, "
        f"{n_events_total:,} events\n"
        f"Models trained on MDC2025 (with field), evaluated on Run1B (no field)",
        fontsize=13, fontweight="bold")

    # 1. Overall match rates
    ax = axes[0, 0]
    metrics_names = ["Reco MR", "Truth MR"]
    x = np.arange(len(metrics_names))
    w = 0.25
    for i, name in enumerate(["BFS", "SimpleEdgeNet", "CaloClusterNetV1"]):
        a = agg[name]
        vals = [a["reco_match_rate"] * 100, a["truth_match_rate"] * 100]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=method_short[name],
                      color=method_colors[name], alpha=0.8)
        for j, v in enumerate(vals):
            ax.text(x[j] + (i - 1) * w, v + 0.5, f"{v:.1f}%",
                    ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.set_ylabel("%")
    ax.set_title("Match Rates")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(0, 105)

    # 2. Splits and merges
    ax = axes[0, 1]
    metrics_names = ["Splits", "Merges"]
    x = np.arange(len(metrics_names))
    for i, name in enumerate(["BFS", "SimpleEdgeNet", "CaloClusterNetV1"]):
        a = agg[name]
        vals = [a["n_split"], a["n_merged"]]
        ax.bar(x + (i - 1) * w, vals, w, label=method_short[name],
               color=method_colors[name], alpha=0.8)
        for j, v in enumerate(vals):
            ax.text(x[j] + (i - 1) * w, v + max(vals) * 0.02, f"{v:,}",
                    ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.set_title("Splits & Merges")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    # 3. Purity distribution
    ax = axes[0, 2]
    bins_hist = np.linspace(0.5, 1.0, 60)
    for name in ["BFS", "SimpleEdgeNet", "CaloClusterNetV1"]:
        ax.hist(agg[name]["purities"], bins=bins_hist, alpha=0.4,
                label=f"{method_short[name]} ({agg[name]['mean_purity']:.4f})",
                color=method_colors[name], edgecolor="white")
    ax.set_xlabel("Purity")
    ax.set_ylabel("Count")
    ax.set_title("Cluster Purity Distribution")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # 4. Energy-binned truth match rate
    ax = axes[1, 0]
    x_e = np.arange(len(energy_labels))
    for i, name in enumerate(["BFS", "SimpleEdgeNet", "CaloClusterNetV1"]):
        bm = binned_match_rate(all_detail[name], "energy",
                               energy_bins, energy_labels)
        mr = [b["match_rate"] * 100 for b in bm]
        ax.bar(x_e + (i - 1) * w, mr, w, label=method_short[name],
               color=method_colors[name], alpha=0.8)
        for j, v in enumerate(mr):
            ax.text(x_e[j] + (i - 1) * w, v + 1, f"{v:.1f}%",
                    ha="center", fontsize=7)
    ax.set_xticks(x_e)
    ax.set_xticklabels(energy_labels)
    ax.set_ylabel("Truth match rate (%)")
    ax.set_title("Truth Match Rate by Energy")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(0, 105)

    # 5. Multiplicity-binned truth match rate
    ax = axes[1, 1]
    x_m = np.arange(len(mult_labels))
    for i, name in enumerate(["BFS", "SimpleEdgeNet", "CaloClusterNetV1"]):
        bm = binned_match_rate(all_detail[name], "n_hits",
                               mult_bins, mult_labels)
        mr = [b["match_rate"] * 100 for b in bm]
        ax.bar(x_m + (i - 1) * w, mr, w, label=method_short[name],
               color=method_colors[name], alpha=0.8)
        for j, v in enumerate(mr):
            ax.text(x_m[j] + (i - 1) * w, v + 1, f"{v:.1f}%",
                    ha="center", fontsize=7)
    ax.set_xticks(x_m)
    ax.set_xticklabels(mult_labels)
    ax.set_ylabel("Truth match rate (%)")
    ax.set_title("Truth Match Rate by Hit Count")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(0, 105)

    # 6. Summary table
    ax = axes[1, 2]
    ax.axis("off")
    table_data = []
    for metric, key, fmt in [
        ("Reco match rate", "reco_match_rate", "{:.1%}"),
        ("Truth match rate", "truth_match_rate", "{:.1%}"),
        ("Mean purity", "mean_purity", "{:.4f}"),
        ("Mean completeness", "mean_completeness", "{:.4f}"),
        ("Splits", "n_split", "{:,}"),
        ("Merges", "n_merged", "{:,}"),
    ]:
        row = [metric]
        for name in ["BFS", "SimpleEdgeNet", "CaloClusterNetV1"]:
            row.append(fmt.format(agg[name][key]))
        table_data.append(row)

    table = ax.table(
        cellText=table_data,
        colLabels=["Metric", "BFS", "SEN (0.26)", "CCNv1 (0.20)"],
        cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax.set_title("Run1B Summary", pad=20)

    plt.tight_layout()
    plot_path = out_dir / "run1b_evaluation.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
