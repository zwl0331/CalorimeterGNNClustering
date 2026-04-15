#!/usr/bin/env python3
"""
Cluster-level physics evaluation: energy, centroid, and time residuals.

For each matched reco<->truth cluster pair, computes:
  - Energy residual:  dE = E_reco - E_truth
  - Centroid displacement: dr = |centroid_reco - centroid_truth|  (x-y plane)
  - Time residual: dt = t_reco - t_truth  (seed hit = most energetic)

Evaluates BFS + both GNN models against calo-entrant MC truth.

Outputs:
  - outputs/cluster_physics_eval/cluster_residuals.csv   (per-cluster detail)
  - outputs/cluster_physics_eval/summary.txt             (aggregate statistics)
  - outputs/cluster_physics_eval/residual_plots.png      (comparison histograms)

Usage:
    source setup_env.sh
    OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 python3 -u scripts/evaluate_cluster_physics.py
    # Or with Run1B data:
    python3 scripts/evaluate_cluster_physics.py --root-dir /exp/mu2e/data/users/wzhou2/GNN/root_files_run1b --file-list splits/val_files.txt --n-events 500
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


# ── Cluster physics computation ─────────────────────────────────────────────

def compute_cluster_physics(labels, positions, energies, times):
    """Compute energy, centroid, and time for each cluster.

    Parameters
    ----------
    labels : ndarray (N,) — cluster IDs, -1 = unassigned
    positions : ndarray (N, 2) — (x, y) in mm
    energies : ndarray (N,) — MeV
    times : ndarray (N,) — ns

    Returns
    -------
    dict: cluster_id -> {energy, centroid_x, centroid_y, time, n_hits}
    """
    result = {}
    for cid in np.unique(labels):
        if cid < 0:
            continue
        mask = labels == cid
        e = energies[mask]
        total_e = e.sum()
        if total_e <= 0:
            continue
        w = e / total_e
        pos = positions[mask]
        t = times[mask]

        # Seed time = time of most energetic hit (Offline convention)
        seed_time = float(t[np.argmax(e)])

        result[int(cid)] = {
            "energy": float(total_e),
            "centroid_x": float(np.dot(w, pos[:, 0])),
            "centroid_y": float(np.dot(w, pos[:, 1])),
            "time": seed_time,
            "n_hits": int(mask.sum()),
        }
    return result


# ── Matching and residual computation ────────────────────────────────────────

def match_and_compute_residuals(pred_labels, truth_labels, positions,
                                energies, times, method_name):
    """Match reco to truth clusters and compute physics residuals.

    Uses greedy energy-weighted matching (purity > 0.5, completeness > 0.5).

    Returns list of dicts, one per matched pair.
    """
    pred_physics = compute_cluster_physics(pred_labels, positions, energies, times)
    truth_physics = compute_cluster_physics(truth_labels, positions, energies, times)

    pred_ids = sorted(pred_physics.keys())
    truth_ids = sorted(truth_physics.keys())

    if not pred_ids or not truth_ids:
        return []

    # Build energy overlap matrix
    overlap = defaultdict(lambda: defaultdict(float))
    pred_energy = defaultdict(float)
    truth_energy = {tid: truth_physics[tid]["energy"] for tid in truth_ids}

    for i in range(len(energies)):
        e = energies[i]
        p = pred_labels[i]
        t = truth_labels[i]
        if p >= 0:
            pred_energy[p] += e
        if p >= 0 and t >= 0:
            overlap[p][t] += e

    # Greedy matching
    records = []
    matched_truth = set()

    for pid in pred_ids:
        if pid not in overlap:
            continue
        best_tid = max(overlap[pid], key=lambda t: overlap[pid][t])
        if best_tid in matched_truth:
            continue
        shared = overlap[pid][best_tid]
        pur = shared / pred_energy[pid] if pred_energy[pid] > 0 else 0
        comp = shared / truth_energy[best_tid] if truth_energy[best_tid] > 0 else 0

        if pur > 0.5 and comp > 0.5:
            matched_truth.add(best_tid)
            rp = pred_physics[pid]
            tp = truth_physics[best_tid]

            dE = rp["energy"] - tp["energy"]
            E_ratio = rp["energy"] / tp["energy"] if tp["energy"] > 0 else 0
            dr = np.sqrt((rp["centroid_x"] - tp["centroid_x"])**2 +
                         (rp["centroid_y"] - tp["centroid_y"])**2)
            dt = rp["time"] - tp["time"]

            records.append({
                "method": method_name,
                "truth_energy": tp["energy"],
                "truth_nhits": tp["n_hits"],
                "reco_energy": rp["energy"],
                "reco_nhits": rp["n_hits"],
                "dE": dE,
                "E_ratio": E_ratio,
                "dr": dr,
                "dt": dt,
                "purity": pur,
                "completeness": comp,
            })

    return records


# ── MC truth cluster builder ────────────────────────────────────────────────

def build_mc_truth_clusters(simids, edeps, disks, nhits,
                            calo_root_map, purity_thresh=0.7):
    """Build MC truth cluster labels using calo-entrant truth."""
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


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Cluster-level physics evaluation: energy, centroid, time")
    parser.add_argument("--root-dir", type=str,
                        default="/exp/mu2e/data/users/wzhou2/GNN/root_files_v2")
    parser.add_argument("--file-list", type=str, default="splits/val_files.txt",
                        help="File listing ROOT files to process")
    parser.add_argument("--n-events", type=int, default=500,
                        help="Max events per file")
    parser.add_argument("--output-dir", type=str,
                        default="outputs/cluster_physics_eval")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load models ──
    configs = {
        "SimpleEdgeNet": ("configs/default.yaml",
                          "outputs/runs/simple_edge_net_v2/checkpoints/best_model.pt"),
        "CaloClusterNet": ("configs/calo_cluster_net.yaml",
                           "outputs/runs/calo_cluster_net_v2_stage1/checkpoints/best_model.pt"),
    }

    models = {}
    tau_edges = {}
    tau_nodes = {}
    for name, (cfg_path, ckpt_path) in configs.items():
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        model = build_model(cfg)
        ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device).eval()
        models[name] = model
        tau_edges[name] = cfg["inference"]["tau_edge"]
        # Only use tau_node if trained with lambda_node > 0
        has_node = name == "CaloClusterNet"
        lam_node = cfg.get("train", {}).get("lambda_node", 0.0)
        tau_nodes[name] = cfg["inference"].get("tau_node") if (has_node and lam_node > 0) else None
        print(f"Loaded {name}: epoch {ckpt['epoch']}, tau_edge={tau_edges[name]}")

    # ── Load shared resources ──
    with open("configs/default.yaml") as f:
        base_cfg = yaml.safe_load(f)
    stats = load_stats(base_cfg["data"]["normalization_stats"])
    crystal_map = load_crystal_map("data/crystal_geometry.csv")
    crystal_disk_map = {cid: disk for cid, (disk, _, _) in crystal_map.items()}
    graph_cfg = base_cfg["graph"]

    # ── Load file list ──
    with open(args.file_list) as f:
        file_list = [l.strip() for l in f if l.strip()]

    # Resolve paths: try root-dir + filename
    root_dir = Path(args.root_dir)
    root_files = []
    for fpath in file_list:
        # Try as-is first, then just the filename in root_dir
        p = Path(fpath)
        if p.exists():
            root_files.append(str(p))
        else:
            local = root_dir / p.name
            if local.exists():
                root_files.append(str(local))
            else:
                # Try matching by subrun ID
                import re
                m = re.search(r'001\d+_(\d+)', p.name)
                if m:
                    subrun = m.group(1)
                    matches = list(root_dir.glob(f"*{subrun}*.root"))
                    if matches:
                        root_files.append(str(matches[0]))
                        continue
                print(f"  WARNING: cannot find {p.name} in {root_dir}, skipping")

    if not root_files:
        # Fall back: use all ROOT files in root_dir
        root_files = sorted(str(p) for p in root_dir.glob("*.root"))
        print(f"Using all {len(root_files)} ROOT files from {root_dir}")
    else:
        print(f"Found {len(root_files)} / {len(file_list)} ROOT files")

    branches = [
        "calohits.crystalId_", "calohits.eDep_", "calohits.time_",
        "calohits.clusterIdx_",
        "calohits.crystalPos_.fCoordinates.fX",
        "calohits.crystalPos_.fCoordinates.fY",
        "calohitsmc.simParticleIds", "calohitsmc.eDeps",
        "calomcsim.id", "calomcsim.ancestorSimIds",
    ]

    # ── Process events ──
    all_records = []  # list of dicts
    n_disk_graphs = 0
    t0 = time.time()

    for fi, fpath in enumerate(root_files):
        fname = Path(fpath).name
        print(f"  [{fi+1}/{len(root_files)}] {fname}...", end=" ", flush=True)

        tree = uproot.open(fpath + ":EventNtuple/ntuple")
        arrays = tree.arrays(branches, entry_stop=args.n_events)
        n_events = len(arrays)

        for ev in range(n_events):
            nhits = len(arrays["calohits.crystalId_"][ev])
            if nhits == 0:
                continue

            cryids = np.array(arrays["calohits.crystalId_"][ev], dtype=np.int64)
            energies = np.array(arrays["calohits.eDep_"][ev], dtype=np.float64)
            times = np.array(arrays["calohits.time_"][ev], dtype=np.float64)
            bfs_idx = np.array(arrays["calohits.clusterIdx_"][ev], dtype=np.int64)
            xs = np.array(arrays["calohits.crystalPos_.fCoordinates.fX"][ev],
                          dtype=np.float64)
            ys = np.array(arrays["calohits.crystalPos_.fCoordinates.fY"][ev],
                          dtype=np.float64)
            simids = arrays["calohitsmc.simParticleIds"][ev]
            edeps_mc = arrays["calohitsmc.eDeps"][ev]

            # Build calo-entrant root map
            sim_ids_evt = arrays["calomcsim.id"][ev]
            ancestor_ids_evt = arrays["calomcsim.ancestorSimIds"][ev]
            calo_root_map = build_calo_root_map(
                sim_ids_evt, ancestor_ids_evt,
                simids, cryids, crystal_disk_map)

            disks = np.array([crystal_map[int(c)][0] if int(c) in crystal_map
                              else -1 for c in cryids], dtype=np.int64)

            # Fallback positions from crystal geometry
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
                d_bfs = bfs_idx[dm]
                d_disks = np.full(n_disk, disk_id, dtype=np.int64)

                disk_indices = np.where(dm)[0]
                d_simids = [list(simids[i]) for i in disk_indices]
                d_edeps = [list(edeps_mc[i]) for i in disk_indices]

                # MC truth clusters
                mc_truth = build_mc_truth_clusters(
                    d_simids, d_edeps, d_disks, n_disk, calo_root_map)

                # ── BFS residuals ──
                bfs_recs = match_and_compute_residuals(
                    d_bfs, mc_truth, d_pos, d_e, d_t, "BFS")
                all_records.extend(bfs_recs)

                # ── GNN residuals ──
                edge_index, _ = build_graph(
                    d_pos, d_t,
                    r_max=graph_cfg["r_max_mm"], dt_max=graph_cfg["dt_max_ns"],
                    k_min=graph_cfg["k_min"], k_max=graph_cfg["k_max"])

                if edge_index.shape[1] == 0:
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

                for model_name, model in models.items():
                    with torch.no_grad():
                        output = model(data.to(device))

                    if isinstance(output, dict):
                        logits_np = output["edge_logits"].cpu().numpy()
                        nl = output.get("node_logits")
                        node_logits_np = nl.cpu().numpy() if nl is not None else None
                    else:
                        logits_np = output.cpu().numpy()
                        node_logits_np = None

                    gnn_labels, _ = reconstruct_clusters(
                        edge_index=edge_index,
                        edge_logits=logits_np,
                        n_nodes=n_disk,
                        energies=d_e,
                        tau_edge=tau_edges[model_name],
                        min_hits=1, min_energy_mev=0.0,
                        node_logits=node_logits_np,
                        tau_node=tau_nodes[model_name],
                    )

                    gnn_recs = match_and_compute_residuals(
                        gnn_labels, mc_truth, d_pos, d_e, d_t, model_name)
                    all_records.extend(gnn_recs)

                n_disk_graphs += 1

        print(f"{n_events} events")

    elapsed = time.time() - t0
    print(f"\nProcessed {n_disk_graphs} disk-graphs from {len(root_files)} files "
          f"in {elapsed:.1f}s")

    # ── Save per-cluster CSV ──
    csv_path = out_dir / "cluster_residuals.csv"
    fieldnames = ["method", "truth_energy", "truth_nhits", "reco_energy",
                  "reco_nhits", "dE", "E_ratio", "dr", "dt",
                  "purity", "completeness"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_records:
            row = {k: (f"{v:.6f}" if isinstance(v, float) else v)
                   for k, v in r.items()}
            writer.writerow(row)
    print(f"Saved {len(all_records)} cluster residuals to {csv_path}")

    # ── Summary statistics ──
    methods = ["BFS", "SimpleEdgeNet", "CaloClusterNet"]
    summary_lines = []

    def add(line):
        summary_lines.append(line)
        print(line)

    add(f"\n{'='*70}")
    add(f"  Cluster-Level Physics Evaluation")
    add(f"  {n_disk_graphs} disk-graphs, {len(root_files)} files, "
        f"{args.n_events} events/file")
    add(f"{'='*70}")

    energy_bins = [(0, 50, "<50 MeV"), (50, 200, "50-200 MeV"),
                   (200, float("inf"), ">200 MeV")]
    mult_bins = [(1, 2, "1 hit"), (2, 4, "2-3 hits"), (4, float("inf"), "4+ hits")]

    for method in methods:
        recs = [r for r in all_records if r["method"] == method]
        if not recs:
            add(f"\n  {method}: no matched clusters")
            continue

        dE = np.array([r["dE"] for r in recs])
        dr = np.array([r["dr"] for r in recs])
        dt = np.array([r["dt"] for r in recs])
        E_ratio = np.array([r["E_ratio"] for r in recs])

        add(f"\n  {method} ({len(recs)} matched clusters)")
        add(f"  {'─'*60}")
        add(f"  {'Metric':<25} {'Mean':>10} {'Median':>10} {'Std':>10} {'90th%':>10}")
        add(f"  {'─'*60}")
        add(f"  {'dE (MeV)':<25} {dE.mean():>10.3f} {np.median(dE):>10.3f} "
            f"{dE.std():>10.3f} {np.percentile(np.abs(dE), 90):>10.3f}")
        add(f"  {'|dE| (MeV)':<25} {np.abs(dE).mean():>10.3f} "
            f"{np.median(np.abs(dE)):>10.3f} "
            f"{'':>10} {np.percentile(np.abs(dE), 90):>10.3f}")
        add(f"  {'E_reco/E_truth':<25} {E_ratio.mean():>10.4f} "
            f"{np.median(E_ratio):>10.4f} {E_ratio.std():>10.4f} {'':>10}")
        add(f"  {'dr (mm)':<25} {dr.mean():>10.3f} {np.median(dr):>10.3f} "
            f"{dr.std():>10.3f} {np.percentile(dr, 90):>10.3f}")
        add(f"  {'dt (ns)':<25} {dt.mean():>10.3f} {np.median(dt):>10.3f} "
            f"{dt.std():>10.3f} {np.percentile(np.abs(dt), 90):>10.3f}")

        # Quality cut fractions
        n = len(recs)
        add(f"\n  Quality cuts:")
        add(f"    |dE| > 10 MeV: {(np.abs(dE) > 10).sum():>6d} / {n} "
            f"({(np.abs(dE) > 10).mean():.1%})")
        add(f"    dr > 10 mm:    {(dr > 10).sum():>6d} / {n} "
            f"({(dr > 10).mean():.1%})")
        add(f"    |dt| > 1 ns:   {(np.abs(dt) > 1).sum():>6d} / {n} "
            f"({(np.abs(dt) > 1).mean():.1%})")

        # Energy-binned breakdown
        add(f"\n  Energy-binned |dE| and dr:")
        add(f"  {'Bin':<15} {'N':>6} {'mean|dE|':>10} {'meanDr':>10} {'mean|dt|':>10}")
        for lo, hi, label in energy_bins:
            sub = [r for r in recs if lo <= r["truth_energy"] < hi]
            if not sub:
                add(f"  {label:<15} {'0':>6}")
                continue
            s_dE = np.abs(np.array([r["dE"] for r in sub]))
            s_dr = np.array([r["dr"] for r in sub])
            s_dt = np.abs(np.array([r["dt"] for r in sub]))
            add(f"  {label:<15} {len(sub):>6} {s_dE.mean():>10.3f} "
                f"{s_dr.mean():>10.3f} {s_dt.mean():>10.3f}")

        # Multiplicity-binned breakdown
        add(f"\n  Multiplicity-binned |dE| and dr:")
        add(f"  {'Bin':<15} {'N':>6} {'mean|dE|':>10} {'meanDr':>10} {'mean|dt|':>10}")
        for lo, hi, label in mult_bins:
            sub = [r for r in recs if lo <= r["truth_nhits"] < hi]
            if not sub:
                add(f"  {label:<15} {'0':>6}")
                continue
            s_dE = np.abs(np.array([r["dE"] for r in sub]))
            s_dr = np.array([r["dr"] for r in sub])
            s_dt = np.abs(np.array([r["dt"] for r in sub]))
            add(f"  {label:<15} {len(sub):>6} {s_dE.mean():>10.3f} "
                f"{s_dr.mean():>10.3f} {s_dt.mean():>10.3f}")

    # Save summary
    summary_path = out_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"\nSaved summary to {summary_path}")

    # ── Plots ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"BFS": "coral", "SimpleEdgeNet": "steelblue",
              "CaloClusterNet": "seagreen"}

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(f"Cluster-Level Physics Residuals\n"
                 f"{n_disk_graphs} disk-graphs, {len(root_files)} files",
                 fontsize=14, fontweight="bold")

    # ── Row 1: residual distributions (non-zero only, log y-scale) ──

    # 1. dE histogram — all values, log scale, fine bins
    ax = axes[0, 0]
    bins_dE = np.linspace(-50, 50, 200)
    for method in methods:
        recs = [r for r in all_records if r["method"] == method]
        if recs:
            vals = np.array([r["dE"] for r in recs])
            ax.hist(vals, bins=bins_dE, alpha=0.5,
                    label=f"{method} (n={len(vals):,})",
                    color=colors[method], edgecolor="none")
    ax.set_yscale("log")
    ax.set_xlabel("dE = E_reco − E_truth (MeV)")
    ax.set_ylabel("Clusters (log scale)")
    ax.set_title("Energy Residual")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.axvline(0, color="black", linestyle="--", alpha=0.3)

    # 2. dr histogram — all values, log scale, fine bins
    ax = axes[0, 1]
    bins_dr = np.linspace(0, 80, 200)
    for method in methods:
        recs = [r for r in all_records if r["method"] == method]
        if recs:
            vals = np.array([r["dr"] for r in recs])
            ax.hist(vals, bins=bins_dr, alpha=0.5,
                    label=f"{method} (n={len(vals):,})",
                    color=colors[method], edgecolor="none")
    ax.set_yscale("log")
    ax.set_xlabel("Centroid displacement dr (mm)")
    ax.set_ylabel("Clusters (log scale)")
    ax.set_title("Centroid Displacement")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 3. dt histogram — all values, log scale, fine bins
    ax = axes[0, 2]
    bins_dt = np.linspace(-10, 10, 200)
    for method in methods:
        recs = [r for r in all_records if r["method"] == method]
        if recs:
            vals = np.array([r["dt"] for r in recs])
            ax.hist(vals, bins=bins_dt, alpha=0.5,
                    label=f"{method} (n={len(vals):,})",
                    color=colors[method], edgecolor="none")
    ax.set_yscale("log")
    ax.set_xlabel("dt = t_reco − t_truth (ns)")
    ax.set_ylabel("Clusters (log scale)")
    ax.set_title("Time Residual")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.axvline(0, color="black", linestyle="--", alpha=0.3)

    # ── Row 2: energy-dependent residuals + summary ──

    # 4. |dE| vs truth energy (binned means)
    ax = axes[1, 0]
    for method in methods:
        recs = [r for r in all_records if r["method"] == method]
        if not recs:
            continue
        te = np.array([r["truth_energy"] for r in recs])
        ade = np.abs(np.array([r["dE"] for r in recs]))
        bin_edges = [0, 20, 40, 60, 80, 100, 150, 200, 300]
        centers, means = [], []
        for i in range(len(bin_edges) - 1):
            mask = (te >= bin_edges[i]) & (te < bin_edges[i+1])
            if mask.sum() >= 5:
                centers.append((bin_edges[i] + bin_edges[i+1]) / 2)
                means.append(ade[mask].mean())
        ax.plot(centers, means, 'o-', color=colors[method], label=method,
                markersize=5, linewidth=1.5)
    ax.set_xlabel("Truth cluster energy (MeV)")
    ax.set_ylabel("Mean |dE| (MeV)")
    ax.set_title("|dE| vs Truth Energy")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # 5. dr vs truth energy (binned means)
    ax = axes[1, 1]
    for method in methods:
        recs = [r for r in all_records if r["method"] == method]
        if not recs:
            continue
        te = np.array([r["truth_energy"] for r in recs])
        dr_vals = np.array([r["dr"] for r in recs])
        bin_edges = [0, 20, 40, 60, 80, 100, 150, 200, 300]
        centers, means = [], []
        for i in range(len(bin_edges) - 1):
            mask = (te >= bin_edges[i]) & (te < bin_edges[i+1])
            if mask.sum() >= 5:
                centers.append((bin_edges[i] + bin_edges[i+1]) / 2)
                means.append(dr_vals[mask].mean())
        ax.plot(centers, means, 'o-', color=colors[method], label=method,
                markersize=5, linewidth=1.5)
    ax.set_xlabel("Truth cluster energy (MeV)")
    ax.set_ylabel("Mean centroid displacement (mm)")
    ax.set_title("Centroid Displacement vs Truth Energy")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # 6. Summary comparison bar chart
    ax = axes[1, 2]
    metrics_labels = ["Mean |dE|\n(MeV)", "Mean dr\n(mm)", "Mean |dt|\n(ns)",
                      "|dE|>10MeV\n(%)", "dr>10mm\n(%)"]
    x = np.arange(len(metrics_labels))
    w = 0.25
    for mi, method in enumerate(methods):
        recs = [r for r in all_records if r["method"] == method]
        if not recs:
            continue
        dE = np.abs(np.array([r["dE"] for r in recs]))
        dr_v = np.array([r["dr"] for r in recs])
        dt_v = np.abs(np.array([r["dt"] for r in recs]))
        n = len(recs)
        vals = [dE.mean(), dr_v.mean(), dt_v.mean(),
                (dE > 10).sum() / n * 100, (dr_v > 10).sum() / n * 100]
        offset = (mi - 1) * w
        bars = ax.bar(x + offset, vals, w, label=method,
                      color=colors[method], alpha=0.8)
        for i, v in enumerate(vals):
            fmt = f"{v:.2f}" if i < 3 else f"{v:.1f}%"
            ax.text(x[i] + offset, v + max(vals) * 0.02, fmt,
                    ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_labels, fontsize=9)
    ax.set_title("Summary Comparison")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plot_path = out_dir / "residual_plots.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Saved plots to {plot_path}")


if __name__ == "__main__":
    main()
