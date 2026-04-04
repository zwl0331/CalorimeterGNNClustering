#!/usr/bin/env python3
"""Task 11d: Re-evaluate existing models against calo-entrant truth.

Reads v2 ROOT files (with calomcsim.ancestorSimIds), builds graphs
on the fly, runs GNN inference, and evaluates against BOTH old
(SimParticle) and new (calo-entrant) truth definitions.

This answers the key question: what fraction of the old "merge errors"
disappear when truth is redefined at the primary-shower level?

Usage:
    source setup_env.sh
    OMP_NUM_THREADS=4 python3 scripts/evaluate_new_truth.py
    OMP_NUM_THREADS=4 python3 scripts/evaluate_new_truth.py --split val --max-events 500
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
from src.data.truth_labels import assign_mc_truth
from src.data.truth_labels_primary import assign_mc_truth_primary, build_calo_root_map
from src.geometry.crystal_geometry import load_crystal_map
from src.inference.cluster_reco import reconstruct_clusters
from src.models import build_model
from torch_geometric.data import Data


def build_mc_truth_clusters_old(simids, edeps, disks, nhits, purity_thresh=0.7):
    """Build old SimParticle-level truth clusters."""
    truth_labels = np.full(nhits, -1, dtype=np.int64)
    cluster_map = {}
    next_label = 0
    for i in range(nhits):
        sids = list(simids[i])
        deps = list(edeps[i])
        if len(sids) == 0 or sum(deps) <= 0:
            continue
        deps_arr = np.array(deps)
        best = int(np.argmax(deps_arr))
        purity = deps_arr[best] / deps_arr.sum()
        if purity < purity_thresh:
            continue
        key = (int(disks[i]), int(sids[best]))
        if key not in cluster_map:
            cluster_map[key] = next_label
            next_label += 1
        truth_labels[i] = cluster_map[key]
    return truth_labels


def build_mc_truth_clusters_new(simids, edeps, disks, nhits,
                                 calo_root_map, purity_thresh=0.7):
    """Build new calo-entrant truth clusters."""
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
        # Group by calo-root
        root_edep = {}
        for pid, dep in zip(pids, deps):
            root = calo_root_map.get((int(pid), disk), int(pid))
            root_edep[root] = root_edep.get(root, 0.0) + float(dep)
        best_root = max(root_edep, key=root_edep.get)
        purity = root_edep[best_root] / total_e
        if purity < purity_thresh:
            continue
        key = (disk, best_root)
        if key not in cluster_map:
            cluster_map[key] = next_label
            next_label += 1
        truth_labels[i] = cluster_map[key]
    return truth_labels


def match_clusters(pred_labels, truth_labels, energies):
    """Energy-weighted greedy matching. Returns aggregate dict."""
    pred_ids = sorted(set(pred_labels[pred_labels >= 0].tolist()))
    truth_ids = sorted(set(truth_labels[truth_labels >= 0].tolist()))

    truth_energy = {}
    truth_nhits = {}
    for tid in truth_ids:
        tmask = truth_labels == tid
        truth_energy[tid] = float(energies[tmask].sum())
        truth_nhits[tid] = int(tmask.sum())

    if not pred_ids or not truth_ids:
        return {
            "n_pred": len(pred_ids), "n_truth": len(truth_ids),
            "n_matched_pred": 0, "n_matched_truth": 0,
            "purities": [], "completenesses": [],
            "n_split": 0, "n_merged": 0,
        }

    overlap = defaultdict(lambda: defaultdict(float))
    pred_energy = defaultdict(float)
    for i in range(len(energies)):
        e = energies[i]
        p, t = pred_labels[i], truth_labels[i]
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

    return {
        "n_pred": len(pred_ids), "n_truth": len(truth_ids),
        "n_matched_pred": len(purities), "n_matched_truth": len(matched_truth),
        "purities": purities, "completenesses": completenesses,
        "n_split": n_split, "n_merged": n_merged,
    }


def aggregate_results(results_list):
    """Aggregate per-graph match results."""
    all_pur = [p for r in results_list for p in r["purities"]]
    all_comp = [c for r in results_list for c in r["completenesses"]]
    n_pred = sum(r["n_pred"] for r in results_list)
    n_truth = sum(r["n_truth"] for r in results_list)
    n_mp = sum(r["n_matched_pred"] for r in results_list)
    n_mt = sum(r["n_matched_truth"] for r in results_list)
    n_split = sum(r["n_split"] for r in results_list)
    n_merged = sum(r["n_merged"] for r in results_list)
    return {
        "n_pred": n_pred, "n_truth": n_truth,
        "n_matched_pred": n_mp, "n_matched_truth": n_mt,
        "reco_match_rate": n_mp / n_pred if n_pred > 0 else 0,
        "truth_match_rate": n_mt / n_truth if n_truth > 0 else 0,
        "mean_purity": float(np.mean(all_pur)) if all_pur else 0,
        "mean_completeness": float(np.mean(all_comp)) if all_comp else 0,
        "n_split": n_split, "n_merged": n_merged,
    }


def v1_to_v2_path(v1_name, v2_dir):
    """Map a v1 ROOT file name to its v2 equivalent.

    v1: nts.mu2e.FlateMinusMix1BBTriggered.MDC2025-002.001430_XXXXXXXX.root
    v2: mcs.mu2e.FlateMinusMix1BBTriggered.MDC2025af_best_v1_1.001430_XXXXXXXX.root
    """
    # Extract the sequence suffix (last part: 001430_XXXXXXXX)
    stem = Path(v1_name).stem
    parts = stem.split(".")
    seq = parts[-1]  # e.g. "001430_00000044"
    # Find matching v2 file
    v2_dir = Path(v2_dir)
    matches = list(v2_dir.glob(f"mcs.*{seq}.root"))
    if matches:
        return matches[0]
    return None


def load_model_and_config(config_path, checkpoint_path, device):
    """Load a model + config. Returns (model, cfg, tau_edge, tau_node)."""
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
    lambda_node = cfg.get("train", {}).get("lambda_node", 0.0)
    tau_node_cfg = inf_cfg.get("tau_node")
    tau_node = tau_node_cfg if (has_node_head and lambda_node > 0) else None

    return model, cfg, tau_edge, tau_node


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate models against old and new truth definitions")
    parser.add_argument("--v2-dir", default="/exp/mu2e/data/users/wzhou2/GNN/root_files_v2")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--max-events", type=int, default=500)
    parser.add_argument("--output-dir", default="outputs/new_truth_eval")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    crystal_map = load_crystal_map("data/crystal_geometry.csv")
    crystal_disk_map = {cid: info[0] for cid, info in crystal_map.items()}

    # Load models
    models = {}
    se_ckpt = "outputs/runs/simple_edge_net_v1/checkpoints/best_model.pt"
    ccn_ckpt = "outputs/runs/calo_cluster_net_v1_stage1/checkpoints/best_model.pt"
    if Path(se_ckpt).exists():
        m, c, te, tn = load_model_and_config("configs/default.yaml", se_ckpt, device)
        models["SimpleEdgeNet"] = (m, c, te, tn)
        print(f"SimpleEdgeNet: tau_edge={te}")
    if Path(ccn_ckpt).exists():
        m, c, te, tn = load_model_and_config(
            "configs/calo_cluster_net_v1.yaml", ccn_ckpt, device)
        models["CaloClusterNetV1"] = (m, c, te, tn)
        print(f"CaloClusterNetV1: tau_edge={te}")

    # Load normalization stats (use SimpleEdgeNet config as reference)
    ref_cfg = list(models.values())[0][1] if models else None
    if ref_cfg is None:
        print("ERROR: No model checkpoints found")
        return
    stats = load_stats(ref_cfg["data"]["normalization_stats"])
    graph_cfg = ref_cfg["graph"]

    # Find v2 files for the split
    split_file = f"splits/{args.split}_files.txt"
    with open(split_file) as f:
        v1_files = [l.strip() for l in f if l.strip()]

    v2_files = []
    for v1 in v1_files:
        v2 = v1_to_v2_path(v1, args.v2_dir)
        if v2 and v2.stat().st_size >= 1800 * 1024 * 1024:
            # Verify the file is readable
            try:
                uproot.open(f"{v2}:EventNtuple/ntuple")
                v2_files.append(v2)
            except Exception:
                print(f"  SKIP (corrupt): {v2.name}")
    print(f"Split '{args.split}': {len(v1_files)} files, "
          f"{len(v2_files)} with complete v2 ROOT files")

    if not v2_files:
        print("ERROR: No complete v2 files available for this split")
        return

    branches = [
        "calohits.crystalId_", "calohits.eDep_", "calohits.time_",
        "calohits.clusterIdx_",
        "calohits.crystalPos_.fCoordinates.fX",
        "calohits.crystalPos_.fCoordinates.fY",
        "calohitsmc.simParticleIds", "calohitsmc.eDeps",
        "calomcsim.id", "calomcsim.ancestorSimIds",
    ]

    # Results: {method: {truth_type: [per-graph results]}}
    # methods: "BFS", "SimpleEdgeNet", "CaloClusterNetV1"
    # truth types: "old", "new"
    results = defaultdict(lambda: defaultdict(list))
    n_graphs = 0
    n_events_total = 0
    t0 = time.time()

    for fi, v2_path in enumerate(v2_files):
        print(f"  [{fi+1}/{len(v2_files)}] {v2_path.name}...", end=" ", flush=True)

        tree = uproot.open(f"{v2_path}:EventNtuple/ntuple")
        arrays = tree.arrays(branches, entry_stop=args.max_events)
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
            xs = np.array(arrays["calohits.crystalPos_.fCoordinates.fX"][ev], dtype=np.float64)
            ys = np.array(arrays["calohits.crystalPos_.fCoordinates.fY"][ev], dtype=np.float64)
            simids = arrays["calohitsmc.simParticleIds"][ev]
            edeps_mc = arrays["calohitsmc.eDeps"][ev]

            disks = np.array([crystal_disk_map.get(int(c), -1)
                              for c in cryids], dtype=np.int64)

            if np.all(xs == 0) and np.all(ys == 0):
                for i, c in enumerate(cryids):
                    if int(c) in crystal_map:
                        _, xs[i], ys[i] = crystal_map[int(c)]

            # Build calo-root map for this event
            sim_ids_evt = arrays["calomcsim.id"][ev]
            anc_evt = arrays["calomcsim.ancestorSimIds"][ev]
            calo_root_map = build_calo_root_map(
                sim_ids_evt, anc_evt, simids, cryids, crystal_disk_map)

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

                # Build both truth cluster sets
                truth_old = build_mc_truth_clusters_old(
                    d_simids, d_edeps, d_disks, n_disk)
                truth_new = build_mc_truth_clusters_new(
                    d_simids, d_edeps, d_disks, n_disk, calo_root_map)

                # BFS vs both truths
                results["BFS"]["old"].append(match_clusters(d_cidx, truth_old, d_e))
                results["BFS"]["new"].append(match_clusters(d_cidx, truth_new, d_e))

                # GNN models
                edge_index, _ = build_graph(
                    d_pos, d_t,
                    r_max=graph_cfg["r_max_mm"], dt_max=graph_cfg["dt_max_ns"],
                    k_min=graph_cfg["k_min"], k_max=graph_cfg["k_max"])

                if edge_index.shape[1] == 0:
                    gnn_labels = np.arange(n_disk)
                    for mname in models:
                        results[mname]["old"].append(
                            match_clusters(gnn_labels, truth_old, d_e))
                        results[mname]["new"].append(
                            match_clusters(gnn_labels, truth_new, d_e))
                    n_graphs += 1
                    continue

                node_feat = compute_node_features(d_pos, d_t, d_e)
                edge_feat = compute_edge_features(d_pos, d_t, d_e, edge_index)

                data = Data(
                    x=torch.from_numpy(node_feat),
                    edge_index=torch.from_numpy(edge_index),
                    edge_attr=torch.from_numpy(edge_feat),
                )
                normalize_graph(data, stats)

                for mname, (model, cfg_m, tau_edge, tau_node) in models.items():
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
                        tau_edge=tau_edge,
                        min_hits=1,
                        min_energy_mev=0.0,
                        node_logits=node_logits_np,
                        tau_node=tau_node,
                    )

                    results[mname]["old"].append(
                        match_clusters(gnn_labels, truth_old, d_e))
                    results[mname]["new"].append(
                        match_clusters(gnn_labels, truth_new, d_e))

                n_graphs += 1

        print(f"{n_events} events")

    elapsed = time.time() - t0
    print(f"\nProcessed {n_graphs} disk-graphs from {n_events_total} events "
          f"({len(v2_files)} files) in {elapsed:.1f}s")

    # Print comparison table
    print(f"\n{'='*80}")
    print(f"OLD TRUTH (SimParticle) vs NEW TRUTH (calo-entrant)")
    print(f"{'='*80}")

    header = f"{'Method':<20} {'Truth':<6} {'RecoMR':>8} {'TruthMR':>8} " \
             f"{'Purity':>8} {'Compl':>8} {'Splits':>7} {'Merges':>7} {'N_truth':>8}"
    print(header)
    print("-" * len(header))

    out_rows = []
    for mname in ["BFS"] + list(models.keys()):
        for ttype in ["old", "new"]:
            if not results[mname][ttype]:
                continue
            agg = aggregate_results(results[mname][ttype])
            line = (f"{mname:<20} {ttype:<6} "
                    f"{agg['reco_match_rate']:>7.1%} "
                    f"{agg['truth_match_rate']:>7.1%} "
                    f"{agg['mean_purity']:>8.4f} "
                    f"{agg['mean_completeness']:>8.4f} "
                    f"{agg['n_split']:>7d} "
                    f"{agg['n_merged']:>7d} "
                    f"{agg['n_truth']:>8d}")
            print(line)
            out_rows.append({
                "method": mname, "truth": ttype,
                "reco_match_rate": f"{agg['reco_match_rate']:.4f}",
                "truth_match_rate": f"{agg['truth_match_rate']:.4f}",
                "mean_purity": f"{agg['mean_purity']:.4f}",
                "mean_completeness": f"{agg['mean_completeness']:.4f}",
                "n_split": agg["n_split"], "n_merged": agg["n_merged"],
                "n_truth": agg["n_truth"], "n_pred": agg["n_pred"],
            })
        print()

    # Delta table
    print(f"\n{'='*80}")
    print("DELTA (new - old truth): positive = improvement")
    print(f"{'='*80}")
    for mname in ["BFS"] + list(models.keys()):
        old_r = results[mname].get("old", [])
        new_r = results[mname].get("new", [])
        if not old_r or not new_r:
            continue
        old_a = aggregate_results(old_r)
        new_a = aggregate_results(new_r)
        d_tmr = new_a["truth_match_rate"] - old_a["truth_match_rate"]
        d_rmr = new_a["reco_match_rate"] - old_a["reco_match_rate"]
        d_pur = new_a["mean_purity"] - old_a["mean_purity"]
        d_comp = new_a["mean_completeness"] - old_a["mean_completeness"]
        d_split = new_a["n_split"] - old_a["n_split"]
        d_merge = new_a["n_merged"] - old_a["n_merged"]
        print(f"  {mname:<20} TMR: {d_tmr:>+.1%}  RMR: {d_rmr:>+.1%}  "
              f"Pur: {d_pur:>+.4f}  Comp: {d_comp:>+.4f}  "
              f"Splits: {d_split:>+d}  Merges: {d_merge:>+d}")

    # Save CSV
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "new_truth_comparison.csv"
    if out_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=out_rows[0].keys())
            writer.writeheader()
            writer.writerows(out_rows)
        print(f"\nSaved results to {csv_path}")


if __name__ == "__main__":
    main()
