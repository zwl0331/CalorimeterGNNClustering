#!/usr/bin/env python3
"""
Failure audit for edge-classification GNN clustering.

Answers five questions:
  1. Are merges caused by a few bridge edges, or many?
  2. Where do bad bridge edges live (spatial, energy, time)?
  3. How does threshold choice affect the merge/split balance?
  4. Are failures concentrated in tiny truth objects?
  5. Is the truth definition creating artificial errors?

Usage:
    source setup_env.sh
    python3 scripts/failure_audit.py
    python3 scripts/failure_audit.py --config configs/calo_cluster_net.yaml \
        --checkpoint outputs/runs/calo_cluster_net_v2_stage1/checkpoints/best_model.pt
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import yaml

from src.data.dataset import CaloGraphDataset
from src.data.normalization import load_stats, normalize_graph
from src.inference.cluster_reco import symmetrize_edge_scores
from src.models import build_model


def load_model_and_data(args):
    """Load model, config, and val dataset."""
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device)

    # Load val data
    val_packed = Path(cfg["data"]["processed_dir"]) / "val.pt"
    val_files = [l.strip() for l in open(cfg["data"]["splits"]["val"]) if l.strip()]
    val_dataset = CaloGraphDataset(
        cfg["data"]["processed_dir"], file_list=val_files, preload=True,
        packed_path=val_packed if val_packed.exists() else None,
    )

    # Normalize
    stats = load_stats(cfg["data"]["normalization_stats"])
    for data in val_dataset._cache:
        normalize_graph(data, stats)

    # Load model
    model = build_model(cfg)
    ckpt_path = args.checkpoint
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    tau_edge = cfg["inference"]["tau_edge"]

    return model, val_dataset, cfg, device, tau_edge


def run_inference(model, data, device):
    """Run model forward pass, return edge probs and logits."""
    data_dev = data.clone().to(device)
    with torch.no_grad():
        output = model(data_dev)
    if isinstance(output, dict):
        logits = output["edge_logits"].cpu().numpy()
    else:
        logits = output.cpu().numpy()
    probs = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    return logits, probs


def analyze_graph(data, logits, probs, tau_edge):
    """Analyze a single graph for merge/split failures.

    Returns a dict with per-graph audit results, or None if graph is trivial.
    """
    edge_index = data.edge_index.numpy()
    truth = data.hit_truth_cluster.numpy()
    n_nodes = data.x.shape[0]

    # Raw node features (pre-normalization stored in data, but we have
    # normalized data — use what we can)
    # x features: [log_energy, time, x, y, radial_dist, relative_energy]
    # These are z-scored, but relative comparisons still work

    # Symmetrize edges
    ei_sym, ep_sym = symmetrize_edge_scores(edge_index, probs)
    src_s, dst_s = ei_sym[0], ei_sym[1]

    # Threshold → connected components
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    keep = ep_sym >= tau_edge
    if keep.sum() == 0:
        return None

    src_k, dst_k = src_s[keep], dst_s[keep]
    src_both = np.concatenate([src_k, dst_k])
    dst_both = np.concatenate([dst_k, src_k])
    adj = coo_matrix(
        (np.ones(len(src_both)), (src_both, dst_both)),
        shape=(n_nodes, n_nodes),
    )
    _, pred_labels = connected_components(adj, directed=False)

    # Identify truth clusters (exclude unassigned = -1)
    truth_ids = np.unique(truth[truth >= 0])
    pred_ids = np.unique(pred_labels)

    # === Find merges ===
    # A merge = one predicted cluster containing hits from 2+ truth clusters
    merges = []
    for pid in pred_ids:
        pmask = pred_labels == pid
        truth_in_pred = truth[pmask]
        # Exclude unassigned
        valid_truth = truth_in_pred[truth_in_pred >= 0]
        if len(valid_truth) == 0:
            continue
        unique_truth = np.unique(valid_truth)
        if len(unique_truth) <= 1:
            continue

        # This is a merge: pred cluster `pid` fuses truth clusters `unique_truth`
        # Find bridge edges: predicted-positive edges crossing truth boundaries
        bridge_edges = []
        for e_idx in range(len(src_s)):
            if ep_sym[e_idx] < tau_edge:
                continue
            i, j = src_s[e_idx], dst_s[e_idx]
            if pred_labels[i] != pid or pred_labels[j] != pid:
                continue
            ti, tj = truth[i], truth[j]
            if ti < 0 or tj < 0:
                continue
            if ti != tj:
                bridge_edges.append({
                    "src": int(i), "dst": int(j),
                    "score": float(ep_sym[e_idx]),
                    "truth_src": int(ti), "truth_dst": int(tj),
                })

        # Truth cluster properties
        fused_clusters = []
        for tid in unique_truth:
            tmask = truth == tid
            fused_clusters.append({
                "truth_id": int(tid),
                "n_hits": int(tmask.sum()),
            })

        merges.append({
            "pred_id": int(pid),
            "n_truth_fused": len(unique_truth),
            "n_bridge_edges": len(bridge_edges),
            "bridge_edges": bridge_edges,
            "bridge_scores": [be["score"] for be in bridge_edges],
            "fused_clusters": fused_clusters,
        })

    # === Find splits ===
    # A split = one truth cluster whose hits end up in 2+ predicted clusters
    splits = []
    for tid in truth_ids:
        tmask = truth == tid
        preds_for_truth = pred_labels[tmask]
        unique_preds = np.unique(preds_for_truth)
        if len(unique_preds) <= 1:
            continue
        splits.append({
            "truth_id": int(tid),
            "n_hits": int(tmask.sum()),
            "n_pred_fragments": len(unique_preds),
        })

    # === Edge-level stats for all symmetric edges ===
    # Classify each edge by truth
    n_tp = 0  # true positive: same truth, predicted positive
    n_fp = 0  # false positive: different truth, predicted positive
    n_fn = 0  # false negative: same truth, predicted negative
    n_tn = 0
    fp_scores = []
    fn_scores = []

    for e_idx in range(len(src_s)):
        i, j = src_s[e_idx], dst_s[e_idx]
        ti, tj = truth[i], truth[j]
        if ti < 0 or tj < 0:
            continue  # skip unassigned
        same = (ti == tj)
        positive = (ep_sym[e_idx] >= tau_edge)
        if same and positive:
            n_tp += 1
        elif not same and positive:
            n_fp += 1
            fp_scores.append(float(ep_sym[e_idx]))
        elif same and not positive:
            n_fn += 1
            fn_scores.append(float(ep_sym[e_idx]))
        else:
            n_tn += 1

    # === Truth cluster size distribution ===
    truth_sizes = {}
    for tid in truth_ids:
        truth_sizes[int(tid)] = int((truth == tid).sum())

    return {
        "n_nodes": n_nodes,
        "n_truth_clusters": len(truth_ids),
        "n_pred_clusters": len(pred_ids),
        "merges": merges,
        "splits": splits,
        "n_tp": n_tp, "n_fp": n_fp, "n_fn": n_fn, "n_tn": n_tn,
        "fp_scores": fp_scores,
        "fn_scores": fn_scores,
        "truth_sizes": truth_sizes,
    }


def threshold_sweep(model, dataset, device, thresholds):
    """Sweep thresholds and count merges/splits at each."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    results = []
    # Pre-compute all edge probs
    all_data = []
    for data in dataset:
        _, probs = run_inference(model, data, device)
        ei = data.edge_index.numpy()
        ei_sym, ep_sym = symmetrize_edge_scores(ei, probs)
        all_data.append((data, ei_sym, ep_sym))

    for tau in thresholds:
        total_merges = 0
        total_splits = 0
        total_fp = 0
        total_fn = 0
        total_tp = 0
        for data, ei_sym, ep_sym in all_data:
            truth = data.hit_truth_cluster.numpy()
            n_nodes = data.x.shape[0]
            keep = ep_sym >= tau
            if keep.sum() == 0:
                # All nodes isolated → count splits
                for tid in np.unique(truth[truth >= 0]):
                    if (truth == tid).sum() > 1:
                        total_splits += 1
                continue

            src_k, dst_k = ei_sym[0, keep], ei_sym[1, keep]
            sb = np.concatenate([src_k, dst_k])
            db = np.concatenate([dst_k, src_k])
            adj = coo_matrix((np.ones(len(sb)), (sb, db)), shape=(n_nodes, n_nodes))
            _, pred = connected_components(adj, directed=False)

            truth_ids = np.unique(truth[truth >= 0])
            # Merges
            for pid in np.unique(pred):
                pmask = pred == pid
                vt = truth[pmask]
                vt = vt[vt >= 0]
                if len(np.unique(vt)) > 1:
                    total_merges += 1
            # Splits
            for tid in truth_ids:
                tmask = truth == tid
                if len(np.unique(pred[tmask])) > 1:
                    total_splits += 1

            # FP/FN
            src_s, dst_s = ei_sym[0], ei_sym[1]
            for e_idx in range(len(src_s)):
                i, j = src_s[e_idx], dst_s[e_idx]
                ti, tj = truth[i], truth[j]
                if ti < 0 or tj < 0:
                    continue
                same = (ti == tj)
                pos = (ep_sym[e_idx] >= tau)
                if same and pos:
                    total_tp += 1
                elif not same and pos:
                    total_fp += 1
                elif same and not pos:
                    total_fn += 1

        prec = total_tp / max(total_tp + total_fp, 1)
        rec = total_tp / max(total_tp + total_fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-10)
        results.append({
            "tau": tau, "merges": total_merges, "splits": total_splits,
            "fp": total_fp, "fn": total_fn, "tp": total_tp,
            "precision": prec, "recall": rec, "f1": f1,
        })
        print(f"  τ={tau:.2f}: merges={total_merges}, splits={total_splits}, "
              f"P={prec:.3f} R={rec:.3f} F1={f1:.3f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="GNN failure audit")
    parser.add_argument("--config", default="configs/calo_cluster_net.yaml")
    parser.add_argument("--checkpoint",
                        default="outputs/runs/calo_cluster_net_v2_stage1/checkpoints/best_model.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-graphs", type=int, default=None,
                        help="Limit number of graphs to analyze (default: all)")
    args = parser.parse_args()

    model, val_dataset, cfg, device, tau_edge = load_model_and_data(args)
    print(f"Model: {cfg['model']['name']}, τ_edge={tau_edge}")
    print(f"Val graphs: {len(val_dataset)}")
    print(f"Device: {device}")

    out_dir = Path("outputs/failure_audit")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ====================================================================
    # Run per-graph analysis
    # ====================================================================
    print(f"\n{'='*70}")
    print("Running per-graph failure analysis...")
    print(f"{'='*70}")

    all_results = []
    n_graphs = args.max_graphs or len(val_dataset)
    for i in range(min(n_graphs, len(val_dataset))):
        data = val_dataset[i]
        logits, probs = run_inference(model, data, device)
        result = analyze_graph(data, logits, probs, tau_edge)
        if result is not None:
            all_results.append(result)
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{n_graphs} graphs processed...")

    print(f"  {len(all_results)} graphs analyzed (non-trivial)")

    # ====================================================================
    # Q1: Are merges caused by a few bridge edges?
    # ====================================================================
    print(f"\n{'='*70}")
    print("Q1: Bridge edge analysis for merges")
    print(f"{'='*70}")

    all_merges = []
    for r in all_results:
        all_merges.extend(r["merges"])

    n_total_merges = len(all_merges)
    if n_total_merges > 0:
        bridge_counts = [m["n_bridge_edges"] for m in all_merges]
        bridge_scores_flat = []
        for m in all_merges:
            bridge_scores_flat.extend(m["bridge_scores"])

        print(f"Total merges: {n_total_merges}")
        print(f"Bridge edges per merge:")
        print(f"  mean={np.mean(bridge_counts):.1f}, "
              f"median={np.median(bridge_counts):.0f}, "
              f"max={np.max(bridge_counts)}")
        for n in [1, 2, 3, 4, 5]:
            pct = 100 * np.mean(np.array(bridge_counts) == n)
            print(f"  exactly {n}: {pct:.1f}%")
        pct_le2 = 100 * np.mean(np.array(bridge_counts) <= 2)
        print(f"  <=2 bridge edges: {pct_le2:.1f}%")

        if bridge_scores_flat:
            scores = np.array(bridge_scores_flat)
            print(f"\nBridge edge scores:")
            print(f"  mean={scores.mean():.3f}, median={np.median(scores):.3f}")
            print(f"  min={scores.min():.3f}, max={scores.max():.3f}")
            near_thresh = np.mean((scores >= tau_edge) & (scores < tau_edge + 0.1))
            confident = np.mean(scores >= 0.8)
            print(f"  near threshold ({tau_edge:.2f}-{tau_edge+0.1:.2f}): "
                  f"{100*near_thresh:.1f}%")
            print(f"  confident (>=0.8): {100*confident:.1f}%")
    else:
        print("No merges found.")

    # ====================================================================
    # Q2: Where do bad bridge edges live?
    # ====================================================================
    print(f"\n{'='*70}")
    print("Q2: False positive edge properties")
    print(f"{'='*70}")

    all_fp_scores = []
    for r in all_results:
        all_fp_scores.extend(r["fp_scores"])

    total_fp = sum(r["n_fp"] for r in all_results)
    total_tp = sum(r["n_tp"] for r in all_results)
    total_fn = sum(r["n_fn"] for r in all_results)
    total_tn = sum(r["n_tn"] for r in all_results)

    print(f"Edge classification (all val, τ={tau_edge}):")
    print(f"  TP={total_tp}, FP={total_fp}, FN={total_fn}, TN={total_tn}")
    prec = total_tp / max(total_tp + total_fp, 1)
    rec = total_tp / max(total_tp + total_fn, 1)
    print(f"  Precision={prec:.4f}, Recall={rec:.4f}")

    if all_fp_scores:
        fps = np.array(all_fp_scores)
        print(f"\nFalse positive edge scores (N={len(fps)}):")
        print(f"  mean={fps.mean():.3f}, median={np.median(fps):.3f}")
        for lo, hi in [(0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7),
                       (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]:
            n = np.sum((fps >= lo) & (fps < hi))
            print(f"  [{lo:.1f}, {hi:.1f}): {n} ({100*n/len(fps):.1f}%)")

    # ====================================================================
    # Q3: Threshold sweep
    # ====================================================================
    print(f"\n{'='*70}")
    print("Q3: Threshold sweep (merge/split trade-off)")
    print(f"{'='*70}")

    thresholds = [0.20, 0.25, 0.30, 0.34, 0.40, 0.45, 0.50,
                  0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    sweep = threshold_sweep(model, val_dataset, device, thresholds)

    # ====================================================================
    # Q4: Are failures concentrated in tiny truth objects?
    # ====================================================================
    print(f"\n{'='*70}")
    print("Q4: Failure stratification by truth cluster size")
    print(f"{'='*70}")

    # Collect all truth clusters and whether they were part of a merge or split
    merge_truth_ids = set()
    for r in all_results:
        for m in r["merges"]:
            for fc in m["fused_clusters"]:
                merge_truth_ids.add((id(r), fc["truth_id"]))

    split_truth_ids = set()
    for r in all_results:
        for s in r["splits"]:
            split_truth_ids.add((id(r), s["truth_id"]))

    # Stratify by size
    size_stats = defaultdict(lambda: {"total": 0, "in_merge": 0, "in_split": 0})
    for r in all_results:
        for tid, sz in r["truth_sizes"].items():
            bucket = str(sz) if sz <= 5 else "6+"
            size_stats[bucket]["total"] += 1
            if (id(r), tid) in merge_truth_ids:
                size_stats[bucket]["in_merge"] += 1
            if (id(r), tid) in split_truth_ids:
                size_stats[bucket]["in_split"] += 1

    print(f"{'Size':>6} {'Total':>8} {'Merged':>8} {'%':>6} {'Split':>8} {'%':>6}")
    for sz in ["1", "2", "3", "4", "5", "6+"]:
        s = size_stats[sz]
        t = s["total"]
        m = s["in_merge"]
        sp = s["in_split"]
        mp = 100 * m / max(t, 1)
        spp = 100 * sp / max(t, 1)
        print(f"{sz:>6} {t:>8} {m:>8} {mp:>5.1f}% {sp:>8} {spp:>5.1f}%")

    # ====================================================================
    # Q5: Are "merge errors" actually ambiguous physics?
    # ====================================================================
    print(f"\n{'='*70}")
    print("Q5: Merge anatomy — are fused clusters physically close?")
    print(f"{'='*70}")

    # For merges, characterize the fused truth clusters
    fused_pair_sizes = []
    n_fused_2 = 0  # merges fusing exactly 2 truth clusters
    n_fused_3plus = 0
    n_both_small = 0  # both fused clusters have <=2 hits
    n_one_singleton = 0  # at least one is a single-hit cluster

    for m in all_merges:
        sizes = [fc["n_hits"] for fc in m["fused_clusters"]]
        fused_pair_sizes.append(sizes)
        if len(sizes) == 2:
            n_fused_2 += 1
            if max(sizes) <= 2:
                n_both_small += 1
        else:
            n_fused_3plus += 1
        if min(sizes) == 1:
            n_one_singleton += 1

    if n_total_merges > 0:
        print(f"Total merges: {n_total_merges}")
        print(f"  fusing exactly 2 truth clusters: {n_fused_2} "
              f"({100*n_fused_2/n_total_merges:.1f}%)")
        print(f"  fusing 3+ truth clusters: {n_fused_3plus} "
              f"({100*n_fused_3plus/n_total_merges:.1f}%)")
        print(f"  at least one singleton: {n_one_singleton} "
              f"({100*n_one_singleton/n_total_merges:.1f}%)")
        print(f"  both <=2 hits (pairwise): {n_both_small} "
              f"({100*n_both_small/n_total_merges:.1f}%)")

    # ====================================================================
    # Summary
    # ====================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    total_truth = sum(len(r["truth_sizes"]) for r in all_results)
    total_merges_clusters = sum(
        sum(len(m["fused_clusters"]) for m in r["merges"])
        for r in all_results
    )
    total_splits_count = sum(len(r["splits"]) for r in all_results)

    print(f"Graphs analyzed: {len(all_results)}")
    print(f"Total truth clusters: {total_truth}")
    print(f"Total merge events: {n_total_merges}")
    print(f"Total split events: {total_splits_count}")
    if n_total_merges > 0:
        median_bridges = int(np.median(bridge_counts))
        print(f"Median bridge edges per merge: {median_bridges}")
        le2 = np.mean(np.array(bridge_counts) <= 2)
        print(f"Merges caused by <=2 bridge edges: {100*le2:.1f}%")
        if bridge_scores_flat:
            print(f"Median bridge edge score: {np.median(bridge_scores_flat):.3f} "
                  f"(threshold: {tau_edge})")

    # Save raw results
    summary = {
        "model": cfg["model"]["name"],
        "tau_edge": tau_edge,
        "n_graphs": len(all_results),
        "total_truth_clusters": total_truth,
        "total_merges": n_total_merges,
        "total_splits": total_splits_count,
        "bridge_counts": bridge_counts if n_total_merges > 0 else [],
        "bridge_scores": bridge_scores_flat if n_total_merges > 0 else [],
        "fp_scores": all_fp_scores,
        "threshold_sweep": sweep,
        "size_stratification": dict(size_stats),
    }
    summary_path = out_dir / "audit_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nFull results saved to {summary_path}")


if __name__ == "__main__":
    main()
