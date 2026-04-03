#!/usr/bin/env python3
"""
Threshold tuning for GNN edge classification on the validation set.

Sweeps tau_edge to find the optimal edge probability threshold:
  1. Coarse grid: 0.1 to 0.9 in steps of 0.1
  2. Fine grid: +/-0.1 around coarse optimum in steps of 0.02

For each threshold, computes:
  - Edge-level pairwise F1 (precision, recall)
  - Cluster-level purity, completeness (energy-weighted matching)
  - Truth match rate, reco match rate
  - Number of merges and splits

Saves results to outputs/threshold_sweep/sweep_results.csv and generates plot.

Usage:
    source setup_env.sh
    python3 scripts/tune_threshold.py
    python3 scripts/tune_threshold.py --checkpoint path/to/best_model.pt
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

from src.data.normalization import load_stats, normalize_graph
from src.inference.cluster_reco import reconstruct_clusters
from src.models import build_model


def match_clusters_energy(pred_labels, truth_labels, energies):
    """Energy-weighted greedy cluster matching (same logic as head-to-head)."""
    pred_ids = set(pred_labels[pred_labels >= 0].tolist())
    truth_ids = set(truth_labels[truth_labels >= 0].tolist())

    if not pred_ids or not truth_ids:
        return {
            "n_pred": len(pred_ids), "n_truth": len(truth_ids),
            "n_matched_pred": 0, "n_matched_truth": 0,
            "purities": [], "completenesses": [],
            "n_split": 0, "n_merged": 0,
        }

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
    n_split = sum(1 for ps in truth_to_pred.values() if len(ps) > 1)

    # Merges: pred cluster overlapping >1 truth cluster significantly
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


def evaluate_threshold(graphs_info, threshold, min_hits=2, min_energy_mev=10.0,
                       tau_node=None):
    """Evaluate a single threshold across all precomputed graph data."""
    cluster_results = []
    # Accumulators for edge-level pairwise metrics
    total_tp, total_fp, total_fn = 0, 0, 0

    for g in graphs_info:
        # Reconstruct clusters at this threshold
        cluster_labels, _ = reconstruct_clusters(
            edge_index=g["edge_index"],
            edge_logits=g["logits"],
            n_nodes=g["n_nodes"],
            energies=g["energies"],
            tau_edge=threshold,
            min_hits=min_hits,
            min_energy_mev=min_energy_mev,
            node_logits=g.get("node_logits"),
            tau_node=tau_node,
        )
        cluster_results.append(
            match_clusters_energy(cluster_labels, g["truth_labels"], g["energies"])
        )

        # Edge-level pairwise metrics (on masked edges only)
        probs = g["edge_probs"]
        y = g["y_masked"]
        preds = (probs >= threshold).astype(np.int32)
        total_tp += ((preds == 1) & (y == 1)).sum()
        total_fp += ((preds == 1) & (y == 0)).sum()
        total_fn += ((preds == 0) & (y == 1)).sum()

    # Aggregate pairwise metrics
    pw_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    pw_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    pw_f1 = 2 * pw_prec * pw_rec / (pw_prec + pw_rec) if (pw_prec + pw_rec) > 0 else 0.0

    # Aggregate cluster metrics
    all_pur = [p for r in cluster_results for p in r["purities"]]
    all_comp = [c for r in cluster_results for c in r["completenesses"]]
    n_pred = sum(r["n_pred"] for r in cluster_results)
    n_truth = sum(r["n_truth"] for r in cluster_results)
    n_matched_pred = sum(r["n_matched_pred"] for r in cluster_results)
    n_matched_truth = sum(r["n_matched_truth"] for r in cluster_results)
    n_split = sum(r["n_split"] for r in cluster_results)
    n_merged = sum(r["n_merged"] for r in cluster_results)

    return {
        "tau_edge": threshold,
        "pairwise_precision": float(pw_prec),
        "pairwise_recall": float(pw_rec),
        "pairwise_f1": float(pw_f1),
        "mean_purity": float(np.mean(all_pur)) if all_pur else 0.0,
        "mean_completeness": float(np.mean(all_comp)) if all_comp else 0.0,
        "reco_match_rate": n_matched_pred / n_pred if n_pred > 0 else 0.0,
        "truth_match_rate": n_matched_truth / n_truth if n_truth > 0 else 0.0,
        "n_pred": n_pred,
        "n_truth": n_truth,
        "n_matched_pred": n_matched_pred,
        "n_matched_truth": n_matched_truth,
        "n_split": n_split,
        "n_merged": n_merged,
    }


def main():
    parser = argparse.ArgumentParser(description="Tune edge threshold on validation set")
    parser.add_argument("--checkpoint", type=str,
                        default="outputs/runs/simple_edge_net_v1/checkpoints/best_model.pt")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: outputs/threshold_sweep_<model>)")
    parser.add_argument("--tau-node", type=float, default=None,
                        help="Fixed node saliency threshold (default: from config if model "
                             "supports it)")
    parser.add_argument("--min-hits", type=int, default=1,
                        help="Min hits per cluster (default 1 for tuning; "
                             "production uses 2)")
    parser.add_argument("--min-energy", type=float, default=0.0,
                        help="Min energy per cluster in MeV (default 0 for tuning; "
                             "production uses 10)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = build_model(cfg)
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    model_name = cfg["model"].get("name", "SimpleEdgeNet")
    print(f"Model: {model_name}")
    print(f"Loaded from epoch {ckpt['epoch']} (val F1={ckpt['val_f1']:.4f})")

    # Node saliency threshold (only when explicitly requested via --tau-node,
    # since the node head may be untrained in early stages)
    tau_node = args.tau_node
    if tau_node is not None:
        print(f"tau_node = {tau_node}")
    else:
        print("tau_node = disabled (pass --tau-node to enable)")

    # Output directory (model-specific default)
    if args.output_dir is None:
        suffix = model_name.lower()
        args.output_dir = f"outputs/threshold_sweep_{suffix}"

    # Load val graphs (un-normalized packed file)
    val_packed = Path(cfg["data"]["processed_dir"]) / "val.pt"
    if not val_packed.exists():
        print(f"ERROR: {val_packed} not found. Run scripts/pack_graphs.py first.")
        sys.exit(1)
    print(f"Loading val graphs from {val_packed}...")
    val_graphs = torch.load(val_packed, weights_only=False)
    print(f"  {len(val_graphs)} graphs loaded")

    # Load normalization stats
    stats = load_stats(cfg["data"]["normalization_stats"])

    # Run inference on all val graphs once; precompute everything needed for sweeps
    print("Running model inference on all val graphs...")
    t0 = time.time()
    graphs_info = []

    for idx, data in enumerate(val_graphs):
        # Extract raw energies BEFORE normalization (x[:,0] = log(1+E))
        log_e = data.x[:, 0].numpy()
        energies = np.exp(log_e) - 1.0
        truth_labels = data.hit_truth_cluster.numpy()

        # Normalize a clone for model input
        data_norm = data.clone()
        normalize_graph(data_norm, stats)

        # Forward pass
        with torch.no_grad():
            output = model(data_norm.to(device))

        # Handle both dict (CaloClusterNetV1) and tensor (SimpleEdgeNet) output
        if isinstance(output, dict):
            logits_np = output["edge_logits"].cpu().numpy()
            nl = output.get("node_logits")
            node_logits_np = nl.cpu().numpy() if nl is not None else None
        else:
            logits_np = output.cpu().numpy()
            node_logits_np = None

        # Precompute sigmoid probs on masked edges for pairwise metrics
        mask = data.edge_mask.bool().numpy()
        all_probs = 1.0 / (1.0 + np.exp(-logits_np.astype(np.float64)))

        graphs_info.append({
            "logits": logits_np,
            "node_logits": node_logits_np,
            "edge_index": data.edge_index.numpy(),
            "n_nodes": data.x.shape[0],
            "energies": energies,
            "truth_labels": truth_labels,
            "edge_probs": all_probs[mask],
            "y_masked": data.y_edge[mask].numpy().astype(np.int32),
        })

        if (idx + 1) % 1000 == 0:
            print(f"  {idx + 1}/{len(val_graphs)}...")

    elapsed = time.time() - t0
    print(f"Inference done: {len(graphs_info)} graphs in {elapsed:.1f}s")

    # --- Coarse sweep ---
    coarse_thresholds = list(np.arange(0.1, 0.95, 0.1))
    min_hits = args.min_hits
    min_energy = args.min_energy
    print(f"\nCluster cleanup: min_hits={min_hits}, min_energy={min_energy} MeV")

    print(f"\nCoarse sweep ({len(coarse_thresholds)} thresholds):")
    coarse_results = []
    for tau in coarse_thresholds:
        t1 = time.time()
        result = evaluate_threshold(graphs_info, tau, min_hits, min_energy,
                                    tau_node=tau_node)
        dt = time.time() - t1
        coarse_results.append(result)
        print(f"  tau={tau:.2f}  F1={result['pairwise_f1']:.4f}  "
              f"truth_match={result['truth_match_rate']:.4f}  "
              f"purity={result['mean_purity']:.4f}  "
              f"compl={result['mean_completeness']:.4f}  "
              f"splits={result['n_split']:>5d}  merges={result['n_merged']:>5d}  "
              f"({dt:.1f}s)")

    best_coarse = max(coarse_results, key=lambda r: r["pairwise_f1"])
    print(f"\nBest coarse: tau={best_coarse['tau_edge']:.2f} "
          f"(F1={best_coarse['pairwise_f1']:.4f}, "
          f"truth_match={best_coarse['truth_match_rate']:.4f})")

    # --- Fine sweep around optimum ---
    fine_lo = max(0.02, best_coarse["tau_edge"] - 0.1)
    fine_hi = min(0.98, best_coarse["tau_edge"] + 0.1)
    fine_thresholds = list(np.arange(fine_lo, fine_hi + 0.005, 0.02))
    # Remove values already covered by coarse sweep
    fine_thresholds = [t for t in fine_thresholds
                       if not any(abs(t - c) < 0.005 for c in coarse_thresholds)]

    print(f"\nFine sweep ({len(fine_thresholds)} thresholds "
          f"in [{fine_lo:.2f}, {fine_hi:.2f}]):")
    fine_results = []
    for tau in fine_thresholds:
        t1 = time.time()
        result = evaluate_threshold(graphs_info, tau, min_hits, min_energy,
                                    tau_node=tau_node)
        dt = time.time() - t1
        fine_results.append(result)
        print(f"  tau={tau:.2f}  F1={result['pairwise_f1']:.4f}  "
              f"truth_match={result['truth_match_rate']:.4f}  "
              f"purity={result['mean_purity']:.4f}  "
              f"compl={result['mean_completeness']:.4f}  "
              f"splits={result['n_split']:>5d}  merges={result['n_merged']:>5d}  "
              f"({dt:.1f}s)")

    # --- Combine, find best, report ---
    all_results = coarse_results + fine_results
    all_results.sort(key=lambda r: r["tau_edge"])

    best = max(all_results, key=lambda r: r["pairwise_f1"])
    best_tau = best["tau_edge"]

    print(f"\n{'='*70}")
    print(f"  OPTIMAL tau_edge = {best_tau:.2f}  (maximizes pairwise F1)")
    print(f"{'='*70}")
    print(f"  Pairwise F1:       {best['pairwise_f1']:.4f}  "
          f"(P={best['pairwise_precision']:.4f}, R={best['pairwise_recall']:.4f})")
    print(f"  Truth match rate:  {best['truth_match_rate']:.4f}")
    print(f"  Reco match rate:   {best['reco_match_rate']:.4f}")
    print(f"  Mean purity:       {best['mean_purity']:.4f}")
    print(f"  Mean completeness: {best['mean_completeness']:.4f}")
    print(f"  Reco clusters:     {best['n_pred']}  (truth: {best['n_truth']})")
    print(f"  Splits: {best['n_split']}   Merges: {best['n_merged']}")
    print(f"{'='*70}")

    # Also report metrics with production cleanup at the optimal threshold
    inf_cfg = cfg["inference"]
    prod_min_hits = inf_cfg.get("min_hits", 2)
    prod_min_energy = inf_cfg.get("min_energy_mev", 10.0)
    if min_hits != prod_min_hits or min_energy != prod_min_energy:
        print(f"\nWith production cleanup (min_hits={prod_min_hits}, "
              f"min_energy={prod_min_energy} MeV):")
        prod = evaluate_threshold(graphs_info, best_tau,
                                  prod_min_hits, prod_min_energy,
                                  tau_node=tau_node)
        print(f"  Truth match rate:  {prod['truth_match_rate']:.4f}")
        print(f"  Reco match rate:   {prod['reco_match_rate']:.4f}")
        print(f"  Mean purity:       {prod['mean_purity']:.4f}")
        print(f"  Mean completeness: {prod['mean_completeness']:.4f}")
        print(f"  Reco clusters:     {prod['n_pred']}  (truth: {prod['n_truth']})")
        print(f"  Splits: {prod['n_split']}   Merges: {prod['n_merged']}")

    # --- Save CSV ---
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "sweep_results.csv"
    fieldnames = list(all_results[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            row = {}
            for k, v in r.items():
                row[k] = f"{v:.6f}" if isinstance(v, float) else v
            writer.writerow(row)
    print(f"\nSaved sweep results to {csv_path}")

    # --- Plot ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    taus = [r["tau_edge"] for r in all_results]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(f"Threshold Sweep on Validation Set — "
                 f"Optimal $\\tau_{{edge}}$ = {best_tau:.2f}",
                 fontsize=14, fontweight="bold")

    # 1. Pairwise edge metrics
    ax = axes[0, 0]
    ax.plot(taus, [r["pairwise_f1"] for r in all_results],
            "o-", label="F1", color="steelblue", linewidth=2, markersize=5)
    ax.plot(taus, [r["pairwise_precision"] for r in all_results],
            "s--", label="Precision", color="coral", alpha=0.7, markersize=4)
    ax.plot(taus, [r["pairwise_recall"] for r in all_results],
            "^--", label="Recall", color="green", alpha=0.7, markersize=4)
    ax.axvline(best_tau, color="red", linestyle=":", alpha=0.5,
               label=f"Optimal ({best_tau:.2f})")
    ax.set_xlabel(r"$\tau_{edge}$")
    ax.set_ylabel("Score")
    ax.set_title("Pairwise Edge Metrics")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)

    # 2. Cluster match rates
    ax = axes[0, 1]
    ax.plot(taus, [r["truth_match_rate"] * 100 for r in all_results],
            "o-", label="Truth match rate", color="steelblue", linewidth=2, markersize=5)
    ax.plot(taus, [r["reco_match_rate"] * 100 for r in all_results],
            "s-", label="Reco match rate", color="coral", linewidth=2, markersize=5)
    ax.axvline(best_tau, color="red", linestyle=":", alpha=0.5)
    ax.set_xlabel(r"$\tau_{edge}$")
    ax.set_ylabel("Match rate (%)")
    ax.set_title("Cluster Match Rates")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)

    # 3. Purity & completeness
    ax = axes[0, 2]
    ax.plot(taus, [r["mean_purity"] for r in all_results],
            "o-", label="Purity", color="steelblue", linewidth=2, markersize=5)
    ax.plot(taus, [r["mean_completeness"] for r in all_results],
            "s-", label="Completeness", color="coral", linewidth=2, markersize=5)
    ax.axvline(best_tau, color="red", linestyle=":", alpha=0.5)
    ax.set_xlabel(r"$\tau_{edge}$")
    ax.set_ylabel("Score")
    ax.set_title("Mean Purity & Completeness")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)

    # 4. Splits & merges
    ax = axes[1, 0]
    ax.plot(taus, [r["n_split"] for r in all_results],
            "o-", label="Splits", color="orange", linewidth=2, markersize=5)
    ax.plot(taus, [r["n_merged"] for r in all_results],
            "s-", label="Merges", color="purple", linewidth=2, markersize=5)
    ax.axvline(best_tau, color="red", linestyle=":", alpha=0.5)
    ax.set_xlabel(r"$\tau_{edge}$")
    ax.set_ylabel("Count")
    ax.set_title("Cluster Splits & Merges")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)

    # 5. Number of predicted clusters vs truth
    ax = axes[1, 1]
    ax.plot(taus, [r["n_pred"] for r in all_results],
            "o-", color="steelblue", linewidth=2, markersize=5, label="Predicted")
    n_truth = all_results[0]["n_truth"]
    ax.axhline(n_truth, color="red", linestyle="--", alpha=0.7,
               label=f"Truth ({n_truth:,})")
    ax.axvline(best_tau, color="red", linestyle=":", alpha=0.5)
    ax.set_xlabel(r"$\tau_{edge}$")
    ax.set_ylabel("Number of clusters")
    ax.set_title("Predicted vs Truth Clusters")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)

    # 6. Summary table
    ax = axes[1, 2]
    ax.axis("off")
    table_data = [
        [r"Optimal $\tau_{edge}$", f"{best_tau:.2f}"],
        ["Pairwise F1", f"{best['pairwise_f1']:.4f}"],
        ["Pairwise P / R",
         f"{best['pairwise_precision']:.4f} / {best['pairwise_recall']:.4f}"],
        ["Truth match rate", f"{best['truth_match_rate']:.1%}"],
        ["Reco match rate", f"{best['reco_match_rate']:.1%}"],
        ["Mean purity", f"{best['mean_purity']:.4f}"],
        ["Mean completeness", f"{best['mean_completeness']:.4f}"],
        ["Splits", f"{best['n_split']:,}"],
        ["Merges", f"{best['n_merged']:,}"],
        ["Pred / Truth clusters", f"{best['n_pred']:,} / {best['n_truth']:,}"],
    ]
    table = ax.table(cellText=table_data, colLabels=["Metric", "Value"],
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)
    ax.set_title("Optimal Threshold Summary", pad=20)

    plt.tight_layout()
    plot_path = out_dir / "threshold_sweep.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {plot_path}")

    print(f"\nTo freeze this threshold, update configs/default.yaml:")
    print(f"  inference:")
    print(f"    tau_edge: {best_tau:.2f}")


if __name__ == "__main__":
    main()
