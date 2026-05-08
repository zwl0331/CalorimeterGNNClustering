#!/usr/bin/env python3
"""
Sweep bfs_expand_cut on the validation set at a fixed tau_edge.

Run AFTER scripts/tune_threshold.py has frozen tau_edge.

Mirrors the v2 §7.3 sweep table (None / 5 / 10 / 15 MeV) but extended for
MixLow regime where the optimum may differ from EC=10. For each EC value:
  - Standard cluster-match metrics (TMR, RMR, purity, completeness, splits, merges)
  - Downstream cluster physics on val (E_reco >= 50 MeV cut):
      mean |dE| (energy residual)
      mean 2D dr (centroid displacement)

Inputs come from data/processed_*/val.pt: energies (from x[:, 0] = log(1+E)),
2D positions (x[:, 2:4]), truth labels (hit_truth_cluster).

Saves results to <output-dir>/sweep_bfs_expand_cut.csv.

Usage:
    source setup_env.sh
    python3 scripts/sweep_bfs_expand_cut.py \
        --config configs/run1b_mixlow_default.yaml \
        --checkpoint outputs/runs/simple_edge_net_run1b_mixlow/checkpoints/best_model.pt \
        --tau-edge 0.34 \
        --output-dir outputs/sweep_bfs_ec_simpleedgenet_run1b_mixlow
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


def match_clusters(pred_labels, truth_labels, energies, xs, ys):
    """Energy-weighted greedy match. Returns standard metrics + per-match physics
    (energies/dr in 2D for matched pairs). 2D centroids are sufficient for
    relative ranking on val.
    """
    pred_ids = set(pred_labels[pred_labels >= 0].tolist())
    truth_ids = set(truth_labels[truth_labels >= 0].tolist())

    if not pred_ids or not truth_ids:
        return {
            "n_pred": len(pred_ids), "n_truth": len(truth_ids),
            "n_matched_pred": 0, "n_matched_truth": 0,
            "purities": [], "completenesses": [],
            "n_split": 0, "n_merged": 0,
            "matches": [],
        }

    overlap = defaultdict(lambda: defaultdict(float))
    pred_energy = defaultdict(float)
    truth_energy = defaultdict(float)
    pred_cx = defaultdict(float)
    pred_cy = defaultdict(float)
    truth_cx = defaultdict(float)
    truth_cy = defaultdict(float)

    for i in range(len(energies)):
        e = energies[i]
        p = pred_labels[i]
        t = truth_labels[i]
        if p >= 0:
            pred_energy[p] += e
            pred_cx[p] += e * xs[i]
            pred_cy[p] += e * ys[i]
        if t >= 0:
            truth_energy[t] += e
            truth_cx[t] += e * xs[i]
            truth_cy[t] += e * ys[i]
        if p >= 0 and t >= 0:
            overlap[p][t] += e

    purities, completenesses = [], []
    matched_truth = set()
    matches = []
    for p in sorted(pred_ids):
        if p not in overlap:
            continue
        best_t = max(overlap[p], key=lambda t: overlap[p][t])
        shared = overlap[p][best_t]
        e_reco = pred_energy[p]
        e_truth = truth_energy[best_t]
        pur = shared / e_reco if e_reco > 0 else 0
        comp = shared / e_truth if e_truth > 0 else 0
        if pur > 0.5 and comp > 0.5:
            purities.append(pur)
            completenesses.append(comp)
            matched_truth.add(best_t)
            cx_r = pred_cx[p] / e_reco
            cy_r = pred_cy[p] / e_reco
            cx_t = truth_cx[best_t] / e_truth
            cy_t = truth_cy[best_t] / e_truth
            dr = float(np.hypot(cx_r - cx_t, cy_r - cy_t))
            matches.append({
                "e_reco": float(e_reco),
                "e_truth": float(e_truth),
                "dE": float(e_reco - e_truth),
                "dr": dr,
            })

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

    return {
        "n_pred": len(pred_ids), "n_truth": len(truth_ids),
        "n_matched_pred": len(purities), "n_matched_truth": len(matched_truth),
        "purities": purities, "completenesses": completenesses,
        "n_split": n_split, "n_merged": n_merged,
        "matches": matches,
    }


def evaluate_ec(graphs_info, tau_edge, bfs_ec, min_hits, min_energy_mev,
                tau_node=None):
    """Evaluate a single bfs_expand_cut value over precomputed graph data."""
    cluster_results = []
    for g in graphs_info:
        labels, _ = reconstruct_clusters(
            edge_index=g["edge_index"],
            edge_logits=g["logits"],
            n_nodes=g["n_nodes"],
            energies=g["energies"],
            tau_edge=tau_edge,
            min_hits=min_hits,
            min_energy_mev=min_energy_mev,
            node_logits=g.get("node_logits"),
            tau_node=tau_node,
            bfs_expand_cut=bfs_ec,
        )
        cluster_results.append(
            match_clusters(labels, g["truth_labels"], g["energies"],
                           g["xs"], g["ys"])
        )

    all_pur = [p for r in cluster_results for p in r["purities"]]
    all_comp = [c for r in cluster_results for c in r["completenesses"]]
    n_pred = sum(r["n_pred"] for r in cluster_results)
    n_truth = sum(r["n_truth"] for r in cluster_results)
    n_matched_pred = sum(r["n_matched_pred"] for r in cluster_results)
    n_matched_truth = sum(r["n_matched_truth"] for r in cluster_results)
    n_split = sum(r["n_split"] for r in cluster_results)
    n_merged = sum(r["n_merged"] for r in cluster_results)

    all_matches = [m for r in cluster_results for m in r["matches"]]
    if all_matches:
        ar = np.array([m["e_reco"] for m in all_matches])
        at = np.array([m["e_truth"] for m in all_matches])
        ad = np.array([m["dE"] for m in all_matches])
        adr = np.array([m["dr"] for m in all_matches])
        # downstream cut on E_reco
        mask_ds = ar >= 50.0
        ds_n = int(mask_ds.sum())
        ds_mean_abs_dE = float(np.mean(np.abs(ad[mask_ds]))) if ds_n else 0.0
        ds_mean_dr = float(np.mean(adr[mask_ds])) if ds_n else 0.0
        all_n = len(all_matches)
        all_mean_abs_dE = float(np.mean(np.abs(ad)))
        all_mean_dr = float(np.mean(adr))
    else:
        ds_n = all_n = 0
        ds_mean_abs_dE = ds_mean_dr = all_mean_abs_dE = all_mean_dr = 0.0

    return {
        "bfs_expand_cut": bfs_ec if bfs_ec is not None else -1.0,
        "n_pred": n_pred,
        "n_truth": n_truth,
        "n_matched_pred": n_matched_pred,
        "n_matched_truth": n_matched_truth,
        "reco_match_rate": n_matched_pred / n_pred if n_pred > 0 else 0.0,
        "truth_match_rate": n_matched_truth / n_truth if n_truth > 0 else 0.0,
        "mean_purity": float(np.mean(all_pur)) if all_pur else 0.0,
        "mean_completeness": float(np.mean(all_comp)) if all_comp else 0.0,
        "n_split": n_split,
        "n_merged": n_merged,
        "all_n": all_n,
        "all_mean_abs_dE": all_mean_abs_dE,
        "all_mean_dr": all_mean_dr,
        "ds_n": ds_n,
        "ds_mean_abs_dE": ds_mean_abs_dE,
        "ds_mean_dr": ds_mean_dr,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Sweep bfs_expand_cut at a fixed tau_edge")
    parser.add_argument("--config", type=str, required=True,
                        help="Model config (provides processed_dir + norm stats)")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tau-edge", type=float, required=True,
                        help="Fixed edge threshold (from prior tune_threshold run)")
    parser.add_argument("--tau-node", type=float, default=None,
                        help="Fixed node saliency threshold (default: not used)")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--ec-values", type=str,
                        default="None,3,5,7,10,12,15,20",
                        help="Comma-separated EC values; 'None' for bare CC")
    parser.add_argument("--min-hits", type=int, default=2)
    parser.add_argument("--min-energy", type=float, default=10.0)
    parser.add_argument("--data-pack", type=str, default=None,
                        help="Override packed graphs path "
                             "(default: <processed_dir>/val.pt). "
                             "Use for cross-eval on test or other splits.")
    parser.add_argument("--norm-config", type=str, default=None,
                        help="Override config used for normalization stats "
                             "(default: --config). Useful for cross-eval where "
                             "model was trained on different stats.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_model(cfg)
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    model_name = cfg["model"].get("name", "SimpleEdgeNet")
    print(f"Model: {model_name}")
    print(f"Loaded from epoch {ckpt['epoch']} (val F1={ckpt['val_f1']:.4f})")
    print(f"Fixed tau_edge = {args.tau_edge}")
    print(f"Min cluster: hits={args.min_hits}, energy={args.min_energy} MeV")

    if args.data_pack:
        val_packed = Path(args.data_pack)
    else:
        val_packed = Path(cfg["data"]["processed_dir"]) / "val.pt"
    if not val_packed.exists():
        print(f"ERROR: {val_packed} not found.")
        sys.exit(1)
    print(f"Loading graphs from {val_packed}...")
    val_graphs = torch.load(val_packed, weights_only=False)
    print(f"  {len(val_graphs)} graphs loaded")

    if args.norm_config:
        with open(args.norm_config) as f:
            norm_cfg = yaml.safe_load(f)
        stats_path = norm_cfg["data"]["normalization_stats"]
    else:
        stats_path = cfg["data"]["normalization_stats"]
    print(f"Loading normalization stats from {stats_path}")
    stats = load_stats(stats_path)

    print("Running model inference on all val graphs...")
    t0 = time.time()
    graphs_info = []
    for idx, data in enumerate(val_graphs):
        log_e = data.x[:, 0].numpy()
        energies = np.exp(log_e) - 1.0
        xs = data.x[:, 2].numpy().astype(np.float64)
        ys = data.x[:, 3].numpy().astype(np.float64)
        truth_labels = data.hit_truth_cluster.numpy()

        data_norm = data.clone()
        normalize_graph(data_norm, stats)
        with torch.no_grad():
            output = model(data_norm.to(device))
        if isinstance(output, dict):
            logits_np = output["edge_logits"].cpu().numpy()
            nl = output.get("node_logits")
            node_logits_np = nl.cpu().numpy() if nl is not None else None
        else:
            logits_np = output.cpu().numpy()
            node_logits_np = None

        graphs_info.append({
            "logits": logits_np,
            "node_logits": node_logits_np,
            "edge_index": data.edge_index.numpy(),
            "n_nodes": data.x.shape[0],
            "energies": energies,
            "xs": xs,
            "ys": ys,
            "truth_labels": truth_labels,
        })
        if (idx + 1) % 1000 == 0:
            print(f"  {idx + 1}/{len(val_graphs)}...")
    print(f"Inference done in {time.time() - t0:.1f}s")

    # Parse EC values
    ec_values = []
    for tok in args.ec_values.split(","):
        tok = tok.strip()
        if tok.lower() == "none":
            ec_values.append(None)
        else:
            ec_values.append(float(tok))

    print(f"\nSweeping bfs_expand_cut over: {ec_values}")
    print()
    header = (f"{'EC':>5}  {'TMR':>6}  {'RMR':>6}  {'Pur':>6}  "
              f"{'Compl':>6}  {'Splits':>7}  {'Merges':>7}  "
              f"{'all|dE|':>8}  {'all_dr':>7}  {'ds_n':>6}  "
              f"{'ds|dE|':>8}  {'ds_dr':>7}")
    print(header)
    print("-" * len(header))

    results = []
    for ec in ec_values:
        t1 = time.time()
        r = evaluate_ec(graphs_info, args.tau_edge, ec,
                        args.min_hits, args.min_energy,
                        tau_node=args.tau_node)
        results.append(r)
        ec_str = "None" if ec is None else f"{ec:.0f}"
        print(f"{ec_str:>5}  {r['truth_match_rate']:.4f}  "
              f"{r['reco_match_rate']:.4f}  {r['mean_purity']:.4f}  "
              f"{r['mean_completeness']:.4f}  {r['n_split']:>7d}  "
              f"{r['n_merged']:>7d}  {r['all_mean_abs_dE']:>8.4f}  "
              f"{r['all_mean_dr']:>7.3f}  {r['ds_n']:>6d}  "
              f"{r['ds_mean_abs_dE']:>8.4f}  {r['ds_mean_dr']:>7.3f}  "
              f"({time.time() - t1:.1f}s)")

    # Save CSV
    csv_path = out_dir / "sweep_bfs_expand_cut.csv"
    fieldnames = ["bfs_expand_cut", "tau_edge", "n_pred", "n_truth",
                  "n_matched_pred", "n_matched_truth",
                  "truth_match_rate", "reco_match_rate",
                  "mean_purity", "mean_completeness",
                  "n_split", "n_merged",
                  "all_n", "all_mean_abs_dE", "all_mean_dr",
                  "ds_n", "ds_mean_abs_dE", "ds_mean_dr"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            row = {k: r[k] for k in r if k in fieldnames}
            row["tau_edge"] = args.tau_edge
            w.writerow(row)
    print(f"\nSaved to {csv_path}")

    # Pick best by ds_mean_abs_dE among rows with TMR within 0.5pp of bare CC TMR
    bare = next((r for r in results if r["bfs_expand_cut"] == -1.0), results[0])
    bare_tmr = bare["truth_match_rate"]
    candidates = [r for r in results if r["truth_match_rate"] >= bare_tmr - 0.005]
    if candidates:
        best = min(candidates, key=lambda r: r["ds_mean_abs_dE"]
                   if r["ds_n"] > 0 else float("inf"))
        ec_v = best["bfs_expand_cut"]
        ec_label = "None" if ec_v < 0 else f"{ec_v:.1f}"
        print(f"\nRecommended: bfs_expand_cut = {ec_label}  "
              f"(downstream |dE| = {best['ds_mean_abs_dE']:.4f}, "
              f"TMR = {best['truth_match_rate']:.4f}, "
              f"splits = {best['n_split']})")


if __name__ == "__main__":
    main()
