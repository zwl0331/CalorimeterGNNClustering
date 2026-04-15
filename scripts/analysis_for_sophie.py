#!/usr/bin/env python3
"""
Comprehensive analysis of GNN clustering on MDC2025 v2 (with pileup).
Produces detailed diagnostic plots for comparison with Sophie's results.

Runs GNN inference on packed test graphs to produce:
  1. Truth vs reco cluster-size histograms
  2. Truth vs reco cluster-energy histograms
  3. Split/merge rate by cluster size and energy
  4. Per-cluster purity and completeness distributions
  5. Detailed summary statistics

Usage:
    source setup_env.sh
    python3 scripts/analysis_for_sophie.py
"""

import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import yaml

from src.data.normalization import load_stats, normalize_graph
from src.inference.cluster_reco import reconstruct_clusters
from src.models import build_model


def analyze_clusters(reco_labels, truth_labels, energies):
    """Detailed per-cluster analysis for one graph."""
    reco_ids = sorted(set(reco_labels[reco_labels >= 0].tolist()))
    truth_ids = sorted(set(truth_labels[truth_labels >= 0].tolist()))

    # Truth cluster properties
    truth_clusters = []
    for tid in truth_ids:
        mask = truth_labels == tid
        truth_clusters.append({
            "n_hits": int(mask.sum()),
            "energy": float(energies[mask].sum()),
            "max_hit_energy": float(energies[mask].max()),
        })

    # Reco cluster properties
    reco_clusters = []
    for rid in reco_ids:
        mask = reco_labels == rid
        reco_clusters.append({
            "n_hits": int(mask.sum()),
            "energy": float(energies[mask].sum()),
        })

    # Overlap matrix for matching
    overlap = defaultdict(lambda: defaultdict(float))
    reco_energy = defaultdict(float)
    truth_energy = {tid: float(energies[truth_labels == tid].sum()) for tid in truth_ids}

    for i in range(len(energies)):
        e = energies[i]
        r = reco_labels[i]
        t = truth_labels[i]
        if r >= 0:
            reco_energy[r] += e
        if r >= 0 and t >= 0:
            overlap[r][t] += e

    # Greedy matching
    matched_truth = set()
    matched_reco = set()
    match_purities = []
    match_completenesses = []

    for r in sorted(reco_ids):
        if r not in overlap:
            continue
        best_t = max(overlap[r], key=lambda t: overlap[r][t])
        shared = overlap[r][best_t]
        pur = shared / reco_energy[r] if reco_energy[r] > 0 else 0
        comp = shared / truth_energy[best_t] if truth_energy[best_t] > 0 else 0
        if pur > 0.5 and comp > 0.5:
            match_purities.append(pur)
            match_completenesses.append(comp)
            matched_truth.add(best_t)
            matched_reco.add(r)

    # Per-truth-cluster: matched, split, merged
    truth_detail = []
    for i, tid in enumerate(truth_ids):
        # Which reco clusters overlap this truth cluster significantly?
        overlapping_reco = []
        for r in reco_ids:
            if r in overlap and tid in overlap[r]:
                frac_of_reco = overlap[r][tid] / reco_energy[r] if reco_energy[r] > 0 else 0
                frac_of_truth = overlap[r][tid] / truth_energy[tid] if truth_energy[tid] > 0 else 0
                if frac_of_truth > 0.1:
                    overlapping_reco.append((r, frac_of_reco, frac_of_truth))

        is_matched = tid in matched_truth
        is_split = len([r for r, fr, ft in overlapping_reco if ft > 0.2]) > 1
        is_merged = not is_matched and len(overlapping_reco) > 0

        truth_detail.append({
            **truth_clusters[i],
            "matched": is_matched,
            "split": is_split,
            "merged": is_merged,
            "n_overlapping_reco": len(overlapping_reco),
        })

    # Per-reco-cluster: check for merges
    reco_detail = []
    for i, rid in enumerate(reco_ids):
        sig_truth = [t for t, e in overlap[rid].items()
                     if reco_energy[rid] > 0 and e / reco_energy[rid] > 0.1]
        reco_detail.append({
            **reco_clusters[i],
            "is_merge": len(sig_truth) > 1,
            "n_truth_contrib": len(sig_truth),
        })

    return {
        "truth_clusters": truth_detail,
        "reco_clusters": reco_detail,
        "purities": match_purities,
        "completenesses": match_completenesses,
        "n_matched_truth": len(matched_truth),
        "n_matched_reco": len(matched_reco),
    }


def main():
    device = torch.device("cpu")

    # Load models
    models = {}
    for name, cfg_path, ckpt_path in [
        ("SimpleEdgeNet", "configs/default.yaml",
         "outputs/runs/simple_edge_net_v2/checkpoints/best_model.pt"),
        ("CaloClusterNet", "configs/calo_cluster_net.yaml",
         "outputs/runs/calo_cluster_net_v2_stage1/checkpoints/best_model.pt"),
    ]:
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        model = build_model(cfg)
        ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device).eval()
        tau_edge = cfg["inference"]["tau_edge"]
        models[name] = {"model": model, "tau_edge": tau_edge}
        print(f"  {name}: tau_edge={tau_edge}")

    stats = load_stats("data/normalization_stats.pt")

    # Load test graphs
    graphs = torch.load("data/processed/test.pt", weights_only=False)
    print(f"\nLoaded {len(graphs)} test graphs")

    # Run analysis for each model
    results = {}
    for model_name, m in models.items():
        print(f"\nAnalyzing {model_name}...")
        all_truth = []
        all_reco = []
        all_pur = []
        all_comp = []

        for gi, g in enumerate(graphs):
            if gi % 1000 == 0:
                print(f"  {gi}/{len(graphs)}", flush=True)

            truth_labels = g.hit_truth_cluster.numpy()
            # Recover raw energies before normalization (x[:, 0] = log(1+E))
            raw_energies = (torch.exp(g.x[:, 0]) - 1).numpy()

            # Normalize for model input
            g_norm = g.clone()
            normalize_graph(g_norm, stats)
            with torch.no_grad():
                output = m["model"](g_norm.to(device))

            if isinstance(output, dict):
                logits = output["edge_logits"].cpu().numpy()
            else:
                logits = output.cpu().numpy()

            edge_index = g.edge_index.numpy()
            n_nodes = g.x.shape[0]

            reco_labels, _ = reconstruct_clusters(
                edge_index=edge_index,
                edge_logits=logits,
                n_nodes=n_nodes,
                energies=raw_energies,
                tau_edge=m["tau_edge"],
                min_hits=1,
                min_energy_mev=0.0,
            )

            result = analyze_clusters(reco_labels, truth_labels, raw_energies)
            all_truth.extend(result["truth_clusters"])
            all_reco.extend(result["reco_clusters"])
            all_pur.extend(result["purities"])
            all_comp.extend(result["completenesses"])

        results[model_name] = {
            "truth": all_truth,
            "reco": all_reco,
            "purities": all_pur,
            "completenesses": all_comp,
        }

    # Print summary stats
    for model_name, res in results.items():
        truth = res["truth"]
        reco = res["reco"]
        n_truth = len(truth)
        n_reco = len(reco)
        n_matched = sum(1 for t in truth if t["matched"])
        n_split = sum(1 for t in truth if t["split"])
        n_merged = sum(1 for r in reco if r["is_merge"])

        print(f"\n{'='*60}")
        print(f"  {model_name} — MDC2025 v2 Test Set ({len(graphs)} graphs)")
        print(f"{'='*60}")
        print(f"  Truth clusters:    {n_truth:,}")
        print(f"  Reco clusters:     {n_reco:,}")
        print(f"  Matched truth:     {n_matched:,} ({n_matched/n_truth:.1%})")
        print(f"  Split truth:       {n_split:,} ({n_split/n_truth:.2%})")
        print(f"  Merged reco:       {n_merged:,} ({n_merged/n_reco:.2%}" if n_reco > 0 else f"  Merged reco:       0")
        print(f"  Mean purity:       {np.mean(res['purities']):.4f}")
        print(f"  Mean completeness: {np.mean(res['completenesses']):.4f}")

        # Split rate by cluster size
        print(f"\n  Split rate by truth cluster size:")
        print(f"  {'Size':<8} {'Total':>8} {'Split':>8} {'Rate':>8}")
        for sz in [1, 2, 3, 4, 5, "6+"]:
            if sz == "6+":
                in_bin = [t for t in truth if t["n_hits"] >= 6]
            else:
                in_bin = [t for t in truth if t["n_hits"] == sz]
            n = len(in_bin)
            s = sum(1 for t in in_bin if t["split"])
            print(f"  {str(sz):<8} {n:>8} {s:>8} {s/n*100 if n else 0:>7.1f}%")

        # Split rate by energy
        print(f"\n  Split rate by truth cluster energy:")
        print(f"  {'Energy':<15} {'Total':>8} {'Split':>8} {'Rate':>8}")
        for lo, hi, lbl in [(0, 10, "<10 MeV"), (10, 50, "10-50 MeV"),
                            (50, 200, "50-200 MeV"), (200, float("inf"), ">200 MeV")]:
            in_bin = [t for t in truth if lo <= t["energy"] < hi]
            n = len(in_bin)
            s = sum(1 for t in in_bin if t["split"])
            print(f"  {lbl:<15} {n:>8} {s:>8} {s/n*100 if n else 0:>7.1f}%")

    # ── Plots ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path("outputs/analysis_for_sophie")
    out_dir.mkdir(parents=True, exist_ok=True)

    for model_name, res in results.items():
        truth = res["truth"]
        reco = res["reco"]
        short = "SEN" if "Simple" in model_name else "CCN"
        tau = models[model_name]["tau_edge"]

        fig, axes = plt.subplots(3, 3, figsize=(20, 18))
        fig.suptitle(
            f"MDC2025 v2 (Pileup) — {model_name} (τ={tau})\n"
            f"{len(graphs):,} test disk-graphs, calo-entrant truth",
            fontsize=14, fontweight="bold")

        # 1. Hits per cluster: truth vs reco
        ax = axes[0, 0]
        truth_nhits = [t["n_hits"] for t in truth]
        reco_nhits = [r["n_hits"] for r in reco]
        bins_h = np.arange(0.5, 16.5, 1)
        ax.hist(truth_nhits, bins=bins_h, alpha=0.6, label=f"Truth (n={len(truth):,})",
                color="forestgreen", edgecolor="white")
        ax.hist(reco_nhits, bins=bins_h, alpha=0.6, label=f"Reco (n={len(reco):,})",
                color="steelblue", edgecolor="white")
        ax.set_xlabel("Hits per cluster")
        ax.set_ylabel("Count")
        ax.set_title("Cluster Size: Truth vs Reco")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_yscale("log")

        # 2. Energy per cluster: truth vs reco
        ax = axes[0, 1]
        truth_energy = [t["energy"] for t in truth]
        reco_energy = [r["energy"] for r in reco]
        bins_e = np.linspace(0, 300, 60)
        ax.hist(truth_energy, bins=bins_e, alpha=0.6, label="Truth",
                color="forestgreen", edgecolor="white")
        ax.hist(reco_energy, bins=bins_e, alpha=0.6, label="Reco",
                color="steelblue", edgecolor="white")
        ax.set_xlabel("Total cluster energy (MeV)")
        ax.set_ylabel("Count")
        ax.set_title("Cluster Energy: Truth vs Reco")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_yscale("log")

        # 3. Purity distribution
        ax = axes[0, 2]
        ax.hist(res["purities"], bins=np.linspace(0.5, 1.0, 60),
                color="steelblue", edgecolor="white", alpha=0.8)
        ax.axvline(np.mean(res["purities"]), color="red", linestyle="--",
                   label=f"Mean: {np.mean(res['purities']):.4f}")
        ax.set_xlabel("Purity")
        ax.set_ylabel("Count")
        ax.set_title("Matched Cluster Purity")
        ax.legend()
        ax.grid(alpha=0.3)

        # 4. Completeness distribution
        ax = axes[1, 0]
        ax.hist(res["completenesses"], bins=np.linspace(0.5, 1.0, 60),
                color="seagreen", edgecolor="white", alpha=0.8)
        ax.axvline(np.mean(res["completenesses"]), color="red", linestyle="--",
                   label=f"Mean: {np.mean(res['completenesses']):.4f}")
        ax.set_xlabel("Completeness")
        ax.set_ylabel("Count")
        ax.set_title("Matched Cluster Completeness")
        ax.legend()
        ax.grid(alpha=0.3)

        # 5. Truth match rate by cluster size
        ax = axes[1, 1]
        size_bins = list(range(1, 11))
        match_rates = []
        counts = []
        for sz in size_bins:
            in_bin = [t for t in truth if t["n_hits"] == sz]
            if in_bin:
                match_rates.append(sum(1 for t in in_bin if t["matched"]) / len(in_bin) * 100)
                counts.append(len(in_bin))
            else:
                match_rates.append(0)
                counts.append(0)
        ax.bar(size_bins, match_rates, color="steelblue", alpha=0.8)
        for i, (mr, c) in enumerate(zip(match_rates, counts)):
            if c > 0:
                ax.text(size_bins[i], mr + 1, f"{c:,}", ha="center", fontsize=7)
        ax.set_xlabel("Truth cluster size (hits)")
        ax.set_ylabel("Truth match rate (%)")
        ax.set_title("Truth Match Rate by Cluster Size\n(numbers = cluster count)")
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.3, axis="y")

        # 6. Truth match rate by energy
        ax = axes[1, 2]
        energy_edges = [0, 5, 10, 20, 50, 100, 200, 500]
        energy_labels = ["0-5", "5-10", "10-20", "20-50", "50-100", "100-200", "200+"]
        mr_energy = []
        cnt_energy = []
        for i in range(len(energy_edges) - 1):
            lo, hi = energy_edges[i], energy_edges[i + 1]
            in_bin = [t for t in truth if lo <= t["energy"] < hi]
            if in_bin:
                mr_energy.append(sum(1 for t in in_bin if t["matched"]) / len(in_bin) * 100)
                cnt_energy.append(len(in_bin))
            else:
                mr_energy.append(0)
                cnt_energy.append(0)
        x_pos = np.arange(len(energy_labels))
        ax.bar(x_pos, mr_energy, color="seagreen", alpha=0.8)
        for i, (mr, c) in enumerate(zip(mr_energy, cnt_energy)):
            if c > 0:
                ax.text(i, mr + 1, f"{c:,}", ha="center", fontsize=7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(energy_labels)
        ax.set_xlabel("Truth cluster energy (MeV)")
        ax.set_ylabel("Truth match rate (%)")
        ax.set_title("Truth Match Rate by Energy\n(numbers = cluster count)")
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.3, axis="y")

        # 7. Split rate by cluster size
        ax = axes[2, 0]
        split_rates = []
        for sz in size_bins:
            in_bin = [t for t in truth if t["n_hits"] == sz]
            if in_bin:
                split_rates.append(sum(1 for t in in_bin if t["split"]) / len(in_bin) * 100)
            else:
                split_rates.append(0)
        ax.bar(size_bins, split_rates, color="coral", alpha=0.8)
        ax.set_xlabel("Truth cluster size (hits)")
        ax.set_ylabel("Split rate (%)")
        ax.set_title("Split Rate by Truth Cluster Size")
        ax.grid(alpha=0.3, axis="y")

        # 8. Merge rate by reco cluster size
        ax = axes[2, 1]
        reco_sizes = [r["n_hits"] for r in reco]
        reco_size_bins = list(range(1, 11))
        merge_rates = []
        for sz in reco_size_bins:
            in_bin = [r for r in reco if r["n_hits"] == sz]
            if in_bin:
                merge_rates.append(sum(1 for r in in_bin if r["is_merge"]) / len(in_bin) * 100)
            else:
                merge_rates.append(0)
        ax.bar(reco_size_bins, merge_rates, color="darkorange", alpha=0.8)
        ax.set_xlabel("Reco cluster size (hits)")
        ax.set_ylabel("Merge rate (%)")
        ax.set_title("Merge Rate by Reco Cluster Size")
        ax.grid(alpha=0.3, axis="y")

        # 9. Summary table
        ax = axes[2, 2]
        ax.axis("off")
        n_truth = len(truth)
        n_reco = len(reco)
        n_matched = sum(1 for t in truth if t["matched"])
        n_split = sum(1 for t in truth if t["split"])
        n_merged = sum(1 for r in reco if r["is_merge"])
        table_data = [
            ["Test disk-graphs", f"{len(graphs):,}"],
            ["Truth clusters", f"{n_truth:,}"],
            ["Reco clusters", f"{n_reco:,}"],
            ["Truth match rate", f"{n_matched/n_truth:.1%}"],
            ["Mean purity", f"{np.mean(res['purities']):.4f}"],
            ["Mean completeness", f"{np.mean(res['completenesses']):.4f}"],
            ["Splits (truth)", f"{n_split:,} ({n_split/n_truth:.2%})"],
            ["Merges (reco)", f"{n_merged:,} ({n_merged/n_reco:.2%})"],
            ["Singleton truth", f"{sum(1 for t in truth if t['n_hits']==1):,} "
             f"({sum(1 for t in truth if t['n_hits']==1)/n_truth:.1%})"],
        ]
        table = ax.table(cellText=table_data, colLabels=["Metric", "Value"],
                         cellLoc="center", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 1.6)
        ax.set_title(f"{model_name} Summary", pad=20)

        plt.tight_layout()
        plot_path = out_dir / f"analysis_{short.lower()}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {plot_path}")

    print(f"\nAll plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
