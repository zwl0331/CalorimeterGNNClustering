"""
Graph construction gate validation.

Checks whether the graph builder connects truth-cluster hits well enough
for edge classification to succeed.  Three metrics:

  1. Pair recall: fraction of same-truth-cluster hit pairs connected by a
     direct edge.  Target: >99%.
  2. Cluster connectivity recall: fraction of truth clusters whose hits
     form a connected subgraph.  Target: >95%.
  3. Stratified recall: pair recall broken down by cluster energy,
     hit multiplicity, and radial position.

Usage:
    source setup_env.sh
    python3 scripts/graph_gate.py                         # all processed graphs
    python3 scripts/graph_gate.py --max-graphs 100        # quick check
    python3 scripts/graph_gate.py --processed-dir data/processed/ --out-dir outputs/graph_gate/
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_graphs(processed_dir, max_graphs=None):
    """Load processed .pt graph files."""
    import torch
    pt_files = sorted(Path(processed_dir).glob("*.pt"))
    pt_files = [f for f in pt_files if not f.name.startswith("diagnostics")]
    if max_graphs:
        pt_files = pt_files[:max_graphs]
    graphs = []
    for f in pt_files:
        graphs.append(torch.load(f, weights_only=False))
    return graphs


def edge_set_from_index(edge_index):
    """Build set of (src, dst) pairs from edge_index tensor."""
    ei = edge_index.numpy()
    return set(zip(ei[0].tolist(), ei[1].tolist()))


def analyze_graph(data):
    """Compute gate metrics for a single graph.

    Returns a dict with per-cluster stats and overall pair recall.
    """
    truth = data.hit_truth_cluster.numpy()
    edge_index = data.edge_index.numpy()
    node_feat = data.x.numpy()

    edges = edge_set_from_index(data.edge_index)

    # Group hits by truth cluster (skip -1 = ambiguous)
    cluster_hits = defaultdict(list)
    for i, tc in enumerate(truth):
        if tc >= 0:
            cluster_hits[tc].append(i)

    total_pairs = 0
    connected_pairs = 0
    cluster_results = []

    for tc, hits in cluster_hits.items():
        n = len(hits)
        if n < 2:
            continue

        # All ordered pairs within this truth cluster
        n_pairs = n * (n - 1)  # directed pairs
        n_connected = 0
        for i in range(len(hits)):
            for j in range(len(hits)):
                if i == j:
                    continue
                if (hits[i], hits[j]) in edges:
                    n_connected += 1

        pair_recall = n_connected / n_pairs if n_pairs > 0 else 1.0

        # Connectivity: BFS from first hit, check if all cluster hits reachable
        adj = defaultdict(set)
        for i in range(len(hits)):
            for j in range(len(hits)):
                if i != j and (hits[i], hits[j]) in edges:
                    adj[hits[i]].add(hits[j])

        visited = set()
        stack = [hits[0]]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            for nb in adj.get(node, set()):
                if nb not in visited:
                    stack.append(nb)
        is_connected = all(h in visited for h in hits)

        # Cluster properties for stratification
        hit_energies = np.exp(node_feat[hits, 0]) - 1  # undo log1p
        total_energy = hit_energies.sum()
        hit_positions = node_feat[hits, 2:4]  # x, y columns
        mean_r = np.sqrt(hit_positions[:, 0]**2 + hit_positions[:, 1]**2).mean()

        cluster_results.append({
            "truth_cluster": tc,
            "n_hits": n,
            "n_pairs": n_pairs,
            "n_connected": n_connected,
            "pair_recall": pair_recall,
            "is_connected": is_connected,
            "total_energy": float(total_energy),
            "mean_radial_pos": float(mean_r),
        })

        total_pairs += n_pairs
        connected_pairs += n_connected

    overall_pair_recall = connected_pairs / total_pairs if total_pairs > 0 else 1.0

    return {
        "overall_pair_recall": overall_pair_recall,
        "total_pairs": total_pairs,
        "connected_pairs": connected_pairs,
        "cluster_results": cluster_results,
    }


def stratified_stats(all_cluster_results, key, bin_edges):
    """Compute pair recall in bins of a given cluster property."""
    rows = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        subset = [c for c in all_cluster_results
                  if lo <= c[key] < hi]
        n_clusters = len(subset)
        n_pairs = sum(c["n_pairs"] for c in subset)
        n_conn = sum(c["n_connected"] for c in subset)
        n_connected_clusters = sum(1 for c in subset if c["is_connected"])
        pair_recall = n_conn / n_pairs if n_pairs > 0 else float("nan")
        conn_recall = n_connected_clusters / n_clusters if n_clusters > 0 else float("nan")
        rows.append({
            "bin_lo": lo, "bin_hi": hi,
            "n_clusters": n_clusters,
            "n_pairs": n_pairs,
            "pair_recall": pair_recall,
            "connectivity_recall": conn_recall,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Graph construction gate validation")
    parser.add_argument("--processed-dir", type=str, default="data/processed/",
                        help="Directory with processed .pt files")
    parser.add_argument("--max-graphs", type=int, default=None,
                        help="Max graphs to analyze")
    parser.add_argument("--out-dir", type=str, default="outputs/graph_gate/",
                        help="Output directory for results")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading graphs from {args.processed_dir}...")
    graphs = load_graphs(args.processed_dir, args.max_graphs)
    print(f"Loaded {len(graphs)} graphs")

    if not graphs:
        print("ERROR: No graphs found.", file=sys.stderr)
        sys.exit(1)

    all_cluster_results = []
    total_pairs = 0
    total_connected = 0
    total_clusters = 0
    total_connected_clusters = 0

    for gi, data in enumerate(graphs):
        result = analyze_graph(data)
        all_cluster_results.extend(result["cluster_results"])
        total_pairs += result["total_pairs"]
        total_connected += result["connected_pairs"]
        for cr in result["cluster_results"]:
            total_clusters += 1
            if cr["is_connected"]:
                total_connected_clusters += 1

    overall_pair_recall = total_connected / total_pairs if total_pairs > 0 else 1.0
    overall_conn_recall = total_connected_clusters / total_clusters if total_clusters > 0 else 1.0

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"GRAPH CONSTRUCTION GATE RESULTS")
    print(f"{'='*60}")
    print(f"Graphs analyzed:         {len(graphs)}")
    print(f"Truth clusters (≥2 hits): {total_clusters}")
    print(f"Total directed pairs:    {total_pairs}")
    print(f"Connected pairs:         {total_connected}")
    print(f"")
    pair_pass = "PASS" if overall_pair_recall >= 0.99 else "FAIL"
    conn_pass = "PASS" if overall_conn_recall >= 0.95 else "FAIL"
    print(f"Pair recall:             {overall_pair_recall:.4f}  (target ≥0.99)  [{pair_pass}]")
    print(f"Cluster connectivity:    {overall_conn_recall:.4f}  (target ≥0.95)  [{conn_pass}]")
    print(f"{'='*60}")

    # ── Stratified by cluster energy ─────────────────────────────
    energy_bins = [0, 10, 25, 50, 100, 200, 500, float("inf")]
    energy_strat = stratified_stats(all_cluster_results, "total_energy", energy_bins)
    print(f"\nStratified by cluster energy (MeV):")
    print(f"  {'Bin':>15s}  {'Clusters':>8s}  {'Pairs':>8s}  {'PairRecall':>10s}  {'ConnRecall':>10s}")
    for r in energy_strat:
        hi_str = f"{r['bin_hi']:.0f}" if r['bin_hi'] < float("inf") else "inf"
        bin_str = f"[{r['bin_lo']:.0f}, {hi_str})"
        pr = f"{r['pair_recall']:.4f}" if not np.isnan(r['pair_recall']) else "N/A"
        cr = f"{r['connectivity_recall']:.4f}" if not np.isnan(r['connectivity_recall']) else "N/A"
        print(f"  {bin_str:>15s}  {r['n_clusters']:>8d}  {r['n_pairs']:>8d}  {pr:>10s}  {cr:>10s}")

    # ── Stratified by hit multiplicity ───────────────────────────
    mult_bins = [2, 3, 4, 5, 7, 10, 20, 50, 1000]
    mult_strat = stratified_stats(all_cluster_results, "n_hits", mult_bins)
    print(f"\nStratified by hit multiplicity:")
    print(f"  {'Bin':>15s}  {'Clusters':>8s}  {'Pairs':>8s}  {'PairRecall':>10s}  {'ConnRecall':>10s}")
    for r in mult_strat:
        hi_str = f"{r['bin_hi']:.0f}"
        bin_str = f"[{r['bin_lo']:.0f}, {hi_str})"
        pr = f"{r['pair_recall']:.4f}" if not np.isnan(r['pair_recall']) else "N/A"
        cr = f"{r['connectivity_recall']:.4f}" if not np.isnan(r['connectivity_recall']) else "N/A"
        print(f"  {bin_str:>15s}  {r['n_clusters']:>8d}  {r['n_pairs']:>8d}  {pr:>10s}  {cr:>10s}")

    # ── Stratified by radial position ────────────────────────────
    rad_bins = [0, 100, 200, 300, 400, 500, 700]
    rad_strat = stratified_stats(all_cluster_results, "mean_radial_pos", rad_bins)
    print(f"\nStratified by mean radial position (mm):")
    print(f"  {'Bin':>15s}  {'Clusters':>8s}  {'Pairs':>8s}  {'PairRecall':>10s}  {'ConnRecall':>10s}")
    for r in rad_strat:
        bin_str = f"[{r['bin_lo']:.0f}, {r['bin_hi']:.0f})"
        pr = f"{r['pair_recall']:.4f}" if not np.isnan(r['pair_recall']) else "N/A"
        cr = f"{r['connectivity_recall']:.4f}" if not np.isnan(r['connectivity_recall']) else "N/A"
        print(f"  {bin_str:>15s}  {r['n_clusters']:>8d}  {r['n_pairs']:>8d}  {pr:>10s}  {cr:>10s}")

    # ── Save detailed per-cluster results ────────────────────────
    csv_path = out_dir / "gate_per_cluster.csv"
    if all_cluster_results:
        fieldnames = ["truth_cluster", "n_hits", "n_pairs", "n_connected",
                      "pair_recall", "is_connected", "total_energy", "mean_radial_pos"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_cluster_results)
        print(f"\nPer-cluster details saved to {csv_path}")

    # ── Save summary ─────────────────────────────────────────────
    summary_path = out_dir / "gate_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Graph Construction Gate Results\n")
        f.write(f"{'='*40}\n")
        f.write(f"Graphs: {len(graphs)}\n")
        f.write(f"Truth clusters (≥2 hits): {total_clusters}\n")
        f.write(f"Total directed pairs: {total_pairs}\n")
        f.write(f"Connected pairs: {total_connected}\n")
        f.write(f"Pair recall: {overall_pair_recall:.6f}\n")
        f.write(f"Cluster connectivity recall: {overall_conn_recall:.6f}\n")
        f.write(f"Pair recall gate (≥0.99): {pair_pass}\n")
        f.write(f"Connectivity gate (≥0.95): {conn_pass}\n")
    print(f"Summary saved to {summary_path}")

    # Return exit code based on gate pass
    if pair_pass == "FAIL" or conn_pass == "FAIL":
        print("\n⚠ GATE NOT PASSED — review graph construction parameters")
        return 1
    else:
        print("\n✓ GATE PASSED")
        return 0


if __name__ == "__main__":
    sys.exit(main())
