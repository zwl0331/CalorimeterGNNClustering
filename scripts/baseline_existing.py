"""
Baseline benchmark: evaluate BFS clustering against MC truth.

Reads MDC2025-002 EventNtuple files (with calohitsmc MC truth branches)
and computes how well the existing BFS reco clusters match MC truth clusters.

MC truth cluster definition:
  - For each hit, find the dominant SimParticle (highest energy deposit).
  - Group hits by dominant SimParticle ID → truth clusters (per disk).
  - Hits where dominant SimParticle purity < 0.7 are marked ambiguous.

BFS ↔ MC truth matching:
  - Greedy energy-weighted overlap matching.
  - Purity   = E_shared / E_reco
  - Completeness = E_shared / E_truth
  - Matched if purity > 0.5 AND completeness > 0.5

Usage:
    source setup_env.sh
    python3 scripts/baseline_existing.py [--n-files 5] [--n-events 10000]
"""

import argparse
import glob
import sys
import time
from collections import defaultdict
from pathlib import Path

import awkward as ak
import numpy as np
import uproot


def get_mc_truth_files():
    """Get MDC2025-002 file list."""
    pattern = "/pnfs/mu2e/tape/phy-nts/nts/mu2e/FlateMinusMix1BBTriggered/MDC2025-002/root/*/*/*.root"
    return sorted(glob.glob(pattern))


def build_truth_clusters(hit_simids, hit_edeps, hit_energies, hit_disks):
    """
    Build MC truth clusters from per-hit SimParticle info.

    Returns:
        truth_labels: array of int, truth cluster label per hit (-1 = ambiguous)
        n_ambiguous: number of ambiguous hits
    """
    nhits = len(hit_energies)
    truth_labels = np.full(nhits, -1, dtype=np.int64)

    # For each hit, find dominant SimParticle
    dom_simid = np.full(nhits, -1, dtype=np.int64)
    dom_purity = np.zeros(nhits)

    for i in range(nhits):
        simids = np.array(hit_simids[i])
        edeps = np.array(hit_edeps[i])
        if len(simids) == 0:
            continue
        total = edeps.sum()
        if total <= 0:
            continue
        best = np.argmax(edeps)
        dom_simid[i] = simids[best]
        dom_purity[i] = edeps[best] / total

    # Group by (disk, dominant SimParticle) → truth cluster
    # Only non-ambiguous hits (purity >= 0.7)
    cluster_map = {}  # (disk, simid) -> cluster_label
    next_label = 0

    for i in range(nhits):
        if dom_purity[i] < 0.7 or dom_simid[i] < 0:
            continue  # ambiguous
        key = (int(hit_disks[i]), int(dom_simid[i]))
        if key not in cluster_map:
            cluster_map[key] = next_label
            next_label += 1
        truth_labels[i] = cluster_map[key]

    n_ambiguous = int(np.sum(truth_labels < 0))
    return truth_labels, n_ambiguous


def match_clusters(reco_labels, truth_labels, hit_energies):
    """
    Greedy energy-weighted matching between reco and truth clusters.

    Returns dict with matching stats.
    """
    # Get valid hits (both reco and truth assigned)
    valid = (reco_labels >= 0) & (truth_labels >= 0)

    reco_ids = set(reco_labels[reco_labels >= 0].tolist())
    truth_ids = set(truth_labels[truth_labels >= 0].tolist())

    if not reco_ids or not truth_ids:
        return {
            "n_reco": len(reco_ids),
            "n_truth": len(truth_ids),
            "n_matched": 0,
            "mean_purity": 0.0,
            "mean_completeness": 0.0,
            "n_split": 0,
            "n_merged": 0,
        }

    # Build energy overlap matrix
    overlap = defaultdict(lambda: defaultdict(float))
    reco_energy = defaultdict(float)
    truth_energy = defaultdict(float)

    for i in range(len(hit_energies)):
        e = hit_energies[i]
        r = reco_labels[i]
        t = truth_labels[i]
        if r >= 0:
            reco_energy[r] += e
        if t >= 0:
            truth_energy[t] += e
        if r >= 0 and t >= 0:
            overlap[r][t] += e

    # Greedy matching: for each reco cluster, find best truth match
    matches = []
    matched_truth = set()
    purities = []
    completenesses = []

    for r in sorted(reco_ids):
        if r not in overlap:
            continue
        best_t = max(overlap[r], key=lambda t: overlap[r][t])
        shared_e = overlap[r][best_t]
        purity = shared_e / reco_energy[r] if reco_energy[r] > 0 else 0
        completeness = shared_e / truth_energy[best_t] if truth_energy[best_t] > 0 else 0

        if purity > 0.5 and completeness > 0.5:
            matches.append((r, best_t, purity, completeness))
            matched_truth.add(best_t)
            purities.append(purity)
            completenesses.append(completeness)

    # Count splits: truth cluster matched by multiple reco clusters
    truth_to_reco = defaultdict(list)
    for r, t, p, c in matches:
        truth_to_reco[t].append(r)
    n_split = sum(1 for t, rs in truth_to_reco.items() if len(rs) > 1)

    # Count merges: reco cluster matched to multiple truth clusters
    reco_to_truth = defaultdict(list)
    for r in sorted(reco_ids):
        if r not in overlap:
            continue
        for t, e in overlap[r].items():
            if t in truth_energy and e / reco_energy[r] > 0.1:
                reco_to_truth[r].append(t)
    n_merged = sum(1 for r, ts in reco_to_truth.items() if len(ts) > 1)

    return {
        "n_reco": len(reco_ids),
        "n_truth": len(truth_ids),
        "n_matched": len(matches),
        "mean_purity": float(np.mean(purities)) if purities else 0.0,
        "mean_completeness": float(np.mean(completenesses)) if completenesses else 0.0,
        "n_split": n_split,
        "n_merged": n_merged,
    }


def process_file(filepath, max_events=None):
    """Process one ROOT file, return per-event matching stats."""
    tree = uproot.open(filepath + ":EventNtuple/ntuple")

    branches = [
        "calohits.crystalId_", "calohits.eDep_", "calohits.time_",
        "calohits.clusterIdx_",
        "caloclusters.diskID_", "caloclusters.energyDep_",
        "caloclusters.size_",
        "calohitsmc.simParticleIds", "calohitsmc.eDeps", "calohitsmc.nsim",
    ]

    data = tree.arrays(branches, entry_stop=max_events)
    n_events = len(data)

    # Load crystal geometry for disk lookup
    from src.geometry.crystal_geometry import load_crystal_map
    crystal_map = load_crystal_map()

    results = []
    for ev in range(n_events):
        nhits = len(data["calohits.crystalId_"][ev])
        if nhits == 0:
            continue

        hit_cryids = np.array(data["calohits.crystalId_"][ev])
        hit_energies = np.array(data["calohits.eDep_"][ev])
        hit_clusteridx = np.array(data["calohits.clusterIdx_"][ev])
        hit_simids = data["calohitsmc.simParticleIds"][ev]
        hit_edeps_mc = data["calohitsmc.eDeps"][ev]

        # Get disk per hit from crystal geometry
        hit_disks = np.array([crystal_map[int(c)][0] if int(c) in crystal_map else -1
                              for c in hit_cryids])

        # Build truth clusters
        truth_labels, n_ambiguous = build_truth_clusters(
            hit_simids, hit_edeps_mc, hit_energies, hit_disks
        )

        # BFS reco labels = clusterIdx_
        reco_labels = hit_clusteridx.copy()

        # Match
        stats = match_clusters(reco_labels, truth_labels, hit_energies)
        stats["n_hits"] = nhits
        stats["n_ambiguous"] = n_ambiguous
        stats["event_idx"] = ev
        results.append(stats)

    return results


def main():
    parser = argparse.ArgumentParser(description="BFS baseline benchmark vs MC truth")
    parser.add_argument("--n-files", type=int, default=5, help="Number of ROOT files to process")
    parser.add_argument("--n-events", type=int, default=None, help="Max events per file")
    parser.add_argument("--output", type=str, default="data/baseline/bfs_benchmark.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    files = get_mc_truth_files()
    if not files:
        print("ERROR: No MDC2025-002 files found. Check path.", file=sys.stderr)
        sys.exit(1)

    files = files[:args.n_files]
    print(f"Processing {len(files)} files...")

    all_results = []
    t0 = time.time()
    for i, f in enumerate(files):
        print(f"  [{i+1}/{len(files)}] {Path(f).name}...", end=" ", flush=True)
        ft0 = time.time()
        results = process_file(f, max_events=args.n_events)
        all_results.extend(results)
        print(f"{len(results)} events ({time.time()-ft0:.1f}s)")

    elapsed = time.time() - t0
    print(f"\nProcessed {len(all_results)} events in {elapsed:.1f}s")

    if not all_results:
        print("No events with hits found.")
        return

    # Aggregate stats
    n_matched = sum(r["n_matched"] for r in all_results)
    n_reco = sum(r["n_reco"] for r in all_results)
    n_truth = sum(r["n_truth"] for r in all_results)
    purities = [r["mean_purity"] for r in all_results if r["n_matched"] > 0]
    completenesses = [r["mean_completeness"] for r in all_results if r["n_matched"] > 0]
    n_split = sum(r["n_split"] for r in all_results)
    n_merged = sum(r["n_merged"] for r in all_results)
    n_ambiguous = sum(r["n_ambiguous"] for r in all_results)
    n_hits = sum(r["n_hits"] for r in all_results)

    # Events with hits
    events_with_clusters = sum(1 for r in all_results if r["n_reco"] > 0)

    print(f"\n{'='*60}")
    print(f"BFS Baseline Benchmark (MDC2025-002 MC truth)")
    print(f"{'='*60}")
    print(f"Events processed:       {len(all_results)}")
    print(f"Events with clusters:   {events_with_clusters}")
    print(f"Total hits:             {n_hits}")
    print(f"Ambiguous hits:         {n_ambiguous} ({100*n_ambiguous/n_hits:.1f}%)")
    print(f"Total reco clusters:    {n_reco}")
    print(f"Total truth clusters:   {n_truth}")
    print(f"Matched clusters:       {n_matched}")
    print(f"Reco match rate:        {100*n_matched/n_reco:.1f}%" if n_reco else "N/A")
    print(f"Truth match rate:       {100*n_matched/n_truth:.1f}%" if n_truth else "N/A")
    print(f"Mean purity:            {np.mean(purities):.3f}" if purities else "N/A")
    print(f"Mean completeness:      {np.mean(completenesses):.3f}" if completenesses else "N/A")
    print(f"Split truth clusters:   {n_split}")
    print(f"Merged reco clusters:   {n_merged}")

    # Save per-event CSV
    outpath = Path(args.output)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with open(outpath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nPer-event results saved to {outpath}")


if __name__ == "__main__":
    main()
