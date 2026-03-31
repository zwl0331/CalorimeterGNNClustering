"""
Extract per-cluster detail from BFS reco clusters in MDC2025-002.

Saves: data/baseline/bfs_cluster_summary.csv with columns:
  file_idx, event_idx, cluster_idx, disk, energy, time, cog_x, cog_y, cog_z,
  n_hits, mc_purity, mc_completeness, dominant_simid

Usage:
    source setup_env.sh
    python3 scripts/baseline_cluster_detail.py [--n-files 3] [--n-events 5000]
"""

import argparse
import csv
import glob
import sys
import time
from collections import defaultdict
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import awkward as ak
import numpy as np
import uproot


def get_mc_truth_files():
    """Get MDC2025-002 file list."""
    pattern = "/pnfs/mu2e/tape/phy-nts/nts/mu2e/FlateMinusMix1BBTriggered/MDC2025-002/root/*/*/*.root"
    return sorted(glob.glob(pattern))


def process_file(filepath, file_idx):
    """Extract per-cluster detail from one ROOT file."""
    tree = uproot.open(filepath + ":EventNtuple/ntuple")

    branches = [
        # Cluster info
        "caloclusters.diskID_",
        "caloclusters.energyDep_",
        "caloclusters.time_",
        "caloclusters.cog_.fCoordinates.fX",
        "caloclusters.cog_.fCoordinates.fY",
        "caloclusters.cog_.fCoordinates.fZ",
        "caloclusters.size_",
        "caloclusters.hits_",
        # Hit info
        "calohits.crystalId_",
        "calohits.eDep_",
        "calohits.clusterIdx_",
        # MC truth
        "calohitsmc.simParticleIds",
        "calohitsmc.eDeps",
    ]

    data = tree.arrays(branches)
    n_events = len(data)

    from src.geometry.crystal_geometry import load_crystal_map
    crystal_map = load_crystal_map()

    rows = []
    for ev in range(n_events):
        n_clusters = len(data["caloclusters.diskID_"][ev])
        if n_clusters == 0:
            continue

        hit_energies = np.array(data["calohits.eDep_"][ev])
        hit_cryids = np.array(data["calohits.crystalId_"][ev])
        hit_clusteridx = np.array(data["calohits.clusterIdx_"][ev])
        hit_simids = data["calohitsmc.simParticleIds"][ev]
        hit_edeps_mc = data["calohitsmc.eDeps"][ev]

        # Get disk per hit
        hit_disks = np.array([crystal_map[int(c)][0] if int(c) in crystal_map else -1
                              for c in hit_cryids])

        # Build per-hit dominant SimParticle
        nhits = len(hit_energies)
        dom_simid = np.full(nhits, -1, dtype=np.int64)
        dom_frac = np.zeros(nhits)
        for i in range(nhits):
            simids = np.array(hit_simids[i])
            edeps = np.array(hit_edeps_mc[i])
            if len(simids) == 0:
                continue
            total = edeps.sum()
            if total <= 0:
                continue
            best = np.argmax(edeps)
            dom_simid[i] = simids[best]
            dom_frac[i] = edeps[best] / total

        # Build truth clusters: (disk, dominant_simid) → truth cluster label
        truth_labels = np.full(nhits, -1, dtype=np.int64)
        cluster_map = {}
        next_label = 0
        for i in range(nhits):
            if dom_frac[i] < 0.7 or dom_simid[i] < 0:
                continue
            key = (int(hit_disks[i]), int(dom_simid[i]))
            if key not in cluster_map:
                cluster_map[key] = next_label
                next_label += 1
            truth_labels[i] = cluster_map[key]

        # Build energy-per-truth-cluster
        truth_energy = defaultdict(float)
        for i in range(nhits):
            if truth_labels[i] >= 0:
                truth_energy[truth_labels[i]] += hit_energies[i]

        # Per reco cluster
        for ci in range(n_clusters):
            disk = int(data["caloclusters.diskID_"][ev][ci])
            energy = float(data["caloclusters.energyDep_"][ev][ci])
            t = float(data["caloclusters.time_"][ev][ci])
            cog_x = float(data["caloclusters.cog_.fCoordinates.fX"][ev][ci])
            cog_y = float(data["caloclusters.cog_.fCoordinates.fY"][ev][ci])
            cog_z = float(data["caloclusters.cog_.fCoordinates.fZ"][ev][ci])
            n_hits_cl = int(data["caloclusters.size_"][ev][ci])

            # Find hits in this cluster
            hit_mask = hit_clusteridx == ci
            cl_energies = hit_energies[hit_mask]
            cl_truth = truth_labels[hit_mask]
            cl_dom_simid = dom_simid[hit_mask]

            # Purity: fraction of reco cluster energy from dominant truth cluster
            if cl_energies.sum() > 0 and len(cl_truth) > 0:
                # Count energy per truth cluster in this reco cluster
                truth_contrib = defaultdict(float)
                for j in range(len(cl_energies)):
                    tl = cl_truth[j]
                    if tl >= 0:
                        truth_contrib[tl] += cl_energies[j]
                if truth_contrib:
                    best_truth = max(truth_contrib, key=truth_contrib.get)
                    purity = truth_contrib[best_truth] / cl_energies.sum()
                    completeness = (truth_contrib[best_truth] / truth_energy[best_truth]
                                    if truth_energy[best_truth] > 0 else 0)
                else:
                    purity = 0.0
                    completeness = 0.0
                    best_truth = -1
            else:
                purity = 0.0
                completeness = 0.0
                best_truth = -1

            # Dominant SimParticle in this reco cluster (by energy)
            simid_energy = defaultdict(float)
            for j in range(len(cl_energies)):
                sid = cl_dom_simid[j]
                if sid >= 0:
                    simid_energy[sid] += cl_energies[j]
            dominant_sim = max(simid_energy, key=simid_energy.get) if simid_energy else -1

            rows.append({
                "file_idx": file_idx,
                "event_idx": ev,
                "cluster_idx": ci,
                "disk": disk,
                "energy": round(energy, 4),
                "time": round(t, 4),
                "cog_x": round(cog_x, 2),
                "cog_y": round(cog_y, 2),
                "cog_z": round(cog_z, 2),
                "n_hits": n_hits_cl,
                "mc_purity": round(purity, 4),
                "mc_completeness": round(completeness, 4),
                "dominant_simid": int(dominant_sim),
            })

    return rows


def main():
    parser = argparse.ArgumentParser(description="Extract per-cluster BFS detail")
    parser.add_argument("--n-files", type=int, default=3)
    parser.add_argument("--n-events", type=int, default=None)
    parser.add_argument("--output", type=str, default="data/baseline/bfs_cluster_summary.csv")
    args = parser.parse_args()

    files = get_mc_truth_files()
    if not files:
        print("ERROR: No MDC2025-002 files found.", file=sys.stderr)
        sys.exit(1)

    files = files[:args.n_files]
    print(f"Processing {len(files)} files...")

    all_rows = []
    t0 = time.time()
    for i, f in enumerate(files):
        print(f"  [{i+1}/{len(files)}] {Path(f).name}...", end=" ", flush=True)
        ft0 = time.time()
        rows = process_file(f, file_idx=i)
        all_rows.extend(rows)
        print(f"{len(rows)} clusters ({time.time()-ft0:.1f}s)")

    elapsed = time.time() - t0
    print(f"\nExtracted {len(all_rows)} clusters from {len(files)} files in {elapsed:.1f}s")

    if not all_rows:
        print("No clusters found.")
        return

    outpath = Path(args.output)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Saved to {outpath}")

    # Quick summary
    energies = [r["energy"] for r in all_rows]
    purities = [r["mc_purity"] for r in all_rows]
    nhits = [r["n_hits"] for r in all_rows]
    print(f"\nPer-cluster summary:")
    print(f"  Energy: mean={np.mean(energies):.1f} MeV, median={np.median(energies):.1f} MeV")
    print(f"  Purity: mean={np.mean(purities):.3f}, median={np.median(purities):.3f}")
    print(f"  Hits:   mean={np.mean(nhits):.1f}, median={np.median(nhits):.0f}")


if __name__ == "__main__":
    main()
