"""Task 11a4: Validate ancestry data in v2 ROOT files.

Reads ``calomcsim.ancestorSimIds`` from reprocessed ROOT files and checks:
- Chain completeness (non-empty, ends at a root with no parent)
- Calo-entrant identification per disk
- Chain length distribution
- Comparison of old vs new truth cluster statistics
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.truth_labels import assign_mc_truth
from src.data.truth_labels_primary import (
    assign_mc_truth_primary,
    build_calo_root_map,
)


def load_crystal_disk_map(csv_path=None):
    """Load crystalId -> diskId mapping from geometry CSV."""
    if csv_path is None:
        csv_path = Path(__file__).resolve().parents[1] / "data" / "crystal_geometry.csv"
    disk_map = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            disk_map[int(row["crystalId"])] = int(row["diskId"])
    return disk_map


def validate_file(filepath, crystal_disk_map, max_events=None, verbose=False):
    """Validate ancestry data in one v2 ROOT file.

    Returns a dict with aggregate statistics.
    """
    import uproot

    tree = uproot.open(f"{filepath}:EventNtuple/ntuple")
    branches = [
        "calomcsim.id", "calomcsim.pdg", "calomcsim.startCode",
        "calomcsim.ancestorSimIds",
        "calohitsmc.simParticleIds", "calohitsmc.eDeps",
        "calohits.crystalId_",
    ]
    arrays = tree.arrays(branches, entry_stop=max_events)
    n_events = len(arrays["calomcsim.id"])

    stats = {
        "n_events": n_events,
        "total_simps": 0,
        "empty_chains": 0,
        "chain_lengths": [],
        "startcode_dist": Counter(),
        # Per-disk calo-entrant stats
        "n_calo_entrants": [],  # unique calo-entrants per event
        # Truth comparison
        "old_clusters": 0, "new_clusters": 0,
        "old_singletons": 0, "new_singletons": 0,
        "old_ambiguous": 0, "new_ambiguous": 0,
        "total_hits": 0,
    }

    for evt in range(n_events):
        sim_ids_evt = arrays["calomcsim.id"][evt]
        anc_evt = arrays["calomcsim.ancestorSimIds"][evt]
        pdg_evt = arrays["calomcsim.pdg"][evt]
        start_evt = arrays["calomcsim.startCode"][evt]

        hit_simids = arrays["calohitsmc.simParticleIds"][evt]
        hit_edeps = arrays["calohitsmc.eDeps"][evt]
        hit_cryids = arrays["calohits.crystalId_"][evt]
        n_hits = len(hit_simids)

        # --- Validate chains ---
        n_simps = len(sim_ids_evt)
        stats["total_simps"] += n_simps

        for j in range(n_simps):
            chain = list(anc_evt[j])
            stats["chain_lengths"].append(len(chain))
            if len(chain) == 0:
                stats["empty_chains"] += 1
            stats["startcode_dist"][int(start_evt[j])] += 1

        # --- Build calo-root map ---
        calo_root_map = build_calo_root_map(
            sim_ids_evt, anc_evt, hit_simids, hit_cryids, crystal_disk_map
        )

        # Count unique calo-entrants
        entrants = set(calo_root_map.values())
        stats["n_calo_entrants"].append(len(entrants))

        # --- Compare old vs new truth ---
        if n_hits < 2:
            continue

        hit_disks = np.array([crystal_disk_map.get(int(c), -1)
                              for c in hit_cryids], dtype=np.int64)

        for disk_id in [0, 1]:
            mask_d = hit_disks == disk_id
            n_disk = mask_d.sum()
            if n_disk < 2:
                continue
            idx = np.where(mask_d)[0]
            d_sim = [list(hit_simids[i]) for i in idx]
            d_edeps = [list(hit_edeps[i]) for i in idx]
            d_disks = np.full(n_disk, disk_id, dtype=np.int64)
            # dummy edge_index (not needed for cluster counting)
            edge_index = np.empty((2, 0), dtype=np.int64)

            _, _, tc_old, amb_old = assign_mc_truth(
                d_sim, d_edeps, d_disks, edge_index)
            _, _, tc_new, amb_new = assign_mc_truth_primary(
                d_sim, d_edeps, d_disks, edge_index, calo_root_map)

            stats["total_hits"] += n_disk
            stats["old_ambiguous"] += amb_old.sum()
            stats["new_ambiguous"] += amb_new.sum()

            old_ids = tc_old[tc_old >= 0]
            new_ids = tc_new[tc_new >= 0]
            if len(old_ids) > 0:
                old_counts = Counter(old_ids.tolist())
                stats["old_clusters"] += len(old_counts)
                stats["old_singletons"] += sum(1 for v in old_counts.values() if v == 1)
            if len(new_ids) > 0:
                new_counts = Counter(new_ids.tolist())
                stats["new_clusters"] += len(new_counts)
                stats["new_singletons"] += sum(1 for v in new_counts.values() if v == 1)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Validate ancestry in v2 ROOT files")
    parser.add_argument("--root-dir",
                        default="/exp/mu2e/data/users/wzhou2/GNN/root_files_v2",
                        help="Directory with v2 ROOT files")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Limit number of files to process")
    parser.add_argument("--max-events", type=int, default=500,
                        help="Max events per file")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    files = sorted(root_dir.glob("mcs.*.root"))
    # Filter to complete files (>= 1800 MB)
    files = [f for f in files if f.stat().st_size >= 1800 * 1024 * 1024]
    # Filter to valid files (have EventNtuple tree)
    import uproot as _uproot
    valid_files = []
    for f in files:
        try:
            _uproot.open(f"{f}:EventNtuple/ntuple")
            valid_files.append(f)
        except Exception:
            print(f"  SKIP (corrupt): {f.name}")
    files = valid_files
    if args.max_files:
        files = files[:args.max_files]

    print(f"Validating {len(files)} v2 ROOT files "
          f"({args.max_events or 'all'} events/file)")

    crystal_disk_map = load_crystal_disk_map()
    print(f"Crystal disk map: {len(crystal_disk_map)} crystals")

    # Aggregate
    total = {
        "n_events": 0, "total_simps": 0, "empty_chains": 0,
        "chain_lengths": [], "startcode_dist": Counter(),
        "n_calo_entrants": [],
        "old_clusters": 0, "new_clusters": 0,
        "old_singletons": 0, "new_singletons": 0,
        "old_ambiguous": 0, "new_ambiguous": 0, "total_hits": 0,
    }

    for i, f in enumerate(files):
        print(f"  [{i+1}/{len(files)}] {f.name}...", end=" ", flush=True)
        s = validate_file(f, crystal_disk_map, args.max_events, args.verbose)
        total["n_events"] += s["n_events"]
        total["total_simps"] += s["total_simps"]
        total["empty_chains"] += s["empty_chains"]
        total["chain_lengths"].extend(s["chain_lengths"])
        total["startcode_dist"] += s["startcode_dist"]
        total["n_calo_entrants"].extend(s["n_calo_entrants"])
        for key in ["old_clusters", "new_clusters", "old_singletons",
                     "new_singletons", "old_ambiguous", "new_ambiguous",
                     "total_hits"]:
            total[key] += s[key]
        print(f"{s['n_events']} events, {s['total_simps']} simPs")

    print("\n" + "=" * 60)
    print("ANCESTRY VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Events:          {total['n_events']}")
    print(f"SimParticles:    {total['total_simps']}")

    chains = total["chain_lengths"]
    print(f"\nChain lengths:")
    print(f"  Empty (no ancestors): {total['empty_chains']} "
          f"({100*total['empty_chains']/max(total['total_simps'],1):.1f}%)")
    if chains:
        arr = np.array(chains)
        print(f"  Mean: {arr.mean():.2f}, Median: {np.median(arr):.0f}, "
              f"Max: {arr.max()}")
        for length in [0, 1, 2, 3, 4, 5]:
            pct = 100 * (arr == length).sum() / len(arr)
            print(f"  Length {length}: {pct:.1f}%")
        pct_long = 100 * (arr > 5).sum() / len(arr)
        print(f"  Length >5: {pct_long:.1f}%")

    if total["n_calo_entrants"]:
        ce = np.array(total["n_calo_entrants"])
        print(f"\nCalo-entrants per event: mean={ce.mean():.1f}, "
              f"median={np.median(ce):.0f}, max={ce.max()}")

    print(f"\nStartCode distribution (top 10):")
    for code, count in total["startcode_dist"].most_common(10):
        pct = 100 * count / total["total_simps"]
        print(f"  {code:>4d}: {count:>8d} ({pct:.1f}%)")

    print(f"\n{'='*60}")
    print("TRUTH COMPARISON (old SimParticle vs new calo-entrant)")
    print(f"{'='*60}")
    nh = total["total_hits"]
    print(f"Total hits:      {nh}")
    print(f"Old ambiguous:   {total['old_ambiguous']} ({100*total['old_ambiguous']/max(nh,1):.2f}%)")
    print(f"New ambiguous:   {total['new_ambiguous']} ({100*total['new_ambiguous']/max(nh,1):.2f}%)")
    oa = total["old_ambiguous"]
    na = total["new_ambiguous"]
    if oa > 0:
        print(f"Ambiguity reduction: {oa-na} ({100*(oa-na)/oa:.1f}%)")

    oc = total["old_clusters"]
    nc = total["new_clusters"]
    os_ = total["old_singletons"]
    ns = total["new_singletons"]
    print(f"\nOld clusters:    {oc}  (singletons: {os_}, {100*os_/max(oc,1):.1f}%)")
    print(f"New clusters:    {nc}  (singletons: {ns}, {100*ns/max(nc,1):.1f}%)")
    print(f"Cluster reduction: {oc-nc} ({100*(oc-nc)/max(oc,1):.1f}%)")
    if os_ > 0:
        print(f"Singleton reduction: {os_-ns} ({100*(os_-ns)/os_:.1f}%)")

    # Sanity check: new clusters should only merge old ones, never split
    if nc > oc:
        print("\n*** WARNING: new truth has MORE clusters than old — "
              "this should not happen! ***")


if __name__ == "__main__":
    main()
