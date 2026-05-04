"""Stage-3 parity check for the C++ GNN clustering pipeline.

Reads the TTree produced by Offline/CaloCluster/.../CaloHitGraphParityDump
(which carries per-event-disk CaloHit metadata + the C++ GNN cluster
labels), replays each disk-graph through the Python pipeline (graph
builder + CaloClusterNet PyTorch model + cluster_reco with the frozen
CCN+BFS10 recipe), and asserts byte-identical cluster labels.

Usage:

  source setup_env.sh
  mu2e -c Offline/CaloCluster/fcl/from_mcs-gnn-test.fcl \
       -s <mcs.art> -n 100 -T parity_dump.root        (in working_dir)
  python3 scripts/compare_parity_dump.py parity_dump.root

The Python labels are compared up to permutation of cluster IDs (Python
and C++ may pick different IDs for the same partition; what matters is
that two hits in the same Python cluster are also in the same C++
cluster, and vice versa). Equivalently: the partitioning of node
indices into clusters must match. -1 (dropped) is treated as its own
"cluster" for this comparison.

Task 16g, Stage 3.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import uproot

from src.data.graph_builder import (
    build_graph,
    compute_edge_features,
    compute_node_features,
)
from src.data.normalization import load_stats
from src.geometry.crystal_geometry import load_crystal_map
from src.inference.cluster_reco import reconstruct_clusters
from src.models.calo_cluster_net_deploy import CaloClusterNetDeploy


TAU_EDGE       = 0.20
BFS_EXPAND_CUT = 10.0
MIN_HITS       = 2
MIN_ENERGY_MEV = 10.0
R_MAX          = 210.0
DT_MAX         = 25.0
K_MIN          = 3
K_MAX          = 20


def _canonicalise_labels(labels: np.ndarray) -> np.ndarray:
    """Re-map cluster IDs to first-appearance order so two label
    vectors that describe the same partition compare equal element-wise.
    -1 stays -1.
    """
    out = np.full_like(labels, -1)
    nxt = 0
    seen: dict[int, int] = {}
    for i, lab in enumerate(labels):
        if lab < 0:
            continue
        if lab not in seen:
            seen[lab] = nxt
            nxt += 1
        out[i] = seen[lab]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dump_root", type=Path,
                        help="parity_dump.root produced by from_mcs-gnn-test.fcl")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("outputs/runs/calo_cluster_net_v2_stage1/checkpoints/best_model.pt"))
    parser.add_argument("--norm-stats", type=Path,
                        default=Path("data/normalization_stats.pt"))
    parser.add_argument("--crystal-map", type=Path,
                        default=Path("data/crystal_geometry.csv"))
    parser.add_argument("--max-disagree", type=int, default=10,
                        help="Print details for up to this many mismatched disk-graphs.")
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    model = CaloClusterNetDeploy.from_checkpoint(args.checkpoint).eval()
    norm = load_stats(args.norm_stats)
    crystal_map = load_crystal_map(args.crystal_map)
    # load_crystal_map returns dict[int, tuple[disk, x, y]].
    pos_lookup = {int(cid): (float(v[1]), float(v[2]))
                  for cid, v in crystal_map.items()}

    print(f"Reading dump: {args.dump_root}")
    f = uproot.open(args.dump_root)
    t = f["parityDump/parity"]
    arrays = t.arrays([
        "eventID", "diskID", "nHits",
        "crystalID", "time_ns", "eDep_MeV", "gnnLabel",
    ], library="np")
    n_entries = t.num_entries
    print(f"  entries: {n_entries}")

    n_match = 0
    n_disagree = 0
    n_disagree_printed = 0
    total_hits = 0
    mismatched_hits = 0

    for i in range(n_entries):
        eid       = int(arrays["eventID"][i])
        disk      = int(arrays["diskID"][i])
        crystals  = arrays["crystalID"][i]
        times     = arrays["time_ns"][i].astype(np.float64)
        energies  = arrays["eDep_MeV"][i].astype(np.float64)
        cpp_labels = arrays["gnnLabel"][i].astype(np.int64)

        n = len(crystals)
        total_hits += n
        if n == 0:
            continue

        # Reconstruct positions from crystalID via the geometry map.
        positions = np.array(
            [pos_lookup[int(c)] for c in crystals],
            dtype=np.float64,
        )

        # Graph + features.
        edge_index, _ = build_graph(positions, times, r_max=R_MAX,
                                    dt_max=DT_MAX, k_min=K_MIN, k_max=K_MAX)
        node_feat = compute_node_features(positions, times, energies)
        edge_feat = compute_edge_features(positions, times, energies, edge_index)
        node_mean = np.asarray(norm["node_mean"], dtype=np.float32)
        node_std  = np.asarray(norm["node_std"],  dtype=np.float32)
        edge_mean = np.asarray(norm["edge_mean"], dtype=np.float32)
        edge_std  = np.asarray(norm["edge_std"],  dtype=np.float32)
        x  = (node_feat - node_mean) / node_std
        ea = (edge_feat - edge_mean) / edge_std

        x_t  = torch.from_numpy(np.asarray(x,  dtype=np.float32))
        ei_t = torch.from_numpy(np.asarray(edge_index, dtype=np.int64))
        ea_t = torch.from_numpy(np.asarray(ea, dtype=np.float32))

        with torch.no_grad():
            edge_logits = model(x_t, ei_t, ea_t).numpy()

        py_labels, _ = reconstruct_clusters(
            edge_index=edge_index,
            edge_logits=torch.from_numpy(edge_logits),
            n_nodes=n,
            energies=energies.astype(np.float32),
            tau_edge=TAU_EDGE,
            min_hits=MIN_HITS,
            min_energy_mev=MIN_ENERGY_MEV,
            symmetrize=True,
            bfs_expand_cut=BFS_EXPAND_CUT,
        )
        py_labels = py_labels.astype(np.int64)

        # Canonicalise both label arrays to first-appearance order so
        # the comparison is on the partition, not the assigned IDs.
        py_canon  = _canonicalise_labels(py_labels)
        cpp_canon = _canonicalise_labels(cpp_labels)

        if not np.array_equal(py_canon, cpp_canon):
            n_disagree += 1
            diffs = int((py_canon != cpp_canon).sum())
            mismatched_hits += diffs
            if n_disagree_printed < args.max_disagree:
                n_disagree_printed += 1
                print(f"  [diff] eid={eid} disk={disk} n={n}: {diffs} hits differ")
                print(f"         py : {py_canon.tolist()}")
                print(f"         cpp: {cpp_canon.tolist()}")
        else:
            n_match += 1

    print()
    print("=== Summary ===")
    print(f"disk-graphs:        {n_entries}")
    print(f"matched:            {n_match}")
    print(f"disagree:           {n_disagree}")
    print(f"hits compared:      {total_hits}")
    print(f"mismatched hits:    {mismatched_hits}")
    if n_disagree == 0:
        print("[PASS] all disk-graphs match Python cluster_labels byte-exactly")
        return 0
    else:
        print("[FAIL] cluster-label parity broken")
        return 1


if __name__ == "__main__":
    sys.exit(main())
