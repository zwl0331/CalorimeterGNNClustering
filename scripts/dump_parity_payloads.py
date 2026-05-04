"""Dump per-graph payloads for the C++ parity tests (Task 16g).

For each of the first --n-graphs non-trivial val graphs, writes a JSON
record with:

  - n_nodes, n_edges
  - edge_index   (flat 2*E int64)
  - edge_logits  (E floats from PyTorch CaloClusterNetDeploy)
  - energies     (N raw MeV, recovered from log_e in the packed graph
                  via expm1; matches the per-hit energies a real C++
                  CaloHitGraphMaker would pass into the assembler)
  - cluster_labels (N int64 — Python reference labels from
                    cluster_reco.reconstruct_clusters with the frozen
                    CCN+BFS10 recipe)

The C++ test (Offline/CaloCluster/test/test_GnnClusterAssembler.cc)
reads this file, replays GnnClusterAssembler::assemble on each record,
and asserts byte-identical cluster_labels.

Usage:

  python3 scripts/dump_parity_payloads.py
  python3 scripts/dump_parity_payloads.py --n-graphs 200 --output <path>

Step 16g (Stage 2: cluster-maker parity).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.inference.cluster_reco import reconstruct_clusters
from src.models.calo_cluster_net_deploy import CaloClusterNetDeploy


# Recipe values must match the C++ FHiCL defaults / per-instance values
# in Offline/CaloCluster/src/CaloClusterMakerGNN_module.cc.
TAU_EDGE_CCN     = 0.20
BFS_EXPAND_CUT   = 10.0
MIN_HITS         = 2
MIN_ENERGY_MEV   = 10.0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("outputs/runs/calo_cluster_net_v2_stage1/checkpoints/best_model.pt"),
    )
    parser.add_argument(
        "--val-pt", type=Path, default=Path("data/processed/val.pt"),
    )
    parser.add_argument(
        "--n-graphs", type=int, default=100,
        help="Number of non-trivial val graphs to dump (smallest disks first).",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("tests/parity/calo_cluster_net_v2_stage1.parity.json"),
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    model = CaloClusterNetDeploy.from_checkpoint(args.checkpoint).eval()
    print(f"Loading val graphs: {args.val_pt}")
    graphs = torch.load(args.val_pt, weights_only=False, map_location="cpu")

    payloads = []
    n_seen = 0
    n_dumped = 0
    while n_dumped < args.n_graphs and n_seen < len(graphs):
        g = graphs[n_seen]
        n_seen += 1
        N = int(g.x.size(0))
        E = int(g.edge_index.size(1))
        if N < 2 or E < 2:
            continue

        with torch.no_grad():
            edge_logits = model(g.x, g.edge_index, g.edge_attr).numpy()

        # Recover raw MeV energies from the normalised log(1+E) column.
        # The packed graph carries z-scored x, but column 0 is log_e
        # before z-scoring is applied — so we have to invert with the
        # train-split node_mean/std from data/normalization_stats.pt.
        # Simpler: we'd ideally read the un-normalised energies from the
        # ROOT file; here we just store what the test consumer needs to
        # reproduce the assembler call exactly. The assembler only uses
        # raw MeV for seed-order, expand-cut, and min-energy cleanup, so
        # dump from the original raw column.
        # The packed val graphs store `x[:, 0] = log1p(e_raw)` BEFORE
        # normalisation only for v1; v2 packs are post-normalisation, so
        # we instead recover from a parallel raw-energy field. The
        # CaloGraphDataset packs already include `hit_truth_cluster` and
        # `n_hits` but not raw energies. Pull from x_raw if available;
        # otherwise reconstruct via z-score inversion.
        if hasattr(g, "x_raw") and g.x_raw is not None:
            energies_mev = g.x_raw[:, 0].numpy().astype(np.float32)
        else:
            stats = torch.load("data/normalization_stats.pt",
                               weights_only=True, map_location="cpu")
            mu0  = float(stats["node_mean"][0].item())
            sig0 = float(stats["node_std"][0].item())
            log1p_e = g.x[:, 0].numpy().astype(np.float64) * sig0 + mu0
            energies_mev = (np.exp(log1p_e) - 1.0).astype(np.float32)

        cluster_labels, _ = reconstruct_clusters(
            edge_index=g.edge_index,
            edge_logits=torch.from_numpy(edge_logits),
            n_nodes=N,
            energies=energies_mev,
            tau_edge=TAU_EDGE_CCN,
            min_hits=MIN_HITS,
            min_energy_mev=MIN_ENERGY_MEV,
            symmetrize=True,
            bfs_expand_cut=BFS_EXPAND_CUT,
        )

        # Flatten edge_index in row-major (src row first, then dst row)
        # — exact layout the C++ CaloHitGraph carries.
        ei_flat = (g.edge_index.numpy()
                   .astype(np.int64)
                   .reshape(2 * E)
                   .tolist())

        payloads.append({
            "n_nodes":      N,
            "n_edges":      E,
            "edge_index":   ei_flat,
            "edge_logits":  edge_logits.astype(np.float32).tolist(),
            "energies":     energies_mev.tolist(),
            "cluster_labels": cluster_labels.astype(np.int64).tolist(),
        })
        n_dumped += 1

    record = {
        "schema_version": 1,
        "model_version":  "calo-cluster-net-v2-stage1",
        "tau_edge":       TAU_EDGE_CCN,
        "bfs_expand_cut": BFS_EXPAND_CUT,
        "min_hits":       MIN_HITS,
        "min_energy_mev": MIN_ENERGY_MEV,
        "graphs":         payloads,
    }

    with args.output.open("w") as f:
        json.dump(record, f)

    size = args.output.stat().st_size
    print(f"Wrote {args.output}  ({size:,} bytes)")
    print(f"  n_graphs = {len(payloads)}")
    print(f"  total nodes = {sum(p['n_nodes'] for p in payloads):,}")
    print(f"  total edges = {sum(p['n_edges'] for p in payloads):,}")


if __name__ == "__main__":
    main()
