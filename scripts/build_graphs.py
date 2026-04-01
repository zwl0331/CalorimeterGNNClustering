"""
Build per-disk PyG graph files from EventNtuple ROOT files.

Reads ROOT files listed in a split file, extracts per-disk graphs,
and saves them as .pt files to data/processed/.

Usage:
    source setup_env.sh
    python3 scripts/build_graphs.py --split train --n-files 5
    python3 scripts/build_graphs.py --split train   # all files in split
    python3 scripts/build_graphs.py --split all --n-files 3  # ignore splits, use first N files
    python3 scripts/build_graphs.py --split train --root-dir /exp/mu2e/data/users/wzhou2/GNN/root_files
"""

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import yaml

from src.data.dataset import extract_events_from_file
from src.data.normalization import compute_normalization_stats, save_stats
from src.geometry.crystal_geometry import load_crystal_map


def load_file_list(split_name, config):
    """Load file list for a given split."""
    if split_name == "all":
        import glob
        pattern = "/pnfs/mu2e/tape/phy-nts/nts/mu2e/FlateMinusMix1BBTriggered/MDC2025-002/root/*/*/*.root"
        return sorted(glob.glob(pattern))

    split_path = Path(config["data"]["splits"][split_name])
    if not split_path.exists():
        print(f"ERROR: Split file not found: {split_path}", file=sys.stderr)
        sys.exit(1)
    with open(split_path) as f:
        return [line.strip() for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Build PyG graphs from ROOT files")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "val", "test", "all"],
                        help="Which split to process")
    parser.add_argument("--n-files", type=int, default=None,
                        help="Max number of files to process")
    parser.add_argument("--n-events", type=int, default=None,
                        help="Max events per file")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Config file path")
    parser.add_argument("--root-dir", type=str, default=None,
                        help="Local directory containing ROOT files (remaps split paths)")
    parser.add_argument("--compute-norm", action="store_true",
                        help="Compute normalization stats (train split only)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    files = load_file_list(args.split, config)
    if args.root_dir:
        root_dir = Path(args.root_dir)
        files = [str(root_dir / Path(f).name) for f in files]
    if args.n_files:
        files = files[:args.n_files]

    print(f"Processing {len(files)} files from '{args.split}' split")

    crystal_map = load_crystal_map(config["data"]["crystal_geometry"])
    graph_cfg = config["graph"]
    truth_mode = config["data"].get("truth_mode", "bfs_pseudo")
    out_dir = Path(config["data"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    total_graphs = 0
    total_nodes = 0
    total_edges = 0
    all_diags = []
    saved_paths = []

    t0 = time.time()
    for fi, filepath in enumerate(files):
        fname = Path(filepath).stem
        print(f"  [{fi+1}/{len(files)}] {Path(filepath).name}...", end=" ", flush=True)
        ft0 = time.time()
        file_graphs = 0

        for data, ev_idx, disk_id, diag in extract_events_from_file(
            filepath, crystal_map, graph_cfg,
            truth_mode=truth_mode, max_events=args.n_events,
        ):
            out_name = f"{fname}_evt{ev_idx:06d}_disk{disk_id}.pt"
            out_path = out_dir / out_name
            torch.save(data, out_path)
            saved_paths.append(out_path)

            total_graphs += 1
            total_nodes += diag["n_nodes"]
            total_edges += diag["n_edges"]
            all_diags.append(diag)
            file_graphs += 1

        dt = time.time() - ft0
        print(f"{file_graphs} graphs ({dt:.1f}s)")

    elapsed = time.time() - t0
    print(f"\nDone: {total_graphs} graphs from {len(files)} files in {elapsed:.1f}s")
    print(f"  Total nodes: {total_nodes}, Total edges: {total_edges}")
    if all_diags:
        avg_deg = np.mean([d["avg_degree"] for d in all_diags])
        iso = sum(d["n_isolated"] for d in all_diags)
        print(f"  Avg degree: {avg_deg:.1f}, Isolated nodes: {iso}")

    # Save diagnostics
    diag_path = out_dir / f"diagnostics_{args.split}.csv"
    if all_diags:
        with open(diag_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_diags[0].keys())
            writer.writeheader()
            writer.writerows(all_diags)
        print(f"  Diagnostics: {diag_path}")

    # Compute normalization stats if requested (train only)
    if args.compute_norm:
        if args.split != "train":
            print("WARNING: Normalization stats should only be computed from train split!")
        print("\nComputing normalization stats...")
        graphs = [torch.load(p, weights_only=False) for p in saved_paths]
        stats = compute_normalization_stats(graphs)
        save_stats(stats, config["data"]["normalization_stats"])


if __name__ == "__main__":
    main()
