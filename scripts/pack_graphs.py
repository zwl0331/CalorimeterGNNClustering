#!/usr/bin/env python3
"""
Pack individual .pt graph files into one file per split.

Reduces ~29K torch.load() calls to 1, eliminating NFS I/O bottleneck.

Usage:
    python3 scripts/pack_graphs.py
    python3 scripts/pack_graphs.py --splits train val  # specific splits

Output:
    data/processed/train.pt  (list of Data objects)
    data/processed/val.pt
    data/processed/test.pt
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml

from src.data.dataset import CaloGraphDataset


def pack_split(processed_dir, split_files, split_name):
    """Load all graphs for a split and save as a single list."""
    t0 = time.time()
    ds = CaloGraphDataset(processed_dir, file_list=split_files)
    n = len(ds)
    print(f"  {split_name}: {n} graphs, loading...", end=" ", flush=True)

    graphs = [ds.get(i) for i in range(n)]
    elapsed_load = time.time() - t0
    print(f"loaded in {elapsed_load:.1f}s,", end=" ", flush=True)

    out_path = Path(processed_dir) / f"{split_name}.pt"
    torch.save(graphs, out_path)
    size_mb = out_path.stat().st_size / 1e6
    print(f"saved {size_mb:.1f} MB to {out_path} ({time.time() - t0:.1f}s total)")
    return n


def main():
    parser = argparse.ArgumentParser(description="Pack graph files into split bundles")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    processed_dir = cfg["data"]["processed_dir"]

    for split in args.splits:
        split_path = cfg["data"]["splits"][split]
        with open(split_path) as f:
            file_list = [line.strip() for line in f if line.strip()]
        pack_split(processed_dir, file_list, split)

    print("\nDone. Update train_gnn.py to use packed=True or load directly:")
    print("  graphs = torch.load('data/processed/train.pt', weights_only=False)")


if __name__ == "__main__":
    main()
