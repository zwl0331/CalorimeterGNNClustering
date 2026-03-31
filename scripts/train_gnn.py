#!/usr/bin/env python3
"""
Train a GNN for calorimeter edge classification.

Usage:
    python3 scripts/train_gnn.py --config configs/default.yaml
    python3 scripts/train_gnn.py --config configs/default.yaml --epochs 20 --device cpu
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml

from src.data.dataset import CaloGraphDataset
from src.data.normalization import load_stats, normalize_graph
from src.models.simple_edge_net import SimpleEdgeNet
from src.training.losses import compute_class_weights
from src.training.trainer import Trainer


def load_split_files(split_path):
    """Load file stems from a split file."""
    with open(split_path) as f:
        return [line.strip() for line in f if line.strip()]


def get_git_hash():
    """Get current git hash, or 'unknown'."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def build_model(cfg):
    """Instantiate model from config."""
    model_cfg = cfg["model"]
    name = model_cfg.get("name", "SimpleEdgeNet")

    if name == "SimpleEdgeNet":
        return SimpleEdgeNet(
            node_dim=6,
            edge_dim=8,
            hidden_dim=model_cfg.get("hidden_dim", 64),
            n_mp_layers=model_cfg.get("n_mp_layers", 3),
            dropout=model_cfg.get("dropout", 0.1),
        )
    else:
        raise ValueError(f"Unknown model: {name}")


def main():
    parser = argparse.ArgumentParser(description="Train GNN for calorimeter clustering")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: 'cpu', 'cuda', or 'cuda:0'. Auto-detects if omitted.")
    parser.add_argument("--epochs", type=int, default=None, help="Override max epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--run-name", type=str, default=None, help="Run directory name")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg["train"]
    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Run directory
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg["output"]["run_dir"]) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config + metadata
    meta = {
        "git_hash": get_git_hash(),
        "config_path": args.config,
        "device": str(device),
        "timestamp": datetime.now().isoformat(),
    }
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Load normalization stats
    stats_path = cfg["data"]["normalization_stats"]
    print(f"Loading normalization stats from {stats_path}")
    stats = load_stats(stats_path)

    # Normalization transform
    def norm_transform(data):
        return normalize_graph(data, stats)

    # Load datasets
    processed_dir = cfg["data"]["processed_dir"]
    train_files = load_split_files(cfg["data"]["splits"]["train"])
    val_files = load_split_files(cfg["data"]["splits"]["val"])

    print(f"Loading train dataset from {processed_dir}")
    train_dataset = CaloGraphDataset(processed_dir, file_list=train_files,
                                     transform=norm_transform)
    print(f"Loading val dataset from {processed_dir}")
    val_dataset = CaloGraphDataset(processed_dir, file_list=val_files,
                                   transform=norm_transform)

    print(f"  Train: {len(train_dataset)} graphs")
    print(f"  Val:   {len(val_dataset)} graphs")

    if len(train_dataset) == 0:
        print("ERROR: No training graphs found. Run build_graphs.py first.")
        sys.exit(1)

    # Compute class weights from training set (unnormalized for label counting)
    train_raw = CaloGraphDataset(processed_dir, file_list=train_files)
    cw = compute_class_weights(train_raw)
    pos_weight = cw["pos_weight"]
    print(f"  Class balance: {cw['n_pos']} pos, {cw['n_neg']} neg "
          f"(pos_weight={pos_weight.item():.3f})")

    # Build model
    model = build_model(cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {cfg['model']['name']}")
    print(f"  Parameters: {n_params:,}")
    print(f"  Hidden dim: {cfg['model']['hidden_dim']}")
    print(f"  MP layers:  {cfg['model']['n_mp_layers']}")

    # Train
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        cfg=train_cfg,
        pos_weight=pos_weight,
        device=device,
        run_dir=run_dir,
    )

    print("\n" + "=" * 70)
    trainer.fit()
    print("=" * 70)


if __name__ == "__main__":
    main()
