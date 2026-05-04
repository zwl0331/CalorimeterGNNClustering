"""Export train-split z-score statistics to a JSON sidecar for the C++ deployment.

The training pipeline writes `data/normalization_stats.pt` (a torch blob).
The C++ `art::EDProducer` consuming the `.onnx` should not need a LibTorch
dependency to read 28 floats — this script writes the same stats out as a
plain JSON file alongside the `.onnx`.

Default input/output match the v2_stage1 deployment artifact:

  python3 scripts/export_norm_stats.py
  python3 scripts/export_norm_stats.py --stats <pt-path> --output <json-path>

This is step 16a in docs/plan.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

# Names match the tables in docs/onnx_deployment.md §3. Order is the
# canonical feature index — the C++ side should assert against this.
NODE_FEATURE_NAMES = ["log_e", "t", "x", "y", "r", "e_rel"]
EDGE_FEATURE_NAMES = ["dx", "dy", "d", "dt", "dlog_e", "asym_e", "logsum_e", "dr"]


def stats_to_dict(stats: dict) -> dict:
    """Convert a torch-blob stats dict to a JSON-ready dict.

    Layout is deliberately flat and self-describing so a C++ JSON parser
    can pick out exactly the fields it needs.
    """
    node_mean = stats["node_mean"].tolist()
    node_std = stats["node_std"].tolist()
    edge_mean = stats["edge_mean"].tolist()
    edge_std = stats["edge_std"].tolist()

    if len(node_mean) != len(NODE_FEATURE_NAMES):
        raise ValueError(
            f"Expected {len(NODE_FEATURE_NAMES)} node features, got {len(node_mean)}"
        )
    if len(edge_mean) != len(EDGE_FEATURE_NAMES):
        raise ValueError(
            f"Expected {len(EDGE_FEATURE_NAMES)} edge features, got {len(edge_mean)}"
        )

    return {
        "schema_version": 1,
        "node_features": NODE_FEATURE_NAMES,
        "edge_features": EDGE_FEATURE_NAMES,
        "node_mean": node_mean,
        "node_std": node_std,
        "edge_mean": edge_mean,
        "edge_std": edge_std,
        "node_count": int(stats["node_count"]),
        "edge_count": int(stats["edge_count"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--stats", type=Path,
        default=Path("data/normalization_stats.pt"),
        help="Input torch blob with node/edge mean/std tensors.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("outputs/onnx/calo_cluster_net_v2_stage1.norm.json"),
        help="JSON sidecar destination (next to the .onnx).",
    )
    args = parser.parse_args()

    stats = torch.load(args.stats, weights_only=True)
    payload = stats_to_dict(stats)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    size = args.output.stat().st_size
    print(f"Read   {args.stats}")
    print(f"Wrote  {args.output}  ({size} bytes)")
    print(f"  node features: {payload['node_features']}")
    print(f"  edge features: {payload['edge_features']}")
    print(f"  node_count={payload['node_count']:,}  edge_count={payload['edge_count']:,}")


if __name__ == "__main__":
    main()
