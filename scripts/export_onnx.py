"""Export a trained CaloClusterNet to ONNX for C++ deployment.

Wraps the checkpoint in CaloClusterNetDeploy (edge head only, tensor I/O)
and traces it with torch.onnx.export using a real normalised val graph
as dummy input. Hit count N and edge count E are marked dynamic so the
exported graph handles any disk-graph size.

Defaults target the winning CCN+BFS10 checkpoint (v2_stage1).

  python3 scripts/export_onnx.py
  python3 scripts/export_onnx.py --checkpoint <path> --output <path>

This is step 15b in docs/plan.md. Parity validation against PyTorch
(step 15c) lives in a separate script.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.models.calo_cluster_net_deploy import CaloClusterNetDeploy


def pick_example_graph(val_pt_path: Path, min_edges: int = 20):
    """Return (x, edge_index, edge_attr) from the first non-trivial val graph.

    The tracer bakes in the example's shapes as the defaults for any
    static dimensions. We pass a graph with reasonable N and E so the
    trace doesn't happen to pick up degenerate shortcuts.
    """
    graphs = torch.load(val_pt_path, weights_only=False, map_location="cpu")
    for g in graphs:
        if g.edge_index.size(1) >= min_edges and g.x.size(0) >= 5:
            return g.x.contiguous(), g.edge_index.contiguous(), g.edge_attr.contiguous()
    raise RuntimeError(
        f"No graph with >= {min_edges} edges found in {val_pt_path}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("outputs/runs/calo_cluster_net_v2_stage1/checkpoints/best_model.pt"),
    )
    parser.add_argument(
        "--val-pt", type=Path, default=Path("data/processed/val.pt"),
        help="Packed val graphs for dummy input.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("outputs/onnx/calo_cluster_net_v2_stage1.onnx"),
    )
    parser.add_argument(
        "--opset", type=int, default=17,
        help="ONNX opset version. 17+ is supported by ONNX Runtime 1.17+.",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    model = CaloClusterNetDeploy.from_checkpoint(args.checkpoint)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model.__class__.__name__}  ({n_params:,} params, eval={not model.training})")

    print(f"Example input from: {args.val_pt}")
    x, ei, ea = pick_example_graph(args.val_pt)
    N, E = x.size(0), ei.size(1)
    print(f"  x          : {tuple(x.shape)}  dtype={x.dtype}")
    print(f"  edge_index : {tuple(ei.shape)}  dtype={ei.dtype}")
    print(f"  edge_attr  : {tuple(ea.shape)}  dtype={ea.dtype}")
    print(f"  (N={N}, E={E})")

    with torch.no_grad():
        out = model(x, ei, ea)
    print(
        f"  PyTorch edge_logits: shape={tuple(out.shape)}  "
        f"mean={out.mean().item():.4f}  std={out.std().item():.4f}  "
        f"min={out.min().item():.3f}  max={out.max().item():.3f}"
    )

    print(f"Exporting to: {args.output}  (opset {args.opset})")
    torch.onnx.export(
        model,
        (x, ei, ea),
        str(args.output),
        input_names=["x", "edge_index", "edge_attr"],
        output_names=["edge_logits"],
        dynamic_axes={
            "x": {0: "N"},
            "edge_index": {1: "E"},
            "edge_attr": {0: "E"},
            "edge_logits": {0: "E"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )
    size_mb = args.output.stat().st_size / (1024 * 1024)
    print(f"Wrote {args.output}  ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
