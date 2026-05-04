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

import onnx
import torch

from src.models.calo_cluster_net_deploy import CaloClusterNetDeploy
from src.models.simple_edge_net_deploy import SimpleEdgeNetDeploy

# Per-model presets — used when --model is one of the recognised values.
# Each entry pins the deploy wrapper, the run-dir checkpoint, the output
# filename, and the version string stamped into metadata_props.
MODEL_PRESETS = {
    "ccn": {
        "wrapper":       CaloClusterNetDeploy,
        "checkpoint":    Path("outputs/runs/calo_cluster_net_v2_stage1/checkpoints/best_model.pt"),
        "output":        Path("outputs/onnx/calo_cluster_net_v2_stage1.onnx"),
        "model_version": "calo-cluster-net-v2-stage1",
    },
    "sen": {
        "wrapper":       SimpleEdgeNetDeploy,
        "checkpoint":    Path("outputs/runs/simple_edge_net_v2/checkpoints/best_model.pt"),
        "output":        Path("outputs/onnx/simple_edge_net_v2.onnx"),
        "model_version": "simple-edge-net-v2",
    },
}

# Defaults stamped into the ONNX `metadata_props` map after export. The
# C++ deployment asserts these at session load (FHiCL passes the
# expected values); a mismatch aborts the job. Bump `model_version`
# on any layout-breaking change — new feature set, retrained weights,
# opset bump. Bump `node_features` / `edge_features` whenever the
# normalised feature columns change order or meaning, so the
# `CaloHitGraphMaker` can fail loudly instead of feeding scrambled
# tensors into the model.
DEFAULT_MODEL_VERSION = "calo-cluster-net-v2-stage1"
NODE_FEATURE_NAMES = ["log_e", "t", "x", "y", "r", "e_rel"]
EDGE_FEATURE_NAMES = ["dx", "dy", "d", "dt", "dlog_e", "asym_e", "logsum_e", "dr"]


def stamp_metadata_props(onnx_path: Path, version: str,
                         node_features: list[str], edge_features: list[str]) -> None:
    """Set the deployment-contract entries on an ONNX file.

    Idempotent for the keys we own: removes any existing entries with
    the same keys before appending. Other entries (e.g. PyTorch's own
    producer info) are preserved.
    """
    keys_we_own = {"model_version", "node_features", "edge_features"}
    m = onnx.load(str(onnx_path))
    keep = [p for p in m.metadata_props if p.key not in keys_we_own]
    del m.metadata_props[:]
    m.metadata_props.extend(keep)
    for key, value in (
        ("model_version", version),
        ("node_features", ",".join(node_features)),
        ("edge_features", ",".join(edge_features)),
    ):
        e = m.metadata_props.add()
        e.key = key
        e.value = value
    onnx.save(m, str(onnx_path))


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
        "--model", choices=sorted(MODEL_PRESETS), default="ccn",
        help="Which model to export. 'ccn' = CaloClusterNet (default); "
             "'sen' = SimpleEdgeNet.",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="Checkpoint path; defaults to the preset for --model.",
    )
    parser.add_argument(
        "--val-pt", type=Path, default=Path("data/processed/val.pt"),
        help="Packed val graphs for dummy input.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output .onnx path; defaults to the preset for --model.",
    )
    parser.add_argument(
        "--opset", type=int, default=17,
        help="ONNX opset version. 17+ is supported by ONNX Runtime 1.17+.",
    )
    parser.add_argument(
        "--model-version", type=str, default=None,
        help="Stamped into ONNX metadata_props['model_version']. "
             "Defaults to the preset for --model. C++ asserts this at "
             "session load via FHiCL.",
    )
    args = parser.parse_args()

    preset = MODEL_PRESETS[args.model]
    if args.checkpoint    is None: args.checkpoint    = preset["checkpoint"]
    if args.output        is None: args.output        = preset["output"]
    if args.model_version is None: args.model_version = preset["model_version"]
    Wrapper = preset["wrapper"]

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}  (model={args.model})")
    model = Wrapper.from_checkpoint(args.checkpoint)
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
    stamp_metadata_props(
        args.output, args.model_version,
        NODE_FEATURE_NAMES, EDGE_FEATURE_NAMES,
    )
    size_mb = args.output.stat().st_size / (1024 * 1024)
    print(f"Wrote {args.output}  ({size_mb:.2f} MB)")
    print(f"  metadata_props.model_version  = {args.model_version!r}")
    print(f"  metadata_props.node_features  = {','.join(NODE_FEATURE_NAMES)!r}")
    print(f"  metadata_props.edge_features  = {','.join(EDGE_FEATURE_NAMES)!r}")


if __name__ == "__main__":
    main()
