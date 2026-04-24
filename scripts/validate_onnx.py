"""Validate parity: PyTorch CaloClusterNetDeploy vs exported ONNX model.

Runs both runtimes over every non-trivial graph in data/processed/val.pt
and reports:

  - max and percentile edge-logit abs diffs across the whole val set
  - dynamic-axes coverage (smallest and largest (N, E) actually exercised)
  - per-edge threshold-decision flips at tau_edge (proxy for cluster
    assembly parity — if logits match and thresholds don't flip, the
    downstream deterministic post-processing agrees by construction)
  - per-graph timing for both runtimes on CPU

Exits 0 if max abs diff <= --tol (default 1e-5) AND no threshold flips.

This is step 15c in docs/plan.md. Prerequisites: 15a (deploy wrapper),
15b (export script already run so that the .onnx file exists).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import onnxruntime as ort
import torch

from src.models.calo_cluster_net_deploy import CaloClusterNetDeploy


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x.astype(np.float64)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("outputs/runs/calo_cluster_net_v2_stage1/checkpoints/best_model.pt"),
    )
    parser.add_argument(
        "--onnx", type=Path,
        default=Path("outputs/onnx/calo_cluster_net_v2_stage1.onnx"),
    )
    parser.add_argument(
        "--val-pt", type=Path, default=Path("data/processed/val.pt"),
    )
    parser.add_argument(
        "--tol", type=float, default=1e-5,
        help="Max allowable abs diff in edge logits. Default 1e-5.",
    )
    parser.add_argument(
        "--tau-edge", type=float, default=0.20,
        help="Edge threshold for the decision-flip proxy. "
             "Default 0.20 (v2_stage1 frozen value).",
    )
    parser.add_argument(
        "--n-graphs", type=int, default=None,
        help="Cap number of graphs to check. Default: all.",
    )
    args = parser.parse_args()

    if not args.onnx.exists():
        print(f"ONNX model not found: {args.onnx}", file=sys.stderr)
        print("Run scripts/export_onnx.py first (15b).", file=sys.stderr)
        return 2

    print(f"PyTorch checkpoint: {args.checkpoint}")
    model = CaloClusterNetDeploy.from_checkpoint(args.checkpoint)

    print(f"ONNX model:         {args.onnx}")
    sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])

    print(f"Val graphs:         {args.val_pt}")
    val = torch.load(args.val_pt, weights_only=False, map_location="cpu")
    total = len(val)
    if args.n_graphs:
        val = val[: args.n_graphs]
    print(f"Checking {len(val)} of {total} graphs  (tau_edge={args.tau_edge})")

    all_diffs: list[np.ndarray] = []
    per_graph_max: list[float] = []
    pytorch_ms: list[float] = []
    onnx_ms: list[float] = []
    ns: list[int] = []
    es: list[int] = []
    flips = 0
    n_edges_total = 0
    n_skipped = 0
    max_diff = 0.0

    for i, g in enumerate(val):
        E = int(g.edge_index.size(1))
        if E == 0:
            n_skipped += 1
            continue

        x_np = g.x.numpy()
        ei_np = g.edge_index.numpy()
        ea_np = g.edge_attr.numpy()

        t0 = time.perf_counter()
        with torch.no_grad():
            ref = model(g.x, g.edge_index, g.edge_attr).numpy()
        pytorch_ms.append((time.perf_counter() - t0) * 1e3)

        t0 = time.perf_counter()
        test = sess.run(
            ["edge_logits"],
            {"x": x_np, "edge_index": ei_np, "edge_attr": ea_np},
        )[0]
        onnx_ms.append((time.perf_counter() - t0) * 1e3)

        diff = np.abs(ref - test)
        all_diffs.append(diff)
        per_graph_max.append(float(diff.max()))
        max_diff = max(max_diff, float(diff.max()))

        # Threshold-decision agreement at tau_edge (proxy for cluster parity).
        ref_above = _sigmoid(ref) >= args.tau_edge
        test_above = _sigmoid(test) >= args.tau_edge
        flips += int((ref_above != test_above).sum())
        n_edges_total += E

        ns.append(int(g.x.size(0)))
        es.append(E)

        if (i + 1) % 500 == 0:
            print(f"  [{i + 1}/{len(val)}]  running max diff = {max_diff:.2e}  flips = {flips}")

    flat = np.concatenate(all_diffs) if all_diffs else np.array([0.0])
    n_checked = len(per_graph_max)

    print("\n=== Parity (edge logits) ===")
    print(f"Graphs checked:  {n_checked}  (skipped {n_skipped} with E=0)")
    print(f"Total edges:     {n_edges_total:,}")
    print(f"Max abs diff:    {max_diff:.3e}")
    print(f"Mean abs diff:   {flat.mean():.3e}")
    for p in (50, 90, 99, 99.9):
        print(f"p{p:<5g}abs diff:  {np.percentile(flat, p):.3e}")

    print("\n=== Dynamic-axes coverage ===")
    print(f"N (hits):   min={min(ns)}  max={max(ns)}  median={int(np.median(ns))}")
    print(f"E (edges):  min={min(es)}  max={max(es)}  median={int(np.median(es))}")

    print(f"\n=== Threshold decisions at tau={args.tau_edge} ===")
    flip_pct = 100.0 * flips / n_edges_total if n_edges_total else 0.0
    print(f"Flipped decisions: {flips} / {n_edges_total:,}  ({flip_pct:.4f}%)")

    print("\n=== CPU timing ===")
    py = np.array(pytorch_ms)
    on = np.array(onnx_ms)
    print(f"PyTorch  mean={py.mean():.2f} ms  median={np.median(py):.2f} ms  total={py.sum() / 1e3:.1f} s")
    print(f"ONNX RT  mean={on.mean():.2f} ms  median={np.median(on):.2f} ms  total={on.sum() / 1e3:.1f} s")
    if np.median(on) > 0:
        print(f"Speedup  (PyTorch / ONNX median): {np.median(py) / np.median(on):.2f}x")

    ok_logits = max_diff <= args.tol
    ok_flips = flips == 0
    print()
    if ok_logits and ok_flips:
        print(f"PASS: max diff {max_diff:.3e} <= {args.tol:.0e}, no threshold flips.")
        return 0
    if not ok_logits:
        print(f"FAIL: max abs diff {max_diff:.3e} > tol {args.tol:.0e}")
    if not ok_flips:
        print(f"FAIL: {flips} threshold flips at tau={args.tau_edge} "
              f"(expected 0 given matched logits)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
