"""
CaloClusterNetDeploy — inference-only wrapper for ONNX export.

CaloClusterNet was designed for multi-task training: it takes a PyG
``Data`` object and returns a dict ``{edge_logits, node_logits}``. Neither
is friendly for ``torch.onnx.export``:

  - ``Data`` objects can't be traced; exporters need plain tensors in.
  - The node-saliency head is unused in the deployed recipe (CCN+BFS10
    uses only edge logits), and in the ``v2_stage1`` checkpoint the
    node head was trained with ``lambda_node=0`` — its weights are
    never-supervised noise that would only confuse a C++ caller.

This wrapper composes a trained ``CaloClusterNet`` but exposes only the
edge path with a tensor-in / tensor-out forward signature.
"""

from pathlib import Path

import torch
import torch.nn as nn
import yaml

from src.models import build_model
from src.models.calo_cluster_net import CaloClusterNet


class CaloClusterNetDeploy(nn.Module):
    """Edge-only inference wrapper around a trained ``CaloClusterNet``.

    Reuses the submodules of ``full_model`` by reference (no weight copy),
    so the wrapper stays in sync if ``full_model`` is moved to another
    device. The node-saliency head is omitted entirely.
    """

    def __init__(self, full_model: CaloClusterNet):
        super().__init__()
        self.node_encoder = full_model.node_encoder
        self.edge_encoder = full_model.edge_encoder
        self.mp_blocks = full_model.mp_blocks
        self.edge_head = full_model.edge_head

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        """Edge-logit inference.

        Parameters
        ----------
        x : Tensor (N, 6)
            Per-hit node features (already normalised).
        edge_index : Tensor (2, E)
            Directed edge list.
        edge_attr : Tensor (E, 8)
            Per-edge features (already normalised).

        Returns
        -------
        edge_logits : Tensor (E,)
            Raw logits. Apply sigmoid and threshold externally.
        """
        h = self.node_encoder(x)
        e = self.edge_encoder(edge_attr)
        for block in self.mp_blocks:
            h, e = block(h, e, edge_index)
        return self.edge_head(h, e, edge_index)

    @classmethod
    def from_checkpoint(cls, checkpoint_path, map_location="cpu"):
        """Load a deploy wrapper from a trained CaloClusterNet checkpoint.

        Reads ``config.yaml`` from the run directory alongside the
        checkpoint to rebuild the full model with the original
        hyperparameters, then strips the node head.

        Parameters
        ----------
        checkpoint_path : str or Path
            Path to a ``best_model.pt`` produced by ``train_gnn.py``.
            Expected layout: ``<run_dir>/checkpoints/best_model.pt`` with
            the config at ``<run_dir>/config.yaml``.
        map_location : str or torch.device
            Passed through to ``torch.load``. Defaults to CPU.

        Returns
        -------
        CaloClusterNetDeploy
            In eval mode, ready for inference or ONNX export.
        """
        checkpoint_path = Path(checkpoint_path)
        run_dir = checkpoint_path.parent.parent
        with open(run_dir / "config.yaml") as f:
            cfg = yaml.safe_load(f)

        if cfg["model"].get("name") != "CaloClusterNet":
            raise ValueError(
                f"Deploy wrapper only supports CaloClusterNet, got "
                f"{cfg['model'].get('name')} in {run_dir / 'config.yaml'}"
            )

        full = build_model(cfg)
        ckpt = torch.load(checkpoint_path, weights_only=False,
                          map_location=map_location)
        full.load_state_dict(ckpt["model_state_dict"])
        full.eval()

        wrapper = cls(full)
        wrapper.eval()
        return wrapper
