"""
SimpleEdgeNetDeploy -- inference-only wrapper for ONNX export.

SimpleEdgeNet's forward takes a PyG ``Data`` object, which
``torch.onnx.export`` can't trace. This wrapper composes a trained
``SimpleEdgeNet`` and exposes a tensor-in / tensor-out forward signature
matching the ONNX deployment contract used by both this model and
CaloClusterNet (see calorimeter/GNN/docs/onnx_deployment.md).

The model has no node head and no multi-task output, so the wrapper
is mostly a thin pass-through that re-implements the message-passing
loop with explicit tensor arguments.
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.utils import scatter
import yaml

from src.models import build_model
from src.models.simple_edge_net import SimpleEdgeNet


class SimpleEdgeNetDeploy(nn.Module):
    """Tensor-API inference wrapper around a trained ``SimpleEdgeNet``.

    Reuses the submodules of ``full_model`` by reference (no weight copy).
    """

    def __init__(self, full_model: SimpleEdgeNet):
        super().__init__()
        self.n_mp_layers  = full_model.n_mp_layers
        self.node_encoder = full_model.node_encoder
        self.edge_encoder = full_model.edge_encoder
        self.edge_updates = full_model.edge_updates
        self.node_updates = full_model.node_updates
        self.edge_head    = full_model.edge_head

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        """Edge-logit inference.

        Parameters
        ----------
        x : Tensor (N, 6)
            Per-hit node features (already z-score normalised).
        edge_index : Tensor (2, E)
            Directed edge list.
        edge_attr : Tensor (E, 8)
            Per-edge features (already z-score normalised).

        Returns
        -------
        edge_logits : Tensor (E,)
            Raw logits. Apply sigmoid + threshold externally.
        """
        h = self.node_encoder(x)
        e = self.edge_encoder(edge_attr)
        src, dst = edge_index

        for k in range(self.n_mp_layers):
            e_in = torch.cat([h[src], h[dst], e], dim=1)
            e = self.edge_updates[k](e_in)
            agg = scatter(e, dst, dim=0, dim_size=h.size(0), reduce="sum")
            h_in = torch.cat([h, agg], dim=1)
            h = self.node_updates[k](h_in)

        edge_repr = torch.cat([h[src], h[dst], e], dim=1)
        edge_logits = self.edge_head(edge_repr).squeeze(-1)
        return edge_logits

    @classmethod
    def from_checkpoint(cls, checkpoint_path, map_location="cpu"):
        """Load a deploy wrapper from a trained SimpleEdgeNet checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        run_dir = checkpoint_path.parent.parent
        with open(run_dir / "config.yaml") as f:
            cfg = yaml.safe_load(f)

        if cfg["model"].get("name") != "SimpleEdgeNet":
            raise ValueError(
                f"Deploy wrapper only supports SimpleEdgeNet, got "
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
