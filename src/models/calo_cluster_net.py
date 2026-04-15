"""
CaloClusterNet — multi-task edge-centric GNN for calorimeter clustering.

Architecture:
  - Node encoder: MLP(6 → hidden → hidden)
  - Edge encoder: MLP(8 → hidden → hidden)
  - N × EdgeAwareResBlock (residual MP with gated aggregation + global context)
  - Node saliency head → q_i logit
  - Edge clustering head → s_ij logit  ([h_i, h_j, e_ij, |h_i-h_j|])

Outputs raw logits — apply sigmoid in loss/inference.

Forward returns a dict:
  {"edge_logits": Tensor(E,), "node_logits": Tensor(N,)}
so that the trainer can compute multi-task loss.
"""

import torch
import torch.nn as nn

from src.models.layers import EdgeAwareResBlock
from src.models.heads import NodeSaliencyHead, EdgeClusteringHead


class CaloClusterNet(nn.Module):
    """Multi-task edge-centric GNN for calorimeter clustering.

    Parameters
    ----------
    node_dim : int
        Input node feature dimension (default 6).
    edge_dim : int
        Input edge feature dimension (default 8).
    hidden_dim : int
        Hidden dimension throughout (default 96).
    n_mp_layers : int
        Number of EdgeAwareResBlock rounds (default 4).
    dropout : float
        Dropout rate in all MLPs (default 0.1).
    """

    def __init__(self, node_dim=6, edge_dim=8, hidden_dim=96, n_mp_layers=4,
                 dropout=0.1):
        super().__init__()
        self.n_mp_layers = n_mp_layers

        # Encoders
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Message-passing blocks
        self.mp_blocks = nn.ModuleList([
            EdgeAwareResBlock(hidden_dim=hidden_dim, dropout=dropout)
            for _ in range(n_mp_layers)
        ])

        # Output heads
        self.node_head = NodeSaliencyHead(hidden_dim=hidden_dim, dropout=dropout)
        self.edge_head = EdgeClusteringHead(hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, data):
        """Forward pass.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Must have x, edge_index, edge_attr.

        Returns
        -------
        dict with:
            edge_logits : Tensor (E,)
            node_logits : Tensor (N,)
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        # Encode
        h = self.node_encoder(x)
        e = self.edge_encoder(edge_attr)

        # Message passing
        for block in self.mp_blocks:
            h, e = block(h, e, edge_index)

        # Heads
        edge_logits = self.edge_head(h, e, edge_index)
        node_logits = self.node_head(h)

        return {"edge_logits": edge_logits, "node_logits": node_logits}
