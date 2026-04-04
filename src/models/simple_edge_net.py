"""
SimpleEdgeNet — lightweight edge-classification GNN baseline.

Architecture:
  - Node encoder: MLP(6 → hidden → hidden)
  - Edge encoder: MLP(8 → hidden → hidden)
  - N message-passing rounds (sum aggregation, no gating, no residual)
  - Edge head: MLP([h_i || h_j || e_ij] → 2*hidden → hidden → 1)

Outputs raw logits (no sigmoid) — apply sigmoid in loss/inference.
"""

import torch
import torch.nn as nn
from torch_geometric.utils import scatter


class MLP(nn.Module):
    """Simple multi-layer perceptron with ReLU."""

    def __init__(self, dims, dropout=0.0):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """Apply the MLP layers sequentially to input ``x``."""
        return self.net(x)


class SimpleEdgeNet(nn.Module):
    """Edge-classification GNN with simple sum message passing.

    Parameters
    ----------
    node_dim : int
        Input node feature dimension (default 6).
    edge_dim : int
        Input edge feature dimension (default 8).
    hidden_dim : int
        Hidden dimension throughout (default 64).
    n_mp_layers : int
        Number of message-passing rounds (default 3).
    dropout : float
        Dropout rate in MLPs (default 0.1).
    """

    def __init__(self, node_dim=6, edge_dim=8, hidden_dim=64, n_mp_layers=3,
                 dropout=0.1):
        super().__init__()
        self.n_mp_layers = n_mp_layers

        # Encoders
        self.node_encoder = MLP([node_dim, hidden_dim, hidden_dim], dropout=dropout)
        self.edge_encoder = MLP([edge_dim, hidden_dim, hidden_dim], dropout=dropout)

        # Message-passing layers
        # Each round: edge update MLP + node update MLP
        self.edge_updates = nn.ModuleList()
        self.node_updates = nn.ModuleList()
        for _ in range(n_mp_layers):
            # Edge update: [h_i, h_j, e_ij] -> hidden
            self.edge_updates.append(
                MLP([3 * hidden_dim, 2 * hidden_dim, hidden_dim], dropout=dropout)
            )
            # Node update: [h_i, aggregated_messages] -> hidden
            self.node_updates.append(
                MLP([2 * hidden_dim, 2 * hidden_dim, hidden_dim], dropout=dropout)
            )

        # Edge classification head: [h_i, h_j, e_ij] -> 1
        self.edge_head = MLP(
            [3 * hidden_dim, 2 * hidden_dim, hidden_dim, 1],
            dropout=dropout,
        )

    def forward(self, data):
        """Forward pass.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Must have x, edge_index, edge_attr.

        Returns
        -------
        edge_logits : Tensor, shape (E,)
            Raw logits for edge classification (same-cluster probability
            after sigmoid).
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        # Encode
        h = self.node_encoder(x)             # (N, hidden)
        e = self.edge_encoder(edge_attr)      # (E, hidden)

        src, dst = edge_index  # src -> dst

        # Message passing
        for k in range(self.n_mp_layers):
            # Edge update
            e_in = torch.cat([h[src], h[dst], e], dim=1)  # (E, 3*hidden)
            e = self.edge_updates[k](e_in)                 # (E, hidden)

            # Aggregate messages to destination nodes (sum)
            agg = scatter(e, dst, dim=0, dim_size=h.size(0), reduce="sum")

            # Node update
            h_in = torch.cat([h, agg], dim=1)              # (N, 2*hidden)
            h = self.node_updates[k](h_in)                 # (N, hidden)

        # Edge classification
        edge_repr = torch.cat([h[src], h[dst], e], dim=1)  # (E, 3*hidden)
        edge_logits = self.edge_head(edge_repr).squeeze(-1) # (E,)

        return edge_logits
