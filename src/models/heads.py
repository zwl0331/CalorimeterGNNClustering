"""
Output heads for CaloClusterNet.

NodeSaliencyHead — per-node binary classification (signal vs noise).
EdgeClusteringHead — per-edge binary classification (same cluster).
"""

import torch
import torch.nn as nn


class NodeSaliencyHead(nn.Module):
    """Predict per-node saliency (signal probability).

    Architecture: Linear(hidden, 64) → GELU → Dropout → Linear(64, 1)
    Returns raw logits (apply sigmoid externally).
    """

    def __init__(self, hidden_dim=96, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, h):
        """
        Parameters
        ----------
        h : Tensor (N, hidden_dim)

        Returns
        -------
        logits : Tensor (N,)
        """
        return self.net(h).squeeze(-1)


class EdgeClusteringHead(nn.Module):
    """Predict per-edge same-cluster probability.

    Input: [h_i, h_j, e_ij, |h_i - h_j|] → 4*hidden_dim
    Architecture: MLP → 1 logit
    Returns raw logits (apply sigmoid externally).
    """

    def __init__(self, hidden_dim=96, dropout=0.1):
        super().__init__()
        input_dim = 4 * hidden_dim  # h_i, h_j, e_ij, |h_i - h_j|
        self.net = nn.Sequential(
            nn.Linear(input_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h, e, edge_index):
        """
        Parameters
        ----------
        h : Tensor (N, hidden_dim)
        e : Tensor (E, hidden_dim)
        edge_index : Tensor (2, E)

        Returns
        -------
        logits : Tensor (E,)
        """
        src, dst = edge_index
        h_src = h[src]
        h_dst = h[dst]
        diff = torch.abs(h_src - h_dst)
        edge_repr = torch.cat([h_src, h_dst, e, diff], dim=1)
        return self.net(edge_repr).squeeze(-1)
