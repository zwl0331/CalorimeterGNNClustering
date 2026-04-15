"""
EdgeAwareResBlock — residual message-passing block for CaloClusterNet.

Each block performs four steps:
  A. Edge update (residual):  m_ij = MLP([h_i, h_j, e_ij]); e_ij ← LN(e_ij + m_ij)
  B. Gated aggregation:       g_ij = σ(Linear(e_ij)); a_i = Σ_j g_ji · e_ji
  C. Node update (residual):  u_i = MLP([h_i, a_i]); h_i ← LN(h_i + u_i)
  D. Global context:          c = mean(h_i); h_i ← h_i + Linear(c)
"""

import torch
import torch.nn as nn
from torch_geometric.utils import scatter


class EdgeAwareResBlock(nn.Module):
    """One round of edge-aware message passing with residual connections.

    Parameters
    ----------
    hidden_dim : int
        Dimension of node and edge embeddings (default 96).
    dropout : float
        Dropout rate in MLPs (default 0.1).
    """

    def __init__(self, hidden_dim=96, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # A: Edge update — [h_i, h_j, e_ij] (3*hidden) → hidden
        self.edge_mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.edge_norm = nn.LayerNorm(hidden_dim)

        # B: Gated aggregation — gate per edge
        self.gate_linear = nn.Linear(hidden_dim, 1)

        # C: Node update — [h_i, a_i] (2*hidden) → hidden
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.node_norm = nn.LayerNorm(hidden_dim)

        # D: Global context — project mean node embedding back
        self.global_linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, h, e, edge_index):
        """Forward pass.

        Parameters
        ----------
        h : Tensor (N, hidden_dim)
            Node embeddings.
        e : Tensor (E, hidden_dim)
            Edge embeddings.
        edge_index : Tensor (2, E)
            src → dst edge list.

        Returns
        -------
        h : Tensor (N, hidden_dim)
            Updated node embeddings.
        e : Tensor (E, hidden_dim)
            Updated edge embeddings.
        """
        src, dst = edge_index

        # A: Edge update with residual
        e_in = torch.cat([h[src], h[dst], e], dim=1)
        e = self.edge_norm(e + self.edge_mlp(e_in))

        # B: Gated aggregation (messages flow dst ← src, aggregate over src)
        gate = torch.sigmoid(self.gate_linear(e))  # (E, 1)
        msg = gate * e                               # (E, hidden)
        agg = scatter(msg, dst, dim=0, dim_size=h.size(0), reduce="sum")

        # C: Node update with residual
        h_in = torch.cat([h, agg], dim=1)
        h = self.node_norm(h + self.node_mlp(h_in))

        # D: Global context injection
        ctx = h.mean(dim=0, keepdim=True)            # (1, hidden)
        h = h + self.global_linear(ctx)

        return h, e
