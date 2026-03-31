"""
Compute and apply global normalization statistics for node and edge features.

Statistics are computed from the TRAINING split only, then applied to all splits.
Saved to data/normalization_stats.pt.
"""

from pathlib import Path

import torch
import numpy as np


def compute_normalization_stats(dataset):
    """Compute per-feature mean and std from a dataset of PyG Data objects.

    Parameters
    ----------
    dataset : iterable of Data
        Training split graphs.

    Returns
    -------
    stats : dict
        Keys: node_mean, node_std, edge_mean, edge_std (each a Tensor).
    """
    node_sum = None
    node_sq_sum = None
    node_count = 0

    edge_sum = None
    edge_sq_sum = None
    edge_count = 0

    for data in dataset:
        x = data.x.double()
        n = x.shape[0]
        if node_sum is None:
            node_sum = torch.zeros(x.shape[1], dtype=torch.float64)
            node_sq_sum = torch.zeros(x.shape[1], dtype=torch.float64)
        node_sum += x.sum(dim=0)
        node_sq_sum += (x ** 2).sum(dim=0)
        node_count += n

        ea = data.edge_attr.double()
        m = ea.shape[0]
        if edge_sum is None:
            edge_sum = torch.zeros(ea.shape[1], dtype=torch.float64)
            edge_sq_sum = torch.zeros(ea.shape[1], dtype=torch.float64)
        edge_sum += ea.sum(dim=0)
        edge_sq_sum += (ea ** 2).sum(dim=0)
        edge_count += m

    if node_count == 0:
        raise ValueError("No data to compute normalization stats.")

    node_mean = (node_sum / node_count).float()
    node_std = torch.sqrt(node_sq_sum / node_count - node_mean.double() ** 2).float()
    node_std = torch.clamp(node_std, min=1e-6)

    edge_mean = (edge_sum / edge_count).float()
    edge_std = torch.sqrt(edge_sq_sum / edge_count - edge_mean.double() ** 2).float()
    edge_std = torch.clamp(edge_std, min=1e-6)

    return {
        "node_mean": node_mean,
        "node_std": node_std,
        "edge_mean": edge_mean,
        "edge_std": edge_std,
        "node_count": node_count,
        "edge_count": edge_count,
    }


def save_stats(stats, path="data/normalization_stats.pt"):
    """Save normalization stats to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stats, path)
    print(f"Saved normalization stats to {path}")
    print(f"  Node features: {stats['node_mean'].shape[0]}-dim, "
          f"from {stats['node_count']} nodes")
    print(f"  Edge features: {stats['edge_mean'].shape[0]}-dim, "
          f"from {stats['edge_count']} edges")


def load_stats(path="data/normalization_stats.pt"):
    """Load normalization stats from disk."""
    return torch.load(path, weights_only=True)


def normalize_graph(data, stats):
    """Apply z-score normalization to a PyG Data object (in-place).

    Parameters
    ----------
    data : Data
        Graph with x (node features) and edge_attr (edge features).
    stats : dict
        From compute_normalization_stats / load_stats.

    Returns
    -------
    data : Data
        Same object, modified in-place.
    """
    data.x = (data.x - stats["node_mean"]) / stats["node_std"]
    data.edge_attr = (data.edge_attr - stats["edge_mean"]) / stats["edge_std"]
    return data
