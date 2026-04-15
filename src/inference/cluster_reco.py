"""
Cluster reconstruction from GNN edge predictions.

Given edge logits from SimpleEdgeNet:
  1. Sigmoid → probabilities
  2. Symmetrize directed edge scores (mean of p_ij and p_ji)
  3. Threshold: keep edges with p > tau_edge
  4. Connected components → cluster labels
  5. Cleanup: remove clusters below min_hits or min_energy
"""

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def symmetrize_edge_scores(edge_index, edge_probs):
    """Average directed edge scores to get undirected scores.

    For each pair (i, j) that appears in both directions, the undirected
    score is mean(p_ij, p_ji).  Edges that appear in only one direction
    keep their original score.

    Parameters
    ----------
    edge_index : ndarray (2, E)
    edge_probs : ndarray (E,)

    Returns
    -------
    edge_index_undir : ndarray (2, E_undir)
        Undirected edges (i < j).
    edge_probs_undir : ndarray (E_undir,)
    """
    src, dst = edge_index[0], edge_index[1]

    # Build dict: (min, max) → list of probs
    pair_scores = {}
    for k in range(len(src)):
        key = (min(src[k], dst[k]), max(src[k], dst[k]))
        if key not in pair_scores:
            pair_scores[key] = []
        pair_scores[key].append(edge_probs[k])

    n_undir = len(pair_scores)
    ei_undir = np.empty((2, n_undir), dtype=np.int64)
    ep_undir = np.empty(n_undir, dtype=np.float64)

    for idx, (key, scores) in enumerate(pair_scores.items()):
        ei_undir[0, idx] = key[0]
        ei_undir[1, idx] = key[1]
        ep_undir[idx] = np.mean(scores)

    return ei_undir, ep_undir


def reconstruct_clusters(edge_index, edge_logits, n_nodes, energies=None,
                         tau_edge=0.5, min_hits=2, min_energy_mev=10.0,
                         symmetrize=True, node_logits=None, tau_node=None,
                         saliency_prune=False):
    """Reconstruct clusters from edge predictions.

    Parameters
    ----------
    edge_index : Tensor or ndarray (2, E)
        Directed edge list.
    edge_logits : Tensor or ndarray (E,)
        Raw logits (pre-sigmoid) from the model.
    n_nodes : int
        Number of nodes in the graph.
    energies : Tensor or ndarray (N,), optional
        Hit energies in MeV (raw, not log-transformed). Used for min_energy
        cleanup. If None, energy cleanup is skipped.
    tau_edge : float
        Edge classification threshold on probabilities.
    min_hits : int
        Minimum hits per cluster. Smaller clusters get label -1.
    min_energy_mev : float
        Minimum total energy per cluster (MeV). Below-threshold clusters
        get label -1.
    symmetrize : bool
        If True, average directed scores before thresholding.
    node_logits : Tensor or ndarray (N,), optional
        Raw node saliency logits from CaloClusterNet. If provided with
        tau_node, used for bridge suppression and/or post-clustering pruning.
    tau_node : float or None
        Node saliency threshold. When saliency_prune is False, suppresses
        edges where BOTH endpoints are non-salient (bridge suppression).
        When saliency_prune is True, removes non-salient hits from clusters
        after connected components (post-clustering pruning).
    saliency_prune : bool
        If True, after connected components, remove hits with saliency below
        tau_node from their clusters. This trims stray pileup hits that were
        absorbed via edges to salient cluster members.

    Returns
    -------
    cluster_labels : ndarray (N,)
        Integer cluster ID per node. -1 = unclustered.
    edge_probs : ndarray (E,)
        Sigmoid probabilities (useful for downstream analysis).
    """
    # Convert to numpy
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.cpu().numpy()
    if isinstance(edge_logits, torch.Tensor):
        edge_logits = edge_logits.detach().cpu().numpy()
    if energies is not None and isinstance(energies, torch.Tensor):
        energies = energies.cpu().numpy()
    if node_logits is not None and isinstance(node_logits, torch.Tensor):
        node_logits = node_logits.detach().cpu().numpy()

    edge_probs_raw = 1.0 / (1.0 + np.exp(-edge_logits.astype(np.float64)))

    # Compute node saliency probabilities (used for bridge suppression or pruning)
    node_probs = None
    if node_logits is not None and tau_node is not None:
        node_probs = 1.0 / (1.0 + np.exp(-node_logits.astype(np.float64)))

    # Bridge suppression (pre-clustering): zero out edges where BOTH endpoints
    # are non-salient. Only used when saliency_prune is False.
    if node_probs is not None and not saliency_prune:
        src, dst = edge_index[0], edge_index[1]
        both_non_salient = (node_probs[src] < tau_node) & (node_probs[dst] < tau_node)
        edge_probs_raw[both_non_salient] = 0.0

    if symmetrize:
        ei_sym, ep_sym = symmetrize_edge_scores(edge_index, edge_probs_raw)
    else:
        ei_sym, ep_sym = edge_index, edge_probs_raw

    # Threshold
    keep = ep_sym >= tau_edge

    cluster_labels = np.full(n_nodes, -1, dtype=np.int64)

    if keep.sum() == 0:
        return cluster_labels, edge_probs_raw

    src = ei_sym[0, keep]
    dst = ei_sym[1, keep]

    # Build symmetric adjacency (for connected_components)
    src_both = np.concatenate([src, dst])
    dst_both = np.concatenate([dst, src])
    vals = np.ones(len(src_both), dtype=np.float32)
    adj = coo_matrix((vals, (src_both, dst_both)), shape=(n_nodes, n_nodes))

    n_components, labels = connected_components(adj, directed=False)

    # Assign labels, then apply cleanup
    cluster_labels[:] = labels

    # Cleanup: min_hits
    for cid in range(n_components):
        mask = cluster_labels == cid
        if mask.sum() < min_hits:
            cluster_labels[mask] = -1

    # Cleanup: min_energy
    if energies is not None:
        for cid in np.unique(cluster_labels):
            if cid == -1:
                continue
            mask = cluster_labels == cid
            if energies[mask].sum() < min_energy_mev:
                cluster_labels[mask] = -1

    # Saliency pruning (post-clustering): remove non-salient hits from clusters.
    # This trims stray pileup hits that were absorbed into clusters via edges
    # to salient cluster members — the failure mode that bridge suppression misses.
    if saliency_prune and node_probs is not None:
        for i in range(n_nodes):
            if cluster_labels[i] >= 0 and node_probs[i] < tau_node:
                cluster_labels[i] = -1

    # Relabel to contiguous IDs (0, 1, 2, ...)
    valid_ids = np.unique(cluster_labels[cluster_labels >= 0])
    remap = {old: new for new, old in enumerate(valid_ids)}
    result = np.full_like(cluster_labels, -1)
    for old, new in remap.items():
        result[cluster_labels == old] = new
    cluster_labels = result

    return cluster_labels, edge_probs_raw


def predict_clusters(model, data, device="cpu", tau_edge=0.5,
                     min_hits=2, min_energy_mev=10.0, tau_node=None):
    """Run model inference and return cluster labels.

    Convenience wrapper: forward pass → reconstruct_clusters.

    Parameters
    ----------
    model : nn.Module
        Trained SimpleEdgeNet or CaloClusterNet.
    data : torch_geometric.data.Data
        Single graph with x, edge_index, edge_attr.
    device : str or torch.device
    tau_edge, min_hits, min_energy_mev : see reconstruct_clusters.
    tau_node : float or None
        Node saliency threshold. Only used with models that return
        node_logits (CaloClusterNet).

    Returns
    -------
    cluster_labels : ndarray (N,)
    edge_probs : ndarray (E,)
    """
    model.eval()
    data_dev = data.clone().to(device)
    with torch.no_grad():
        output = model(data_dev)

    if isinstance(output, dict):
        logits = output["edge_logits"]
        node_logits = output.get("node_logits")
    else:
        logits = output
        node_logits = None

    # Use raw energies if available (node feature 0 is log(1+E))
    energies = None
    if data.x is not None:
        log_e = data.x[:, 0].cpu().numpy()
        energies = np.exp(log_e) - 1.0

    return reconstruct_clusters(
        edge_index=data.edge_index,
        edge_logits=logits.cpu(),
        n_nodes=data.x.shape[0],
        energies=energies,
        tau_edge=tau_edge,
        min_hits=min_hits,
        min_energy_mev=min_energy_mev,
        node_logits=node_logits.cpu() if node_logits is not None else None,
        tau_node=tau_node,
    )
