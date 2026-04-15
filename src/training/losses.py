"""
Loss functions for edge classification training.

Supports:
  - Class-balanced BCE (per-edge weights inversely proportional to class frequency)
  - Optional minority subsampling during training
  - Edge mask to exclude ambiguous/unassigned edges from loss
"""

import torch
import torch.nn.functional as F


def compute_class_weights(dataset):
    """Compute class balance info from training dataset.

    Returns
    -------
    dict with n_pos, n_neg, pos_weight (neg/pos ratio for BCE).
    """
    n_pos = 0
    n_neg = 0
    for data in dataset:
        mask = data.edge_mask.bool()
        labels = data.y_edge[mask]
        n_pos += (labels == 1).sum().item()
        n_neg += (labels == 0).sum().item()

    total = n_pos + n_neg
    # pos_weight for BCE: ratio neg/pos — upweights positives when they are rare
    # When positives dominate (pos_weight < 1), this downweights them
    pos_weight = torch.tensor(n_neg / n_pos) if n_pos > 0 else torch.tensor(1.0)

    return {
        "n_pos": n_pos,
        "n_neg": n_neg,
        "pos_weight": pos_weight,
    }


def masked_bce_loss(logits, targets, mask, pos_weight=None):
    """Compute BCE loss on masked edges with optional class reweighting.

    Parameters
    ----------
    logits : Tensor (E,)
        Raw edge logits (pre-sigmoid).
    targets : Tensor (E,)
        Binary edge labels.
    mask : Tensor (E,)
        Boolean mask — loss computed only where True.
    pos_weight : Tensor or None
        Weight for positive class in BCE. Values < 1 downweight positives,
        > 1 upweight positives.

    Returns
    -------
    loss : scalar Tensor
    """
    m = mask.bool()
    logits_m = logits[m]
    targets_m = targets[m].float()

    if logits_m.numel() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    loss = F.binary_cross_entropy_with_logits(
        logits_m, targets_m,
        pos_weight=pos_weight,
    )
    return loss


def node_saliency_loss(node_logits, y_node):
    """BCE loss for node saliency prediction.

    Parameters
    ----------
    node_logits : Tensor (N,)
        Raw logits for node saliency.
    y_node : Tensor (N,)
        Binary node labels (1 = signal hit with known truth cluster, 0 = noise/unassigned).

    Returns
    -------
    loss : scalar Tensor
    """
    targets = (y_node >= 0).float()  # y_node >= 0 means assigned to a truth cluster

    if node_logits.numel() == 0:
        return torch.tensor(0.0, device=node_logits.device, requires_grad=True)

    return F.binary_cross_entropy_with_logits(node_logits, targets)


def consistency_loss(edge_logits, node_logits, edge_index):
    """Consistency regularizer between node saliency and edge predictions.

    If both endpoints of an edge are predicted as non-salient (low q_i, q_j),
    the edge should also be predicted as negative. Penalizes high edge
    probability when both node saliencies are low.

    L_cons = mean( σ(s_ij) · (1 - σ(q_i)) · (1 - σ(q_j)) )

    Parameters
    ----------
    edge_logits : Tensor (E,)
    node_logits : Tensor (N,)
    edge_index : Tensor (2, E)

    Returns
    -------
    loss : scalar Tensor
    """
    if edge_logits.numel() == 0:
        return torch.tensor(0.0, device=edge_logits.device, requires_grad=True)

    src, dst = edge_index
    p_edge = torch.sigmoid(edge_logits)
    q_src = torch.sigmoid(node_logits[src])
    q_dst = torch.sigmoid(node_logits[dst])

    # Penalize: edge says "same cluster" but both nodes say "noise"
    penalty = p_edge * (1.0 - q_src) * (1.0 - q_dst)
    return penalty.mean()


def multitask_loss(model_output, batch, pos_weight=None,
                   lambda_edge=1.0, lambda_node=0.0, lambda_cons=0.0):
    """Compute multi-task loss for CaloClusterNet.

    Parameters
    ----------
    model_output : dict or Tensor
        If dict: {"edge_logits": ..., "node_logits": ...}
        If Tensor: edge logits only (SimpleEdgeNet compatibility).
    batch : PyG Batch
    pos_weight : Tensor or None
    lambda_edge, lambda_node, lambda_cons : float
        Loss weights for each term.

    Returns
    -------
    total_loss : scalar Tensor
    loss_dict : dict with individual loss values (for logging)
    """
    if isinstance(model_output, torch.Tensor):
        # SimpleEdgeNet compatibility — edge-only loss
        edge_logits = model_output
        l_edge = masked_bce_loss(edge_logits, batch.y_edge, batch.edge_mask,
                                 pos_weight=pos_weight)
        return l_edge, {"edge_loss": l_edge.item(), "total_loss": l_edge.item()}

    edge_logits = model_output["edge_logits"]
    node_logits = model_output["node_logits"]

    l_edge = masked_bce_loss(edge_logits, batch.y_edge, batch.edge_mask,
                             pos_weight=pos_weight)

    loss_dict = {"edge_loss": l_edge.item()}
    total = lambda_edge * l_edge

    if lambda_node > 0 and node_logits is not None:
        l_node = node_saliency_loss(node_logits, batch.y_node)
        loss_dict["node_loss"] = l_node.item()
        total = total + lambda_node * l_node

    if lambda_cons > 0 and node_logits is not None:
        l_cons = consistency_loss(edge_logits, node_logits, batch.edge_index)
        loss_dict["cons_loss"] = l_cons.item()
        total = total + lambda_cons * l_cons

    loss_dict["total_loss"] = total.item()
    return total, loss_dict
