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
