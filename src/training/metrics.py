"""
Evaluation metrics for edge classification and cluster quality.

Edge metrics: precision, recall, F1, ROC AUC, PR AUC.
Cluster metrics: purity, completeness, ARI.
"""

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def node_saliency_metrics(node_logits, y_node, threshold=0.5):
    """Compute node saliency classification metrics.

    Parameters
    ----------
    node_logits : Tensor (N,)
        Raw logits for node saliency.
    y_node : Tensor (N,)
        Node labels. >= 0 means signal (assigned to truth cluster), -1 means noise.
    threshold : float
        Classification threshold on sigmoid(logits).

    Returns
    -------
    dict with precision, recall, f1, accuracy, n_signal, n_noise.
    """
    probs = torch.sigmoid(node_logits).cpu().numpy()
    targets = (y_node >= 0).cpu().numpy().astype(int)

    preds = (probs >= threshold).astype(int)

    tp = ((preds == 1) & (targets == 1)).sum()
    fp = ((preds == 1) & (targets == 0)).sum()
    fn = ((preds == 0) & (targets == 1)).sum()
    tn = ((preds == 0) & (targets == 0)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(targets) if len(targets) > 0 else 0.0

    return {
        "node_precision": float(precision),
        "node_recall": float(recall),
        "node_f1": float(f1),
        "node_accuracy": float(accuracy),
        "n_signal": int(targets.sum()),
        "n_noise": int((1 - targets).sum()),
    }


def edge_metrics(logits, targets, mask, threshold=0.5):
    """Compute edge classification metrics on masked edges.

    Parameters
    ----------
    logits : Tensor (E,)
        Raw logits (pre-sigmoid).
    targets : Tensor (E,)
        Binary labels.
    mask : Tensor (E,)
        Boolean mask.
    threshold : float
        Classification threshold on sigmoid(logits).

    Returns
    -------
    dict with precision, recall, f1, accuracy, n_pos, n_neg.
    """
    m = mask.bool()
    probs = torch.sigmoid(logits[m]).cpu().numpy()
    y = targets[m].cpu().numpy().astype(int)

    preds = (probs >= threshold).astype(int)

    tp = ((preds == 1) & (y == 1)).sum()
    fp = ((preds == 1) & (y == 0)).sum()
    fn = ((preds == 0) & (y == 1)).sum()
    tn = ((preds == 0) & (y == 0)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(y) if len(y) > 0 else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "n_pos": int(y.sum()),
        "n_neg": int((1 - y).sum()),
    }


def edge_auc(logits, targets, mask):
    """Compute ROC AUC and PR AUC on masked edges.

    Returns dict with roc_auc, pr_auc. Returns 0.0 if single-class.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    m = mask.bool()
    probs = torch.sigmoid(logits[m]).detach().cpu().numpy()
    y = targets[m].cpu().numpy().astype(int)

    if len(np.unique(y)) < 2:
        return {"roc_auc": 0.0, "pr_auc": 0.0}

    return {
        "roc_auc": float(roc_auc_score(y, probs)),
        "pr_auc": float(average_precision_score(y, probs)),
    }


def cluster_metrics_from_edges(edge_index, edge_probs, edge_mask, hit_truth_cluster,
                               n_nodes, threshold=0.5):
    """Reconstruct clusters from predicted edges and compare to truth.

    Parameters
    ----------
    edge_index : Tensor (2, E)
    edge_probs : Tensor (E,)
        Sigmoid probabilities.
    edge_mask : Tensor (E,)
    hit_truth_cluster : Tensor (N,)
        Per-node truth cluster ID (-1 = unassigned).
    n_nodes : int
    threshold : float

    Returns
    -------
    dict with purity, completeness, n_pred_clusters, n_truth_clusters.
    """
    ei = edge_index.cpu().numpy()
    probs = edge_probs.cpu().numpy()
    mask = edge_mask.bool().cpu().numpy()
    truth = hit_truth_cluster.cpu().numpy()

    # Build predicted adjacency from positive edges
    pred_pos = (probs >= threshold) & mask
    if pred_pos.sum() == 0:
        return {"purity": 0.0, "completeness": 0.0,
                "n_pred_clusters": 0, "n_truth_clusters": 0}

    src = ei[0, pred_pos]
    dst = ei[1, pred_pos]
    # Symmetrize
    src_sym = np.concatenate([src, dst])
    dst_sym = np.concatenate([dst, src])
    vals = np.ones(len(src_sym), dtype=np.float32)

    adj = coo_matrix((vals, (src_sym, dst_sym)), shape=(n_nodes, n_nodes))
    n_components, labels = connected_components(adj, directed=False)

    # Compute purity and completeness for assigned hits
    assigned = truth >= 0
    if assigned.sum() == 0:
        return {"purity": 0.0, "completeness": 0.0,
                "n_pred_clusters": n_components, "n_truth_clusters": 0}

    truth_ids = np.unique(truth[assigned])
    pred_ids = np.unique(labels[assigned])

    # Purity: for each predicted cluster, fraction of hits from dominant truth cluster
    purities = []
    for pid in pred_ids:
        pmask = (labels == pid) & assigned
        if pmask.sum() == 0:
            continue
        tc = truth[pmask]
        dominant_count = np.bincount(tc).max()
        purities.append(dominant_count / pmask.sum())

    # Completeness: for each truth cluster, fraction of hits in dominant predicted cluster
    completenesses = []
    for tid in truth_ids:
        tmask = (truth == tid) & assigned
        if tmask.sum() == 0:
            continue
        pc = labels[tmask]
        dominant_count = np.bincount(pc).max()
        completenesses.append(dominant_count / tmask.sum())

    purity = float(np.mean(purities)) if purities else 0.0
    completeness = float(np.mean(completenesses)) if completenesses else 0.0

    return {
        "purity": purity,
        "completeness": completeness,
        "n_pred_clusters": int(len(pred_ids)),
        "n_truth_clusters": int(len(truth_ids)),
    }
