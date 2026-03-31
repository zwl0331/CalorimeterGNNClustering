"""Truth labeling for edge classification in GNN calorimeter clustering.

Two modes:
  1. BFS pseudo-truth: uses reco cluster assignments (calohits.clusterIdx_)
  2. MC truth: uses SimParticle IDs and energy deposits from calohitsmc
"""

import numpy as np


def assign_bfs_truth(cluster_idx, edge_index):
    """Assign edge labels from BFS reco cluster assignments.

    Args:
        cluster_idx: np.ndarray of shape (n_hits,), int. -1 = unassigned.
        edge_index: np.ndarray of shape (2, n_edges), int.

    Returns:
        y_edge: np.ndarray of shape (n_edges,), int (0 or 1).
            1 if both hits belong to the same cluster.
        edge_mask: np.ndarray of shape (n_edges,), bool.
            True = valid edge (both hits assigned). Use to mask the loss.
    """
    cluster_idx = np.asarray(cluster_idx)
    edge_index = np.asarray(edge_index)

    src = edge_index[0]
    dst = edge_index[1]

    ci_src = cluster_idx[src]
    ci_dst = cluster_idx[dst]

    # Both hits must be assigned (not -1)
    edge_mask = (ci_src != -1) & (ci_dst != -1)

    # Same cluster → positive edge
    y_edge = (ci_src == ci_dst).astype(np.int64)

    # Masked edges get label 0 (won't matter since they're masked)
    y_edge[~edge_mask] = 0

    return y_edge, edge_mask


def assign_mc_truth(sim_particle_ids, edeps, hit_disks, edge_index,
                    purity_threshold=0.7):
    """Assign edge labels from MC truth SimParticle information.

    For each hit the dominant SimParticle is the one with the largest energy
    deposit.  A hit is *ambiguous* when the dominant particle's share of the
    total deposit is below ``purity_threshold``.

    Truth clusters group non-ambiguous hits that share the same dominant
    SimParticle **and** the same disk.

    Args:
        sim_particle_ids: list of lists — per-hit SimParticle IDs.
        edeps: list of lists — per-hit energy deposits (aligned with
            sim_particle_ids).
        hit_disks: np.ndarray of shape (n_hits,), int.
        edge_index: np.ndarray of shape (2, n_edges), int.
        purity_threshold: float, default 0.7.

    Returns:
        y_edge: np.ndarray of shape (n_edges,), int (0 or 1).
        edge_mask: np.ndarray of shape (n_edges,), bool (True = valid).
        hit_truth_cluster: np.ndarray of shape (n_hits,), int.
            Unique truth-cluster ID per hit; -1 for ambiguous hits.
        is_ambiguous: np.ndarray of shape (n_hits,), bool.
    """
    hit_disks = np.asarray(hit_disks)
    edge_index = np.asarray(edge_index)
    n_hits = len(sim_particle_ids)

    dominant_pid = np.full(n_hits, -1, dtype=np.int64)
    is_ambiguous = np.ones(n_hits, dtype=bool)

    for i in range(n_hits):
        pids = sim_particle_ids[i]
        deps = edeps[i]
        if len(pids) == 0 or len(deps) == 0:
            # No SimParticle info → ambiguous
            continue
        deps_arr = np.asarray(deps, dtype=np.float64)
        total = deps_arr.sum()
        if total <= 0:
            continue
        best = np.argmax(deps_arr)
        purity = deps_arr[best] / total
        if purity >= purity_threshold:
            is_ambiguous[i] = False
            dominant_pid[i] = pids[best]

    # Build truth cluster IDs: unique (dominant_pid, disk) pairs for
    # non-ambiguous hits.
    hit_truth_cluster = np.full(n_hits, -1, dtype=np.int64)
    cluster_map = {}  # (pid, disk) -> cluster_id
    next_id = 0
    for i in range(n_hits):
        if is_ambiguous[i]:
            continue
        key = (int(dominant_pid[i]), int(hit_disks[i]))
        if key not in cluster_map:
            cluster_map[key] = next_id
            next_id += 1
        hit_truth_cluster[i] = cluster_map[key]

    # Edge labels
    src = edge_index[0]
    dst = edge_index[1]

    amb_src = is_ambiguous[src]
    amb_dst = is_ambiguous[dst]
    edge_mask = ~(amb_src | amb_dst)  # valid only if both non-ambiguous

    tc_src = hit_truth_cluster[src]
    tc_dst = hit_truth_cluster[dst]
    y_edge = ((tc_src == tc_dst) & (tc_src != -1)).astype(np.int64)
    y_edge[~edge_mask] = 0

    return y_edge, edge_mask, hit_truth_cluster, is_ambiguous
