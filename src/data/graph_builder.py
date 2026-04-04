"""
Graph construction for calorimeter hit clustering.

Builds per-disk graphs using a hybrid radius + kNN strategy:
  1. Radius graph: connect hits within r_max (mm) in disk-local (x,y)
  2. Time filter: drop edges with |Δt| > dt_max (ns)
  3. kNN fallback: isolated nodes get edges to k_min nearest time-compatible neighbors
  4. Degree cap: keep at most k_max nearest neighbors per node

Uses scipy.spatial.cKDTree (torch-cluster is NOT available).
"""

import numpy as np
from scipy.spatial import cKDTree


def build_graph(positions, times, r_max=150.0, dt_max=25.0,
                k_min=3, k_max=20):
    """Build a graph for one disk of one event.

    Parameters
    ----------
    positions : ndarray, shape (n, 2)
        Hit (x, y) positions in disk-local frame (mm).
    times : ndarray, shape (n,)
        Hit times (ns).
    r_max : float
        Spatial radius cutoff (mm).
    dt_max : float
        Maximum |Δt| for any edge (ns).
    k_min : int
        Minimum neighbors for isolated nodes (kNN fallback).
    k_max : int
        Maximum degree cap per node.

    Returns
    -------
    edge_index : ndarray, shape (2, n_edges)
        Directed edge list (src, dst).
    diagnostics : dict
        Graph statistics: n_nodes, n_edges, avg_degree, n_isolated,
        min_degree, max_degree.
    """
    n = len(positions)
    if n == 0:
        return np.empty((2, 0), dtype=np.int64), _empty_diagnostics(0)
    if n == 1:
        return np.empty((2, 0), dtype=np.int64), _empty_diagnostics(1)

    positions = np.asarray(positions, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)

    tree = cKDTree(positions)

    # Step 1: radius graph
    pairs = tree.query_pairs(r=r_max, output_type='ndarray')
    if len(pairs) == 0:
        src_list, dst_list = [], []
    else:
        # Filter by time
        dt = np.abs(times[pairs[:, 0]] - times[pairs[:, 1]])
        mask = dt <= dt_max
        pairs = pairs[mask]
        # Make directed (both directions)
        src_list = list(pairs[:, 0]) + list(pairs[:, 1])
        dst_list = list(pairs[:, 1]) + list(pairs[:, 0])

    # Track degree per node
    degree = np.zeros(n, dtype=np.int64)
    for s in src_list:
        degree[s] += 1

    # Step 2: kNN fallback for isolated/low-degree nodes
    isolated = np.where(degree < k_min)[0]
    if len(isolated) > 0:
        # Query more neighbors than needed to account for time filtering
        k_query = min(k_min * 3, n)
        dists, indices = tree.query(positions[isolated], k=k_query)

        for local_i, node_i in enumerate(isolated):
            added = 0
            for j_pos in range(k_query):
                node_j = indices[local_i, j_pos]
                if node_j == node_i:
                    continue
                if np.abs(times[node_i] - times[node_j]) > dt_max:
                    continue
                # Add edge if not already present
                src_list.append(node_i)
                dst_list.append(node_j)
                src_list.append(node_j)
                dst_list.append(node_i)
                added += 1
                if degree[node_i] + added >= k_min:
                    break

    # Build edge_index and deduplicate
    if not src_list:
        return np.empty((2, 0), dtype=np.int64), _empty_diagnostics(n)

    edge_index = np.stack([np.array(src_list, dtype=np.int64),
                           np.array(dst_list, dtype=np.int64)])
    edge_index = _deduplicate(edge_index)

    # Step 3: degree cap — keep k_max nearest per node
    if k_max is not None and k_max > 0:
        edge_index = _cap_degree(edge_index, positions, k_max)

    diagnostics = _compute_diagnostics(edge_index, n)
    return edge_index, diagnostics


def _deduplicate(edge_index):
    """Remove duplicate directed edges, keeping the first occurrence."""
    combined = edge_index[0] * (edge_index[1].max() + 1) + edge_index[1]
    _, unique_idx = np.unique(combined, return_index=True)
    return edge_index[:, unique_idx]


def _cap_degree(edge_index, positions, k_max):
    """Keep at most k_max nearest neighbors per source node."""
    src, dst = edge_index
    n = max(src.max(), dst.max()) + 1

    # Compute distances for all edges
    dx = positions[src, 0] - positions[dst, 0]
    dy = positions[src, 1] - positions[dst, 1]
    dists = np.sqrt(dx**2 + dy**2)

    keep = np.ones(len(src), dtype=bool)

    for node in range(n):
        node_mask = src == node
        if node_mask.sum() <= k_max:
            continue
        node_indices = np.where(node_mask)[0]
        node_dists = dists[node_indices]
        # Keep k_max nearest
        sorted_idx = np.argsort(node_dists)
        drop = node_indices[sorted_idx[k_max:]]
        keep[drop] = False

    return edge_index[:, keep]


def _compute_diagnostics(edge_index, n_nodes):
    """Compute graph statistics: node/edge counts, degree distribution."""
    if edge_index.shape[1] == 0:
        return _empty_diagnostics(n_nodes)

    degree = np.bincount(edge_index[0], minlength=n_nodes)
    return {
        "n_nodes": n_nodes,
        "n_edges": edge_index.shape[1],
        "avg_degree": float(degree.mean()),
        "min_degree": int(degree.min()),
        "max_degree": int(degree.max()),
        "n_isolated": int((degree == 0).sum()),
    }


def _empty_diagnostics(n_nodes):
    """Return zero-valued diagnostics dict for a graph with no edges."""
    return {
        "n_nodes": n_nodes,
        "n_edges": 0,
        "avg_degree": 0.0,
        "min_degree": 0,
        "max_degree": 0,
        "n_isolated": n_nodes,
    }


def compute_edge_features(positions, times, energies, edge_index):
    """Compute 8-dim edge features for directed edges.

    Parameters
    ----------
    positions : ndarray, shape (n, 2)
        Hit (x, y) in mm.
    times : ndarray, shape (n,)
        Hit times in ns.
    energies : ndarray, shape (n,)
        Hit energies in MeV.
    edge_index : ndarray, shape (2, n_edges)
        Directed edges.

    Returns
    -------
    edge_attr : ndarray, shape (n_edges, 8)
    """
    src, dst = edge_index
    if len(src) == 0:
        return np.empty((0, 8), dtype=np.float32)

    x_s, y_s = positions[src, 0], positions[src, 1]
    x_d, y_d = positions[dst, 0], positions[dst, 1]
    t_s, t_d = times[src], times[dst]
    e_s, e_d = energies[src], energies[dst]

    dx = x_s - x_d
    dy = y_s - y_d
    dist = np.sqrt(dx**2 + dy**2)
    dt = t_s - t_d

    log_e_s = np.log1p(e_s)
    log_e_d = np.log1p(e_d)
    d_log_e = log_e_s - log_e_d

    e_sum = e_s + e_d
    e_asym = np.where(e_sum > 0, (e_s - e_d) / e_sum, 0.0)
    log_sum_e = np.log1p(e_sum)

    r_s = np.sqrt(x_s**2 + y_s**2)
    r_d = np.sqrt(x_d**2 + y_d**2)
    dr = r_s - r_d

    edge_attr = np.stack([dx, dy, dist, dt, d_log_e, e_asym, log_sum_e, dr],
                         axis=1).astype(np.float32)
    return edge_attr


def compute_node_features(positions, times, energies):
    """Compute 6-dim node features.

    Parameters
    ----------
    positions : ndarray, shape (n, 2)
        Hit (x, y) in mm.
    times : ndarray, shape (n,)
        Hit times in ns.
    energies : ndarray, shape (n,)
        Hit energies in MeV.

    Returns
    -------
    node_feat : ndarray, shape (n, 6)
    """
    n = len(energies)
    if n == 0:
        return np.empty((0, 6), dtype=np.float32)

    log_e = np.log1p(energies)
    r = np.sqrt(positions[:, 0]**2 + positions[:, 1]**2)
    e_max = energies.max()
    rel_e = energies / e_max if e_max > 0 else np.zeros(n)

    node_feat = np.stack([log_e, times, positions[:, 0], positions[:, 1],
                          r, rel_e], axis=1).astype(np.float32)
    return node_feat
