"""
Cluster postprocessing: compute per-cluster physics features.

Given cluster labels and hit-level data, compute per-cluster:
  - Hit list, n_hits, total energy
  - Energy-weighted centroid (x, y), energy-weighted time
  - RMS spatial width, max-hit energy fraction
"""

import numpy as np


def compute_cluster_features(cluster_labels, positions, energies, times):
    """Compute physics features for each reconstructed cluster.

    Parameters
    ----------
    cluster_labels : ndarray (N,)
        Integer cluster ID per hit. -1 = unclustered.
    positions : ndarray (N, 2)
        Hit positions (x, y) in mm.
    energies : ndarray (N,)
        Hit energies in MeV.
    times : ndarray (N,)
        Hit times in ns.

    Returns
    -------
    clusters : list of dict
        One dict per cluster with keys:
        - cluster_id : int
        - hit_indices : list of int
        - n_hits : int
        - total_energy : float (MeV)
        - centroid_x, centroid_y : float (energy-weighted, mm)
        - time : float (energy-weighted, ns)
        - rms_width : float (energy-weighted RMS distance from centroid, mm)
        - max_hit_fraction : float (energy of most energetic hit / total)
    """
    valid_ids = np.unique(cluster_labels[cluster_labels >= 0])
    clusters = []

    for cid in valid_ids:
        mask = cluster_labels == cid
        idx = np.where(mask)[0]

        e = energies[idx]
        pos = positions[idx]
        t = times[idx]

        total_e = e.sum()
        if total_e <= 0:
            continue

        w = e / total_e
        cx = np.dot(w, pos[:, 0])
        cy = np.dot(w, pos[:, 1])
        ct = np.dot(w, t)

        dx = pos[:, 0] - cx
        dy = pos[:, 1] - cy
        rms = np.sqrt(np.dot(w, dx**2 + dy**2))

        clusters.append({
            "cluster_id": int(cid),
            "hit_indices": idx.tolist(),
            "n_hits": int(len(idx)),
            "total_energy": float(total_e),
            "centroid_x": float(cx),
            "centroid_y": float(cy),
            "time": float(ct),
            "rms_width": float(rms),
            "max_hit_fraction": float(e.max() / total_e),
        })

    return clusters


def compute_summary_statistics(clusters):
    """Compute aggregate statistics over all clusters.

    Parameters
    ----------
    clusters : list of dict
        Output from compute_cluster_features.

    Returns
    -------
    dict with n_clusters, mean/median n_hits, mean/median energy.
    """
    if not clusters:
        return {
            "n_clusters": 0,
            "mean_n_hits": 0.0,
            "median_n_hits": 0.0,
            "mean_energy": 0.0,
            "median_energy": 0.0,
        }

    sizes = np.array([c["n_hits"] for c in clusters])
    energies = np.array([c["total_energy"] for c in clusters])

    return {
        "n_clusters": len(clusters),
        "mean_n_hits": float(sizes.mean()),
        "median_n_hits": float(np.median(sizes)),
        "mean_energy": float(energies.mean()),
        "median_energy": float(np.median(energies)),
    }
