"""
PyG Dataset for calorimeter hit graphs.

Reads EventNtuple ROOT files (MDC2025-002 format), extracts per-disk
graphs as PyG Data objects with node/edge features and truth labels.

Two usage modes:
  1. process_and_save(): build graphs from ROOT files, save as .pt files
  2. CaloGraphDataset: PyG Dataset that loads saved .pt files
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data, Dataset

from src.data.graph_builder import (
    build_graph,
    compute_edge_features,
    compute_node_features,
)
from src.data.truth_labels import assign_bfs_truth, assign_mc_truth
from src.geometry.crystal_geometry import load_crystal_map


# MDC2025-002 branches
_BRANCHES_RECO = [
    "calohits.crystalId_",
    "calohits.eDep_",
    "calohits.time_",
    "calohits.clusterIdx_",
    "calohits.crystalPos_.fCoordinates.fX",
    "calohits.crystalPos_.fCoordinates.fY",
]

_BRANCHES_MC = [
    "calohitsmc.simParticleIds",
    "calohitsmc.eDeps",
    "calohitsmc.nsim",
]


def extract_events_from_file(filepath, crystal_map, graph_cfg, truth_mode="bfs_pseudo",
                             max_events=None):
    """Read one ROOT file and yield PyG Data objects (one per disk per event).

    Parameters
    ----------
    filepath : str or Path
        Path to ROOT file.
    crystal_map : dict
        crystalId -> (diskId, x_mm, y_mm).
    graph_cfg : dict
        Graph construction parameters (r_max_mm, dt_max_ns, k_min, k_max).
    truth_mode : str
        "bfs_pseudo" or "mc_truth".
    max_events : int or None
        Limit number of events to read.

    Yields
    ------
    data : torch_geometric.data.Data
        Per-disk graph with node/edge features and labels.
    event_idx : int
    disk_id : int
    diagnostics : dict
    """
    import uproot

    branches = list(_BRANCHES_RECO)
    if truth_mode == "mc_truth":
        branches += _BRANCHES_MC

    tree = uproot.open(str(filepath) + ":EventNtuple/ntuple")
    arrays = tree.arrays(branches, entry_stop=max_events)
    n_events = len(arrays)

    r_max = graph_cfg.get("r_max_mm", 150.0)
    dt_max = graph_cfg.get("dt_max_ns", 25.0)
    k_min = graph_cfg.get("k_min", 3)
    k_max = graph_cfg.get("k_max", 20)

    for ev in range(n_events):
        nhits = len(arrays["calohits.crystalId_"][ev])
        if nhits == 0:
            continue

        cryids = np.array(arrays["calohits.crystalId_"][ev], dtype=np.int64)
        energies = np.array(arrays["calohits.eDep_"][ev], dtype=np.float64)
        times = np.array(arrays["calohits.time_"][ev], dtype=np.float64)
        cluster_idx = np.array(arrays["calohits.clusterIdx_"][ev], dtype=np.int64)

        # Get positions — prefer in-file crystal positions (MDC2025-002)
        has_pos = "calohits.crystalPos_.fCoordinates.fX" in arrays.fields
        if has_pos:
            xs = np.array(arrays["calohits.crystalPos_.fCoordinates.fX"][ev], dtype=np.float64)
            ys = np.array(arrays["calohits.crystalPos_.fCoordinates.fY"][ev], dtype=np.float64)
        else:
            xs = np.zeros(nhits, dtype=np.float64)
            ys = np.zeros(nhits, dtype=np.float64)

        # Get disk per hit from crystal geometry
        disks = np.array([crystal_map[int(c)][0] if int(c) in crystal_map else -1
                          for c in cryids], dtype=np.int64)

        # Fall back to crystal_map positions if in-file positions are all zero
        if not has_pos or (np.all(xs == 0) and np.all(ys == 0)):
            for i, c in enumerate(cryids):
                c = int(c)
                if c in crystal_map:
                    _, xs[i], ys[i] = crystal_map[c]

        # MC truth data
        if truth_mode == "mc_truth":
            sim_ids = arrays["calohitsmc.simParticleIds"][ev]
            edeps_mc = arrays["calohitsmc.eDeps"][ev]

        # Process each disk separately
        for disk_id in [0, 1]:
            disk_mask = disks == disk_id
            n_disk = disk_mask.sum()
            if n_disk < 2:
                continue

            d_energies = energies[disk_mask]
            d_times = times[disk_mask]
            d_xs = xs[disk_mask]
            d_ys = ys[disk_mask]
            d_positions = np.stack([d_xs, d_ys], axis=1)
            d_cluster_idx = cluster_idx[disk_mask]

            # Build graph
            edge_index, diag = build_graph(
                d_positions, d_times,
                r_max=r_max, dt_max=dt_max, k_min=k_min, k_max=k_max,
            )

            if edge_index.shape[1] == 0:
                continue

            # Node features (6-dim)
            node_feat = compute_node_features(d_positions, d_times, d_energies)
            # Edge features (8-dim)
            edge_feat = compute_edge_features(d_positions, d_times, d_energies, edge_index)

            # Truth labels
            if truth_mode == "bfs_pseudo":
                y_edge, edge_mask = assign_bfs_truth(d_cluster_idx, edge_index)
                y_node = (d_cluster_idx >= 0).astype(np.int64)
                hit_truth_cluster = d_cluster_idx.copy()
            elif truth_mode == "mc_truth":
                # Get MC data for this disk's hits
                disk_indices = np.where(disk_mask)[0]
                d_sim_ids = [list(sim_ids[i]) for i in disk_indices]
                d_edeps_mc = [list(edeps_mc[i]) for i in disk_indices]
                d_disks = np.full(n_disk, disk_id, dtype=np.int64)

                y_edge, edge_mask, hit_truth_cluster, is_ambiguous = assign_mc_truth(
                    d_sim_ids, d_edeps_mc, d_disks, edge_index,
                )
                y_node = (~is_ambiguous).astype(np.int64)
            else:
                raise ValueError(f"Unknown truth_mode: {truth_mode}")

            # Build PyG Data
            data = Data(
                x=torch.from_numpy(node_feat),
                edge_index=torch.from_numpy(edge_index),
                edge_attr=torch.from_numpy(edge_feat),
                y_edge=torch.from_numpy(y_edge),
                edge_mask=torch.from_numpy(edge_mask),
                y_node=torch.from_numpy(y_node),
                hit_truth_cluster=torch.from_numpy(hit_truth_cluster),
                # Metadata
                n_hits=n_disk,
                disk_id=disk_id,
            )

            yield data, ev, disk_id, diag


class CaloGraphDataset(Dataset):
    """PyG Dataset that loads pre-built .pt graph files.

    Parameters
    ----------
    processed_dir : str or Path
        Directory containing event_XXXXX_disk_Y.pt files.
    file_list : list[str] or None
        If given, only load graphs from these source ROOT files
        (for split-aware loading). Matches on source filename stem.
    """

    def __init__(self, processed_dir, file_list=None, transform=None):
        self._processed_dir = Path(processed_dir)

        # Collect all .pt files
        all_files = sorted(self._processed_dir.glob("*.pt"))

        if file_list is not None:
            # Filter to graphs from specific source files
            allowed_stems = set()
            for f in file_list:
                allowed_stems.add(Path(f).stem)
            self._files = [f for f in all_files
                           if self._source_stem(f) in allowed_stems]
        else:
            self._files = all_files

        # Don't call super().__init__() — we manage our own file list.
        # But PyG Dataset needs certain attributes, so set them manually.
        self._indices = None
        self.transform = transform
        self.pre_transform = None
        self.pre_filter = None

    @staticmethod
    def _source_stem(pt_path):
        """Extract source ROOT file stem from .pt filename."""
        # Filename format: {source_stem}_evt{N}_disk{D}.pt
        name = pt_path.stem
        # Find last _evt and strip from there
        idx = name.rfind("_evt")
        return name[:idx] if idx >= 0 else name

    def len(self):
        return len(self._files)

    def get(self, idx):
        return torch.load(self._files[idx], weights_only=False)

    @property
    def file_paths(self):
        return self._files
