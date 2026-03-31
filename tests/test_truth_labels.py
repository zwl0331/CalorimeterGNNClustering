"""Unit tests for src/data/truth_labels.py."""

import numpy as np
import sys
import unittest
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.truth_labels import assign_bfs_truth, assign_mc_truth


# ───────────────────────────────────────────────────────────────────
# BFS pseudo-truth tests
# ───────────────────────────────────────────────────────────────────

class TestAssignBfsTruth(unittest.TestCase):
    """Tests for assign_bfs_truth()."""

    def test_same_cluster_positive(self):
        """Two hits in the same cluster → edge label 1."""
        cluster_idx = np.array([0, 0, 1])
        edge_index = np.array([[0, 1], [1, 0]])  # 0↔1
        y, mask = assign_bfs_truth(cluster_idx, edge_index)
        assert mask.all(), "Both hits assigned → both edges valid"
        assert (y == 1).all(), "Same cluster → label 1"

    def test_different_cluster_negative(self):
        """Two hits in different clusters → edge label 0."""
        cluster_idx = np.array([0, 1])
        edge_index = np.array([[0, 1], [1, 0]])
        y, mask = assign_bfs_truth(cluster_idx, edge_index)
        assert mask.all()
        assert (y == 0).all()

    def test_unassigned_hit_masked(self):
        """Hit with clusterIdx -1 → edge masked out."""
        cluster_idx = np.array([0, -1, 0])
        edge_index = np.array([[0, 1, 0, 2], [1, 0, 2, 0]])
        y, mask = assign_bfs_truth(cluster_idx, edge_index)
        # Edges 0↔1 involve hit 1 (unassigned) → masked
        assert not mask[0] and not mask[1]
        # Edges 0↔2 are both assigned to cluster 0 → valid, label 1
        assert mask[2] and mask[3]
        assert y[2] == 1 and y[3] == 1

    def test_all_unassigned(self):
        """All hits unassigned → all edges masked."""
        cluster_idx = np.array([-1, -1, -1])
        edge_index = np.array([[0, 1, 2], [1, 2, 0]])
        y, mask = assign_bfs_truth(cluster_idx, edge_index)
        assert not mask.any()

    def test_empty_graph(self):
        """No edges → empty arrays returned."""
        cluster_idx = np.array([0, 1])
        edge_index = np.empty((2, 0), dtype=np.int64)
        y, mask = assign_bfs_truth(cluster_idx, edge_index)
        assert len(y) == 0
        assert len(mask) == 0

    def test_output_shapes(self):
        """Output shapes match number of edges."""
        cluster_idx = np.array([0, 0, 1, 1, 2])
        edge_index = np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])
        y, mask = assign_bfs_truth(cluster_idx, edge_index)
        assert y.shape == (5,)
        assert mask.shape == (5,)

    def test_output_dtypes(self):
        """y_edge is int64, edge_mask is bool."""
        cluster_idx = np.array([0, 0])
        edge_index = np.array([[0], [1]])
        y, mask = assign_bfs_truth(cluster_idx, edge_index)
        assert y.dtype == np.int64
        assert mask.dtype == bool


# ───────────────────────────────────────────────────────────────────
# MC truth tests
# ───────────────────────────────────────────────────────────────────

class TestAssignMcTruth(unittest.TestCase):
    """Tests for assign_mc_truth()."""

    def test_same_particle_positive(self):
        """Two hits from the same SimParticle → edge label 1."""
        sim_ids = [[100], [100]]
        edeps = [[5.0], [3.0]]
        disks = np.array([0, 0])
        edge_index = np.array([[0, 1], [1, 0]])

        y, mask, tc, amb = assign_mc_truth(sim_ids, edeps, disks, edge_index)
        assert mask.all()
        assert (y == 1).all()
        assert not amb.any()
        assert tc[0] == tc[1]  # same truth cluster

    def test_different_particle_negative(self):
        """Two hits from different SimParticles → edge label 0."""
        sim_ids = [[100], [200]]
        edeps = [[5.0], [3.0]]
        disks = np.array([0, 0])
        edge_index = np.array([[0, 1], [1, 0]])

        y, mask, tc, amb = assign_mc_truth(sim_ids, edeps, disks, edge_index)
        assert mask.all()
        assert (y == 0).all()
        assert tc[0] != tc[1]

    def test_ambiguous_hit_masked(self):
        """Hit with purity < threshold → ambiguous, edge masked."""
        # Hit 0: 60% from particle 100, 40% from 200 → ambiguous (< 0.7)
        # Hit 1: 100% from particle 100 → not ambiguous
        sim_ids = [[100, 200], [100]]
        edeps = [[0.6, 0.4], [1.0]]
        disks = np.array([0, 0])
        edge_index = np.array([[0, 1], [1, 0]])

        y, mask, tc, amb = assign_mc_truth(sim_ids, edeps, disks, edge_index)
        assert amb[0] == True, "Hit 0 should be ambiguous"
        assert amb[1] == False, "Hit 1 should not be ambiguous"
        assert not mask.any(), "All edges involve ambiguous hit 0 → masked"

    def test_purity_threshold_boundary(self):
        """Hit with purity == threshold → not ambiguous."""
        sim_ids = [[100, 200]]
        edeps = [[0.7, 0.3]]
        disks = np.array([0])
        edge_index = np.empty((2, 0), dtype=np.int64)

        _, _, tc, amb = assign_mc_truth(sim_ids, edeps, disks, edge_index,
                                        purity_threshold=0.7)
        assert not amb[0], "Purity == threshold should not be ambiguous"
        assert tc[0] >= 0

    def test_purity_just_below_threshold(self):
        """Hit with purity just below threshold → ambiguous."""
        sim_ids = [[100, 200]]
        edeps = [[0.69, 0.31]]
        disks = np.array([0])
        edge_index = np.empty((2, 0), dtype=np.int64)

        _, _, tc, amb = assign_mc_truth(sim_ids, edeps, disks, edge_index,
                                        purity_threshold=0.7)
        assert amb[0], "Purity below threshold should be ambiguous"
        assert tc[0] == -1

    def test_different_disks_different_clusters(self):
        """Same SimParticle on different disks → different truth clusters."""
        sim_ids = [[100], [100]]
        edeps = [[5.0], [3.0]]
        disks = np.array([0, 1])
        edge_index = np.array([[0, 1], [1, 0]])

        y, mask, tc, amb = assign_mc_truth(sim_ids, edeps, disks, edge_index)
        assert mask.all()
        assert (y == 0).all(), "Different disks → different clusters → label 0"
        assert tc[0] != tc[1]

    def test_empty_simparticle_info(self):
        """Hit with no SimParticle info → ambiguous."""
        sim_ids = [[], [100]]
        edeps = [[], [1.0]]
        disks = np.array([0, 0])
        edge_index = np.array([[0, 1], [1, 0]])

        y, mask, tc, amb = assign_mc_truth(sim_ids, edeps, disks, edge_index)
        assert amb[0] == True
        assert tc[0] == -1
        assert not mask.any()

    def test_zero_energy_ambiguous(self):
        """Hit with zero total energy → ambiguous."""
        sim_ids = [[100]]
        edeps = [[0.0]]
        disks = np.array([0])
        edge_index = np.empty((2, 0), dtype=np.int64)

        _, _, tc, amb = assign_mc_truth(sim_ids, edeps, disks, edge_index)
        assert amb[0] == True
        assert tc[0] == -1

    def test_multi_hit_cluster(self):
        """Multiple hits from the same particle → all in same truth cluster."""
        sim_ids = [[10], [10], [10], [20]]
        edeps = [[1.0], [2.0], [3.0], [1.0]]
        disks = np.array([0, 0, 0, 0])
        # Fully connected edges among 0,1,2 + edges to 3
        edge_index = np.array([
            [0, 0, 1, 1, 2, 2, 0, 3],
            [1, 2, 0, 2, 0, 1, 3, 0],
        ])

        y, mask, tc, amb = assign_mc_truth(sim_ids, edeps, disks, edge_index)
        assert mask.all()
        assert not amb.any()
        # Hits 0,1,2 share cluster; hit 3 is different
        assert tc[0] == tc[1] == tc[2]
        assert tc[3] != tc[0]
        # Edges within {0,1,2} → label 1; edges to 3 → label 0
        for idx in range(6):
            assert y[idx] == 1, f"Edge {idx} within same particle should be 1"
        assert y[6] == 0 and y[7] == 0

    def test_output_shapes_and_dtypes(self):
        """Check shapes and dtypes of all outputs."""
        sim_ids = [[100], [200], [100]]
        edeps = [[1.0], [1.0], [1.0]]
        disks = np.array([0, 0, 0])
        edge_index = np.array([[0, 1, 2], [1, 2, 0]])

        y, mask, tc, amb = assign_mc_truth(sim_ids, edeps, disks, edge_index)
        assert y.shape == (3,)
        assert mask.shape == (3,)
        assert tc.shape == (3,)
        assert amb.shape == (3,)
        assert y.dtype == np.int64
        assert mask.dtype == bool
        assert tc.dtype == np.int64
        assert amb.dtype == bool

    def test_custom_purity_threshold(self):
        """Custom purity threshold changes ambiguity classification."""
        sim_ids = [[100, 200]]
        edeps = [[0.55, 0.45]]
        disks = np.array([0])
        edge_index = np.empty((2, 0), dtype=np.int64)

        # With default 0.7 → ambiguous
        _, _, _, amb_strict = assign_mc_truth(sim_ids, edeps, disks, edge_index,
                                              purity_threshold=0.7)
        assert amb_strict[0] == True

        # With threshold 0.5 → not ambiguous
        _, _, _, amb_loose = assign_mc_truth(sim_ids, edeps, disks, edge_index,
                                             purity_threshold=0.5)
        assert amb_loose[0] == False


if __name__ == "__main__":
    unittest.main(verbosity=2)
