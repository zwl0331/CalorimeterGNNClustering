"""Unit tests for src/data/truth_labels_primary.py."""

import numpy as np
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.truth_labels_primary import (
    assign_mc_truth_primary,
    build_calo_root_map,
)


class TestBuildCaloRootMap(unittest.TestCase):
    """Tests for build_calo_root_map()."""

    def test_no_ancestors_self_is_root(self):
        """SimParticle with no ancestors in calo → itself is calo-entrant."""
        sim_ids = [10]
        ancestors = [[]]  # no ancestors in calo
        hit_sim_ids = [[10]]
        hit_cry_ids = [100]
        crystal_disk = {100: 0}

        m = build_calo_root_map(sim_ids, ancestors, hit_sim_ids,
                                hit_cry_ids, crystal_disk)
        self.assertEqual(m[(10, 0)], 10)

    def test_ancestor_in_same_disk(self):
        """SimParticle whose ancestor also deposited in same disk → ancestor is root."""
        sim_ids = [4, 5]
        ancestors = [[], [4]]  # SimP 5's parent is 4
        hit_sim_ids = [[4], [5]]
        hit_cry_ids = [100, 101]
        crystal_disk = {100: 0, 101: 0}

        m = build_calo_root_map(sim_ids, ancestors, hit_sim_ids,
                                hit_cry_ids, crystal_disk)
        self.assertEqual(m[(5, 0)], 4, "SimP 5 should root to 4 on disk 0")
        self.assertEqual(m[(4, 0)], 4, "SimP 4 is its own root")

    def test_cross_disk_secondary_is_own_root(self):
        """Secondary that crosses from disk 0 to disk 1 → own root on disk 1."""
        # SimP 4 deposits in disk 0, SimP 5 (child of 4) deposits in disk 1
        sim_ids = [4, 5]
        ancestors = [[], [4]]
        hit_sim_ids = [[4], [5]]
        hit_cry_ids = [100, 700]
        crystal_disk = {100: 0, 700: 1}

        m = build_calo_root_map(sim_ids, ancestors, hit_sim_ids,
                                hit_cry_ids, crystal_disk)
        self.assertEqual(m[(4, 0)], 4)
        self.assertEqual(m[(5, 1)], 5,
                         "SimP 5 on disk 1 should be its own root (parent 4 not on disk 1)")

    def test_deep_chain(self):
        """Multi-level ancestry: grandchild → child → parent, all on same disk."""
        sim_ids = [1, 2, 3]
        ancestors = [[], [1], [2, 1]]  # 3→2→1
        hit_sim_ids = [[1], [2], [3]]
        hit_cry_ids = [100, 101, 102]
        crystal_disk = {100: 0, 101: 0, 102: 0}

        m = build_calo_root_map(sim_ids, ancestors, hit_sim_ids,
                                hit_cry_ids, crystal_disk)
        self.assertEqual(m[(3, 0)], 1, "SimP 3 should root to 1 (highest in chain on disk 0)")
        self.assertEqual(m[(2, 0)], 1, "SimP 2 should root to 1")
        self.assertEqual(m[(1, 0)], 1)

    def test_gap_in_chain(self):
        """Ancestor chain has gaps (intermediate not in calomcsim)."""
        # SimP 10 → SimP 6 (not in calo) → SimP 4 (in calo)
        sim_ids = [4, 10]
        ancestors = [[], [6, 4]]  # 6 not in sim_ids
        hit_sim_ids = [[4], [10]]
        hit_cry_ids = [100, 101]
        crystal_disk = {100: 0, 101: 0}

        m = build_calo_root_map(sim_ids, ancestors, hit_sim_ids,
                                hit_cry_ids, crystal_disk)
        self.assertEqual(m[(10, 0)], 4,
                         "SimP 10 should root to 4 (skip gap at 6)")

    def test_both_disks(self):
        """SimParticle depositing in both disks → separate entries per disk."""
        sim_ids = [10]
        ancestors = [[]]
        hit_sim_ids = [[10], [10]]
        hit_cry_ids = [100, 700]
        crystal_disk = {100: 0, 700: 1}

        m = build_calo_root_map(sim_ids, ancestors, hit_sim_ids,
                                hit_cry_ids, crystal_disk)
        self.assertEqual(m[(10, 0)], 10)
        self.assertEqual(m[(10, 1)], 10)


class TestAssignMcTruthPrimary(unittest.TestCase):
    """Tests for assign_mc_truth_primary()."""

    def _simple_root_map(self, mapping):
        """Helper: build calo_root_map from {(pid, disk): root} dict."""
        return mapping

    def test_same_shower_positive(self):
        """Two hits from different SimPs but same calo-root → edge label 1."""
        sim_ids = [[4], [5]]
        edeps = [[5.0], [3.0]]
        disks = np.array([0, 0])
        edge_index = np.array([[0, 1], [1, 0]])
        root_map = {(4, 0): 4, (5, 0): 4}

        y, mask, tc, amb = assign_mc_truth_primary(
            sim_ids, edeps, disks, edge_index, root_map)
        self.assertTrue(mask.all())
        self.assertTrue((y == 1).all(),
                        "Same calo-root → same cluster → label 1")
        self.assertEqual(tc[0], tc[1])

    def test_different_showers_negative(self):
        """Two hits from different calo-roots → edge label 0."""
        sim_ids = [[4], [200]]
        edeps = [[5.0], [3.0]]
        disks = np.array([0, 0])
        edge_index = np.array([[0, 1], [1, 0]])
        root_map = {(4, 0): 4, (200, 0): 200}

        y, mask, tc, amb = assign_mc_truth_primary(
            sim_ids, edeps, disks, edge_index, root_map)
        self.assertTrue(mask.all())
        self.assertTrue((y == 0).all())
        self.assertNotEqual(tc[0], tc[1])

    def test_ambiguity_resolved_by_grouping(self):
        """Hit with two SimPs from same root → NOT ambiguous (deposits sum)."""
        # Hit 0: 40% from SimP 5, 60% from SimP 11 — both root to 4
        sim_ids = [[5, 11]]
        edeps = [[0.4, 0.6]]
        disks = np.array([0])
        edge_index = np.empty((2, 0), dtype=np.int64)
        root_map = {(5, 0): 4, (11, 0): 4}

        _, _, tc, amb = assign_mc_truth_primary(
            sim_ids, edeps, disks, edge_index, root_map)
        self.assertFalse(amb[0],
                         "Both SimPs root to 4 → purity 100% → not ambiguous")
        self.assertGreaterEqual(tc[0], 0)

    def test_still_ambiguous_different_roots(self):
        """Hit with two SimPs from different roots below threshold → ambiguous."""
        sim_ids = [[100, 200]]
        edeps = [[0.6, 0.4]]
        disks = np.array([0])
        edge_index = np.empty((2, 0), dtype=np.int64)
        root_map = {(100, 0): 100, (200, 0): 200}

        _, _, tc, amb = assign_mc_truth_primary(
            sim_ids, edeps, disks, edge_index, root_map)
        self.assertTrue(amb[0], "Different roots, purity 60% < 70% → ambiguous")
        self.assertEqual(tc[0], -1)

    def test_cross_disk_separate_clusters(self):
        """Same calo-root on different disks → different truth clusters."""
        sim_ids = [[10], [10]]
        edeps = [[5.0], [3.0]]
        disks = np.array([0, 1])
        edge_index = np.array([[0, 1], [1, 0]])
        root_map = {(10, 0): 10, (10, 1): 10}

        y, mask, tc, amb = assign_mc_truth_primary(
            sim_ids, edeps, disks, edge_index, root_map)
        self.assertTrue(mask.all())
        self.assertTrue((y == 0).all(),
                        "Different disks → different clusters")
        self.assertNotEqual(tc[0], tc[1])

    def test_fallback_when_pid_not_in_root_map(self):
        """SimParticle not in calo_root_map → falls back to pid itself."""
        sim_ids = [[999]]
        edeps = [[1.0]]
        disks = np.array([0])
        edge_index = np.empty((2, 0), dtype=np.int64)
        root_map = {}  # empty

        _, _, tc, amb = assign_mc_truth_primary(
            sim_ids, edeps, disks, edge_index, root_map)
        self.assertFalse(amb[0])
        self.assertGreaterEqual(tc[0], 0)

    def test_empty_hit(self):
        """Hit with no SimParticle info → ambiguous."""
        sim_ids = [[], [100]]
        edeps = [[], [1.0]]
        disks = np.array([0, 0])
        edge_index = np.array([[0, 1], [1, 0]])
        root_map = {(100, 0): 100}

        y, mask, tc, amb = assign_mc_truth_primary(
            sim_ids, edeps, disks, edge_index, root_map)
        self.assertTrue(amb[0])
        self.assertEqual(tc[0], -1)
        self.assertFalse(mask.any())

    def test_output_shapes_and_dtypes(self):
        """Check shapes and dtypes of all outputs."""
        sim_ids = [[10], [20], [10]]
        edeps = [[1.0], [1.0], [1.0]]
        disks = np.array([0, 0, 0])
        edge_index = np.array([[0, 1, 2], [1, 2, 0]])
        root_map = {(10, 0): 10, (20, 0): 20}

        y, mask, tc, amb = assign_mc_truth_primary(
            sim_ids, edeps, disks, edge_index, root_map)
        self.assertEqual(y.shape, (3,))
        self.assertEqual(mask.shape, (3,))
        self.assertEqual(tc.shape, (3,))
        self.assertEqual(amb.shape, (3,))
        self.assertEqual(y.dtype, np.int64)
        self.assertEqual(mask.dtype, bool)
        self.assertEqual(tc.dtype, np.int64)
        self.assertEqual(amb.dtype, bool)

    def test_multi_hit_shower(self):
        """Multiple hits from same shower (different SimPs) → same cluster."""
        # SimP 4 (signal), SimP 5 (brem from 4), SimP 11 (brem from shower)
        sim_ids = [[4], [5], [11], [200]]
        edeps = [[10.0], [2.0], [1.5], [5.0]]
        disks = np.array([0, 0, 0, 0])
        edge_index = np.array([
            [0, 0, 1, 1, 2, 2, 0, 3],
            [1, 2, 0, 2, 0, 1, 3, 0],
        ])
        root_map = {(4, 0): 4, (5, 0): 4, (11, 0): 4, (200, 0): 200}

        y, mask, tc, amb = assign_mc_truth_primary(
            sim_ids, edeps, disks, edge_index, root_map)
        self.assertTrue(mask.all())
        self.assertFalse(amb.any())
        # Hits 0,1,2 share calo-root 4 → same cluster
        self.assertEqual(tc[0], tc[1])
        self.assertEqual(tc[1], tc[2])
        # Hit 3 (root 200) → different cluster
        self.assertNotEqual(tc[0], tc[3])
        # Edges within shower → 1, edges to 200 → 0
        for idx in range(6):
            self.assertEqual(y[idx], 1)
        self.assertEqual(y[6], 0)
        self.assertEqual(y[7], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
