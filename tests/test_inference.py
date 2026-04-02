"""Unit tests for src/inference/cluster_reco.py."""

import unittest
import numpy as np
import torch

from src.inference.cluster_reco import (
    symmetrize_edge_scores,
    reconstruct_clusters,
)


class TestSymmetrizeEdgeScores(unittest.TestCase):
    """Tests for directed → undirected edge score averaging."""

    def test_bidirectional_pair(self):
        """Two directed edges (0→1, 1→0) average to one undirected edge."""
        edge_index = np.array([[0, 1], [1, 0]])
        probs = np.array([0.8, 0.6])
        ei, ep = symmetrize_edge_scores(edge_index, probs)
        self.assertEqual(ei.shape[1], 1)
        self.assertAlmostEqual(ep[0], 0.7, places=5)

    def test_single_direction(self):
        """Edge in one direction only keeps its original score."""
        edge_index = np.array([[0], [1]])
        probs = np.array([0.9])
        ei, ep = symmetrize_edge_scores(edge_index, probs)
        self.assertEqual(ei.shape[1], 1)
        self.assertAlmostEqual(ep[0], 0.9, places=5)

    def test_canonical_ordering(self):
        """Undirected edges always have i < j."""
        edge_index = np.array([[3, 1], [1, 3]])
        probs = np.array([0.4, 0.6])
        ei, ep = symmetrize_edge_scores(edge_index, probs)
        self.assertTrue(ei[0, 0] < ei[1, 0])

    def test_multiple_pairs(self):
        """Multiple pairs symmetrized independently."""
        edge_index = np.array([[0, 1, 2, 3], [1, 0, 3, 2]])
        probs = np.array([0.8, 0.6, 0.9, 0.7])
        ei, ep = symmetrize_edge_scores(edge_index, probs)
        self.assertEqual(ei.shape[1], 2)
        # Sort by first node for consistent comparison
        order = np.argsort(ei[0])
        self.assertAlmostEqual(ep[order[0]], 0.7, places=5)  # (0,1)
        self.assertAlmostEqual(ep[order[1]], 0.8, places=5)  # (2,3)


class TestReconstructClusters(unittest.TestCase):
    """Tests for full cluster reconstruction pipeline."""

    def test_two_clusters(self):
        """4 nodes, 2 pairs connected → 2 clusters."""
        # 0-1 connected, 2-3 connected
        edge_index = np.array([[0, 1, 2, 3], [1, 0, 3, 2]])
        logits = np.array([5.0, 5.0, 5.0, 5.0])  # high sigmoid → ~1.0
        energies = np.array([100.0, 100.0, 100.0, 100.0])

        labels, probs = reconstruct_clusters(
            edge_index, logits, n_nodes=4, energies=energies,
            tau_edge=0.5, min_hits=1, min_energy_mev=0.0,
        )
        # Should have 2 clusters
        self.assertEqual(len(np.unique(labels[labels >= 0])), 2)
        # Nodes 0 and 1 in same cluster
        self.assertEqual(labels[0], labels[1])
        # Nodes 2 and 3 in same cluster
        self.assertEqual(labels[2], labels[3])
        # But different from 0-1
        self.assertNotEqual(labels[0], labels[2])

    def test_threshold_separates(self):
        """Low-confidence edges are filtered out."""
        edge_index = np.array([[0, 1, 1, 2], [1, 0, 2, 1]])
        # 0↔1 high confidence, 1↔2 low confidence
        logits = np.array([5.0, 5.0, -5.0, -5.0])
        energies = np.array([100.0, 100.0, 100.0])

        labels, _ = reconstruct_clusters(
            edge_index, logits, n_nodes=3, energies=energies,
            tau_edge=0.5, min_hits=1, min_energy_mev=0.0,
        )
        self.assertEqual(labels[0], labels[1])
        self.assertNotEqual(labels[0], labels[2])

    def test_min_hits_cleanup(self):
        """Clusters smaller than min_hits get label -1."""
        # Single edge connecting 0-1, node 2 isolated
        edge_index = np.array([[0, 1], [1, 0]])
        logits = np.array([5.0, 5.0])
        energies = np.array([100.0, 100.0, 100.0])

        labels, _ = reconstruct_clusters(
            edge_index, logits, n_nodes=3, energies=energies,
            tau_edge=0.5, min_hits=2, min_energy_mev=0.0,
        )
        # 0-1 cluster kept (2 hits)
        self.assertEqual(labels[0], labels[1])
        self.assertTrue(labels[0] >= 0)
        # Node 2 is isolated → -1
        self.assertEqual(labels[2], -1)

    def test_min_energy_cleanup(self):
        """Clusters below min_energy get label -1."""
        edge_index = np.array([[0, 1, 2, 3], [1, 0, 3, 2]])
        logits = np.array([5.0, 5.0, 5.0, 5.0])
        # Cluster 0-1 has 200 MeV, cluster 2-3 has 5 MeV
        energies = np.array([100.0, 100.0, 2.5, 2.5])

        labels, _ = reconstruct_clusters(
            edge_index, logits, n_nodes=4, energies=energies,
            tau_edge=0.5, min_hits=1, min_energy_mev=10.0,
        )
        # 0-1 kept
        self.assertTrue(labels[0] >= 0)
        self.assertEqual(labels[0], labels[1])
        # 2-3 removed (5 MeV < 10 MeV threshold)
        self.assertEqual(labels[2], -1)
        self.assertEqual(labels[3], -1)

    def test_no_edges_above_threshold(self):
        """All edges below threshold → all nodes unclustered."""
        edge_index = np.array([[0, 1], [1, 0]])
        logits = np.array([-5.0, -5.0])  # sigmoid ~ 0.007

        labels, _ = reconstruct_clusters(
            edge_index, logits, n_nodes=2, tau_edge=0.5,
        )
        self.assertTrue(np.all(labels == -1))

    def test_contiguous_relabeling(self):
        """Cluster IDs are relabeled to 0, 1, 2, ... after cleanup."""
        # 3 disconnected pairs
        edge_index = np.array([[0, 1, 2, 3, 4, 5], [1, 0, 3, 2, 5, 4]])
        logits = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
        energies = np.array([50.0, 50.0, 50.0, 50.0, 50.0, 50.0])

        labels, _ = reconstruct_clusters(
            edge_index, logits, n_nodes=6, energies=energies,
            tau_edge=0.5, min_hits=1, min_energy_mev=0.0,
        )
        valid = labels[labels >= 0]
        unique = np.unique(valid)
        # Should be 0, 1, 2
        np.testing.assert_array_equal(unique, np.arange(len(unique)))

    def test_torch_tensors_accepted(self):
        """Function accepts torch Tensors (auto-converts)."""
        edge_index = torch.tensor([[0, 1], [1, 0]])
        logits = torch.tensor([5.0, 5.0])
        energies = torch.tensor([100.0, 100.0])

        labels, probs = reconstruct_clusters(
            edge_index, logits, n_nodes=2, energies=energies,
            tau_edge=0.5, min_hits=1, min_energy_mev=0.0,
        )
        self.assertEqual(labels[0], labels[1])
        self.assertTrue(labels[0] >= 0)

    def test_no_symmetrize(self):
        """symmetrize=False uses directed scores directly."""
        # Only one direction: 0→1 high, but no 1→0
        edge_index = np.array([[0], [1]])
        logits = np.array([5.0])
        energies = np.array([100.0, 100.0])

        labels, _ = reconstruct_clusters(
            edge_index, logits, n_nodes=2, energies=energies,
            tau_edge=0.5, min_hits=1, min_energy_mev=0.0,
            symmetrize=False,
        )
        # Still connected (directed edge above threshold)
        self.assertEqual(labels[0], labels[1])


if __name__ == "__main__":
    unittest.main()
