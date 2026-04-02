"""Unit tests for src/inference/postprocess.py."""

import unittest
import numpy as np

from src.inference.postprocess import (
    compute_cluster_features,
    compute_summary_statistics,
)


class TestComputeClusterFeatures(unittest.TestCase):
    """Tests for per-cluster feature computation."""

    def test_single_cluster(self):
        """Two hits in one cluster → correct centroid and energy."""
        labels = np.array([0, 0])
        positions = np.array([[0.0, 0.0], [10.0, 0.0]])
        energies = np.array([100.0, 100.0])
        times = np.array([500.0, 500.0])

        clusters = compute_cluster_features(labels, positions, energies, times)
        self.assertEqual(len(clusters), 1)

        c = clusters[0]
        self.assertEqual(c["n_hits"], 2)
        self.assertAlmostEqual(c["total_energy"], 200.0)
        self.assertAlmostEqual(c["centroid_x"], 5.0)  # midpoint
        self.assertAlmostEqual(c["centroid_y"], 0.0)
        self.assertAlmostEqual(c["time"], 500.0)
        self.assertAlmostEqual(c["max_hit_fraction"], 0.5)

    def test_energy_weighted_centroid(self):
        """Centroid pulled toward higher-energy hit."""
        labels = np.array([0, 0])
        positions = np.array([[0.0, 0.0], [10.0, 0.0]])
        energies = np.array([300.0, 100.0])  # 3:1 ratio
        times = np.array([500.0, 600.0])

        clusters = compute_cluster_features(labels, positions, energies, times)
        c = clusters[0]
        # Centroid: (300*0 + 100*10) / 400 = 2.5
        self.assertAlmostEqual(c["centroid_x"], 2.5)
        # Time: (300*500 + 100*600) / 400 = 525
        self.assertAlmostEqual(c["time"], 525.0)
        self.assertAlmostEqual(c["max_hit_fraction"], 0.75)

    def test_rms_width(self):
        """RMS width for symmetric 2-hit cluster."""
        labels = np.array([0, 0])
        positions = np.array([[-5.0, 0.0], [5.0, 0.0]])
        energies = np.array([100.0, 100.0])
        times = np.array([0.0, 0.0])

        clusters = compute_cluster_features(labels, positions, energies, times)
        c = clusters[0]
        # Centroid at (0, 0). Each hit 5mm away. RMS = sqrt(25) = 5.0
        self.assertAlmostEqual(c["rms_width"], 5.0, places=3)

    def test_unclustered_hits_excluded(self):
        """Hits with label -1 are excluded from all clusters."""
        labels = np.array([0, -1, 0])
        positions = np.array([[0.0, 0.0], [100.0, 100.0], [10.0, 0.0]])
        energies = np.array([100.0, 999.0, 100.0])
        times = np.array([0.0, 0.0, 0.0])

        clusters = compute_cluster_features(labels, positions, energies, times)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0]["n_hits"], 2)
        self.assertAlmostEqual(clusters[0]["total_energy"], 200.0)

    def test_multiple_clusters(self):
        """Two separate clusters computed independently."""
        labels = np.array([0, 0, 1, 1])
        positions = np.array([[0.0, 0.0], [1.0, 0.0], [100.0, 0.0], [101.0, 0.0]])
        energies = np.array([50.0, 50.0, 200.0, 200.0])
        times = np.array([10.0, 10.0, 20.0, 20.0])

        clusters = compute_cluster_features(labels, positions, energies, times)
        self.assertEqual(len(clusters), 2)

        c0 = [c for c in clusters if c["cluster_id"] == 0][0]
        c1 = [c for c in clusters if c["cluster_id"] == 1][0]
        self.assertAlmostEqual(c0["total_energy"], 100.0)
        self.assertAlmostEqual(c1["total_energy"], 400.0)

    def test_empty_labels(self):
        """All unclustered → empty list."""
        labels = np.array([-1, -1, -1])
        positions = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        energies = np.array([100.0, 100.0, 100.0])
        times = np.array([0.0, 0.0, 0.0])

        clusters = compute_cluster_features(labels, positions, energies, times)
        self.assertEqual(len(clusters), 0)

    def test_single_hit_cluster(self):
        """Single-hit cluster has RMS width 0."""
        labels = np.array([0])
        positions = np.array([[5.0, 10.0]])
        energies = np.array([50.0])
        times = np.array([100.0])

        clusters = compute_cluster_features(labels, positions, energies, times)
        self.assertEqual(len(clusters), 1)
        self.assertAlmostEqual(clusters[0]["rms_width"], 0.0)
        self.assertAlmostEqual(clusters[0]["max_hit_fraction"], 1.0)

    def test_hit_indices_correct(self):
        """hit_indices maps back to original array positions."""
        labels = np.array([-1, 0, -1, 0, -1])
        positions = np.zeros((5, 2))
        energies = np.array([0.0, 50.0, 0.0, 50.0, 0.0])
        times = np.zeros(5)

        clusters = compute_cluster_features(labels, positions, energies, times)
        self.assertEqual(clusters[0]["hit_indices"], [1, 3])


class TestComputeSummaryStatistics(unittest.TestCase):
    """Tests for aggregate summary statistics."""

    def test_basic_summary(self):
        """Summary over two clusters."""
        clusters = [
            {"n_hits": 3, "total_energy": 100.0},
            {"n_hits": 5, "total_energy": 300.0},
        ]
        s = compute_summary_statistics(clusters)
        self.assertEqual(s["n_clusters"], 2)
        self.assertAlmostEqual(s["mean_n_hits"], 4.0)
        self.assertAlmostEqual(s["median_n_hits"], 4.0)
        self.assertAlmostEqual(s["mean_energy"], 200.0)
        self.assertAlmostEqual(s["median_energy"], 200.0)

    def test_empty_list(self):
        """Empty cluster list returns zeros."""
        s = compute_summary_statistics([])
        self.assertEqual(s["n_clusters"], 0)
        self.assertAlmostEqual(s["mean_n_hits"], 0.0)


if __name__ == "__main__":
    unittest.main()
