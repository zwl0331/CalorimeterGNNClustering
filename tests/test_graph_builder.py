"""Unit tests for src/data/graph_builder.py."""

import numpy as np
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.graph_builder import (
    build_graph,
    compute_edge_features,
    compute_node_features,
)


# ───────────────────────────────────────────────────────────────────
# build_graph tests
# ───────────────────────────────────────────────────────────────────

class TestBuildGraph(unittest.TestCase):
    """Tests for build_graph()."""

    def test_empty_input(self):
        """Zero hits → empty edge_index."""
        positions = np.empty((0, 2))
        times = np.empty(0)
        ei, diag = build_graph(positions, times)
        self.assertEqual(ei.shape, (2, 0))
        self.assertEqual(diag["n_nodes"], 0)
        self.assertEqual(diag["n_edges"], 0)

    def test_single_hit(self):
        """One hit → no edges possible."""
        positions = np.array([[0.0, 0.0]])
        times = np.array([500.0])
        ei, diag = build_graph(positions, times)
        self.assertEqual(ei.shape, (2, 0))
        self.assertEqual(diag["n_nodes"], 1)
        self.assertEqual(diag["n_isolated"], 1)

    def test_two_close_hits_connected(self):
        """Two hits within r_max and dt_max → bidirectional edge."""
        positions = np.array([[0.0, 0.0], [10.0, 0.0]])
        times = np.array([500.0, 502.0])
        ei, diag = build_graph(positions, times, r_max=50.0, dt_max=25.0)
        self.assertEqual(diag["n_edges"], 2)  # bidirectional
        # Both directions present
        edges = set(zip(ei[0].tolist(), ei[1].tolist()))
        self.assertIn((0, 1), edges)
        self.assertIn((1, 0), edges)

    def test_two_distant_hits_not_connected(self):
        """Two hits beyond r_max → no radius edge (may get kNN fallback)."""
        positions = np.array([[0.0, 0.0], [200.0, 0.0]])
        times = np.array([500.0, 502.0])
        # With kNN fallback, they still connect (only 2 nodes, k_min=3)
        ei, diag = build_graph(positions, times, r_max=50.0, dt_max=25.0, k_min=1)
        # With k_min=1, the kNN fallback will connect them since both are isolated
        # But if we set k_min=0, no fallback
        ei2, diag2 = build_graph(positions, times, r_max=50.0, dt_max=25.0,
                                  k_min=0)
        self.assertEqual(diag2["n_edges"], 0)

    def test_time_filter(self):
        """Hits within r_max but beyond dt_max → not connected."""
        positions = np.array([[0.0, 0.0], [10.0, 0.0]])
        times = np.array([500.0, 600.0])  # Δt=100 >> dt_max=25
        ei, diag = build_graph(positions, times, r_max=150.0, dt_max=25.0,
                                k_min=0)
        self.assertEqual(diag["n_edges"], 0)

    def test_knn_fallback_connects_isolated(self):
        """Isolated node beyond r_max gets kNN fallback edges."""
        # Three close hits + one far hit
        positions = np.array([
            [0.0, 0.0], [10.0, 0.0], [0.0, 10.0],  # cluster
            [300.0, 0.0],  # isolated
        ])
        times = np.array([500.0, 501.0, 502.0, 503.0])
        ei, diag = build_graph(positions, times, r_max=50.0, dt_max=25.0,
                                k_min=2)
        # Node 3 should have at least some edges from kNN fallback
        node3_edges = (ei[0] == 3).sum()
        self.assertGreaterEqual(node3_edges, 1)
        self.assertEqual(diag["n_isolated"], 0)

    def test_knn_fallback_respects_time(self):
        """kNN fallback still filters by dt_max."""
        positions = np.array([[0.0, 0.0], [300.0, 0.0]])
        times = np.array([500.0, 600.0])  # Δt=100 >> dt_max=25
        ei, diag = build_graph(positions, times, r_max=50.0, dt_max=25.0,
                                k_min=3)
        # Even with kNN fallback, time filter should block the edge
        self.assertEqual(diag["n_edges"], 0)

    def test_degree_cap(self):
        """Degree cap limits max neighbors per node."""
        # Create a star: one central node connected to many peripheral nodes
        n = 30
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        positions = np.zeros((n + 1, 2))
        positions[0] = [0.0, 0.0]  # center
        positions[1:, 0] = 40.0 * np.cos(angles)
        positions[1:, 1] = 40.0 * np.sin(angles)
        times = np.full(n + 1, 500.0)

        ei, diag = build_graph(positions, times, r_max=150.0, dt_max=25.0,
                                k_min=3, k_max=10)
        # Center node should have at most k_max=10 outgoing edges
        center_out = (ei[0] == 0).sum()
        self.assertLessEqual(center_out, 10)
        self.assertLessEqual(diag["max_degree"], 10)

    def test_directed_edges(self):
        """Edge index contains directed edges (both directions)."""
        positions = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 8.0]])
        times = np.array([500.0, 501.0, 502.0])
        ei, _ = build_graph(positions, times, r_max=150.0, dt_max=25.0)
        edges = set(zip(ei[0].tolist(), ei[1].tolist()))
        # If (i,j) is present, (j,i) should also be present
        for s, d in list(edges):
            self.assertIn((d, s), edges, f"Missing reverse edge ({d},{s})")

    def test_no_self_loops(self):
        """Edge index should not contain self-loops."""
        positions = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 8.0]])
        times = np.array([500.0, 501.0, 502.0])
        ei, _ = build_graph(positions, times, r_max=150.0, dt_max=25.0)
        self.assertTrue(np.all(ei[0] != ei[1]))

    def test_diagnostics_correct(self):
        """Diagnostics match actual edge_index statistics."""
        positions = np.array([
            [0.0, 0.0], [10.0, 0.0], [0.0, 10.0],
            [10.0, 10.0], [50.0, 50.0],
        ])
        times = np.full(5, 500.0)
        ei, diag = build_graph(positions, times, r_max=150.0, dt_max=25.0)
        self.assertEqual(diag["n_nodes"], 5)
        self.assertEqual(diag["n_edges"], ei.shape[1])
        degree = np.bincount(ei[0], minlength=5)
        self.assertEqual(diag["min_degree"], int(degree.min()))
        self.assertEqual(diag["max_degree"], int(degree.max()))
        self.assertAlmostEqual(diag["avg_degree"], float(degree.mean()), places=5)

    def test_edge_index_dtype(self):
        """Edge index has int64 dtype."""
        positions = np.array([[0.0, 0.0], [10.0, 0.0]])
        times = np.array([500.0, 501.0])
        ei, _ = build_graph(positions, times, r_max=150.0, dt_max=25.0)
        self.assertEqual(ei.dtype, np.int64)


# ───────────────────────────────────────────────────────────────────
# compute_node_features tests
# ───────────────────────────────────────────────────────────────────

class TestComputeNodeFeatures(unittest.TestCase):
    """Tests for compute_node_features()."""

    def test_empty(self):
        """Zero hits → shape (0, 6)."""
        nf = compute_node_features(np.empty((0, 2)), np.empty(0), np.empty(0))
        self.assertEqual(nf.shape, (0, 6))

    def test_shape(self):
        """Output shape is (n, 6)."""
        positions = np.array([[0.0, 0.0], [10.0, 20.0], [30.0, 40.0]])
        times = np.array([500.0, 510.0, 520.0])
        energies = np.array([1.0, 2.0, 3.0])
        nf = compute_node_features(positions, times, energies)
        self.assertEqual(nf.shape, (3, 6))
        self.assertEqual(nf.dtype, np.float32)

    def test_feature_values(self):
        """Spot-check individual feature columns."""
        positions = np.array([[3.0, 4.0]])
        times = np.array([500.0])
        energies = np.array([2.0])
        nf = compute_node_features(positions, times, energies)
        self.assertAlmostEqual(nf[0, 0], np.log1p(2.0), places=5)  # log energy
        self.assertAlmostEqual(nf[0, 1], 500.0, places=5)  # time
        self.assertAlmostEqual(nf[0, 2], 3.0, places=5)  # x
        self.assertAlmostEqual(nf[0, 3], 4.0, places=5)  # y
        self.assertAlmostEqual(nf[0, 4], 5.0, places=5)  # radial distance
        self.assertAlmostEqual(nf[0, 5], 1.0, places=5)  # relative energy (max)

    def test_relative_energy(self):
        """Relative energy is E/E_max for each hit."""
        positions = np.array([[0.0, 0.0], [0.0, 0.0]])
        times = np.array([500.0, 500.0])
        energies = np.array([1.0, 4.0])
        nf = compute_node_features(positions, times, energies)
        self.assertAlmostEqual(nf[0, 5], 0.25, places=5)
        self.assertAlmostEqual(nf[1, 5], 1.0, places=5)

    def test_zero_energy(self):
        """All-zero energies → relative energy all zero, no NaN."""
        positions = np.array([[0.0, 0.0]])
        times = np.array([500.0])
        energies = np.array([0.0])
        nf = compute_node_features(positions, times, energies)
        self.assertFalse(np.any(np.isnan(nf)))
        self.assertAlmostEqual(nf[0, 5], 0.0, places=5)


# ───────────────────────────────────────────────────────────────────
# compute_edge_features tests
# ───────────────────────────────────────────────────────────────────

class TestComputeEdgeFeatures(unittest.TestCase):
    """Tests for compute_edge_features()."""

    def test_empty(self):
        """Zero edges → shape (0, 8)."""
        ef = compute_edge_features(
            np.empty((0, 2)), np.empty(0), np.empty(0),
            np.empty((2, 0), dtype=np.int64),
        )
        self.assertEqual(ef.shape, (0, 8))

    def test_shape(self):
        """Output shape is (n_edges, 8)."""
        positions = np.array([[0.0, 0.0], [3.0, 4.0]])
        times = np.array([500.0, 505.0])
        energies = np.array([1.0, 2.0])
        ei = np.array([[0, 1], [1, 0]])
        ef = compute_edge_features(positions, times, energies, ei)
        self.assertEqual(ef.shape, (2, 8))
        self.assertEqual(ef.dtype, np.float32)

    def test_feature_values(self):
        """Spot-check edge feature columns for a single edge."""
        positions = np.array([[0.0, 0.0], [3.0, 4.0]])
        times = np.array([500.0, 505.0])
        energies = np.array([1.0, 3.0])
        ei = np.array([[0], [1]])  # single edge 0→1
        ef = compute_edge_features(positions, times, energies, ei)
        self.assertAlmostEqual(ef[0, 0], -3.0, places=5)  # Δx
        self.assertAlmostEqual(ef[0, 1], -4.0, places=5)  # Δy
        self.assertAlmostEqual(ef[0, 2], 5.0, places=5)  # distance
        self.assertAlmostEqual(ef[0, 3], -5.0, places=5)  # Δt
        # Δ log energy: log(1+1) - log(1+3)
        expected_dle = np.log1p(1.0) - np.log1p(3.0)
        self.assertAlmostEqual(ef[0, 4], expected_dle, places=5)
        # energy asymmetry: (1-3)/(1+3) = -0.5
        self.assertAlmostEqual(ef[0, 5], -0.5, places=5)
        # log summed energy: log(1 + 1 + 3) = log(5)
        self.assertAlmostEqual(ef[0, 6], np.log1p(4.0), places=5)

    def test_antisymmetric_features(self):
        """Δx, Δy, Δt, Δ_log_e, energy_asym flip sign for reversed edge."""
        positions = np.array([[0.0, 0.0], [3.0, 4.0]])
        times = np.array([500.0, 505.0])
        energies = np.array([1.0, 3.0])
        ei = np.array([[0, 1], [1, 0]])
        ef = compute_edge_features(positions, times, energies, ei)
        # Antisymmetric: cols 0,1,3,4,5,7
        for col in [0, 1, 3, 4, 5, 7]:
            self.assertAlmostEqual(ef[0, col], -ef[1, col], places=5,
                                   msg=f"Column {col} should be antisymmetric")
        # Symmetric: cols 2,6 (distance, log summed energy)
        for col in [2, 6]:
            self.assertAlmostEqual(ef[0, col], ef[1, col], places=5,
                                   msg=f"Column {col} should be symmetric")

    def test_no_nan(self):
        """No NaN even with zero energies."""
        positions = np.array([[0.0, 0.0], [10.0, 0.0]])
        times = np.array([500.0, 500.0])
        energies = np.array([0.0, 0.0])
        ei = np.array([[0, 1], [1, 0]])
        ef = compute_edge_features(positions, times, energies, ei)
        self.assertFalse(np.any(np.isnan(ef)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
