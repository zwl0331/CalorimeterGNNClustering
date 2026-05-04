"""Unit tests for the C++ deployment normalisation sidecar export."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from scripts.export_norm_stats import (
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    stats_to_dict,
)


def _example_stats():
    """A stats dict shaped like the real `normalization_stats.pt`."""
    return {
        "node_mean": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        "node_std": torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        "edge_mean": torch.tensor([7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]),
        "edge_std": torch.tensor([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]),
        "node_count": 100,
        "edge_count": 200,
    }


class TestStatsToDict(unittest.TestCase):
    def test_payload_contains_all_required_keys(self):
        payload = stats_to_dict(_example_stats())
        for key in (
            "schema_version", "node_features", "edge_features",
            "node_mean", "node_std", "edge_mean", "edge_std",
            "node_count", "edge_count",
        ):
            self.assertIn(key, payload)

    def test_feature_names_match_canonical_lists(self):
        payload = stats_to_dict(_example_stats())
        self.assertEqual(payload["node_features"], NODE_FEATURE_NAMES)
        self.assertEqual(payload["edge_features"], EDGE_FEATURE_NAMES)
        self.assertEqual(len(payload["node_features"]), 6)
        self.assertEqual(len(payload["edge_features"]), 8)

    def test_values_round_trip_bit_exact(self):
        """Floats must survive tensor → list → tensor with bit-exact equality."""
        stats = _example_stats()
        payload = stats_to_dict(stats)
        for key in ("node_mean", "node_std", "edge_mean", "edge_std"):
            recovered = torch.tensor(payload[key], dtype=stats[key].dtype)
            self.assertTrue(
                torch.equal(recovered, stats[key]),
                f"{key} did not round-trip exactly: {recovered} vs {stats[key]}",
            )

    def test_counts_are_python_ints(self):
        payload = stats_to_dict(_example_stats())
        self.assertIsInstance(payload["node_count"], int)
        self.assertIsInstance(payload["edge_count"], int)
        self.assertEqual(payload["node_count"], 100)
        self.assertEqual(payload["edge_count"], 200)

    def test_wrong_node_dim_raises(self):
        bad = _example_stats()
        bad["node_mean"] = torch.zeros(5)
        bad["node_std"] = torch.ones(5)
        with self.assertRaises(ValueError):
            stats_to_dict(bad)

    def test_wrong_edge_dim_raises(self):
        bad = _example_stats()
        bad["edge_mean"] = torch.zeros(7)
        bad["edge_std"] = torch.ones(7)
        with self.assertRaises(ValueError):
            stats_to_dict(bad)


class TestSidecarFileIfPresent(unittest.TestCase):
    """If the real sidecar exists, it must match the train-split torch blob."""

    def setUp(self):
        self.repo = Path(__file__).resolve().parents[1]
        self.stats_pt = self.repo / "data" / "normalization_stats.pt"
        self.sidecar = self.repo / "outputs" / "onnx" / "calo_cluster_net_v2_stage1.norm.json"

    def test_sidecar_matches_torch_blob(self):
        if not self.stats_pt.exists() or not self.sidecar.exists():
            self.skipTest("sidecar or stats blob not present in this checkout")
        stats = torch.load(self.stats_pt, weights_only=True)
        with self.sidecar.open() as f:
            payload = json.load(f)
        for key in ("node_mean", "node_std", "edge_mean", "edge_std"):
            recovered = torch.tensor(payload[key], dtype=stats[key].dtype)
            self.assertTrue(
                torch.equal(recovered, stats[key]),
                f"sidecar {key} disagrees with {self.stats_pt}",
            )
        self.assertEqual(payload["node_count"], int(stats["node_count"]))
        self.assertEqual(payload["edge_count"], int(stats["edge_count"]))


class TestEndToEndExport(unittest.TestCase):
    """Round-trip via the script entry point: torch blob -> JSON -> dict -> torch."""

    def test_export_then_load_round_trip(self):
        stats = _example_stats()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            pt_path = tmp / "stats.pt"
            json_path = tmp / "stats.norm.json"
            torch.save(stats, pt_path)

            loaded = torch.load(pt_path, weights_only=True)
            payload = stats_to_dict(loaded)
            with json_path.open("w") as f:
                json.dump(payload, f)

            with json_path.open() as f:
                recovered = json.load(f)
            for key in ("node_mean", "node_std", "edge_mean", "edge_std"):
                t = torch.tensor(recovered[key], dtype=stats[key].dtype)
                self.assertTrue(torch.equal(t, stats[key]))


if __name__ == "__main__":
    unittest.main()
