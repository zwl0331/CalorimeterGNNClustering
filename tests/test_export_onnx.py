"""Tests for the metadata_props stamping in scripts/export_onnx.py.

The C++ deployment asserts these keys at session load (FHiCL passes
the expected values). Round-trip via a tiny synthetic ONNX file —
fast, no checkpoint or torch-onnx export needed.
"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import onnx
from onnx import helper, TensorProto

from scripts.export_onnx import (
    DEFAULT_MODEL_VERSION,
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    stamp_metadata_props,
)


def _make_tiny_onnx(path: Path) -> None:
    """Write a minimal valid ONNX model: identity over a 1-D float tensor."""
    inp = helper.make_tensor_value_info("x", TensorProto.FLOAT, [None])
    out = helper.make_tensor_value_info("y", TensorProto.FLOAT, [None])
    node = helper.make_node("Identity", inputs=["x"], outputs=["y"])
    graph = helper.make_graph([node], "tiny", [inp], [out])
    model = helper.make_model(graph, producer_name="test", opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, str(path))


class TestStampMetadataProps(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "tiny.onnx"
        _make_tiny_onnx(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def _props(self) -> dict:
        m = onnx.load(str(self.path))
        return {p.key: p.value for p in m.metadata_props}

    def test_stamps_three_keys(self):
        stamp_metadata_props(self.path, "v1", ["a", "b"], ["x", "y", "z"])
        props = self._props()
        self.assertEqual(props["model_version"], "v1")
        self.assertEqual(props["node_features"], "a,b")
        self.assertEqual(props["edge_features"], "x,y,z")

    def test_idempotent_overwrite(self):
        stamp_metadata_props(self.path, "v1", ["a"], ["x"])
        stamp_metadata_props(self.path, "v2", ["a", "b"], ["x", "y"])
        props = self._props()
        self.assertEqual(props["model_version"], "v2")
        self.assertEqual(props["node_features"], "a,b")
        self.assertEqual(props["edge_features"], "x,y")
        # No duplicates.
        m = onnx.load(str(self.path))
        keys = [p.key for p in m.metadata_props]
        self.assertEqual(len(keys), len(set(keys)))

    def test_preserves_other_metadata(self):
        # Pre-stamp a foreign key the export shouldn't touch.
        m = onnx.load(str(self.path))
        e = m.metadata_props.add()
        e.key = "producer_pytorch_version"
        e.value = "2.5.1"
        onnx.save(m, str(self.path))

        stamp_metadata_props(self.path, "v1", NODE_FEATURE_NAMES, EDGE_FEATURE_NAMES)
        props = self._props()
        self.assertEqual(props["producer_pytorch_version"], "2.5.1")
        self.assertEqual(props["model_version"], "v1")

    def test_canonical_names_match_norm_sidecar(self):
        # Same lists used by scripts/export_norm_stats.py — these have
        # to agree, because the C++ side does both assertions.
        from scripts.export_norm_stats import (
            EDGE_FEATURE_NAMES as SIDECAR_EDGE,
            NODE_FEATURE_NAMES as SIDECAR_NODE,
        )
        self.assertEqual(NODE_FEATURE_NAMES, SIDECAR_NODE)
        self.assertEqual(EDGE_FEATURE_NAMES, SIDECAR_EDGE)

    def test_default_model_version_is_v2_stage1(self):
        # Sanity-check the default the C++ deployment expects.
        self.assertEqual(DEFAULT_MODEL_VERSION, "calo-cluster-net-v2-stage1")


if __name__ == "__main__":
    unittest.main()
