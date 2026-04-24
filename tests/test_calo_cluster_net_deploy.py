"""Unit tests for CaloClusterNetDeploy — the ONNX-export wrapper."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch_geometric.data import Data

from src.models.calo_cluster_net import CaloClusterNet
from src.models.calo_cluster_net_deploy import CaloClusterNetDeploy


def _make_graph(n_nodes=12, n_edges=30, node_dim=6, edge_dim=8, seed=0):
    """Synthetic PyG Data object for testing. No labels/masks needed."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n_nodes, node_dim, generator=gen)
    edge_index = torch.randint(0, n_nodes, (2, n_edges), generator=gen)
    edge_attr = torch.randn(n_edges, edge_dim, generator=gen)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class TestDeployForward(unittest.TestCase):
    """Shape, determinism, and tensor-in/tensor-out contract."""

    def setUp(self):
        torch.manual_seed(0)
        self.full = CaloClusterNet(hidden_dim=32, n_mp_layers=2, dropout=0.0)
        self.full.eval()
        self.wrap = CaloClusterNetDeploy(self.full)
        self.wrap.eval()

    def test_output_is_plain_tensor_not_dict(self):
        g = _make_graph()
        out = self.wrap(g.x, g.edge_index, g.edge_attr)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, (g.edge_index.size(1),))
        self.assertEqual(out.dtype, torch.float32)

    def test_no_node_head_attribute(self):
        """Wrapper should expose encoders/mp/edge head but not node head."""
        self.assertTrue(hasattr(self.wrap, "node_encoder"))
        self.assertTrue(hasattr(self.wrap, "edge_encoder"))
        self.assertTrue(hasattr(self.wrap, "mp_blocks"))
        self.assertTrue(hasattr(self.wrap, "edge_head"))
        self.assertFalse(hasattr(self.wrap, "node_head"))

    def test_eval_mode_deterministic(self):
        g = _make_graph()
        out1 = self.wrap(g.x, g.edge_index, g.edge_attr)
        out2 = self.wrap(g.x, g.edge_index, g.edge_attr)
        self.assertTrue(torch.allclose(out1, out2))

    def test_varied_graph_sizes(self):
        """Wrapper should handle graphs with different N, E (dynamic axes)."""
        for n_nodes, n_edges in [(5, 10), (20, 80), (1, 1), (50, 300)]:
            g = _make_graph(n_nodes=n_nodes, n_edges=n_edges, seed=n_nodes)
            out = self.wrap(g.x, g.edge_index, g.edge_attr)
            self.assertEqual(out.shape, (n_edges,))


class TestParityWithFullModel(unittest.TestCase):
    """Wrapper edge logits must match CaloClusterNet's edge_logits exactly."""

    def test_edge_logits_match_full_model(self):
        torch.manual_seed(42)
        full = CaloClusterNet(hidden_dim=48, n_mp_layers=3, dropout=0.0)
        full.eval()
        wrap = CaloClusterNetDeploy(full)
        wrap.eval()

        g = _make_graph(n_nodes=15, n_edges=40, seed=7)

        with torch.no_grad():
            ref = full(g)["edge_logits"]
            test = wrap(g.x, g.edge_index, g.edge_attr)

        self.assertTrue(torch.allclose(ref, test, atol=1e-6),
                        f"max diff: {(ref - test).abs().max().item():.2e}")

    def test_shared_weights_not_copy(self):
        """Wrapper should reuse submodules by reference, not clone weights."""
        full = CaloClusterNet(hidden_dim=16, n_mp_layers=2, dropout=0.0)
        wrap = CaloClusterNetDeploy(full)
        self.assertIs(wrap.node_encoder, full.node_encoder)
        self.assertIs(wrap.edge_head, full.edge_head)


class TestFromCheckpoint(unittest.TestCase):
    """Loading from a real trained checkpoint (skipped if unavailable)."""

    CHECKPOINT = Path("outputs/runs/calo_cluster_net_v2_stage1/checkpoints/best_model.pt")

    def setUp(self):
        if not self.CHECKPOINT.exists():
            self.skipTest(f"Checkpoint not present: {self.CHECKPOINT}")

    def test_loads_and_runs(self):
        wrap = CaloClusterNetDeploy.from_checkpoint(self.CHECKPOINT)
        self.assertFalse(wrap.training, "wrapper should be in eval mode")

        g = _make_graph(n_nodes=30, n_edges=120, seed=1)
        with torch.no_grad():
            out = wrap(g.x, g.edge_index, g.edge_attr)
        self.assertEqual(out.shape, (120,))
        self.assertTrue(torch.isfinite(out).all())

    def test_matches_full_model_on_same_checkpoint(self):
        """Wrapper and full model loaded from the same ckpt must agree."""
        import yaml
        from src.models import build_model

        run_dir = self.CHECKPOINT.parent.parent
        with open(run_dir / "config.yaml") as f:
            cfg = yaml.safe_load(f)

        full = build_model(cfg)
        ckpt = torch.load(self.CHECKPOINT, weights_only=False, map_location="cpu")
        full.load_state_dict(ckpt["model_state_dict"])
        full.eval()

        wrap = CaloClusterNetDeploy.from_checkpoint(self.CHECKPOINT)

        g = _make_graph(n_nodes=25, n_edges=90, seed=3)
        with torch.no_grad():
            ref = full(g)["edge_logits"]
            test = wrap(g.x, g.edge_index, g.edge_attr)

        self.assertTrue(torch.allclose(ref, test, atol=1e-6),
                        f"max diff: {(ref - test).abs().max().item():.2e}")


class TestRealValGraph(unittest.TestCase):
    """Parity test on a real normalised val graph (skipped if unavailable)."""

    CHECKPOINT = Path("outputs/runs/calo_cluster_net_v2_stage1/checkpoints/best_model.pt")
    VAL_PT = Path("data/processed/val.pt")

    def setUp(self):
        if not self.CHECKPOINT.exists() or not self.VAL_PT.exists():
            self.skipTest("checkpoint or packed val graphs not present")

    def test_parity_on_val_graph(self):
        import yaml
        from src.models import build_model

        run_dir = self.CHECKPOINT.parent.parent
        with open(run_dir / "config.yaml") as f:
            cfg = yaml.safe_load(f)
        full = build_model(cfg)
        ckpt = torch.load(self.CHECKPOINT, weights_only=False, map_location="cpu")
        full.load_state_dict(ckpt["model_state_dict"])
        full.eval()

        wrap = CaloClusterNetDeploy.from_checkpoint(self.CHECKPOINT)

        val_graphs = torch.load(self.VAL_PT, weights_only=False)
        # Check first few non-trivial graphs
        n_checked = 0
        for g in val_graphs[:20]:
            if g.edge_index.size(1) == 0:
                continue
            with torch.no_grad():
                ref = full(g)["edge_logits"]
                test = wrap(g.x, g.edge_index, g.edge_attr)
            max_diff = (ref - test).abs().max().item()
            self.assertTrue(torch.allclose(ref, test, atol=1e-6),
                            f"graph {n_checked}: max diff {max_diff:.2e}")
            n_checked += 1
            if n_checked >= 5:
                break

        self.assertGreater(n_checked, 0, "no graphs exercised")


if __name__ == "__main__":
    unittest.main()
