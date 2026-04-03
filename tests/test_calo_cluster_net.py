"""Unit tests for CaloClusterNetV1 and its components."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch_geometric.data import Data

from src.models.layers import EdgeAwareResBlock
from src.models.heads import NodeSaliencyHead, EdgeClusteringHead
from src.models.calo_cluster_net import CaloClusterNetV1


def _make_graph(n_nodes=10, n_edges=20, node_dim=6, edge_dim=8):
    """Create a synthetic PyG Data object for testing."""
    x = torch.randn(n_nodes, node_dim)
    edge_index = torch.randint(0, n_nodes, (2, n_edges))
    edge_attr = torch.randn(n_edges, edge_dim)
    y_edge = torch.randint(0, 2, (n_edges,)).float()
    y_node = torch.randint(-1, 3, (n_nodes,)).float()
    edge_mask = torch.ones(n_edges, dtype=torch.bool)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                y_edge=y_edge, y_node=y_node, edge_mask=edge_mask)


class TestEdgeAwareResBlock(unittest.TestCase):

    def test_output_shapes(self):
        block = EdgeAwareResBlock(hidden_dim=32, dropout=0.0)
        h = torch.randn(10, 32)
        e = torch.randn(20, 32)
        edge_index = torch.randint(0, 10, (2, 20))

        h_out, e_out = block(h, e, edge_index)
        self.assertEqual(h_out.shape, (10, 32))
        self.assertEqual(e_out.shape, (20, 32))

    def test_residual_connection(self):
        """Edge and node outputs should differ from inputs (residual adds, doesn't replace)."""
        block = EdgeAwareResBlock(hidden_dim=16, dropout=0.0)
        h = torch.randn(5, 16)
        e = torch.randn(8, 16)
        edge_index = torch.randint(0, 5, (2, 8))

        h_out, e_out = block(h, e, edge_index)
        # Outputs shouldn't be identical to inputs (MLP + residual changes them)
        self.assertFalse(torch.allclose(h, h_out, atol=1e-6))
        self.assertFalse(torch.allclose(e, e_out, atol=1e-6))

    def test_gradients_flow(self):
        block = EdgeAwareResBlock(hidden_dim=16, dropout=0.0)
        h = torch.randn(5, 16, requires_grad=True)
        e = torch.randn(8, 16, requires_grad=True)
        edge_index = torch.randint(0, 5, (2, 8))

        h_out, e_out = block(h, e, edge_index)
        loss = h_out.sum() + e_out.sum()
        loss.backward()
        self.assertIsNotNone(h.grad)
        self.assertIsNotNone(e.grad)


class TestNodeSaliencyHead(unittest.TestCase):

    def test_output_shape(self):
        head = NodeSaliencyHead(hidden_dim=32, dropout=0.0)
        h = torch.randn(10, 32)
        out = head(h)
        self.assertEqual(out.shape, (10,))

    def test_output_unbounded(self):
        """Raw logits should be unbounded (no sigmoid applied)."""
        head = NodeSaliencyHead(hidden_dim=32, dropout=0.0)
        h = torch.randn(100, 32) * 10
        out = head(h)
        # At least some outputs should be outside [0, 1]
        self.assertTrue((out < 0).any() or (out > 1).any())


class TestEdgeClusteringHead(unittest.TestCase):

    def test_output_shape(self):
        head = EdgeClusteringHead(hidden_dim=32, dropout=0.0)
        h = torch.randn(10, 32)
        e = torch.randn(20, 32)
        edge_index = torch.randint(0, 10, (2, 20))
        out = head(h, e, edge_index)
        self.assertEqual(out.shape, (20,))


class TestCaloClusterNetV1(unittest.TestCase):

    def test_forward_returns_dict(self):
        model = CaloClusterNetV1(hidden_dim=32, n_mp_layers=2, dropout=0.0)
        data = _make_graph()
        out = model(data)
        self.assertIsInstance(out, dict)
        self.assertIn("edge_logits", out)
        self.assertIn("node_logits", out)

    def test_output_shapes(self):
        model = CaloClusterNetV1(hidden_dim=32, n_mp_layers=2, dropout=0.0)
        data = _make_graph(n_nodes=15, n_edges=30)
        out = model(data)
        self.assertEqual(out["edge_logits"].shape, (30,))
        self.assertEqual(out["node_logits"].shape, (15,))

    def test_default_config(self):
        """Default config: hidden=96, 4 MP layers."""
        model = CaloClusterNetV1()
        data = _make_graph()
        out = model(data)
        self.assertEqual(out["edge_logits"].shape, (20,))
        self.assertEqual(out["node_logits"].shape, (10,))

    def test_param_count_larger_than_simple(self):
        from src.models.simple_edge_net import SimpleEdgeNet
        simple = SimpleEdgeNet(hidden_dim=64, n_mp_layers=3)
        v1 = CaloClusterNetV1(hidden_dim=96, n_mp_layers=4)
        n_simple = sum(p.numel() for p in simple.parameters())
        n_v1 = sum(p.numel() for p in v1.parameters())
        self.assertGreater(n_v1, n_simple)

    def test_gradients_flow_through_all_heads(self):
        model = CaloClusterNetV1(hidden_dim=16, n_mp_layers=2, dropout=0.0)
        data = _make_graph()
        out = model(data)
        loss = out["edge_logits"].sum() + out["node_logits"].sum()
        loss.backward()
        # All parameters should have gradients
        for name, p in model.named_parameters():
            self.assertIsNotNone(p.grad, f"No gradient for {name}")

    def test_single_node_graph(self):
        """Model should handle a graph with 1 node and 0 edges."""
        model = CaloClusterNetV1(hidden_dim=16, n_mp_layers=2, dropout=0.0)
        data = Data(
            x=torch.randn(1, 6),
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            edge_attr=torch.zeros(0, 8),
        )
        out = model(data)
        self.assertEqual(out["edge_logits"].shape, (0,))
        self.assertEqual(out["node_logits"].shape, (1,))

    def test_eval_mode_deterministic(self):
        model = CaloClusterNetV1(hidden_dim=16, n_mp_layers=2, dropout=0.1)
        model.eval()
        data = _make_graph()
        out1 = model(data)
        out2 = model(data)
        self.assertTrue(torch.allclose(out1["edge_logits"], out2["edge_logits"]))
        self.assertTrue(torch.allclose(out1["node_logits"], out2["node_logits"]))

    def test_compatible_with_reconstruct_clusters(self):
        """Edge logits can be fed directly to reconstruct_clusters."""
        from src.inference.cluster_reco import reconstruct_clusters
        model = CaloClusterNetV1(hidden_dim=16, n_mp_layers=2, dropout=0.0)
        model.eval()
        data = _make_graph(n_nodes=10, n_edges=20)
        with torch.no_grad():
            out = model(data)
        labels, probs = reconstruct_clusters(
            data.edge_index, out["edge_logits"], n_nodes=10,
            tau_edge=0.5, min_hits=1, min_energy_mev=0.0,
        )
        self.assertEqual(labels.shape, (10,))

    def test_compatible_with_predict_clusters(self):
        """predict_clusters works with CaloClusterNetV1 (dict output)."""
        from src.inference.cluster_reco import predict_clusters
        model = CaloClusterNetV1(hidden_dim=16, n_mp_layers=2, dropout=0.0)
        data = _make_graph(n_nodes=10, n_edges=20)
        labels, probs = predict_clusters(
            model, data, device="cpu", tau_edge=0.5,
            min_hits=1, min_energy_mev=0.0,
        )
        self.assertEqual(labels.shape, (10,))


class TestMultitaskLoss(unittest.TestCase):

    def test_edge_only_stage(self):
        """lambda_node=0, lambda_cons=0 should give edge-only loss."""
        from src.training.losses import multitask_loss
        data = _make_graph()
        output = {"edge_logits": torch.randn(20), "node_logits": torch.randn(10)}
        loss, ld = multitask_loss(output, data, lambda_edge=1.0,
                                  lambda_node=0.0, lambda_cons=0.0)
        self.assertIn("edge_loss", ld)
        self.assertNotIn("node_loss", ld)
        self.assertNotIn("cons_loss", ld)

    def test_multitask_stage(self):
        """All three loss terms present when all lambdas > 0."""
        from src.training.losses import multitask_loss
        data = _make_graph()
        output = {"edge_logits": torch.randn(20), "node_logits": torch.randn(10)}
        loss, ld = multitask_loss(output, data, lambda_edge=1.0,
                                  lambda_node=0.3, lambda_cons=0.05)
        self.assertIn("edge_loss", ld)
        self.assertIn("node_loss", ld)
        self.assertIn("cons_loss", ld)
        self.assertGreater(loss.item(), 0)

    def test_backward_compatible_with_tensor(self):
        """Tensor input (SimpleEdgeNet) should still work."""
        from src.training.losses import multitask_loss
        data = _make_graph()
        logits = torch.randn(20)
        loss, ld = multitask_loss(logits, data)
        self.assertIn("edge_loss", ld)


if __name__ == "__main__":
    unittest.main()
