"""
Smoke test: verify the environment has all required packages and that a
minimal GNN forward pass works. Run after sourcing setup_env.sh.

    python3 scripts/smoke_test_env.py
"""

import sys

def check(name, fn):
    try:
        result = fn()
        print(f"  {name}: {result}")
        return True
    except Exception as e:
        print(f"  {name}: FAILED — {e}")
        return False

print("=== Package versions ===")
ok = True
ok &= check("python",         lambda: sys.version.split()[0])
ok &= check("torch",          lambda: __import__("torch").__version__)
ok &= check("torch_geometric", lambda: __import__("torch_geometric").__version__)
ok &= check("uproot",         lambda: __import__("uproot").__version__)
ok &= check("numpy",          lambda: __import__("numpy").__version__)
ok &= check("scipy",          lambda: __import__("scipy").__version__)
ok &= check("sklearn",        lambda: __import__("sklearn").__version__)
ok &= check("matplotlib",     lambda: __import__("matplotlib").__version__)

print("\n=== CUDA ===")
import torch
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  Device: {torch.cuda.get_device_name(0)}")
else:
    print("  (expected on login node — use GPU node for training)")

print("\n=== Minimal GNN forward pass ===")
try:
    import torch
    from torch_geometric.data import Data
    from torch_geometric.nn import radius_graph

    # 5 hits, 6 node features (log_E, t, x, y, r, E_rel)
    x = torch.randn(5, 6)
    pos = torch.randn(5, 2)  # (x, y) positions
    edge_index = radius_graph(pos, r=2.0)
    n_edges = edge_index.shape[1]
    edge_attr = torch.randn(n_edges, 10)  # 10 edge features

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    print(f"  Graph: {data.num_nodes} nodes, {data.num_edges} edges")

    # Tiny linear message-passing step
    from torch_geometric.nn import MessagePassing
    class TinyMP(MessagePassing):
        def __init__(self):
            super().__init__(aggr="sum")
            self.lin = torch.nn.Linear(6, 6)
        def forward(self, x, edge_index):
            return self.propagate(edge_index, x=x)
        def message(self, x_j):
            return self.lin(x_j)

    model = TinyMP()
    out = model(data.x, data.edge_index)
    assert out.shape == (5, 6)
    print(f"  Forward pass output shape: {tuple(out.shape)} — OK")
except Exception as e:
    print(f"  FAILED — {e}")
    ok = False

print()
if ok:
    print("All checks passed.")
else:
    print("Some checks FAILED. See above.")
    sys.exit(1)
