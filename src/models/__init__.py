from src.models.simple_edge_net import SimpleEdgeNet
from src.models.calo_cluster_net import CaloClusterNetV1


def build_model(cfg):
    """Instantiate model from config dict.

    Parameters
    ----------
    cfg : dict
        Full config with 'model' section containing 'name', 'hidden_dim',
        'n_mp_layers', 'dropout'.
    """
    model_cfg = cfg["model"]
    name = model_cfg.get("name", "SimpleEdgeNet")

    if name == "SimpleEdgeNet":
        return SimpleEdgeNet(
            node_dim=6, edge_dim=8,
            hidden_dim=model_cfg.get("hidden_dim", 64),
            n_mp_layers=model_cfg.get("n_mp_layers", 3),
            dropout=model_cfg.get("dropout", 0.1),
        )
    elif name == "CaloClusterNetV1":
        return CaloClusterNetV1(
            node_dim=6, edge_dim=8,
            hidden_dim=model_cfg.get("hidden_dim", 96),
            n_mp_layers=model_cfg.get("n_mp_layers", 4),
            dropout=model_cfg.get("dropout", 0.1),
        )
    else:
        raise ValueError(f"Unknown model: {name}")
