"""GNN calorimeter clustering for Mu2e.

Edge-classification GNN that predicts which calorimeter hit pairs belong
to the same physics cluster. Built on PyTorch Geometric.

Subpackages
-----------
data        Dataset loading, graph construction, truth labels, normalization.
geometry    Crystal geometry lookup (crystalId -> disk, x, y).
models      GNN architectures (SimpleEdgeNet, CaloClusterNetV1).
training    Training loop, loss functions, evaluation metrics.
inference   Cluster reconstruction from edge predictions, postprocessing.
"""
