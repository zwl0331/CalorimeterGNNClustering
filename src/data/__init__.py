"""Data pipeline: ROOT files -> PyG graphs with MC truth labels.

Modules
-------
dataset         Extract per-disk graphs from ROOT files; CaloGraphDataset loader.
graph_builder   Hybrid radius + kNN graph construction using scipy cKDTree.
truth_labels    MC truth edge/node labels from SimParticle energy deposits.
normalization   Z-score normalization statistics (train split only).
"""
