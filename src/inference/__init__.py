"""Inference pipeline: edge predictions -> reconstructed clusters.

Modules
-------
cluster_reco    Symmetrize edge scores, threshold, connected components, cleanup.
postprocess     Per-cluster physics features (energy, centroid, time, RMS width).
"""
