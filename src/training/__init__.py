"""Training infrastructure for edge-classification GNN.

Modules
-------
trainer     Train/val loop with early stopping, LR scheduling, checkpointing.
losses      Masked BCE, node saliency loss, consistency regularizer, multi-task loss.
metrics     Edge-level (precision, recall, F1, AUC) and cluster-level (purity, completeness).
"""
