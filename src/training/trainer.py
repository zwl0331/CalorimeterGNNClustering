"""
Training loop for edge classification GNN.

Handles:
  - Train/val epoch loops with DataLoader
  - Weighted BCE with negative subsampling (train only)
  - ReduceLROnPlateau scheduling on val F1
  - Early stopping
  - Checkpointing best model
  - Per-epoch metric logging
"""

import json
import time
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from src.training.losses import masked_bce_loss, multitask_loss, compute_class_weights
from src.training.metrics import edge_metrics, edge_auc, cluster_metrics_from_edges


class Trainer:
    """Edge classification trainer.

    Parameters
    ----------
    model : nn.Module
    train_dataset : Dataset
    val_dataset : Dataset
    cfg : dict
        Training config (from default.yaml 'train' section).
    pos_weight : Tensor or None
    device : torch.device
    run_dir : Path
        Directory for saving checkpoints and logs.
    """

    def __init__(self, model, train_dataset, val_dataset, cfg, pos_weight=None,
                 device=None, run_dir=None):
        self.model = model
        self.cfg = cfg
        self.device = device or torch.device("cpu")
        self.model.to(self.device)

        self.train_loader = DataLoader(
            train_dataset, batch_size=cfg.get("batch_size", 32), shuffle=True,
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=cfg.get("batch_size", 32), shuffle=False,
        )

        self.pos_weight = pos_weight.to(self.device) if pos_weight is not None else None

        # Multi-task loss weights
        self.lambda_edge = cfg.get("lambda_edge", 1.0)
        self.lambda_node = cfg.get("lambda_node", 0.0)
        self.lambda_cons = cfg.get("lambda_cons", 0.0)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.get("lr", 1e-3),
            weight_decay=cfg.get("weight_decay", 1e-4),
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5,
            patience=cfg.get("early_stop_patience", 15) // 3,
            min_lr=1e-6,
        )

        self.epochs = cfg.get("epochs", 100)
        self.patience = cfg.get("early_stop_patience", 15)

        # Logging
        self.run_dir = Path(run_dir) if run_dir else Path("outputs/runs/default")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.history = []
        self.best_val_f1 = 0.0
        self.epochs_without_improvement = 0

    def _extract_edge_logits(self, output):
        """Extract edge logits from model output (Tensor or dict)."""
        if isinstance(output, dict):
            return output["edge_logits"]
        return output

    def train_epoch(self):
        """Run one training epoch. Returns dict of average metrics."""
        self.model.train()
        total_loss = 0.0
        total_sub_losses = {}
        n_batches = 0
        all_logits, all_targets, all_masks = [], [], []

        for batch in self.train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            output = self.model(batch)
            loss, loss_dict = multitask_loss(
                output, batch, pos_weight=self.pos_weight,
                lambda_edge=self.lambda_edge,
                lambda_node=self.lambda_node,
                lambda_cons=self.lambda_cons,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            for k, v in loss_dict.items():
                total_sub_losses[k] = total_sub_losses.get(k, 0.0) + v
            n_batches += 1

            all_logits.append(self._extract_edge_logits(output).detach())
            all_targets.append(batch.y_edge)
            all_masks.append(batch.edge_mask)

        # Aggregate metrics
        logits_cat = torch.cat(all_logits)
        targets_cat = torch.cat(all_targets)
        masks_cat = torch.cat(all_masks)

        em = edge_metrics(logits_cat, targets_cat, masks_cat)
        em["loss"] = total_loss / max(n_batches, 1)
        for k, v in total_sub_losses.items():
            em[k] = v / max(n_batches, 1)
        return em

    @torch.no_grad()
    def val_epoch(self):
        """Run one validation epoch. Returns dict of metrics."""
        self.model.eval()

        if len(self.val_loader.dataset) == 0:
            return {"loss": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                    "accuracy": 0.0, "n_pos": 0, "n_neg": 0,
                    "roc_auc": 0.0, "pr_auc": 0.0}

        total_loss = 0.0
        total_sub_losses = {}
        n_batches = 0
        all_logits, all_targets, all_masks = [], [], []

        for batch in self.val_loader:
            batch = batch.to(self.device)

            output = self.model(batch)
            loss, loss_dict = multitask_loss(
                output, batch, pos_weight=self.pos_weight,
                lambda_edge=self.lambda_edge,
                lambda_node=self.lambda_node,
                lambda_cons=self.lambda_cons,
            )

            total_loss += loss.item()
            for k, v in loss_dict.items():
                total_sub_losses[k] = total_sub_losses.get(k, 0.0) + v
            n_batches += 1

            all_logits.append(self._extract_edge_logits(output))
            all_targets.append(batch.y_edge)
            all_masks.append(batch.edge_mask)

        # Edge metrics (full val set, no subsampling)
        logits_cat = torch.cat(all_logits)
        targets_cat = torch.cat(all_targets)
        masks_cat = torch.cat(all_masks)

        em = edge_metrics(logits_cat, targets_cat, masks_cat)
        auc = edge_auc(logits_cat, targets_cat, masks_cat)
        em.update(auc)
        em["loss"] = total_loss / max(n_batches, 1)
        for k, v in total_sub_losses.items():
            em[k] = v / max(n_batches, 1)
        return em

    def fit(self):
        """Run full training loop. Returns history list."""
        print(f"Training on {self.device} for up to {self.epochs} epochs")
        print(f"  Train graphs: {len(self.train_loader.dataset)}")
        print(f"  Val graphs:   {len(self.val_loader.dataset)}")
        if self.pos_weight is not None:
            print(f"  pos_weight: {self.pos_weight.item():.2f}")
        print(f"  Run dir: {self.run_dir}")
        print()

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()

            train_m = self.train_epoch()
            val_m = self.val_epoch()

            # Step scheduler on val F1
            self.scheduler.step(val_m["f1"])
            lr = self.optimizer.param_groups[0]["lr"]

            elapsed = time.time() - t0

            record = {
                "epoch": epoch,
                "lr": lr,
                "elapsed_s": round(elapsed, 1),
                "train": train_m,
                "val": val_m,
            }
            self.history.append(record)

            # Print progress
            print(
                f"Epoch {epoch:3d} | "
                f"train loss {train_m['loss']:.4f} F1 {train_m['f1']:.3f} | "
                f"val loss {val_m['loss']:.4f} F1 {val_m['f1']:.3f} "
                f"P {val_m['precision']:.3f} R {val_m['recall']:.3f} | "
                f"lr {lr:.1e} | {elapsed:.1f}s"
            )
            if "roc_auc" in val_m:
                print(
                    f"         val ROC-AUC {val_m['roc_auc']:.3f} "
                    f"PR-AUC {val_m['pr_auc']:.3f}"
                )

            # Check for NaN
            if torch.isnan(torch.tensor(train_m["loss"])):
                print("NaN loss detected — stopping training.")
                break

            # Best model checkpoint
            if val_m["f1"] > self.best_val_f1:
                self.best_val_f1 = val_m["f1"]
                self.epochs_without_improvement = 0
                ckpt_path = self.checkpoint_dir / "best_model.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_f1": val_m["f1"],
                    "val_metrics": val_m,
                }, ckpt_path)
                print(f"  -> New best val F1 = {val_m['f1']:.4f}, saved to {ckpt_path}")
            else:
                self.epochs_without_improvement += 1

            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                print(f"Early stopping at epoch {epoch} "
                      f"(no improvement for {self.patience} epochs)")
                break

        # Save history
        history_path = self.run_dir / "history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"\nTraining complete. Best val F1 = {self.best_val_f1:.4f}")
        print(f"History saved to {history_path}")

        return self.history
