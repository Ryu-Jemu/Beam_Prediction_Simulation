"""
Training module with comprehensive logging, checkpointing, and optimization
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
import os
import time
import json
from tqdm import tqdm
from typing import Dict, Optional

from config import Config
from models import BeamPredictorLLM
from utils import (
    circular_mse_loss,
    weighted_circular_mse_loss,
    cosine_similarity_loss,
    combined_loss,
    normalized_angle_mse,
    mae_degrees,
    hit_at_threshold,
    compose_residual_with_baseline,
    compute_statistics_text,
    AverageMeter,
    set_seed,
    count_parameters
)


def get_loss_function(cfg: Config):
    """Get loss function based on config
    
    Args:
        cfg: configuration
    
    Returns:
        loss_fn: loss function
    """
    # Temporal weights (exponential decay)
    if cfg.use_step_weighting:
        weights = torch.pow(
            torch.tensor(cfg.horizon_gamma),
            torch.arange(cfg.H, dtype=torch.float32)
        ).view(1, cfg.H, 1)
    else:
        weights = torch.ones(1, cfg.H, 1)
    
    def loss_fn(pred, target, device):
        w = weights.to(device)
        
        if cfg.loss_type == "weighted_circular":
            return weighted_circular_mse_loss(pred, target, w.squeeze(-1))
        elif cfg.loss_type == "mse":
            return circular_mse_loss(pred, target)
        elif cfg.loss_type == "cosine":
            return cosine_similarity_loss(pred, target)
        elif cfg.loss_type == "combined":
            return combined_loss(pred, target, w.squeeze(-1))
        else:
            raise ValueError(f"Unknown loss type: {cfg.loss_type}")
    
    return loss_fn


def get_optimizer(model: nn.Module, cfg: Config):
    """Get optimizer
    
    Args:
        model: model to optimize
        cfg: configuration
    
    Returns:
        optimizer
    """
    if cfg.optimizer.lower() == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=cfg.betas,
            eps=cfg.eps
        )
    elif cfg.optimizer.lower() == "adam":
        return optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=cfg.betas,
            eps=cfg.eps
        )
    elif cfg.optimizer.lower() == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=0.9,
            weight_decay=cfg.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")


def get_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """Get learning rate scheduler
    
    Args:
        optimizer: optimizer
        cfg: configuration
        steps_per_epoch: number of steps per epoch
    
    Returns:
        scheduler
    """
    if not cfg.use_scheduler:
        return None
    
    if cfg.scheduler_type == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=cfg.n_epochs,
            eta_min=cfg.min_lr
        )
    elif cfg.scheduler_type == "step":
        return StepLR(
            optimizer,
            step_size=cfg.n_epochs // 3,
            gamma=0.1
        )
    elif cfg.scheduler_type == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=cfg.min_lr
        )
    else:
        return None


class Trainer:
    """Trainer for beam prediction model"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: Config,
        device: torch.device
    ):
        """
        Args:
            model: model to train
            train_loader: training dataloader
            val_loader: validation dataloader
            cfg: configuration
            device: torch device
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        
        # Move model to device
        self.model.to(device)
        
        # Optimizer and scheduler
        self.optimizer = get_optimizer(model, cfg)
        self.scheduler = get_scheduler(
            self.optimizer, cfg, len(train_loader)
        )
        
        # Loss function
        self.loss_fn = get_loss_function(cfg)
        
        # Checkpointing
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        
        # Logging
        os.makedirs(cfg.log_dir, exist_ok=True)
        self.log_file = os.path.join(cfg.log_dir, "training_log.json")
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_metrics": [],
            "learning_rate": []
        }
        
        print(f"Model parameters: {count_parameters(model):,}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch
        
        Args:
            epoch: current epoch number
        
        Returns:
            metrics: dict of training metrics
        """
        self.model.train()
        
        loss_meter = AverageMeter()
        nmse_meter = AverageMeter()
        
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.cfg.n_epochs} [Train]",
            leave=False
        )
        
        for batch_idx, batch in enumerate(pbar):
            # Unpack batch
            X, Y, a_past, a_fut, _, _, _, a_baseline = batch
            X = X.to(self.device)
            Y = Y.to(self.device)
            a_past = a_past.to(self.device)
            a_baseline = a_baseline.to(self.device)
            
            batch_size = X.size(0)
            
            # Compute statistics text (use first sample)
            stats_text = compute_statistics_text(a_past[0:1])
            
            # Forward pass
            residual = self.model(X, stats_text)  # [B, H, 2]
            
            # Compose with baseline if enabled
            if self.cfg.use_ctrv_baseline:
                pred = compose_residual_with_baseline(residual, a_baseline)
            else:
                pred = residual
            
            # Compute loss
            loss = self.loss_fn(pred, Y, self.device)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.cfg.grad_clip > 0:
                if self.cfg.grad_clip_type == "norm":
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.cfg.grad_clip
                    )
                else:
                    nn.utils.clip_grad_value_(
                        self.model.parameters(),
                        self.cfg.grad_clip
                    )
            
            self.optimizer.step()
            
            # Metrics
            with torch.no_grad():
                nmse = normalized_angle_mse(pred, Y)
            
            loss_meter.update(loss.item(), batch_size)
            nmse_meter.update(nmse.item(), batch_size)
            
            # Update progress bar
            if batch_idx % self.cfg.log_interval == 0:
                pbar.set_postfix({
                    "loss": f"{loss_meter.avg:.4f}",
                    "nmse": f"{nmse_meter.avg:.4f}"
                })
        
        return {
            "loss": loss_meter.avg,
            "nmse": nmse_meter.avg
        }
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate model
        
        Args:
            epoch: current epoch number
        
        Returns:
            metrics: dict of validation metrics
        """
        self.model.eval()
        
        loss_meter = AverageMeter()
        nmse_meter = AverageMeter()
        mae_meter = AverageMeter()
        hit5_meter = AverageMeter()
        hit10_meter = AverageMeter()
        
        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch}/{self.cfg.n_epochs} [Val]  ",
            leave=False
        )
        
        for batch in pbar:
            # Unpack batch
            X, Y, a_past, a_fut, _, _, _, a_baseline = batch
            X = X.to(self.device)
            Y = Y.to(self.device)
            a_past = a_past.to(self.device)
            a_baseline = a_baseline.to(self.device)
            
            batch_size = X.size(0)
            
            # Statistics text
            stats_text = compute_statistics_text(a_past[0:1])
            
            # Forward pass
            residual = self.model(X, stats_text)
            
            # Compose with baseline
            if self.cfg.use_ctrv_baseline:
                pred = compose_residual_with_baseline(residual, a_baseline)
            else:
                pred = residual
            
            # Metrics
            loss = self.loss_fn(pred, Y, self.device)
            nmse = normalized_angle_mse(pred, Y)
            mae = mae_degrees(pred, Y)
            hit5 = hit_at_threshold(pred, Y, 5.0)
            hit10 = hit_at_threshold(pred, Y, 10.0)
            
            loss_meter.update(loss.item(), batch_size)
            nmse_meter.update(nmse.item(), batch_size)
            mae_meter.update(mae.item(), batch_size)
            hit5_meter.update(hit5.item(), batch_size)
            hit10_meter.update(hit10.item(), batch_size)
            
            pbar.set_postfix({
                "loss": f"{loss_meter.avg:.4f}",
                "mae": f"{mae_meter.avg:.2f}°"
            })
        
        return {
            "loss": loss_meter.avg,
            "nmse": nmse_meter.avg,
            "mae_deg": mae_meter.avg,
            "hit@5": hit5_meter.avg,
            "hit@10": hit10_meter.avg
        }
    
    def train(self) -> Dict[str, float]:
        """Train model for all epochs
        
        Returns:
            best_metrics: dict of best validation metrics
        """
        print(f"\nStarting training for {self.cfg.n_epochs} epochs...")
        print(f"Device: {self.device}")
        print("-" * 70)
        
        for epoch in range(1, self.cfg.n_epochs + 1):
            epoch_start = time.time()
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            if epoch % self.cfg.eval_interval == 0:
                val_metrics = self.validate(epoch)
                val_loss = val_metrics["loss"]
            else:
                val_metrics = None
                val_loss = None
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    if val_loss is not None:
                        self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Logging
            epoch_time = time.time() - epoch_start
            
            print(f"Epoch {epoch:3d}/{self.cfg.n_epochs} | "
                  f"Train Loss: {train_metrics['loss']:.4f} | ",
                  end="")
            
            if val_metrics is not None:
                print(f"Val Loss: {val_metrics['loss']:.4f} | "
                      f"MAE: {val_metrics['mae_deg']:.2f}° | "
                      f"Hit@10: {val_metrics['hit@10']:.3f} | ",
                      end="")
            
            print(f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")
            
            # Save history
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["learning_rate"].append(current_lr)
            if val_metrics is not None:
                self.history["val_loss"].append(val_metrics["loss"])
                self.history["val_metrics"].append(val_metrics)
            
            # Checkpointing
            if val_metrics is not None and self.cfg.save_best:
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch
                    self.patience_counter = 0
                    
                    # Save best model
                    checkpoint_path = os.path.join(
                        self.cfg.checkpoint_dir,
                        "best_model.pt"
                    )
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_loss": val_loss,
                        "val_metrics": val_metrics,
                        "config": self.cfg.to_dict()
                    }, checkpoint_path)
                    
                    print(f"  → Saved best model (val_loss: {val_loss:.4f})")
                else:
                    self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.cfg.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                print(f"Best validation loss: {self.best_val_loss:.4f} "
                      f"at epoch {self.best_epoch}")
                break
        
        # Save training history
        with open(self.log_file, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print("-" * 70)
        print(f"Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f} "
              f"at epoch {self.best_epoch}")
        
        if getattr(self.cfg, "save_visualizations", False):
            try:
                from evaluation import auto_visualize_from_log
                log_path = os.path.join(self.cfg.log_dir, "training_log.json")
                out_dir = getattr(self.cfg, "viz_dir", "./visualizations")
                auto_visualize_from_log(log_path, dict(self.cfg.__dict__), out_dir)
                print(f"[viz] saved to {out_dir}")
            except Exception as e:
                print(f"[viz] failed: {e}")
        
        # Load best model
        if os.path.exists(os.path.join(self.cfg.checkpoint_dir, "best_model.pt")):
            checkpoint = torch.load(
                os.path.join(self.cfg.checkpoint_dir, "best_model.pt"),
                map_location=self.device
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            return checkpoint["val_metrics"]
        
        return {}
