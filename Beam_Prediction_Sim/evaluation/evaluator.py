"""
Evaluation module with autoregressive prediction and detailed metrics
"""
import torch
import torch.nn as nn
import numpy as np
import math
from tqdm import tqdm
from typing import Dict, List, Tuple

# Fixed imports

from config import Config
from utils import (
    normalized_angle_mse,
    mae_degrees,
    hit_at_threshold,
    sincos_to_angle,
    angle_to_sincos,
    compose_residual_with_baseline,
    compute_statistics_text,
    AverageMeter
)
from data import ctrv_predict


class Evaluator:
    """Evaluator for beam prediction model"""
    
    def __init__(
        self,
        model: nn.Module,
        test_loader: torch.utils.data.DataLoader,
        cfg: Config,
        device: torch.device
    ):
        """
        Args:
            model: trained model
            test_loader: test dataloader
            cfg: configuration
            device: torch device
        """
        self.model = model
        self.test_loader = test_loader
        self.cfg = cfg
        self.device = device
        
        self.model.to(device)
        self.model.eval()
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on test set
        
        Returns:
            metrics: dict of test metrics
        """
        print("\nEvaluating model on test set...")
        
        nmse_meter = AverageMeter()
        mae_meter = AverageMeter()
        hit5_meter = AverageMeter()
        hit10_meter = AverageMeter()
        hit15_meter = AverageMeter()
        
        # Per-step metrics
        nmse_per_step = [AverageMeter() for _ in range(self.cfg.H)]
        mae_per_step = [AverageMeter() for _ in range(self.cfg.H)]
        
        # Collect errors for histogram
        all_errors_deg = []
        
        pbar = tqdm(self.test_loader, desc="Testing")
        
        for batch in pbar:
            # Unpack batch
            X, Y, a_past, a_fut, q_all, gain_all, traj, a_baseline = batch
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
            
            # Overall metrics
            nmse = normalized_angle_mse(pred, Y)
            mae = mae_degrees(pred, Y)
            hit5 = hit_at_threshold(pred, Y, 5.0)
            hit10 = hit_at_threshold(pred, Y, 10.0)
            hit15 = hit_at_threshold(pred, Y, 15.0)
            
            nmse_meter.update(nmse.item(), batch_size)
            mae_meter.update(mae.item(), batch_size)
            hit5_meter.update(hit5.item(), batch_size)
            hit10_meter.update(hit10.item(), batch_size)
            hit15_meter.update(hit15.item(), batch_size)
            
            # Per-step metrics
            for h in range(self.cfg.H):
                pred_h = pred[:, h:h+1, :]
                Y_h = Y[:, h:h+1, :]
                nmse_h = normalized_angle_mse(pred_h, Y_h)
                mae_h = mae_degrees(pred_h, Y_h)
                nmse_per_step[h].update(nmse_h.item(), batch_size)
                mae_per_step[h].update(mae_h.item(), batch_size)
            
            # Collect errors
            pred_angles = sincos_to_angle(pred)
            target_angles = sincos_to_angle(Y)
            errors = (pred_angles - target_angles).abs() * 180.0 / math.pi
            all_errors_deg.append(errors.cpu())
            
            pbar.set_postfix({
                "MAE": f"{mae_meter.avg:.2f}°",
                "Hit@10": f"{hit10_meter.avg:.3f}"
            })
        
        # Compile results
        metrics = {
            "nmse": nmse_meter.avg,
            "mae_deg": mae_meter.avg,
            "hit@5": hit5_meter.avg,
            "hit@10": hit10_meter.avg,
            "hit@15": hit15_meter.avg,
            "nmse_per_step": [m.avg for m in nmse_per_step],
            "mae_per_step": [m.avg for m in mae_per_step]
        }
        
        # Print results
        print("\n" + "=" * 70)
        print("TEST RESULTS")
        print("=" * 70)
        print(f"Normalized MSE: {metrics['nmse']:.6f}")
        print(f"MAE (degrees):  {metrics['mae_deg']:.2f}°")
        print(f"Hit@5:          {metrics['hit@5']:.4f}")
        print(f"Hit@10:         {metrics['hit@10']:.4f}")
        print(f"Hit@15:         {metrics['hit@15']:.4f}")
        print("\nPer-step MAE:")
        for h, mae in enumerate(metrics['mae_per_step'], 1):
            print(f"  Step {h:2d}: {mae:.2f}°")
        print("=" * 70)
        
        # Save errors for visualization
        all_errors = torch.cat(all_errors_deg, dim=0).numpy()  # [N, H]
        metrics["error_matrix"] = all_errors
        
        return metrics
    
    @torch.no_grad()
    def evaluate_autoregressive(
        self,
        num_samples: int = 100
    ) -> Dict[str, float]:
        """Evaluate with autoregressive prediction
        
        This is more realistic: use model predictions to update state
        and predict next step iteratively.
        
        Args:
            num_samples: number of samples to test
        
        Returns:
            metrics: dict of AR test metrics
        """
        print(f"\nAutoregressive evaluation on {num_samples} samples...")
        
        dataset = self.test_loader.dataset
        num_samples = min(num_samples, len(dataset))
        
        mae_meter = AverageMeter()
        hit10_meter = AverageMeter()
        
        for idx in tqdm(range(num_samples), desc="AR Testing"):
            # Get sample
            sample = dataset[idx]
            X, Y, a_past, a_fut, q_all, gain_all, traj, a_baseline = sample
            
            # Predict autoregressively
            pred_angles = self._predict_ar_one_sample(
                X, a_past, a_baseline
            )  # [H]
            
            # Convert to sin/cos for metric computation
            pred_sincos = angle_to_sincos(pred_angles).unsqueeze(0)  # [1, H, 2]
            target_sincos = Y.unsqueeze(0).to(self.device)  # [1, H, 2]
            
            # Compute metrics
            mae = mae_degrees(pred_sincos, target_sincos)
            hit10 = hit_at_threshold(pred_sincos, target_sincos, 10.0)
            
            mae_meter.update(mae.item(), 1)
            hit10_meter.update(hit10.item(), 1)
        
        metrics = {
            "ar_mae_deg": mae_meter.avg,
            "ar_hit@10": hit10_meter.avg
        }
        
        print(f"\nAutoregressive Results:")
        print(f"  MAE: {metrics['ar_mae_deg']:.2f}°")
        print(f"  Hit@10: {metrics['ar_hit@10']:.4f}")
        
        return metrics
    
    def _predict_ar_one_sample(
        self,
        X: torch.Tensor,
        a_past: torch.Tensor,
        a_baseline: torch.Tensor
    ) -> torch.Tensor:
        """Predict autoregressively for one sample
        
        Args:
            X: [C, U] input features
            a_past: [U] past AoD angles
            a_baseline: [H] baseline predictions
        
        Returns:
            predictions: [H] predicted AoD angles
        """
        cfg = self.cfg
        device = self.device
        
        # Move to device
        X = X.clone().to(device)
        a_past = a_past.to(device)
        a_baseline = a_baseline.to(device)
        
        # Extract last state
        # Feature indices: [q, x, y, hs, hc, v, r, w, (sin_aod, cos_aod)]
        last_features = X[:, -1].cpu().numpy()
        
        # Denormalize to get actual values
        q_norm, x_norm, y_norm = last_features[0:3]
        hs_val, hc_val = last_features[3:5]
        v_norm, r_norm, w_norm = last_features[5:8]
        
        # Reconstruct state
        L = cfg.area_size_m
        x = x_norm * L
        y = y_norm * L
        heading = math.atan2(hs_val, hc_val)
        v = v_norm * (cfg.speed_max_mps - cfg.speed_min_mps) + cfg.speed_min_mps
        
        # Estimate omega from last features
        if cfg.U >= 2:
            omega = w_norm * math.pi
        else:
            omega = 0.0
        
        predictions = []
        
        for h in range(cfg.H):
            # Compute statistics
            stats_text = compute_statistics_text(a_past.unsqueeze(0))
            
            # Predict (get residual)
            with torch.no_grad():
                residual = self.model(X.unsqueeze(0), stats_text)  # [1, H, 2]
                res_h = residual[0, 0, :]  # [2] - first step
            
            # Compose with baseline
            if cfg.use_ctrv_baseline:
                base_angle = a_baseline[h:h+1].view(1, 1)
                pred_sincos = compose_residual_with_baseline(
                    res_h.view(1, 1, 2),
                    base_angle
                )  # [1, 1, 2]
                pred_angle = sincos_to_angle(pred_sincos).squeeze()
            else:
                pred_angle = sincos_to_angle(res_h)
            
            predictions.append(pred_angle.item())
            
            # Update state with CTRV
            x, y, heading = ctrv_predict(
                x, y, heading, v, omega,
                cfg.delta_t_s, L, use_reflection=True
            )
            
            # Compute predicted beam index from angle
            # (simplified: use direct mapping)
            q_pred = int(((pred_angle.item() + math.pi) / (2 * math.pi)) * (cfg.M - 1))
            q_pred = max(0, min(cfg.M - 1, q_pred))
            q_pred_norm = (q_pred / (cfg.M - 1)) * 2 - 1.0
            
            # Build new feature column
            bs_x, bs_y = L/2, L/2
            r = math.hypot(x - bs_x, y - bs_y)
            
            new_features = torch.tensor([
                q_pred_norm,
                x / L,
                y / L,
                math.sin(heading),
                math.cos(heading),
                (v - cfg.speed_min_mps) / (cfg.speed_max_mps - cfg.speed_min_mps + 1e-8),
                r / (L * math.sqrt(2) + 1e-6),
                omega / math.pi
            ], device=device, dtype=torch.float32)
            
            # Add AoD features if enabled
            if cfg.include_aod_in_features:
                new_features = torch.cat([
                    new_features,
                    torch.tensor([math.sin(pred_angle.item()), 
                                  math.cos(pred_angle.item())], 
                                 device=device)
                ])
            
            # Roll window
            X = torch.cat([X[:, 1:], new_features.view(-1, 1)], dim=1)
            a_past = torch.cat([a_past[1:], pred_angle.view(1)])
        
        return torch.tensor(predictions)


def compute_beam_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    codebook_size: int
) -> float:
    """Compute beam selection accuracy
    
    Args:
        predictions: [B, H, 2] predicted (sin, cos)
        targets: [B, H, 2] target (sin, cos)
        codebook_size: number of beams
    
    Returns:
        accuracy: beam selection accuracy
    """
    pred_angles = sincos_to_angle(predictions)
    target_angles = sincos_to_angle(targets)
    
    # Convert to beam indices
    pred_indices = ((pred_angles + math.pi) / (2 * math.pi) * (codebook_size - 1))
    pred_indices = pred_indices.round().long()
    pred_indices = torch.clamp(pred_indices, 0, codebook_size - 1)
    
    target_indices = ((target_angles + math.pi) / (2 * math.pi) * (codebook_size - 1))
    target_indices = target_indices.round().long()
    target_indices = torch.clamp(target_indices, 0, codebook_size - 1)
    
    accuracy = (pred_indices == target_indices).float().mean().item()
    
    return accuracy
