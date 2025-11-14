"""
Utility functions for beam prediction
Includes: seed setting, device management, metrics, angle operations
"""
import math
import random
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Additional settings for determinism
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device_str: str = "auto") -> torch.device:
    """Get torch device with fallback"""
    if isinstance(device_str, torch.device):
        return device_str
    
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and \
             torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device_str)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ==================== Angle Operations ====================

def wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]"""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def angle_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute angular difference wrapped to [-pi, pi]"""
    return torch.atan2(torch.sin(a - b), torch.cos(a - b))


def sincos_to_angle(sincos: torch.Tensor) -> torch.Tensor:
    """Convert (sin, cos) representation to angle
    Args:
        sincos: [..., 2] tensor with [..., 0] = sin, [..., 1] = cos
    Returns:
        angle: [...] tensor with angles in [-pi, pi]
    """
    return torch.atan2(sincos[..., 0], sincos[..., 1])


def angle_to_sincos(angle: torch.Tensor) -> torch.Tensor:
    """Convert angle to (sin, cos) representation
    Args:
        angle: [...] tensor with angles
    Returns:
        sincos: [..., 2] tensor with [..., 0] = sin, [..., 1] = cos
    """
    return torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)


def normalize_sincos(sincos: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize (sin, cos) to unit circle
    Args:
        sincos: [..., 2] tensor
    Returns:
        normalized sincos: [..., 2] tensor
    """
    norm = torch.sqrt(sincos[..., 0]**2 + sincos[..., 1]**2)
    norm = torch.clamp(norm, min=eps)
    return sincos / norm.unsqueeze(-1)


# ==================== Loss Functions ====================

def circular_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean"
) -> torch.Tensor:
    """Circular MSE loss for angle prediction
    Args:
        pred: [B, H, 2] predicted (sin, cos)
        target: [B, H, 2] target (sin, cos)
        reduction: "mean" | "sum" | "none"
    Returns:
        loss: scalar or [B, H] tensor
    """
    pred_angle = sincos_to_angle(pred)
    target_angle = sincos_to_angle(target)
    
    diff = angle_diff(pred_angle, target_angle)
    loss = (diff ** 2) / (math.pi ** 2)  # Normalize to [0, 1]
    
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


def weighted_circular_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    reduction: str = "mean"
) -> torch.Tensor:
    """Weighted circular MSE loss
    Args:
        pred: [B, H, 2] predicted (sin, cos)
        target: [B, H, 2] target (sin, cos)
        weights: [H] or [B, H] temporal weights
        reduction: "mean" | "sum" | "none"
    Returns:
        loss: scalar
    """
    loss = circular_mse_loss(pred, target, reduction="none")  # [B, H]
    
    if weights.dim() == 1:
        weights = weights.view(1, -1)  # [1, H]
    
    weighted_loss = loss * weights
    
    if reduction == "mean":
        return weighted_loss.mean()
    elif reduction == "sum":
        return weighted_loss.sum()
    else:
        return weighted_loss


def cosine_similarity_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean"
) -> torch.Tensor:
    """Cosine similarity loss for (sin, cos) predictions
    Args:
        pred: [B, H, 2] predicted (sin, cos)
        target: [B, H, 2] target (sin, cos)
        reduction: "mean" | "sum" | "none"
    Returns:
        loss: scalar or [B, H] tensor
    """
    # Normalize
    pred_norm = normalize_sincos(pred)
    target_norm = normalize_sincos(target)
    
    # Cosine similarity: dot product of normalized vectors
    cos_sim = (pred_norm * target_norm).sum(dim=-1)  # [B, H]
    
    # Loss: 1 - cos_sim (range [0, 2])
    loss = 1 - cos_sim
    
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


def combined_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    alpha: float = 0.7,
    beta: float = 0.3
) -> torch.Tensor:
    """Combined circular MSE and cosine similarity loss
    Args:
        pred: [B, H, 2] predicted (sin, cos)
        target: [B, H, 2] target (sin, cos)
        weights: [H] temporal weights
        alpha: weight for circular MSE
        beta: weight for cosine similarity
    Returns:
        loss: scalar
    """
    if weights is not None:
        loss_mse = weighted_circular_mse_loss(pred, target, weights)
    else:
        loss_mse = circular_mse_loss(pred, target)
    
    loss_cos = cosine_similarity_loss(pred, target)
    
    return alpha * loss_mse + beta * loss_cos


# ==================== Metrics ====================

def normalized_angle_mse(
    pred: torch.Tensor,
    target: torch.Tensor
) -> torch.Tensor:
    """Normalized angle MSE metric
    Args:
        pred: [B, H, 2] predicted (sin, cos)
        target: [B, H, 2] target (sin, cos)
    Returns:
        nmse: scalar in [0, 1]
    """
    return circular_mse_loss(pred, target, reduction="mean")


def mae_degrees(
    pred: torch.Tensor,
    target: torch.Tensor
) -> torch.Tensor:
    """Mean absolute error in degrees
    Args:
        pred: [B, H, 2] predicted (sin, cos)
        target: [B, H, 2] target (sin, cos)
    Returns:
        mae: scalar in degrees
    """
    pred_angle = sincos_to_angle(pred)
    target_angle = sincos_to_angle(target)
    
    diff = angle_diff(pred_angle, target_angle).abs()
    return diff.mean() * 180.0 / math.pi


def hit_at_threshold(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold_deg: float = 10.0
) -> torch.Tensor:
    """Hit rate at threshold (accuracy within threshold degrees)
    Args:
        pred: [B, H, 2] predicted (sin, cos)
        target: [B, H, 2] target (sin, cos)
        threshold_deg: threshold in degrees
    Returns:
        hit_rate: scalar in [0, 1]
    """
    pred_angle = sincos_to_angle(pred)
    target_angle = sincos_to_angle(target)
    
    diff_deg = angle_diff(pred_angle, target_angle).abs() * 180.0 / math.pi
    hit = (diff_deg <= threshold_deg).float()
    return hit.mean()


def best_beam_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_beams: int = 64
) -> torch.Tensor:
    """Beam index accuracy (how often predicted beam matches target)
    Args:
        pred: [B, H, 2] predicted (sin, cos)
        target: [B, H, 2] target (sin, cos)
        num_beams: number of beams in codebook
    Returns:
        accuracy: scalar in [0, 1]
    """
    pred_angle = sincos_to_angle(pred)
    target_angle = sincos_to_angle(target)
    
    # Convert angles to beam indices
    pred_idx = angle_to_beam_index(pred_angle, num_beams)
    target_idx = angle_to_beam_index(target_angle, num_beams)
    
    accuracy = (pred_idx == target_idx).float().mean()
    return accuracy


def angle_to_beam_index(angle: torch.Tensor, num_beams: int) -> torch.Tensor:
    """Convert angle to beam index
    Args:
        angle: [...] angles in [-pi, pi]
        num_beams: number of beams
    Returns:
        beam_idx: [...] beam indices in [0, num_beams-1]
    """
    # Normalize to [0, 1]
    normalized = (angle + math.pi) / (2 * math.pi)
    # Convert to index
    idx = (normalized * (num_beams - 1)).round().long()
    idx = torch.clamp(idx, 0, num_beams - 1)
    return idx


# ==================== Residual Composition ====================

def compose_residual_with_baseline(
    residual: torch.Tensor,
    baseline: torch.Tensor
) -> torch.Tensor:
    """Compose residual (sin, cos) with baseline angle
    
    This computes: angle_pred = angle_baseline + angle_residual
    using proper circular arithmetic in (sin, cos) space
    
    Args:
        residual: [B, H, 2] residual (sin, cos)
        baseline: [B, H] baseline angles in radians
    Returns:
        composed: [B, H, 2] composed (sin, cos)
    """
    if baseline.device != residual.device:
        baseline = baseline.to(residual.device)
    
    # Extract residual sin and cos
    sin_r = residual[..., 0]
    cos_r = residual[..., 1]
    
    # Baseline sin and cos
    sin_b = torch.sin(baseline)
    cos_b = torch.cos(baseline)
    
    # Angle addition formula:
    # sin(a + b) = sin(a)cos(b) + cos(a)sin(b)
    # cos(a + b) = cos(a)cos(b) - sin(a)sin(b)
    sin_composed = sin_b * cos_r + cos_b * sin_r
    cos_composed = cos_b * cos_r - sin_b * sin_r
    
    # Stack and normalize
    composed = torch.stack([sin_composed, cos_composed], dim=-1)
    return normalize_sincos(composed)


# ==================== Feature Normalization ====================

class RevIN(nn.Module):
    """Reversible Instance Normalization for time series"""
    
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))
    
    def forward(self, x: torch.Tensor, mode: str = "norm") -> torch.Tensor:
        """
        Args:
            x: [B, C, T] input
            mode: "norm" or "denorm"
        Returns:
            normalized or denormalized x
        """
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Ensure contiguous memory layout for MPS compatibility
        if x.device.type == "mps":
            x = x.contiguous()
        
        return x
    
    def _get_statistics(self, x: torch.Tensor):
        """Compute mean and std along time dimension"""
        self.mean = x.mean(dim=2, keepdim=True).detach()
        self.std = torch.sqrt(
            x.var(dim=2, keepdim=True, unbiased=False) + self.eps
        ).detach()
    
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input"""
        x = (x - self.mean) / self.std
        if self.affine:
            x = x * self.affine_weight.view(1, -1, 1)
            x = x + self.affine_bias.view(1, -1, 1)
        return x
    
    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize input"""
        if self.affine:
            x = x - self.affine_bias.view(1, -1, 1)
            x = x / self.affine_weight.view(1, -1, 1)
        x = x * self.std + self.mean
        return x


# ==================== Statistics for PaP ====================

def compute_statistics_text(
    aod_past: torch.Tensor,
    include_autocorr: bool = True
) -> str:
    """Compute statistics text for Prompt-as-Prefix
    
    Args:
        aod_past: [U] or [B, U] past AoD angles
        include_autocorr: whether to include autocorrelation info
    Returns:
        stats_text: string description of statistics
    """
    if aod_past.dim() == 2:
        # Take first sample if batch
        aod_past = aod_past[0]
    
    aod_past = aod_past.detach().cpu().numpy()
    
    # Basic statistics
    mean_val = float(np.mean(aod_past))
    std_val = float(np.std(aod_past))
    
    # Trend (difference between last and first)
    trend = float(aod_past[-1] - aod_past[0])
    trend_str = "upward" if trend > 0 else "downward"
    
    # Velocity (mean absolute difference)
    if len(aod_past) > 1:
        diff = np.diff(aod_past)
        velocity = float(np.mean(np.abs(diff)))
    else:
        velocity = 0.0
    
    stats_text = (
        f"Past AoD statistics: mean={mean_val:.3f} rad, "
        f"std={std_val:.3f} rad, trend={trend_str} ({trend:.3f} rad), "
        f"velocity={velocity:.4f} rad/step"
    )
    
    if include_autocorr and len(aod_past) > 5:
        # Simple autocorrelation at lag 1
        try:
            autocorr_1 = float(np.corrcoef(aod_past[:-1], aod_past[1:])[0, 1])
            stats_text += f", autocorr(lag1)={autocorr_1:.3f}"
        except:
            pass
    
    return stats_text


# ==================== Logging ====================

class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
