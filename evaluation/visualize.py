"""
Visualization module for beam prediction results
"""
import numpy as np
import math
import matplotlib.pyplot as plt
from typing import Optional, List
import torch
import json, os

# Fixed imports

from config import Config


def sincos_to_angle(sincos):
    """Convert (sin, cos) to angle"""
    if isinstance(sincos, torch.Tensor):
        return torch.atan2(sincos[..., 0], sincos[..., 1])
    else:
        return np.arctan2(sincos[..., 0], sincos[..., 1])


def plot_training_history(
    history: dict,
    save_path: str = "training_history.png"
):
    """Plot training history
    
    Args:
        history: training history dict
        save_path: path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax = axes[0]
    ax.plot(history["train_loss"], label="Train Loss", alpha=0.8)
    if "val_loss" in history and len(history["val_loss"]) > 0:
        # Val loss may be recorded less frequently
        val_epochs = np.arange(1, len(history["val_loss"]) + 1)
        ax.plot(val_epochs, history["val_loss"], label="Val Loss", alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Learning rate plot
    ax = axes[1]
    ax.plot(history["learning_rate"], label="Learning Rate")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved training history plot to {save_path}")


def plot_sample_prediction(
    sample: tuple,
    predictions: torch.Tensor,
    cfg: Config,
    save_path: str = "sample_prediction.png"
):
    """Plot prediction for a single sample
    
    Args:
        sample: (X, Y, a_past, a_fut, q_all, gain_all, traj, a_baseline)
        predictions: [H, 2] predicted (sin, cos)
        cfg: configuration
        save_path: path to save figure
    """
    X, Y, a_past, a_fut, q_all, gain_all, traj, a_baseline = sample
    
    # Convert to numpy
    if isinstance(predictions, torch.Tensor):
        predictions_np = predictions.detach().cpu().numpy()
    else:
        predictions_np = predictions
    if isinstance(Y, torch.Tensor):
        Y_np = Y.detach().cpu().numpy()
    else:
        Y_np = Y
    if isinstance(a_past, torch.Tensor):
        a_past_np = a_past.detach().cpu().numpy()
    else:
        a_past_np = a_past
    if isinstance(a_fut, torch.Tensor):
        a_fut_np = a_fut.detach().cpu().numpy()
    else:
        a_fut_np = a_fut
    if isinstance(traj, torch.Tensor):
        traj = traj.detach().cpu().numpy()
    
    # Convert to angles
    pred_angles = sincos_to_angle(predictions_np)
    target_angles = sincos_to_angle(Y_np)
    
    # Compute errors
    errors = np.abs(pred_angles - target_angles)
    errors = np.minimum(errors, 2*np.pi - errors)  # Wrap to [0, π]
    errors_deg = errors * 180.0 / np.pi
    
    # Metrics
    mae = np.mean(errors_deg)
    hit10 = np.mean(errors_deg <= 10.0)
    
    # Create figure
    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.8])
    
    # Plot 1: Trajectory and AoDs
    ax1 = fig.add_subplot(gs[0])
    L = cfg.area_size_m
    bs_pos = np.array([L/2, L/2])
    
    ax1.set_xlim(0, L)
    ax1.set_ylim(0, L)
    ax1.set_aspect('equal')
    ax1.set_title("Trajectory and Beam Directions")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    
    # Plot trajectory
    U, H = cfg.U, cfg.H
    ax1.plot(traj[:, 0], traj[:, 1], 'k-', alpha=0.3, linewidth=1)
    ax1.plot(traj[:U, 0], traj[:U, 1], 'b-', alpha=0.5, linewidth=2, label="Past")
    ax1.plot(traj[U:, 0], traj[U:, 1], 'g-', alpha=0.5, linewidth=2, label="Future")
    
    # Plot BS
    ax1.scatter(bs_pos[0], bs_pos[1], marker='^', s=200, c='red', 
                edgecolors='black', linewidth=2, label="BS", zorder=10)
    
    # Plot last past position
    ax1.scatter(traj[U-1, 0], traj[U-1, 1], marker='o', s=100, c='blue',
                edgecolors='black', linewidth=1.5, label="Current", zorder=9)
    
    # Plot future positions
    for i in range(0, H, max(1, H//3)):
        ax1.scatter(traj[U+i, 0], traj[U+i, 1], marker='s', s=60, c='green',
                    alpha=0.7, edgecolors='black', linewidth=1, zorder=8)
    
    # Plot beam directions (every other step to avoid clutter)
    ray_length = L * 0.25
    for i in range(0, H, max(1, H//5)):
        # Target
        theta_t = target_angles[i]
        x_t = bs_pos[0] + ray_length * np.cos(theta_t)
        y_t = bs_pos[1] + ray_length * np.sin(theta_t)
        ax1.plot([bs_pos[0], x_t], [bs_pos[1], y_t], 'g:', linewidth=2, alpha=0.6)
        
        # Prediction
        theta_p = pred_angles[i]
        x_p = bs_pos[0] + ray_length * np.cos(theta_p)
        y_p = bs_pos[1] + ray_length * np.sin(theta_p)
        ax1.plot([bs_pos[0], x_p], [bs_pos[1], y_p], 'r--', linewidth=2, alpha=0.6)
    
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(alpha=0.2)
    
    # Plot 2: Angle errors over time
    ax2 = fig.add_subplot(gs[1])
    steps = np.arange(1, H+1)
    ax2.plot(steps, errors_deg, 'o-', linewidth=2, markersize=8, label="Error")
    ax2.axhline(10, color='red', linestyle='--', alpha=0.5, label="10° threshold")
    ax2.set_xlabel("Prediction Step")
    ax2.set_ylabel("Absolute Error (degrees)")
    ax2.set_title(f"Angle Prediction Error\nMAE={mae:.2f}°, Hit@10={hit10:.3f}")
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_ylim(bottom=0)
    
    # Plot 3: Angle timeline
    ax3 = fig.add_subplot(gs[2])
    all_steps = np.arange(-U+1, H+1)
    
    # Past AoDs
    ax3.plot(all_steps[:U], a_past * 180/np.pi, 'b-', linewidth=2, 
             label="Past AoD", alpha=0.7)
    
    # Future: target vs prediction
    future_steps = all_steps[U:]
    ax3.plot(future_steps, target_angles * 180/np.pi, 'g-', linewidth=2, 
             label="Target", alpha=0.7)
    ax3.plot(future_steps, pred_angles * 180/np.pi, 'r--', linewidth=2, 
             label="Prediction", alpha=0.7)
    
    ax3.axvline(0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    ax3.set_xlabel("Time Step")
    ax3.set_ylabel("AoD (degrees)")
    ax3.set_title("Angle of Departure")
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved sample prediction plot to {save_path}")


def plot_error_histogram(
    error_matrix: np.ndarray,
    H: int,
    save_path: str = "error_histogram.png"
):
    """Plot error histogram for each prediction step
    
    Args:
        error_matrix: [N, H] error matrix in degrees
        H: prediction horizon
        save_path: path to save figure
    """
    fig, axes = plt.subplots(1, H, figsize=(3*H, 3), squeeze=False)
    axes = axes.flatten()
    
    for h in range(H):
        ax = axes[h]
        errors_h = error_matrix[:, h]
        
        ax.hist(errors_h, bins=40, density=True, alpha=0.7, edgecolor='black')
        ax.set_xlabel("Error (degrees)")
        ax.set_ylabel("Density")
        ax.set_title(f"Step {h+1}\nMAE={np.mean(errors_h):.2f}°")
        ax.grid(alpha=0.3)
        ax.axvline(np.mean(errors_h), color='red', linestyle='--', 
                   linewidth=2, label=f"Mean")
        ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved error histogram to {save_path}")


def plot_metrics_per_step(
    metrics_per_step: List[float],
    metric_name: str,
    save_path: str = "metrics_per_step.png"
):
    """Plot metrics per prediction step
    
    Args:
        metrics_per_step: list of metric values per step
        metric_name: name of the metric
        save_path: path to save figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    steps = np.arange(1, len(metrics_per_step) + 1)
    ax.plot(steps, metrics_per_step, 'o-', linewidth=2, markersize=8)
    
    ax.set_xlabel("Prediction Step")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} vs Prediction Horizon")
    ax.grid(alpha=0.3)
    ax.set_xticks(steps)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved per-step metrics plot to {save_path}")


def create_all_visualizations(
    cfg: Config,
    history: dict,
    test_metrics: dict,
    test_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    viz_dir: str = "./visualizations"
):
    """Create all visualizations
    
    Args:
        cfg: configuration
        history: training history
        test_metrics: test metrics
        test_loader: test dataloader
        model: trained model
        device: torch device
        viz_dir: directory to save visualizations
    """
    os.makedirs(viz_dir, exist_ok=True)
    
    print("\nCreating visualizations...")
    
    # 1. Training history
    if history:
        plot_training_history(
            history,
            os.path.join(viz_dir, "training_history.png")
        )
    
    # 2. Error histograms
    if "error_matrix" in test_metrics:
        plot_error_histogram(
            test_metrics["error_matrix"],
            cfg.H,
            os.path.join(viz_dir, "error_histograms.png")
        )
    
    # 3. Per-step metrics
    if "mae_per_step" in test_metrics:
        plot_metrics_per_step(
            test_metrics["mae_per_step"],
            "MAE (degrees)",
            os.path.join(viz_dir, "mae_per_step.png")
        )
    
    # 4. Sample predictions
    model.eval()
    dataset = test_loader.dataset
    
    for i in range(min(cfg.viz_samples, getattr(dataset, '__len__', lambda: cfg.viz_samples)())):
        sample = dataset[i]
        X, Y, a_past, *rest = sample
        
        # Predict
        X_batch = X.unsqueeze(0).to(device)
        a_past_batch = a_past.unsqueeze(0).to(device)
        
        with torch.no_grad():
            from utils import compute_statistics_text, compose_residual_with_baseline
            stats_text = compute_statistics_text(a_past_batch[0:1])
            residual = model(X_batch, stats_text)
            
            # Compose with baseline if available
            if len(rest) >= 2:
                a_baseline = rest[-1].unsqueeze(0).to(device)
                if cfg.use_ctrv_baseline:
                    pred = compose_residual_with_baseline(residual, a_baseline)
                else:
                    pred = residual
            else:
                pred = residual
        
        pred = pred.squeeze(0)  # [H, 2]
        
        plot_sample_prediction(
            sample,
            pred,
            cfg,
            os.path.join(viz_dir, f"sample_{i+1}.png")
        )
    
    print(f"Visualizations saved to {viz_dir}")


def _save_line_chart(epochs, y, title, ylabel, filepath, y2=None, label1=None, label2=None):
    plt.figure()
    plt.plot(epochs, y, label=label1 or ylabel)
    if y2 is not None:
        plt.plot(epochs, y2, label=label2 or "")
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    if y2 is not None or label1 is not None:
        plt.legend()
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()

def auto_visualize_from_log(log_path: str, cfg: dict, out_dir: str):
    with open(log_path, "r") as f:
        logs = json.load(f)
    epochs = np.arange(1, len(logs["train_loss"]) + 1)

    # curves
    _save_line_chart(epochs, logs["train_loss"], "Train vs Val Loss", "loss",
                     os.path.join(out_dir, "loss_curves.png"),
                     y2=logs["val_loss"], label1="train_loss", label2="val_loss")

    val_metrics = logs.get("val_metrics", [])
    mae = np.array([m.get("mae_deg", np.nan) for m in val_metrics], dtype=float)
    nmse = np.array([m.get("nmse", np.nan) for m in val_metrics], dtype=float)
    hit5 = np.array([m.get("hit@5", np.nan) for m in val_metrics], dtype=float)
    hit10 = np.array([m.get("hit@10", np.nan) for m in val_metrics], dtype=float)

    _save_line_chart(epochs, mae, "Validation MAE (deg)", "deg",
                     os.path.join(out_dir, "val_mae_deg.png"))
    _save_line_chart(epochs, nmse, "Validation NMSE", "NMSE",
                     os.path.join(out_dir, "val_nmse.png"))
    _save_line_chart(epochs, hit5, "Validation hit@5, hit@10", "ratio",
                     os.path.join(out_dir, "val_hit.png"),
                     y2=hit10, label1="hit@5", label2="hit@10")

    _save_line_chart(epochs, logs["learning_rate"], "Learning Rate Schedule", "lr",
                     os.path.join(out_dir, "lr_schedule.png"))

    # summary
    best = int(np.nanargmin(np.array(logs["val_loss"])) + 1)
    summary = {
        "best_epoch": best,
        "best_val_loss": float(logs["val_loss"][best-1]),
        "val_mae_deg_at_best": float(mae[best-1]),
        "val_hit@5_at_best": float(hit5[best-1]),
        "val_hit@10_at_best": float(hit10[best-1]),
        "val_nmse_at_best": float(nmse[best-1]),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary