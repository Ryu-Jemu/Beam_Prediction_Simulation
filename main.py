"""
Main entry point for beam prediction training and evaluation
"""
import torch
import argparse
import json
import os
from typing import Optional

from config import Config, get_default_config, get_lightweight_config
from data import create_dataloaders, BeamSeqDataset
from models import BeamPredictorLLM
from training import Trainer
from evaluation import Evaluator, create_all_visualizations
from utils import set_seed

from beam_improvements import (
    apply_improvements,
    compute_enhanced_statistics_text,
    validate_improvements
)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Beam Prediction with LLMs")
    
    # Mode
    parser.add_argument("--mode", type=str, default="both",  # Changed: both is default
                        choices=["train", "eval", "both"],
                        help="Mode: train, eval, or both (default: both)")
    
    # Config
    parser.add_argument("--config", type=str, default="default",
                        choices=["default", "lightweight", "tiny", 
                                 "long_trajectory", "extreme_trajectory"],
                        help="Configuration preset (use long_trajectory for increased travel distance)")
    
    parser.add_argument("--use_improvements", action="store_true", default=True,
                        help="Apply all improvements (default: True)")
    parser.add_argument("--use_position_constraints", action="store_true",
                        help="Enable position-based constraints")
    parser.add_argument("--use_enhanced_stats", action="store_true", default=True,
                        help="Use enhanced statistics with FFT")
    parser.add_argument("--use_streaming", action="store_true",
                        help="Use memory-efficient streaming dataset")
    
    # Model
    parser.add_argument("--use_gpt2", action="store_true",
                        help="Use GPT-2 backbone")
    parser.add_argument("--no_gpt2", dest="use_gpt2", action="store_false")
    parser.set_defaults(use_gpt2=None)
    
    # Training
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate")
    
    # Evaluation
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint for evaluation")
    parser.add_argument("--ar_eval", action="store_true",
                        help="Use autoregressive evaluation")
    
    # Data
    parser.add_argument("--n_train", type=int, default=None,
                        help="Number of training samples")
    parser.add_argument("--n_val", type=int, default=None,
                        help="Number of validation samples")
    parser.add_argument("--n_test", type=int, default=None,
                        help="Number of test samples")
    
    # Other
    parser.add_argument("--seed", type=int, default=1337,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cuda, mps, or cpu")
    parser.add_argument("--no_viz", action="store_true",
                        help="Disable visualizations")
    
    return parser.parse_args()


def get_config(args):
    """Get configuration from args
    
    Args:
        args: parsed arguments
    
    Returns:
        cfg: configuration
    """
    # Load base config
    if args.config == "lightweight":
        cfg = get_lightweight_config()
    elif args.config == "tiny":
        from config import get_tiny_config
        cfg = get_tiny_config()
    elif args.config == "long_trajectory":
        from config import get_long_trajectory_config
        cfg = get_long_trajectory_config()
    elif args.config == "extreme_trajectory":
        from config import get_extreme_trajectory_config
        cfg = get_extreme_trajectory_config()
    else:
        cfg = get_default_config()
    
    # Override with command line args
    if args.use_gpt2 is not None:
        cfg.use_gpt2 = args.use_gpt2
    if args.epochs is not None:
        cfg.n_epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.n_train is not None:
        cfg.n_train = args.n_train
    if args.n_val is not None:
        cfg.n_val = args.n_val
    if args.n_test is not None:
        cfg.n_test = args.n_test
    
    cfg.seed = args.seed
    cfg.device = args.device
    cfg.save_visualizations = not args.no_viz
    
    cfg.use_improvements = args.use_improvements
    cfg.use_position_constraints = args.use_position_constraints
    cfg.use_enhanced_stats = args.use_enhanced_stats
    cfg.use_streaming_dataset = args.use_streaming
    
    return cfg


def train_model(cfg: Config, device: torch.device):
    """Train model
    
    Args:
        cfg: configuration
        device: torch device
    
    Returns:
        model: trained model
        history: training history
    """
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(cfg)
    
    # Create model
    print("\nInitializing model...")
    model = BeamPredictorLLM(cfg)
    print(f"Model created successfully")
    print(f"Using GPT-2: {cfg.use_gpt2}")
    
    if cfg.use_improvements:
        print("\n🔧 Applying improvements...")
        cfg, model, (train_loader, val_loader, test_loader) = apply_improvements(
            cfg, model, (train_loader, val_loader, test_loader)
        )
        print("✅ Improvements applied!")
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, cfg, device)

    if cfg.use_enhanced_stats:
        original_compute_stats = trainer.compute_statistics_text if hasattr(trainer, 'compute_statistics_text') else None
        
        def enhanced_stats_wrapper(aod_past: torch.Tensor, include_autocorr: bool = True) -> str:
            return compute_enhanced_statistics_text(aod_past, include_autocorr=include_autocorr)
        
        # Monkey-patch the enhanced statistics
        import utils
        utils.compute_statistics_text = enhanced_stats_wrapper
        print("✅ Enhanced statistics activated")

    # Train
    best_metrics = trainer.train()
    
    # Save config
    config_path = os.path.join(cfg.checkpoint_dir, "config.json")
    with open(config_path, 'w') as f:
        config_dict = cfg.to_dict()
        # Add improvement flags
        config_dict['improvements_applied'] = cfg.use_improvements
        config_dict['position_constraints'] = cfg.use_position_constraints
        config_dict['enhanced_stats'] = cfg.use_enhanced_stats
        json.dump(config_dict, f, indent=2)
    print(f"\nSaved config to {config_path}")
    
    return model, trainer.history


def evaluate_model(
    cfg: Config,
    device: torch.device,
    model: Optional[torch.nn.Module] = None,
    checkpoint_path: Optional[str] = None
):
    """Evaluate model
    
    Args:
        cfg: configuration
        device: torch device
        model: trained model (optional)
        checkpoint_path: path to checkpoint (optional)
    
    Returns:
        test_metrics: test metrics dict
        test_loader: test dataloader
    """
    print("\n" + "="*70)
    print("EVALUATION")
    print("="*70)
    
    # Create test dataloader
    test_dataset = BeamSeqDataset(cfg.n_test, cfg, "test", cfg.seed)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory
    )
    
    # Load model if not provided
    if model is None:
        print("\nLoading model from checkpoint...")
        if checkpoint_path is None:
            checkpoint_path = os.path.join(cfg.checkpoint_dir, "best_model.pt")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        model = BeamPredictorLLM(cfg)
        
        if cfg.use_improvements:
            cfg, model, _ = apply_improvements(cfg, model)    
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if model is not None:
            model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  Val Loss: {checkpoint['val_loss']:.4f}")

    # Create evaluator (ensure model is not None)
    if model is None:
        raise ValueError("Model must be provided for evaluation")
    evaluator = Evaluator(model, test_loader, cfg, device)
    
    # Evaluate
    test_metrics = evaluator.evaluate()
    
    if cfg.use_improvements:
        print("\n📋 Validating improvements...")
        validation_results = validate_improvements(cfg, model, test_loader)
        test_metrics['improvement_validation'] = validation_results

    return test_metrics, test_loader


def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Get configuration
    cfg = get_config(args)
    
    # Set seed
    set_seed(cfg.seed)
    
    # Get device
    device = cfg.get_device()
    print(f"\nUsing device: {device}")
    print(f"Random seed: {cfg.seed}")
    
    # Print configuration
    print("\nConfiguration:")
    print("-" * 70)
    for key, value in cfg.to_dict().items():
        if not key.startswith('_'):
            print(f"  {key:30s}: {value}")
    print("-" * 70)
    
    if cfg.use_improvements:
        print("\n🔧 Improvements Enabled:")
        print(f"  - Position Constraints: {cfg.use_position_constraints}")
        print(f"  - Enhanced Statistics: {cfg.use_enhanced_stats}")
        print(f"  - Streaming Dataset: {cfg.use_streaming_dataset}")
        print("-" * 70)

    # Execute based on mode
    model = None
    history = None
    test_metrics = None
    test_loader = None
    
    if args.mode in ["train", "both"]:
        model, history = train_model(cfg, device)
    
    if args.mode in ["eval", "both"]:
        test_metrics, test_loader = evaluate_model(
            cfg,
            device,
            model=model,
            checkpoint_path=args.checkpoint
        )
        if cfg.use_improvements and 'improvement_validation' in test_metrics:
            print("\n📊 Improvement Metrics:")
            for key, value in test_metrics['improvement_validation'].items():
                print(f"  {key}: {value}")
    
    # Visualizations - ALWAYS create if not explicitly disabled
    if cfg.save_visualizations:
        # Ensure we have necessary data
        if test_metrics is None and args.mode == "train":
            print("\n" + "="*70)
            print("Running evaluation for visualizations...")
            print("="*70)
            test_metrics, test_loader = evaluate_model(cfg, device, model=model)
        
        if test_metrics is not None:
            # Load model if needed
            if model is None:
                checkpoint_path = args.checkpoint or os.path.join(
                    cfg.checkpoint_dir, "best_model.pt"
                )
                if os.path.exists(checkpoint_path):
                    print("\nLoading model for visualization...")
                    model = BeamPredictorLLM(cfg)
                    checkpoint = torch.load(checkpoint_path, map_location=device)
                    model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    print(f"⚠️  Checkpoint not found: {checkpoint_path}")
                    print("Skipping visualizations")
                    model = None
            
            if model is not None:
                # Ensure visualization directory is absolute path
                viz_dir = os.path.abspath(cfg.viz_dir)
                os.makedirs(viz_dir, exist_ok=True)
                print(f"\n{'='*70}")
                print(f"Creating visualizations in: {viz_dir}")
                print(f"{'='*70}")
                
                create_all_visualizations(
                    cfg,
                    history or {},
                    test_metrics,
                    test_loader,
                    model,
                    device,
                    viz_dir=viz_dir
                )
                
                print(f"\n✓ Visualizations saved to: {viz_dir}")
                print(f"  Total files: {len([f for f in os.listdir(viz_dir) if f.endswith('.png')])}")
        else:
            print("\n⚠️  No test metrics available for visualization")
    else:
        print("\n⚠️  Visualizations disabled (--no_viz or save_visualizations=False)")
    
    print("\n" + "="*70)
    print("DONE")
    print("="*70)


if __name__ == "__main__":
    main()
