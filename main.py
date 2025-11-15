"""
Main entry point for beam prediction training and evaluation
"""
import torch
import argparse
import json
import os
import time
import numpy as np
from typing import Optional, Dict

from config import Config, get_default_config, get_lightweight_config
from data import create_dataloaders, BeamSeqDataset
from models import BeamPredictorLLM
from training import Trainer
from evaluation import Evaluator, create_all_visualizations
from utils import set_seed, mae_degrees, hit_at_threshold, sincos_to_angle

from beam_improvements import (
    apply_improvements,
    compute_enhanced_statistics_text,
    validate_improvements
)
from position_aware import (
    PhysicsBasedBeamPredictor,
    HybridBeamPredictor,
    evaluate_predictor
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

    # Enhanced options from run.py
    parser.add_argument("--use_physics", action="store_true", default=True,
                        help="Use physics-based beam prediction (default: True)")
    parser.add_argument("--physics_alpha", type=float, default=0.3,
                        help="Physics-based prediction weight (0-1, default: 0.3)")
    parser.add_argument("--run_complete_eval", action="store_true",
                        help="Run complete evaluation with physics components")

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

    # Enhanced options from run.py
    cfg.use_physics = args.use_physics
    cfg.physics_alpha = args.physics_alpha
    cfg.run_complete_eval = args.run_complete_eval

    return cfg


class EnhancedBeamPredictorLLM(torch.nn.Module):
    """물리 기반 로직이 통합된 향상된 LLM 빔 예측기"""

    def __init__(
        self,
        base_model: BeamPredictorLLM,
        cfg: Config,
        use_physics: bool = True
    ):
        super().__init__()
        self.base_model = base_model
        self.cfg = cfg
        self.use_physics = use_physics

        if use_physics:
            # 물리 기반 예측기 초기화
            self.physics_predictor = PhysicsBasedBeamPredictor(
                num_beams=cfg.M,
                area_size=cfg.area_size_m,
                bs_pos=np.array([cfg.area_size_m/2, cfg.area_size_m/2]),
                dt=cfg.delta_t_s
            )

            # 하이브리드 예측기
            self.hybrid = HybridBeamPredictor(
                self.physics_predictor,
                None,  # base_model은 별도로 처리
                alpha=cfg.physics_alpha
            )

    def forward(
        self,
        x: torch.Tensor,
        stats_text: Optional[str] = None
    ) -> torch.Tensor:
        """순전파

        Args:
            x: [B, C, U] 입력 특징
            stats_text: 통계 텍스트

        Returns:
            predictions: [B, H, 2] (sin, cos) 예측
        """
        B, C, U = x.shape
        device = x.device

        # 1. LLM 기반 예측
        llm_pred = self.base_model(x, stats_text)  # [B, H, 2]

        if not self.use_physics:
            return llm_pred

        # 2. 물리 기반 예측 추가
        physics_pred = torch.zeros_like(llm_pred)

        for b in range(B):
            # 특징에서 위치 정보 추출
            # x[b]: [C, U], features: [q_norm, x_norm, y_norm, ...]
            x_norm = x[b, 1, :].cpu().numpy()  # Normalized x positions
            y_norm = x[b, 2, :].cpu().numpy()  # Normalized y positions

            # 실제 위치로 변환
            positions = np.stack([
                x_norm * self.cfg.area_size_m,
                y_norm * self.cfg.area_size_m
            ], axis=-1)  # [U, 2]

            # 빔 인덱스 추출
            q_norm = x[b, 0, :].cpu().numpy()  # Normalized beam indices
            past_beams = ((q_norm + 1) / 2 * (self.cfg.M - 1)).astype(int)

            # 각 horizon step에 대해 예측
            for h in range(self.cfg.H):
                # 물리 기반 예측
                predicted_beam, info = self.physics_predictor.predict(
                    past_beams[-3:],  # 최근 3개 빔
                    positions[:-1],    # 과거 위치들
                    positions[-1],     # 현재 위치
                    horizon=h+1
                )

                # 빔을 AoD로 변환
                aod = self.physics_predictor._beam_index_to_aod(predicted_beam)
                physics_pred[b, h, 0] = np.sin(aod)
                physics_pred[b, h, 1] = np.cos(aod)

        physics_pred = physics_pred.to(device)

        # 3. 하이브리드 결합
        alpha = self.cfg.physics_alpha  # 물리 가중치
        combined_pred = alpha * physics_pred + (1 - alpha) * llm_pred

        # 정규화
        norm = torch.linalg.norm(combined_pred, dim=-1, keepdim=True).clamp(min=1e-8)
        combined_pred = combined_pred / norm

        return combined_pred


def compute_accurate_mae(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_beams: int = 64
) -> float:
    """정확한 MAE 계산 (실제 각도 차이)

    Args:
        predictions: [B, H, 2] 예측 (sin, cos)
        targets: [B, H, 2] 실제 (sin, cos)
        num_beams: 빔 개수

    Returns:
        mae: 평균 절대 오차 (degrees)
    """
    # (sin, cos)를 각도로 변환
    pred_angles = sincos_to_angle(predictions)  # [B, H]
    target_angles = sincos_to_angle(targets)    # [B, H]

    # 각도 차이 계산 (circular)
    diff = torch.abs(pred_angles - target_angles)

    # [-π, π] 범위로 wrapping
    diff = torch.minimum(diff, 2*np.pi - diff)

    # Degrees로 변환
    mae = torch.mean(diff) * 180.0 / np.pi

    return mae.item()


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
    base_model = BeamPredictorLLM(cfg)

    # Apply improvements first
    if cfg.use_improvements:
        print("\n🔧 Applying improvements...")
        cfg, base_model, (train_loader, val_loader, test_loader) = apply_improvements(
            cfg, base_model, (train_loader, val_loader, test_loader)
        )
        print("✅ Improvements applied!")

    # Create enhanced model with physics if enabled
    if cfg.use_physics:
        print("\n🔬 Creating enhanced model with physics...")
        model = EnhancedBeamPredictorLLM(base_model, cfg, use_physics=True)
        print(f"✅ Enhanced model created (physics weight: {cfg.physics_alpha})")
    else:
        model = base_model

    print(f"Model created successfully")
    print(f"Using GPT-2: {cfg.use_gpt2}")
    print(f"Using physics: {cfg.use_physics}")

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


def run_complete_evaluation(cfg: Config, device: torch.device):
    """완전한 평가 실행"""

    print("\n" + "="*70)
    print("COMPLETE BEAM PREDICTION EVALUATION")
    print("="*70)

    # 1. 데이터 로더 생성
    print("\n[1/5] Creating data loaders...")
    train_loader, val_loader, test_loader = create_dataloaders(cfg)

    # 2. 모델 생성
    print("\n[2/5] Initializing models...")
    base_model = BeamPredictorLLM(cfg)

    # 개선사항 적용
    if cfg.use_improvements:
        print("  Applying improvements...")
        cfg, base_model, _ = apply_improvements(cfg, base_model)

    # 향상된 모델 생성
    model = EnhancedBeamPredictorLLM(base_model, cfg, use_physics=True)
    model = model.to(device)

    print(f"  ✓ Model initialized with physics enhancement")

    # 3. 학습 (간단한 버전)
    print("\n[3/5] Training model...")
    train_model_simple(model, train_loader, val_loader, cfg, device)

    # 4. 평가
    print("\n[4/5] Evaluating model...")
    test_metrics = evaluate_model_enhanced(model, test_loader, cfg, device)

    # 5. 물리 기반 평가
    print("\n[5/5] Physics-based evaluation...")
    physics_metrics = evaluate_physics_component(model, test_loader, cfg, device)

    # 결과 출력
    print_final_results(test_metrics, physics_metrics)

    return test_metrics, physics_metrics


def train_model_simple(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    cfg: Config,
    device: torch.device,
    epochs: int = 5
):
    """간단한 학습 루프"""

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= 10:  # 빠른 테스트를 위해 제한
                break

            X, Y, *rest = batch[:2]
            X = X.to(device)
            Y = Y.to(device)

            # Forward
            pred = model(X)

            # Loss (circular MSE)
            loss = torch.mean((pred - Y) ** 2)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f"  Epoch {epoch}/{epochs}: Loss={train_loss/(batch_idx+1):.4f}")


def evaluate_model_enhanced(
    model: torch.nn.Module,
    test_loader,
    cfg: Config,
    device: torch.device
) -> Dict[str, float]:
    """모델 평가"""

    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= 20:  # 테스트 샘플 제한
                break

            X, Y, *rest = batch[:2]
            X = X.to(device)
            Y = Y.to(device)

            # Predict
            pred = model(X)

            all_predictions.append(pred.cpu())
            all_targets.append(Y.cpu())

    # Concatenate
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)

    # Compute metrics
    mae = compute_accurate_mae(predictions, targets, cfg.M)
    hit5 = hit_at_threshold(predictions, targets, 5.0).item()
    hit10 = hit_at_threshold(predictions, targets, 10.0).item()
    hit15 = hit_at_threshold(predictions, targets, 15.0).item()

    # Per-step MAE
    mae_per_step = []
    for h in range(cfg.H):
        mae_h = compute_accurate_mae(
            predictions[:, h:h+1, :],
            targets[:, h:h+1, :],
            cfg.M
        )
        mae_per_step.append(mae_h)

    return {
        "mae_degrees": mae,
        "hit@5": hit5,
        "hit@10": hit10,
        "hit@15": hit15,
        "mae_per_step": mae_per_step,
        "num_samples": predictions.shape[0]
    }


def evaluate_physics_component(
    model: EnhancedBeamPredictorLLM,
    test_loader,
    cfg: Config,
    device: torch.device
) -> Dict[str, float]:
    """물리 기반 컴포넌트 평가"""

    if not model.use_physics:
        return {}

    physics_predictor = model.physics_predictor

    # 테스트 데이터 수집
    test_samples = []

    for batch_idx, batch in enumerate(test_loader):
        if batch_idx >= 10:
            break

        X, Y, *rest = batch[:8]

        for b in range(X.shape[0]):
            # 특징에서 정보 추출
            x_norm = X[b, 1, :].numpy()
            y_norm = X[b, 2, :].numpy()
            q_norm = X[b, 0, :].numpy()

            positions = np.stack([
                x_norm * cfg.area_size_m,
                y_norm * cfg.area_size_m
            ], axis=-1)

            past_beams = ((q_norm + 1) / 2 * (cfg.M - 1)).astype(int)

            # 실제 미래 빔 계산
            target_angle = sincos_to_angle(Y[b, 0, :]).item()
            actual_beam = int((target_angle + np.pi) / (2*np.pi) * (cfg.M - 1))

            test_samples.append({
                "past_beams": past_beams[-3:],
                "past_positions": positions[:-1],
                "current_pos": positions[-1],
                "actual_beam": actual_beam
            })

    # 평가
    results = evaluate_predictor(
        physics_predictor,
        test_samples,
        verbose=False
    )

    return results


def print_final_results(
    test_metrics: Dict[str, float],
    physics_metrics: Dict[str, float]
):
    """최종 결과 출력"""

    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)

    print("\n📊 Overall Performance:")
    print(f"  MAE (degrees):    {test_metrics['mae_degrees']:.2f}°")
    print(f"  Hit@5:            {test_metrics['hit@5']:.4f}")
    print(f"  Hit@10:           {test_metrics['hit@10']:.4f}")
    print(f"  Hit@15:           {test_metrics['hit@15']:.4f}")
    print(f"  Samples:          {test_metrics['num_samples']}")

    print("\n📈 Per-Step MAE:")
    for i, mae in enumerate(test_metrics['mae_per_step'], 1):
        print(f"  Step {i}: {mae:.2f}°")

    if physics_metrics:
        print("\n🔬 Physics Component:")
        print(f"  MAE (degrees):    {physics_metrics.get('mae_degrees', 0):.2f}°")

        for method in ["adaptive", "bs_crossing", "reflection"]:
            key = f"mae_{method}"
            if key in physics_metrics:
                count = physics_metrics.get(f"count_{method}", 0)
                print(f"  {method:12s}:    {physics_metrics[key]:.2f}° (n={count})")

    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE")
    print("="*70)

    # MAE 설명
    print("\n📝 Note on MAE Calculation:")
    print("  - MAE represents the average angular difference in degrees")
    print("  - Each prediction step's error is calculated independently")
    print("  - The reported MAE is NOT cumulative across steps")
    print("  - Formula: MAE = mean(|predicted_angle - actual_angle|)")
    print("="*70)


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


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

    if cfg.use_physics:
        print("\n🔬 Physics-Based Prediction:")
        print(f"  - Physics Weight: {cfg.physics_alpha}")
        print(f"  - Complete Evaluation: {cfg.run_complete_eval}")
        print("-" * 70)

    # Execute based on mode
    model = None
    history = None
    test_metrics = None
    test_loader = None
    physics_metrics = None
    execution_time = 0

    # Execution time tracking
    start_time = time.time()

    # Check if complete evaluation is requested
    if cfg.run_complete_eval:
        print("\n" + "="*70)
        print("RUNNING COMPLETE ENHANCED EVALUATION")
        print("="*70)
        test_metrics, physics_metrics = run_complete_evaluation(cfg, device)
    else:
        # Standard execution
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

    execution_time = time.time() - start_time
    
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
    
    # Print execution time and save results
    print(f"\n⏱️  Total execution time: {execution_time:.2f} seconds")

    # Save results if we have metrics
    if test_metrics is not None:
        results = {
            "config": cfg.to_dict(),
            "test_metrics": test_metrics,
            "execution_time": execution_time
        }

        if physics_metrics is not None:
            results["physics_metrics"] = physics_metrics

        os.makedirs("./results", exist_ok=True)
        results_file = "./results/beam_prediction_results.json"
        # Convert numpy types to JSON-serializable format
        json_serializable_results = convert_numpy_types(results)
        with open(results_file, "w") as f:
            json.dump(json_serializable_results, f, indent=2)
        print(f"\n💾 Results saved to {results_file}")

    print("\n" + "="*70)
    print("DONE")
    print("="*70)


if __name__ == "__main__":
    main()
