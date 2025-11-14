"""
Configuration V2.0 - Mac-Safe First
Complete redesign prioritizing stability on Mac/MPS
"""
import torch
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Config:
    """Configuration V2.0 - Mac/MPS Safety First"""
    
    # Scenario
    U: int = 20
    H: int = 5
    area_size_m: float = 200.0
    delta_t_s: float = 2
    
    # Mobility
    speed_min_mps: float = 5.0
    speed_max_mps: float = 15.0
    speed_mode: str = "markov"
    heading_turn_deg: float = 10.0
    
    # Channel
    M: int = 64
    d_over_lam: float = 0.5
    L_paths: int = 3
    aod_spread_deg: float = 5.0
    
    # Features
    base_feature_dim: int = 8
    include_aod_in_features: bool = False
    normalize_features: bool = True
    use_revin: bool = False
    
    # Model
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    activation: str = "gelu"
    patch_len: int = 5
    patch_stride: int = 2
    
    # LLM
    use_gpt2: bool = True            # gpt 사용: True
    gpt2_model: str = "gpt2"
    gpt2_dim: int = 768
    gpt2_freeze: bool = True            
    text_prototype_vocab: int = 1500
    use_pap: bool = True
    
    # Baseline
    use_ctrv_baseline: bool = True
    ctrv_weight: float = 0.3
    
    # Training
    n_train: int = 20000
    n_val: int = 2000
    n_test: int = 4000
    batch_size: int = 16
    
    # Optimization
    n_epochs: int = 10
    lr: float = 1e-4
    weight_decay: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    optimizer: str = "adamw"
    
    # Scheduler
    use_scheduler: bool = True
    scheduler_type: str = "cosine"
    min_lr: float = 1e-6
    
    # Loss
    loss_type: str = "weighted_circular"
    use_step_weighting: bool = True
    horizon_gamma: float = 0.95
    
    # Regularization
    grad_clip: float = 1.0
    grad_clip_type: str = "norm"
    
    # Device - MAC SAFE
    seed: int = 1337
    device: str = "auto"
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: int = 2
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_best: bool = True
    early_stopping_patience: int = 10
    
    # Logging
    log_dir: str = "./logs"
    log_interval: int = 10
    eval_interval: int = 1
    save_visualizations: bool = True
    viz_samples: int = 5  # Number of samples to visualize
    viz_dir: str = "./visualizations"  # Visualization directory

    # Improvements
    use_improvements: bool = False
    use_position_constraints: bool = False
    use_enhanced_stats: bool = False
    
    @property
    def feature_dim(self) -> int:
        dim = self.base_feature_dim
        if self.include_aod_in_features:
            dim += 2
        return dim
    
    @property
    def num_patches(self) -> int:
        return (self.U - self.patch_len) // self.patch_stride + 1
    
    def get_device(self) -> torch.device:
        """Mac-safe device selection"""
        if self.device == "auto":
            if torch.cuda.is_available():
                print("✓ CUDA GPU")
                return torch.device("cuda")
            print("✓ CPU (Mac-safe, recommended)")
            print("  Try MPS: --device mps (experimental)")
            return torch.device("cpu")
        
        if self.device == "mps":
            if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                print("⚠️  MPS unavailable → CPU")
                return torch.device("cpu")
            
            print("⚠️  MPS (experimental)")
            print("   If errors: --device cpu")
            self.num_workers = 0
            self.pin_memory = False
            self.persistent_workers = False
            self.use_revin = False
            if self.batch_size > 16:
                print(f"   Reducing batch: {self.batch_size} → 16")
                self.batch_size = 16
            return torch.device("mps")
        
        if self.device == "cuda":
            if not torch.cuda.is_available():
                print("⚠️  CUDA unavailable → CPU")
                return torch.device("cpu")
            print("✓ CUDA GPU")
            return torch.device("cuda")
        
        print("✓ CPU")
        return torch.device("cpu")
    
    def validate(self):
        assert self.U > 0 and self.H > 0
        assert self.batch_size > 0
    
    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


def get_default_config() -> Config:
    return Config()


def get_lightweight_config() -> Config:
    cfg = Config()
    cfg.n_train, cfg.n_val, cfg.n_test = 512, 64, 64
    cfg.n_epochs = 5
    cfg.d_model, cfg.n_heads, cfg.n_layers = 64, 4, 2
    cfg.M, cfg.L_paths, cfg.H = 32, 2, 10
    cfg.batch_size = 8
    return cfg


def get_tiny_config() -> Config:
    cfg = Config()
    cfg.n_train, cfg.n_val, cfg.n_test = 128, 32, 32
    cfg.n_epochs = 3
    cfg.d_model, cfg.n_heads, cfg.n_layers = 32, 2, 1
    cfg.M, cfg.L_paths = 16, 1
    cfg.U, cfg.H = 10, 5
    cfg.batch_size = 4
    return cfg

def get_highperf_config() -> "Config":
    """
    High-performance preset for larger-scale runs.
    Matches the fields used elsewhere in this repo.
    """
    # 기본 프리셋을 가져와 일부만 상향 조정
    try:
        cfg = get_default_config()
    except NameError:
        # get_default_config가 없다면 Config 직접 생성
        try:
            from dataclasses import asdict
            _ = asdict  # noqa: F401
        except Exception:
            pass
        # 최소 필드 가정
        cfg = Config()  # 기존 파일의 Config @dataclass 가정

    # 데이터 크기
    cfg.n_train = getattr(cfg, "n_train", 4096) * 2    # 8192
    cfg.n_val   = getattr(cfg, "n_val",  512)  * 2     # 1024
    cfg.n_test  = getattr(cfg, "n_test", 512)  * 2     # 1024
    cfg.n_epochs = max(getattr(cfg, "n_epochs", 30), 50)

    # 모델 용량
    cfg.d_model  = max(getattr(cfg, "d_model", 128), 256)
    cfg.n_heads  = max(getattr(cfg, "n_heads", 4), 8)
    cfg.n_layers = max(getattr(cfg, "n_layers", 4), 6)

    # 채널/예측 지평
    cfg.M       = max(getattr(cfg, "M", 64), 128)        # 안테나 수
    cfg.L_paths = max(getattr(cfg, "L_paths", 3), 4)     # 다중경로 수
    cfg.H       = max(getattr(cfg, "H", 10), 20)         # 예측 horizon

    # 학습 하이퍼파라미터
    cfg.batch_size = max(getattr(cfg, "batch_size", 16), 32)
    cfg.lr         = min(getattr(cfg, "lr", 5e-4), 3e-4)

    # GPT-2 백본(오프라인이면 폴백되도록 상위 로직이 처리)
    cfg.gpt2_model = getattr(cfg, "gpt2_model", "gpt2-medium")
    cfg.gpt2_dim   = getattr(cfg, "gpt2_dim", 1024)

    return cfg