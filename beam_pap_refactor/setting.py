from dataclasses import dataclass

@dataclass
class Config:
    # ----- device / seed -----
    seed: int = 1337
    device: str = "auto"           # "auto" | "cuda" | "mps" | "cpu"

    # ----- timeline -----
    U: int = 20                    # past length
    H: int = 5                     # horizon length
    delta_t_s: float = 2.0         # seconds per slot (no auto scaling)

    # ----- area / mobility -----
    area_size_m: float = 200.0     # square area side
    speed_mode: str = "fixed"      # "fixed" | "markov"
    speed_min_mps: float = 5.0
    speed_max_mps: float = 20.0
    heading_turn_deg: float = 5.0  # small random turn per step

    # ----- channel / array -----
    M: int = 64                    # ULA elements = beams in DFT codebook
    d_over_lam: float = 0.5        # inter-element spacing / wavelength

    # ----- features -----
    feature_dim: int = 8           # q,x,y,hs,hc,v,r,w
    include_aod_in_features: bool = True  # +[sin aod, cos aod]

    # ----- PaP / model -----
    use_gpt2: bool = False       # keep False unless transformers ready
    use_ctrv_baseline: bool = True
    horizon_gamma: float = 0.9     # loss weight decay across horizon

    # patching / encoder
    patch_len: int = 5
    patch_stride: int = 2
    d_model: int = 128
    gpt2_dim: int = 128
    n_heads: int = 4
    n_layers: int = 2
    pap_vocab_small: int = 8

    # ----- train -----
    n_train: int = 4096
    n_val: int   = 512
    n_test: int  = 512
    batch_size: int = 64
    lr: float = 2e-4
    weight_decay: float = 0.01
    n_epochs: int = 10
    grad_clip: float = 1.0