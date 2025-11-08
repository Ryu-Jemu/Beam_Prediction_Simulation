import math, random, numpy as np, torch

def set_seed(seed:int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def pick_device(pref="auto"):
    if isinstance(pref, torch.device):
        return pref
    if isinstance(pref, str):
        s = pref.lower()
        if s != "auto":
            return torch.device(s)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")

def _angle_diff(a, b):
    return torch.atan2(torch.sin(a-b), torch.cos(a-b))

def normalized_angle_mse(pred_sc: torch.Tensor, tgt_sc: torch.Tensor) -> torch.Tensor:
    pa = torch.atan2(pred_sc[...,0], pred_sc[...,1])
    ta = torch.atan2(tgt_sc[...,0],  tgt_sc[...,1])
    d = _angle_diff(pa, ta)
    return torch.mean((d**2) / (math.pi**2))

def weighted_circular_mse(pred_sc: torch.Tensor, tgt_sc: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    pa = torch.atan2(pred_sc[...,0], pred_sc[...,1])
    ta = torch.atan2(tgt_sc[...,0],  tgt_sc[...,1])
    d2 = _angle_diff(pa, ta)**2
    return torch.mean(d2.unsqueeze(-1) * w) / (math.pi**2)

def mae_deg(pred_sc: torch.Tensor, tgt_sc: torch.Tensor) -> torch.Tensor:
    pa = torch.atan2(pred_sc[...,0], pred_sc[...,1])
    ta = torch.atan2(tgt_sc[...,0],  tgt_sc[...,1])
    return torch.mean(_angle_diff(pa, ta).abs()) * 180.0 / math.pi

def hit_at_deg(pred_sc: torch.Tensor, tgt_sc: torch.Tensor, thr_deg: float=10.0) -> torch.Tensor:
    pa = torch.atan2(pred_sc[...,0], pred_sc[...,1])
    ta = torch.atan2(tgt_sc[...,0],  tgt_sc[...,1])
    ok = (_angle_diff(pa, ta).abs() * 180.0 / math.pi) <= thr_deg
    return ok.float().mean()

def compose_residual_with_baseline(res_sc: torch.Tensor, a_base: torch.Tensor) -> torch.Tensor:
    """
    res_sc: [B,H,2] (sin,cos of residual)
    a_base: [B,H]   (baseline AoD, rad)
    returns [B,H,2]
    """
    if a_base.device != res_sc.device:
        a_base = a_base.to(res_sc.device)
    sr, cr = res_sc[...,0], res_sc[...,1]
    sb, cb = torch.sin(a_base), torch.cos(a_base)
    s = sb*cr + cb*sr
    c = cb*cr - sb*sr
    norm = torch.clamp(torch.sqrt(s*s + c*c), min=1e-8)
    return torch.stack([s/norm, c/norm], dim=-1)