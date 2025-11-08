import math, inspect
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Any
from setting import Config
from mobility import MarkovMobility
from channel import dft_codebook, simulate_sv_channel, optimal_beam_index

# ==================== helpers ====================

def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi

def _call_with_subset(fn, *args, **kwargs):
    sig = inspect.signature(fn)
    allowed = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(*args, **allowed)

def _build_mobility(cfg: Config, rng: np.random.Generator):
    sig = inspect.signature(MarkovMobility)
    params = sig.parameters
    kwargs = {}
    for k in ("area_size", "L", "size", "side", "area", "box_len"):
        if k in params: kwargs[k] = cfg.area_size_m; break
    for k in ("delta_t", "dt", "delta_t_s", "time_step"):
        if k in params: kwargs[k] = cfg.delta_t_s; break
    for k in ("heading_turn_deg", "turn_deg", "turn_std_deg",
              "heading_std_deg", "heading_sigma_deg", "heading_step_deg"):
        if k in params: kwargs[k] = cfg.heading_turn_deg; break
    if "rng" in params: kwargs["rng"] = rng
    elif "seed" in params: kwargs["seed"] = int(rng.integers(2**31 - 1))
    try:
        return MarkovMobility(**kwargs)
    except TypeError:
        pass
    try:
        return MarkovMobility(cfg.area_size_m, cfg.delta_t_s, cfg.heading_turn_deg, rng)
    except TypeError as e:
        raise TypeError(f"MarkovMobility ctor mismatch. params={list(params.keys())}, kwargs={kwargs}. Original: {e}")

def _call_mobility(mm, T: int, speed_mode: str, v0: float, vmin: float, vmax: float):
    if speed_mode == "markov" and hasattr(mm, "simulate_markov"):
        try: return _call_with_subset(mm.simulate_markov, T, v0=v0, v_min=vmin, v_max=vmax)
        except TypeError: pass
    if speed_mode == "fixed" and hasattr(mm, "simulate_fixed"):
        try: return _call_with_subset(mm.simulate_fixed, T, v=v0, v0=v0)
        except TypeError: pass
    if not hasattr(mm, "simulate"):
        raise AttributeError("MarkovMobility에 simulate()/simulate_fixed()/simulate_markov() 중 하나가 필요합니다.")
    fn = mm.simulate
    try:
        return _call_with_subset(fn, T, mode=speed_mode, kind=speed_mode, v=v0, v0=v0, v_min=vmin, v_max=vmax)
    except TypeError:
        pass
    for args in ((T, v0), (T,)):
        try: return fn(*args)
        except TypeError: continue
    raise TypeError("MarkovMobility.simulate 호출 실패: 시그니처를 확인하세요.")

def _simulate_traj_fallback(T:int, cfg:Config, rng:np.random.Generator):
    L  = cfg.area_size_m
    dt = cfg.delta_t_s
    turn = math.radians(cfg.heading_turn_deg)
    x = rng.uniform(0, L); y = rng.uniform(0, L)
    psi = rng.uniform(-math.pi, math.pi)
    v = rng.uniform(cfg.speed_min_mps, cfg.speed_max_mps)
    xs = np.zeros(T, np.float32); ys = np.zeros(T, np.float32)
    hs = np.zeros(T, np.float32); vs = np.zeros(T, np.float32)
    for t in range(T):
        xs[t]=x; ys[t]=y; hs[t]=psi; vs[t]=v
        if cfg.speed_mode=="markov":
            r=rng.random()
            if r<0.25: v=max(cfg.speed_min_mps, v-1.0)
            elif r<0.75: v=v
            else: v=min(cfg.speed_max_mps, v+1.0)
        psi = _wrap(psi + rng.uniform(-turn,+turn))
        xn = x + v*dt*math.cos(psi); yn = y + v*dt*math.sin(psi)
        if xn<0 or xn>L: psi=math.pi-psi; xn = 2*L - xn if xn>L else -xn; xn = 0.0 if xn<0 else (L if xn>L else xn)
        if yn<0 or yn>L: psi=-psi; yn = 2*L - yn if yn>L else -yn; yn = 0.0 if yn<0 else (L if yn>L else yn)
        x,y = xn,yn
    return vs,xs,ys,hs

def _call_sv_channel(M:int, aod:float, rng, cfg:Config):
    try:
        sig = inspect.signature(simulate_sv_channel)
        kw = {}
        if "M" in sig.parameters: kw["M"] = M
        if "d_over_lam" in sig.parameters: kw["d_over_lam"] = getattr(cfg, "d_over_lam", 0.5)
        if "aod_rad" in sig.parameters: kw["aod_rad"] = float(aod)
        if "rng" in sig.parameters: kw["rng"] = rng
        h = simulate_sv_channel(**kw)
    except (TypeError, ValueError):
        try:
            h = simulate_sv_channel(M, getattr(cfg,"d_over_lam",0.5), float(aod))
        except Exception:
            m = np.arange(M, dtype=np.float32)
            d = getattr(cfg,"d_over_lam",0.5)
            k = 2*np.pi * d * np.sin(float(aod))
            h = np.exp(1j*k*m)
    if hasattr(h,"detach"): h = h.detach().cpu().numpy()
    h = np.asarray(h).reshape(-1).astype(np.complex64)
    if h.shape[0]!=M:
        h = h[:M] if h.shape[0]>M else np.pad(h,(0,M-h.shape[0]))
    return h

# ==================== dataset ====================

class BeamSeqDataset(Dataset):
    """
    Returns per item (tensors):
      X:[C,U], Y:[H,2], a_past:[U], a_fut:[H],
      q_all:[U+H], gain_all:[U+H], traj:[U+H,2], a_base:[H]
    """
    def __init__(self, n_samples:int, cfg:Config, split:str="train", seed:int=1337):
        self.cfg = cfg
        self.split = split
        self.rng = np.random.default_rng(seed + (0 if split=="train" else 777 if split=="val" else 888))
        self.T = cfg.U + cfg.H
        self.bs_pos = np.array([cfg.area_size_m/2.0, cfg.area_size_m/2.0], dtype=np.float32)
        self.W = dft_codebook(cfg.M)  # [M,M]
        self.data = [self._make_one(i) for i in range(n_samples)]

    def _simulate_traj(self):
        cfg = self.cfg
        try:
            mm = _build_mobility(cfg, self.rng)
            v0 = self.rng.uniform(cfg.speed_min_mps, cfg.speed_max_mps)
            vs, xs, ys, hs = _call_mobility(mm, self.T, cfg.speed_mode, v0, cfg.speed_min_mps, cfg.speed_max_mps)
            return dict(vs=np.asarray(vs,np.float32), xs=np.asarray(xs,np.float32),
                        ys=np.asarray(ys,np.float32), hs=np.asarray(hs,np.float32))
        except Exception:
            vs,xs,ys,hs = _simulate_traj_fallback(self.T, cfg, self.rng)
            return dict(vs=vs, xs=xs, ys=ys, hs=hs)

    def _aod_from_pos(self, xs, ys):
        bx, by = self.bs_pos
        return np.arctan2(ys - by, xs - bx).astype(np.float32)

    def _ctrv_rollout_with_reflection(self, x0, y0, psi0, v0, w, steps:int):
        L = self.cfg.area_size_m
        dt = float(self.cfg.delta_t_s)
        x,y,psi = float(x0),float(y0),float(psi0)
        a_base=[]
        for _ in range(steps):
            if abs(w) < 1e-6:
                xn = x + v0*dt*math.cos(psi)
                yn = y + v0*dt*math.sin(psi)
            else:
                R = v0 / w
                psi_new = psi + w*dt
                xn = x + R*(math.sin(psi_new)-math.sin(psi))
                yn = y - R*(math.cos(psi_new)-math.cos(psi))
                psi = _wrap(psi_new)
            if xn<0 or xn>L:
                psi = math.pi - psi
                xn = 2*L - xn if xn>L else -xn
                xn = 0.0 if xn<0 else (L if xn>L else xn)
            if yn<0 or yn>L:
                psi = -psi
                yn = 2*L - yn if yn>L else -yn
                yn = 0.0 if yn<0 else (L if yn>L else yn)
            x,y = xn,yn
            a_base.append(np.arctan2(y - self.bs_pos[1], x - self.bs_pos[0]))
        return np.array(a_base, dtype=np.float32)

    def _make_one(self, idx:int) -> Dict[str, Any]:
        cfg = self.cfg
        U,H = cfg.U, cfg.H
        sim = self._simulate_traj()
        xs,ys,hs,vs = sim["xs"], sim["ys"], sim["hs"], sim["vs"]

        aods = self._aod_from_pos(xs, ys)     # [T]
        gains = np.zeros(self.T, np.float32)
        qs    = np.zeros(self.T, np.int64)
        for t in range(self.T):
            h = _call_sv_channel(cfg.M, float(aods[t]), self.rng, cfg)
            try:
                q_star, g = optimal_beam_index(self.W, h)
            except Exception:
                Hw = self.W.conj().T @ h
                pow_ = np.abs(Hw)**2
                q_star = int(np.argmax(pow_)); g = float(pow_[q_star])
            qs[t], gains[t] = q_star, g

        aU = aods[:U]; aF = aods[U:U+H]
        Y = np.stack([np.sin(aF), np.cos(aF)], axis=-1).astype(np.float32)  # [H,2]

        qn  = (qs[:U].astype(np.float32) / float(self.W.shape[1]-1))*2 - 1.0
        xsN = xs[:U]/cfg.area_size_m; ysN = ys[:U]/cfg.area_size_m
        hs_sin = np.sin(hs[:U]); hs_cos = np.cos(hs[:U])
        vN = (vs[:U]-cfg.speed_min_mps)/(cfg.speed_max_mps-cfg.speed_min_mps+1e-8)
        bx,by = self.bs_pos
        r  = np.sqrt((xs[:U]-bx)**2 + (ys[:U]-by)**2)
        rN = r / (cfg.area_size_m*np.sqrt(2)+1e-6)
        w  = np.zeros(U, np.float32)
        if U>=2:
            for i in range(1,U):
                w[i] = _wrap(float(hs[i]-hs[i-1]))/max(1e-8, cfg.delta_t_s)
        wN = np.clip(w/math.pi, -1, 1)

        feats = [qn, xsN, ysN, hs_sin, hs_cos, vN, rN, wN]
        if cfg.include_aod_in_features:
            feats.extend([np.sin(aU), np.cos(aU)])
        X = np.stack(feats, axis=0).astype(np.float32)            # [C,U]

        if cfg.use_ctrv_baseline:
            omega = _wrap(float(hs[U-1]-hs[U-2]))/max(1e-8, cfg.delta_t_s) if U>=2 else 0.0
            a_base = self._ctrv_rollout_with_reflection(xs[U-1], ys[U-1], hs[U-1], vs[U-1], omega, H)
        else:
            a_base = np.zeros(H, np.float32)

        return dict(
            X=X, Y=Y, a_past=aU.astype(np.float32), a_fut=aF.astype(np.float32),
            q_all=qs.astype(np.int64), gain_all=gains.astype(np.float32),
            traj=np.stack([xs, ys], axis=-1).astype(np.float32),
            a_base=a_base.astype(np.float32)
        )

    def __len__(self): return len(self.data)

    def __getitem__(self, i:int):
        d = self.data[i]
        return (
            torch.from_numpy(d["X"]),
            torch.from_numpy(d["Y"]),
            torch.from_numpy(d["a_past"]),
            torch.from_numpy(d["a_fut"]),
            torch.from_numpy(d["q_all"]),
            torch.from_numpy(d["gain_all"]),
            torch.from_numpy(d["traj"]),
            torch.from_numpy(d["a_base"]),
        )