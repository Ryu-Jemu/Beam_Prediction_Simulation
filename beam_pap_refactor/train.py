import math, time, torch
from torch.utils.data import DataLoader
from setting import Config
from dataset import BeamSeqDataset
from model import PaPGPT2Regressor
from utils import (
    set_seed, pick_device,
    weighted_circular_mse, normalized_angle_mse,
    mae_deg, hit_at_deg, compose_residual_with_baseline
)
from visualize import make_summary_figure, make_error_histograms

def compute_stats_text(a_past: torch.Tensor) -> str:
    mu = float(torch.mean(a_past).cpu())
    std = float(torch.std(a_past).cpu())
    return f"past AoD mean={mu:.3f}, std={std:.3f}"

def _q_from_angle(theta: float, Q: int) -> int:
    u = (theta + math.pi) / (2*math.pi)
    q = int(round(u * (Q-1)))
    return max(0, min(Q-1, q))

def _roll_predict_one_sample(cfg: Config, model: PaPGPT2Regressor, sample, device: torch.device):
    # unpack
    X, Y, a_past, a_fut, q_all, gain_all, traj, a_base = sample
    X = X.clone().to(device)              # [C,U]
    a_past = a_past.to(device)            # [U]
    a_base = a_base.to(device)            # [H]

    BQ = cfg.M
    C = cfg.feature_dim + (2 if cfg.include_aod_in_features else 0)
    assert X.size(0) == C, f"feature channels mismatch: X={X.size(0)}, cfg={C}"

    # indices
    ch = {"q":0, "x":1, "y":2, "hs":3, "hc":4, "v":5, "r":6, "w":7}
    next_col = torch.zeros(C, device=device)

    # last state from window (convert back to metric)
    dt = cfg.delta_t_s
    L  = cfg.area_size_m
    x  = float(X[ch["x"], -1].detach().cpu()*L)
    y  = float(X[ch["y"], -1].detach().cpu()*L)
    psi= float(torch.atan2(X[ch["hs"], -1], X[ch["hc"], -1]).detach().cpu())
    v  = float(X[ch["v"], -1].detach().cpu()*(cfg.speed_max_mps-cfg.speed_min_mps)+cfg.speed_min_mps)
    if cfg.U >= 2:
        psi_prev = float(torch.atan2(X[ch["hs"], -2], X[ch["hc"], -2]).detach().cpu())
        w = (((psi - psi_prev) + math.pi)%(2*math.pi) - math.pi) / max(1e-8, dt)
    else:
        w = 0.0

    preds = []
    for k in range(cfg.H):
        with torch.no_grad():
            res_full = model(X.unsqueeze(0), stats_text=compute_stats_text(a_past.unsqueeze(0)))  # [1,H,2]
            res1 = res_full[:, 0:1, :]    # [1,1,2]
        pred_sc_dev = compose_residual_with_baseline(res1, a_base[k:k+1].view(1,1))  # [1,1,2]
        pred_sc = pred_sc_dev.squeeze(0).squeeze(0).detach().cpu()                  # [2]
        preds.append(pred_sc)

        # CTRV + reflections
        if abs(w) < 1e-6:
            xn = x + v*dt*math.cos(psi); yn = y + v*dt*math.sin(psi)
        else:
            R = v/w; psi_new = psi + w*dt
            xn = x + R*(math.sin(psi_new) - math.sin(psi))
            yn = y - R*(math.cos(psi_new) - math.cos(psi))
            psi = ((psi_new + math.pi)%(2*math.pi))-math.pi
        if xn < 0 or xn > L:
            psi = math.pi - psi
            xn = 2*L - xn if xn > L else -xn
            xn = max(0.0, min(L, xn))
        if yn < 0 or yn > L:
            psi = -psi
            yn = 2*L - yn if yn > L else -yn
            yn = max(0.0, min(L, yn))
        x, y = xn, yn

        # build new feature column from predicted AoD + propagated state
        theta = math.atan2(float(pred_sc[0]), float(pred_sc[1]))
        q_pred = _q_from_angle(theta, BQ)
        q_norm = (q_pred/(BQ-1))*2 - 1.0

        xN, yN = x/L, y/L
        vN = (v - cfg.speed_min_mps)/(cfg.speed_max_mps-cfg.speed_min_mps+1e-8)
        rN = (math.hypot(x - L/2, y - L/2)) / (L*math.sqrt(2)+1e-6)
        wN = max(-1.0, min(1.0, w/math.pi))

        next_col.zero_()
        next_col[ch["q"]]  = q_norm
        next_col[ch["x"]]  = xN
        next_col[ch["y"]]  = yN
        next_col[ch["hs"]] = math.sin(psi)
        next_col[ch["hc"]] = math.cos(psi)
        next_col[ch["v"]]  = vN
        next_col[ch["r"]]  = rN
        next_col[ch["w"]]  = wN
        if cfg.include_aod_in_features:
            next_col[-2] = math.sin(theta)
            next_col[-1] = math.cos(theta)

        # roll by 1
        X = torch.cat([X[:, 1:], next_col.view(-1,1)], dim=1)
        a_past = torch.cat([a_past[1:], torch.tensor([theta], device=device)], dim=0)

    return torch.stack(preds, dim=0)  # [H,2] CPU

def config_batch(cfg: Config) -> int:
    return cfg.batch_size

def run():
    cfg = Config()
    raw_dev = pick_device(cfg.device)
    device = raw_dev if isinstance(raw_dev, torch.device) else torch.device("cpu")
    set_seed(cfg.seed)
    print(f"Device: {device.type}")

    tr = BeamSeqDataset(cfg.n_train, cfg, split="train", seed=cfg.seed)
    va = BeamSeqDataset(cfg.n_val,   cfg, split="val",   seed=cfg.seed)
    te = BeamSeqDataset(cfg.n_test,  cfg, split="test",  seed=cfg.seed)

    tr_dl = DataLoader(tr, batch_size=config_batch(cfg), shuffle=True, drop_last=True)
    va_dl = DataLoader(va, batch_size=cfg.batch_size, shuffle=False)
    te_dl = DataLoader(te, batch_size=cfg.batch_size, shuffle=False)

    model = PaPGPT2Regressor(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    w = torch.pow(torch.tensor(cfg.horizon_gamma, device=device), torch.arange(cfg.H, device=device).float()).view(1,cfg.H,1)

    best = float("inf")
    for ep in range(1, cfg.n_epochs+1):
        t0 = time.time()
        model.train()
        tr_loss = tr_nmse = 0.0
        for X, Y, a_past, a_fut, *_ , a_base in tr_dl:
            X = X.to(device); Y = Y.to(device); a_past = a_past.to(device); a_base = a_base.to(device)
            stats = compute_stats_text(a_past)
            res = model(X, stats_text=stats)
            pred = compose_residual_with_baseline(res, a_base)
            loss = weighted_circular_mse(pred, Y, w)
            opt.zero_grad(set_to_none=True); loss.backward()
            if cfg.grad_clip: torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            tr_loss += loss.item()*X.size(0)
            tr_nmse += normalized_angle_mse(pred, Y).item()*X.size(0)
        Ntr = len(tr_dl.dataset); tr_loss/=Ntr; tr_nmse/=Ntr

        model.eval()
        va_nmse = va_mae = va_hit = 0.0
        with torch.no_grad():
            for X, Y, a_past, a_fut, *_ , a_base in va_dl:
                X = X.to(device); Y = Y.to(device); a_past = a_past.to(device); a_base = a_base.to(device)
                stats = compute_stats_text(a_past)
                res = model(X, stats_text=stats); pred = compose_residual_with_baseline(res, a_base)
                va_nmse += normalized_angle_mse(pred, Y).item()*X.size(0)
                va_mae  += mae_deg(pred, Y).item()*X.size(0)
                va_hit  += hit_at_deg(pred, Y, thr_deg=10.0).item()*X.size(0)
        Nva = len(va_dl.dataset); va_nmse/=Nva; va_mae/=Nva; va_hit/=Nva
        dur = time.time()-t0
        print(f"Epoch {ep:02d} | tr_loss={tr_loss:.4f} tr_nMSE={tr_nmse:.4f} | val_nMSE={va_nmse:.4f} MAE={va_mae:.2f}deg hit@10={va_hit:.3f} | time={dur:.1f}s")
        if va_nmse < best:
            best = va_nmse
            torch.save(model.state_dict(), "best.pt")

    # test
    model.load_state_dict(torch.load("best.pt", map_location=device))
    model.eval()
    metrics = {"nMSE":0.0, "MAE_deg":0.0, "hit10":0.0, "best_pred_deg":0.0}
    all_err_deg = []
    with torch.no_grad():
        for X, Y, a_past, a_fut, *_ , a_base in te_dl:
            X = X.to(device); Y = Y.to(device); a_past = a_past.to(device); a_base = a_base.to(device)
            stats = compute_stats_text(a_past)
            res = model(X, stats_text=stats); pred = compose_residual_with_baseline(res, a_base)
            metrics["nMSE"]   += normalized_angle_mse(pred, Y).item()*X.size(0)
            metrics["MAE_deg"]+= mae_deg(pred, Y).item()*X.size(0)
            metrics["hit10"]  += hit_at_deg(pred, Y, 10.0).item()*X.size(0)
            ps, pc = pred[...,0], pred[...,1]; ts, tc = Y[...,0], Y[...,1]
            pa = torch.atan2(ps, pc); ta = torch.atan2(ts, tc)
            d = torch.atan2(torch.sin(pa-ta), torch.cos(pa-ta)).abs() * 180.0 / math.pi
            metrics["best_pred_deg"] += d.min(dim=1).values.mean().item() * X.size(0)
            all_err_deg.append(d.cpu())
    N = len(te_dl.dataset)
    for k in metrics: metrics[k] /= N
    print("Test metrics:", metrics)

    # Viz on first test sample (AR rolling)
    sample = te[0]
    pred_roll = _roll_predict_one_sample(cfg, model, sample, device)
    Xs, Ys, a_past_s, a_fut_s, q_all_s, gain_all_s, traj_s, _a_base = sample
    make_summary_figure(cfg,
                        (Xs, Ys, a_past_s, a_fut_s, q_all_s, gain_all_s, traj_s),
                        pred_roll,
                        savepath="viz_sample.png")
    print("Saved visualization to viz_sample.png")

    err_mat = torch.cat(all_err_deg, dim=0).numpy()
    make_error_histograms(err_mat, cfg.H, savepath="viz_hist.png", bins=36)
    print("Saved histograms to viz_hist.png")