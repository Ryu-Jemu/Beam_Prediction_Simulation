import numpy as np, math, matplotlib.pyplot as plt
from setting import Config

def _to_angle(sc):
    return np.arctan2(sc[...,0], sc[...,1])

def _deg(a): return a*180.0/math.pi

def make_summary_figure(cfg: Config, sample_tuple, pred_roll, savepath="viz_sample.png"):
    """
    sample_tuple: (X, Y, a_past, a_fut, q_all, gain_all, traj)
    pred_roll: [H,2] CPU numpy/torch
    """
    X, Y, a_past, a_fut, q_all, gain_all, traj = sample_tuple
    if hasattr(pred_roll, "detach"): pred_roll = pred_roll.detach().cpu().numpy()
    if hasattr(Y, "detach"): Y = Y.detach().cpu().numpy()
    if hasattr(a_fut, "detach"): a_fut = a_fut.detach().cpu().numpy()
    if hasattr(traj, "detach"): traj = traj.detach().cpu().numpy()

    pa = _to_angle(pred_roll)
    ta = _to_angle(Y)
    err = np.abs(np.arctan2(np.sin(pa-ta), np.cos(pa-ta)))
    mse = np.mean((err**2)/(math.pi**2))
    mae = _deg(np.mean(err))
    hit10 = float(np.mean(_deg(err) <= 10.0))

    L = cfg.area_size_m
    bs = np.array([L/2, L/2])

    fig, axes = plt.subplots(1, 2, figsize=(12,5))
    ax = axes[0]
    ax.set_title("Trajectory and AoD")
    ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_aspect('equal', 'box')
    ax.plot(traj[:,0], traj[:,1], '-', lw=1.0, label="path")
    ax.scatter(bs[0], bs[1], marker='^', s=80, label="BS")

    U = cfg.U; H = cfg.H
    ax.scatter(traj[U-1,0], traj[U-1,1], c='k', s=30, label="t=U-1")
    ax.plot(traj[U:U+H,0], traj[U:U+H,1], 'o-', lw=1.0, label="future")
    for k in range(H):
        theta_p = pa[k]; theta_t = ta[k]
        for theta, sty in [(theta_p,'--'),(theta_t,':')]:
            x2 = bs[0] + (L*0.4)*math.cos(theta)
            y2 = bs[1] + (L*0.4)*math.sin(theta)
            ax.plot([bs[0], x2],[bs[1], y2], sty, alpha=0.6)
    ax.legend(loc="upper right")

    ax2 = axes[1]
    ax2.set_title(f"Angle error per step (deg)\nMSE={mse:.4f}, MAE={mae:.2f}°, hit@10={hit10:.3f}")
    ax2.plot(range(1,H+1), _deg(err), 'o-')
    ax2.set_xlabel("horizon step"); ax2.set_ylabel("abs error (deg)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(savepath, dpi=140)
    plt.close(fig)

def make_error_histograms(err_mat_deg, H:int, savepath="viz_hist.png", bins=36):
    if hasattr(err_mat_deg, "detach"):
        err_mat_deg = err_mat_deg.detach().cpu().numpy()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, H, figsize=(3*H,3))
    if H==1: axes=[axes]
    for k in range(H):
        ax = axes[k]
        ax.hist(err_mat_deg[:,k], bins=bins, density=True, alpha=0.8)
        ax.set_title(f"Step {k+1}")
        ax.set_xlabel("abs error (deg)"); ax.set_ylabel("pdf")
    plt.tight_layout()
    plt.savefig(savepath, dpi=140)
    plt.close(fig)