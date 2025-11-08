
import math, cmath, numpy as np

def ula_response(M, d_over_lam, phi):
    """
    ULA steering vector (receive) with element spacing d and AoD phi (radians).
    a(phi) = [1, e^{j2π d/λ sin φ}, ..., e^{j2π d/λ (M-1) sin φ}]^T
    """
    k = 2*math.pi * d_over_lam * math.sin(phi)
    return np.exp(1j * k * np.arange(M))

def dft_codebook(M):
    """
    DFT codebook beams as steering vectors with spatial frequency grid.
    """
    # spatial freq u in [-1,1], M points
    q = np.arange(M)
    u = (2*q - (M-1)) / M  # approx evenly spaced in [-1,1)
    W = np.exp(1j * math.pi * np.outer(np.arange(M), u))
    # column-wise normalize
    W = W / np.sqrt(M)
    return W  # shape [M, M]

def simulate_sv_channel(M, L_paths, d_over_lam, aod_rad, seed=0):
    """
    Simplified single-cluster SV centered at 'aod_rad' with L lateral rays.
    """
    rng = np.random.default_rng(seed)
    h = np.zeros(M, dtype=np.complex128)
    for l in range(L_paths):
        # small spread around center
        phi_l = aod_rad + rng.normal(0.0, 5.0 * math.pi/180.0)  # 5 deg std
        gain = (rng.normal(0, 1) + 1j*rng.normal(0,1)) / np.sqrt(2*L_paths)
        a = ula_response(M, d_over_lam, phi_l)
        h += gain * a.conj()  # receive vector, conj for inner product with Tx beam
    return h  # [M]

def aod_from_pos(bs_pos, user_pos):
    # AoD from BS to user
    dx = user_pos[0] - bs_pos[0]
    dy = user_pos[1] - bs_pos[1]
    return math.atan2(dy, dx)

def optimal_beam_index(W, h):
    """
    W: [M, Q] codebook with columns f_q
    h: [M] channel vector
    returns: (best_q, best_power)
    """
    # projections for all beams: conj(h)^T W -> [Q]
    proj = np.abs(h.conj() @ W)**2
    q = int(proj.argmax())
    return q, float(proj[q])
