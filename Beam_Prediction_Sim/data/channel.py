"""
Channel simulation module
Includes: ULA response, DFT codebook, SV channel model, optimal beam selection
"""
import math
import numpy as np
from typing import Tuple, Optional


def ula_response(
    M: int,
    d_over_lambda: float,
    aod_rad: float
) -> np.ndarray:
    """ULA (Uniform Linear Array) steering vector
    
    a(φ) = [1, e^{j·2π·(d/λ)·sin(φ)}, ..., e^{j·2π·(d/λ)·(M-1)·sin(φ)}]^T
    
    Args:
        M: number of antenna elements
        d_over_lambda: element spacing normalized by wavelength
        aod_rad: angle of departure in radians
    
    Returns:
        a: [M] complex steering vector
    """
    k = 2 * math.pi * d_over_lambda * math.sin(aod_rad)
    m = np.arange(M, dtype=np.float64)
    a = np.exp(1j * k * m).astype(np.complex128)
    return a


def dft_codebook(M: int) -> np.ndarray:
    """DFT codebook for beamforming
    
    Each column is a beam steering vector for a uniformly spaced
    spatial frequency in the range [-1, 1].
    
    Args:
        M: number of antennas and number of beams
    
    Returns:
        W: [M, M] codebook matrix (columns are beams)
    """
    m = np.arange(M, dtype=np.float64)
    q = np.arange(M, dtype=np.float64)
    
    # Spatial frequency: u ∈ [-1, 1]
    u = (2 * q - (M - 1)) / M
    
    # DFT matrix: W[m, q] = exp(j·π·m·u[q])
    W = np.exp(1j * math.pi * np.outer(m, u))
    
    # Normalize columns
    W = W / np.sqrt(M)
    
    return W.astype(np.complex128)


def simulate_sv_channel(
    M: int,
    d_over_lambda: float,
    aod_rad: float,
    L_paths: int = 3,
    aod_spread_deg: float = 5.0,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """Simulate Saleh-Valenzuela (SV) channel model
    
    h = Σ_l (α_l / √ρ_l) · a*(φ_l)
    
    where φ_l are AoDs around the center AoD with small spread,
    and α_l are complex path gains.
    
    Args:
        M: number of antennas
        d_over_lambda: element spacing / wavelength
        aod_rad: center AoD in radians
        L_paths: number of multipath components
        aod_spread_deg: AoD spread around center (std dev in degrees)
        rng: random number generator
    
    Returns:
        h: [M] complex channel vector
    """
    if rng is None:
        rng = np.random.default_rng()
    
    h = np.zeros(M, dtype=np.complex128)
    aod_spread_rad = math.radians(aod_spread_deg)
    
    for l in range(L_paths):
        # AoD for this path (small spread around center)
        aod_l = aod_rad + rng.normal(0.0, aod_spread_rad)
        
        # Complex gain (Rayleigh fading)
        alpha_real = rng.normal(0, 1)
        alpha_imag = rng.normal(0, 1)
        alpha_l = (alpha_real + 1j * alpha_imag) / math.sqrt(2 * L_paths)
        
        # Path loss (simplified: all paths have similar loss)
        # In more realistic models, this would depend on distance
        rho_l = 1.0
        
        # Steering vector (conjugate for receive array)
        a_l = ula_response(M, d_over_lambda, aod_l)
        
        # Add path contribution
        h += (alpha_l / math.sqrt(rho_l)) * np.conj(a_l)
    
    return h


def optimal_beam_index(
    W: np.ndarray,
    h: np.ndarray
) -> Tuple[int, float]:
    """Find optimal beam from codebook
    
    The optimal beam maximizes |h^H · f_q|^2
    
    Args:
        W: [M, Q] codebook with Q beams (columns)
        h: [M] channel vector
    
    Returns:
        q_opt: index of optimal beam
        gain_opt: beamforming gain (power)
    """
    # Compute beamforming gain for all beams
    # |h^H · f_q|^2 = |conj(h)^T · f_q|^2
    gains = np.abs(np.conj(h) @ W) ** 2  # [Q]
    
    # Find best beam
    q_opt = int(np.argmax(gains))
    gain_opt = float(gains[q_opt])
    
    return q_opt, gain_opt


def aod_from_positions(
    bs_pos: np.ndarray,
    user_pos: np.ndarray
) -> float:
    """Compute AoD from BS to user
    
    Args:
        bs_pos: [2] BS position (x, y)
        user_pos: [2] user position (x, y)
    
    Returns:
        aod: angle of departure in radians [-π, π]
    """
    dx = user_pos[0] - bs_pos[0]
    dy = user_pos[1] - bs_pos[1]
    aod = math.atan2(dy, dx)
    return aod


def beam_index_to_aod(
    q: int,
    Q: int,
    aod_range: Tuple[float, float] = (-math.pi, math.pi)
) -> float:
    """Convert beam index to approximate AoD
    
    Args:
        q: beam index [0, Q-1]
        Q: total number of beams
        aod_range: (min_aod, max_aod) in radians
    
    Returns:
        aod: approximate AoD in radians
    """
    min_aod, max_aod = aod_range
    normalized = q / (Q - 1)  # [0, 1]
    aod = min_aod + normalized * (max_aod - min_aod)
    return aod


def aod_to_beam_index(
    aod: float,
    Q: int,
    aod_range: Tuple[float, float] = (-math.pi, math.pi)
) -> int:
    """Convert AoD to beam index
    
    Args:
        aod: angle of departure in radians
        Q: total number of beams
        aod_range: (min_aod, max_aod) in radians
    
    Returns:
        q: beam index [0, Q-1]
    """
    min_aod, max_aod = aod_range
    
    # Normalize to [0, 1]
    normalized = (aod - min_aod) / (max_aod - min_aod)
    normalized = np.clip(normalized, 0, 1)
    
    # Convert to index
    q = int(round(normalized * (Q - 1)))
    q = np.clip(q, 0, Q - 1)
    
    return q


def compute_beamforming_gain(
    h: np.ndarray,
    f: np.ndarray
) -> float:
    """Compute beamforming gain
    
    gain = |h^H · f|^2
    
    Args:
        h: [M] channel vector
        f: [M] beamforming vector
    
    Returns:
        gain: beamforming power gain
    """
    inner_product = np.vdot(h, f)  # h^H · f
    gain = float(np.abs(inner_product) ** 2)
    return gain


def normalized_beamforming_gain(
    h: np.ndarray,
    f: np.ndarray,
    f_opt: Optional[np.ndarray] = None
) -> float:
    """Compute normalized beamforming gain
    
    normalized_gain = |h^H · f|^2 / |h^H · f_opt|^2
    
    If f_opt is not provided, use optimal beamformer f_opt = h / ||h||
    
    Args:
        h: [M] channel vector
        f: [M] beamforming vector
        f_opt: [M] optimal beamforming vector (optional)
    
    Returns:
        normalized_gain: gain normalized by optimal gain
    """
    gain = compute_beamforming_gain(h, f)
    
    if f_opt is None:
        # Optimal beamformer: matched filter
        h_norm = h / np.linalg.norm(h)
        gain_opt = compute_beamforming_gain(h, h_norm)
    else:
        gain_opt = compute_beamforming_gain(h, f_opt)
    
    return gain / (gain_opt + 1e-10)


def simulate_channel_sequence(
    M: int,
    d_over_lambda: float,
    aods: np.ndarray,
    L_paths: int = 3,
    aod_spread_deg: float = 5.0,
    seed: int = 0
) -> np.ndarray:
    """Simulate a sequence of channel vectors
    
    Args:
        M: number of antennas
        d_over_lambda: element spacing / wavelength
        aods: [T] array of AoDs over time
        L_paths: number of multipath components
        aod_spread_deg: AoD spread
        seed: random seed
    
    Returns:
        H: [T, M] channel matrix (each row is a channel vector)
    """
    T = len(aods)
    H = np.zeros((T, M), dtype=np.complex128)
    rng = np.random.default_rng(seed)
    
    for t in range(T):
        H[t] = simulate_sv_channel(
            M, d_over_lambda, aods[t],
            L_paths, aod_spread_deg, rng
        )
    
    return H


def compute_optimal_beams_sequence(
    W: np.ndarray,
    H: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute optimal beams for a sequence of channels
    
    Args:
        W: [M, Q] codebook
        H: [T, M] channel matrix
    
    Returns:
        beam_indices: [T] optimal beam indices
        gains: [T] beamforming gains
    """
    T = H.shape[0]
    beam_indices = np.zeros(T, dtype=np.int64)
    gains = np.zeros(T, dtype=np.float32)
    
    for t in range(T):
        q_opt, gain_opt = optimal_beam_index(W, H[t])
        beam_indices[t] = q_opt
        gains[t] = gain_opt
    
    return beam_indices, gains
