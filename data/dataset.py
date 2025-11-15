"""
Improved dataset module for beam prediction
Features: robust trajectory generation, better feature engineering, CTRV baseline
"""
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional

from config import Config
# NOTE: use absolute imports so that this module can be imported from a
# package (e.g. via `from data import BeamSeqDataset`) without requiring
# it to live inside a package itself.  Relative imports (e.g. `.mobility`)
# would fail when `dataset.py` resides at the project root.  Here we
# import the necessary classes and functions from the top-level modules.
"""
Dataset module for beam prediction

This module imports several helper classes and functions from the
`mobility` and `channel` modules.  When this file is imported as part
of a package (e.g. `data.dataset`), Python may not automatically find
those top-level modules on the search path, leading to
`ModuleNotFoundError`.  To make the imports robust, we first try the
standard imports and, if they fail, dynamically load the modules
relative to the current file using `importlib`.
"""


# Attempt to import mobility.  Try standard import first.  If that
# fails, try a relative import (in case this module is part of a
# package).  If that also fails, attempt to dynamically load the
# module from common fallback locations.
try:
    from mobility import MarkovMobility, compute_turning_rate, ctrv_rollout  # type: ignore
except ModuleNotFoundError:
    try:
        # Attempt relative import when dataset.py is inside a package
        from .mobility import MarkovMobility, compute_turning_rate, ctrv_rollout  # type: ignore
    except Exception:
        import importlib.util as _importlib_util
        import os as _os
        import sys as _sys

        _module_dir = _os.path.dirname(__file__)
        # List potential paths where mobility.py might reside
        _candidate_paths = [
            _os.path.join(_module_dir, 'mobility.py'),
            _os.path.join(_os.path.abspath(_os.path.join(_module_dir, _os.pardir)), 'mobility.py'),
        ]
        for _mobility_path in _candidate_paths:
            if _os.path.exists(_mobility_path):
                _spec = _importlib_util.spec_from_file_location('mobility', _mobility_path)
                if _spec and _spec.loader:
                    _mobility_module = _importlib_util.module_from_spec(_spec)
                    _sys.modules['mobility'] = _mobility_module
                    _spec.loader.exec_module(_mobility_module)  # type: ignore[attr-defined]
                    MarkovMobility = _mobility_module.MarkovMobility  # type: ignore[attr-defined]
                    compute_turning_rate = _mobility_module.compute_turning_rate  # type: ignore[attr-defined]
                    ctrv_rollout = _mobility_module.ctrv_rollout  # type: ignore[attr-defined]
                    break
        else:
            raise ImportError("Could not locate mobility module in known locations")

# Attempt to import channel.  Similar logic as for mobility.
try:
    from channel import dft_codebook, simulate_sv_channel, optimal_beam_index, aod_from_positions  # type: ignore
except ModuleNotFoundError:
    try:
        from .channel import dft_codebook, simulate_sv_channel, optimal_beam_index, aod_from_positions  # type: ignore
    except Exception:
        import importlib.util as _importlib_util
        import os as _os
        import sys as _sys

        _module_dir = _os.path.dirname(__file__)
        _candidate_paths = [
            _os.path.join(_module_dir, 'channel.py'),
            _os.path.join(_os.path.abspath(_os.path.join(_module_dir, _os.pardir)), 'channel.py'),
        ]
        for _channel_path in _candidate_paths:
            if _os.path.exists(_channel_path):
                _spec2 = _importlib_util.spec_from_file_location('channel', _channel_path)
                if _spec2 and _spec2.loader:
                    _channel_module = _importlib_util.module_from_spec(_spec2)
                    _sys.modules['channel'] = _channel_module
                    _spec2.loader.exec_module(_channel_module)  # type: ignore[attr-defined]
                    dft_codebook = _channel_module.dft_codebook  # type: ignore[attr-defined]
                    simulate_sv_channel = _channel_module.simulate_sv_channel  # type: ignore[attr-defined]
                    optimal_beam_index = _channel_module.optimal_beam_index  # type: ignore[attr-defined]
                    aod_from_positions = _channel_module.aod_from_positions  # type: ignore[attr-defined]
                    break
        else:
            raise ImportError("Could not locate channel module in known locations")


class BeamSeqDataset(torch.utils.data.Dataset):
    """Dataset for beam prediction sequences
    
    Returns per item:
        X: [C, U] input features over past U steps
        Y: [H, 2] target (sin, cos) for future H steps
        a_past: [U] past AoD angles
        a_fut: [H] future AoD angles
        q_all: [U+H] all beam indices
        gain_all: [U+H] all beamforming gains
        traj: [U+H, 2] trajectory (x, y)
        a_baseline: [H] CTRV baseline AoD predictions
    """
    
    def __init__(
        self,
        n_samples: int,
        cfg: Config,
        split: str = "train",
        seed: int = 1337
    ):
        """
        Args:
            n_samples: number of samples to generate
            cfg: configuration
            split: "train" | "val" | "test"
            seed: random seed (different per split)
        """
        self.cfg = cfg
        self.split = split
        self.n_samples = n_samples
        
        # Different seed per split
        if split == "train":
            self.seed = seed
        elif split == "val":
            self.seed = seed + 777
        elif split == "test":
            self.seed = seed + 888
        else:
            self.seed = seed
        
        self.rng = np.random.default_rng(self.seed)
        
        # Timeline
        self.U = cfg.U
        self.H = cfg.H
        self.T = cfg.U + cfg.H
        
        # BS position (center of area)
        self.bs_pos = np.array([
            cfg.area_size_m / 2.0,
            cfg.area_size_m / 2.0
        ], dtype=np.float32)
        
        # DFT codebook
        self.W = dft_codebook(cfg.M)  # [M, M]
        self.Q = self.W.shape[1]
        
        # Generate all samples
        print(f"Generating {n_samples} {split} samples...")
        self.data = [self._generate_sample(i) for i in range(n_samples)]
        print(f"Dataset {split} ready: {len(self.data)} samples")
    
    def _generate_sample(self, idx: int) -> Dict:
        """Generate a single sample"""
        cfg = self.cfg
        # 1. Generate mobility trajectory
        # Pass the sample index to ensure unique random seeds
        traj_data = self._generate_trajectory(idx)
        xs = traj_data["xs"]  # [T]
        ys = traj_data["ys"]  # [T]
        vs = traj_data["vs"]  # [T]
        hs = traj_data["hs"]  # [T]
        
        # 2. Compute AoDs from positions
        aods = self._compute_aods(xs, ys)  # [T]
        
        # 3. Simulate channels and find optimal beams
        qs, gains = self._compute_optimal_beams(aods)  # [T], [T]
        
        # 4. Compute CTRV baseline for future
        if cfg.use_ctrv_baseline:
            a_baseline = self._compute_ctrv_baseline(xs, ys, hs, vs)  # [H]
        else:
            a_baseline = np.zeros(self.H, dtype=np.float32)
        
        # 5. Extract past and future
        a_past = aods[:self.U].astype(np.float32)
        a_fut = aods[self.U:].astype(np.float32)
        
        # 6. Build input features X: [C, U]
        X = self._build_features(xs, ys, hs, vs, qs, aods)
        
        # 7. Build target Y: [H, 2] as (sin, cos)
        Y = np.stack([np.sin(a_fut), np.cos(a_fut)], axis=-1).astype(np.float32)
        
        # 8. Trajectory for visualization
        traj = np.stack([xs, ys], axis=-1).astype(np.float32)  # [T, 2]
        
        return {
            "X": X,
            "Y": Y,
            "a_past": a_past,
            "a_fut": a_fut,
            "q_all": qs.astype(np.int64),
            "gain_all": gains.astype(np.float32),
            "traj": traj,
            "a_baseline": a_baseline
        }
    
    def _generate_trajectory(self, sample_idx: int) -> Dict:
        """Generate mobility trajectory using MarkovMobility

        To ensure that each sample produces a distinct trajectory even when
        the global PRNG state is reset (for example by external calls to
        ``np.random.seed``), we derive the RNG seed from the dataset seed
        and the sample index.  This avoids accidental reuse of the same
        random seed across multiple samples, which would otherwise lead
        to identical trajectories.  Using ``np.random.default_rng`` with
        an explicit seed ensures reproducibility without relying on the
        process-wide RNG state.

        Args:
            sample_idx: the index of the sample being generated

        Returns:
            A dictionary containing position, velocity and heading arrays.
        """
        cfg = self.cfg

        # Derive a deterministic seed from the dataset's base seed and the
        # sample index.  A large prime is used to decorrelate neighbouring
        # indices and avoid simple patterns.  We take the modulus to ensure
        # the seed fits in the 32‑bit range required by default_rng.
        derived_seed = (self.seed + (sample_idx * 2654435761)) % (2**31 - 1)

        try:
            # Create mobility model
            mm = MarkovMobility(
                area_size_m=cfg.area_size_m,
                delta_t_s=cfg.delta_t_s,
                speed_min_mps=cfg.speed_min_mps,
                speed_max_mps=cfg.speed_max_mps,
                heading_turn_deg=cfg.heading_turn_deg,
                speed_mode=cfg.speed_mode,
                reflect_at_boundary=True
            )

            # Generate trajectory with derived seed
            xs, ys, vs, hs = mm.simulate(self.T, seed=derived_seed)
            return {"xs": xs, "ys": ys, "vs": vs, "hs": hs}

        except Exception as e:
            # If MarkovMobility fails (e.g. missing module), fall back
            print(f"Warning: mobility generation failed: {e}")
            return self._generate_trajectory_fallback(sample_idx)
    
    def _generate_trajectory_fallback(self, sample_idx: int) -> Dict:
        """Fallback trajectory generation

        When the standard MarkovMobility simulator cannot be used (for example
        due to missing dependencies), this fallback generates a simple
        random walk with reflection at the boundaries.  To maintain
        reproducibility and uniqueness across samples, the RNG is seeded
        based on the dataset seed and the sample index.

        Args:
            sample_idx: the index of the sample being generated

        Returns:
            A dictionary containing position, velocity and heading arrays.
        """
        cfg = self.cfg
        L = cfg.area_size_m
        dt = cfg.delta_t_s

        # Derive deterministic seed for this sample
        derived_seed = (self.seed + (sample_idx * 2654435761)) % (2**31 - 1)
        rng = np.random.default_rng(derived_seed)

        # Random initial state
        x = rng.uniform(0.1 * L, 0.9 * L)
        y = rng.uniform(0.1 * L, 0.9 * L)
        heading = rng.uniform(-math.pi, math.pi)
        v = rng.uniform(cfg.speed_min_mps, cfg.speed_max_mps)

        xs = np.zeros(self.T, dtype=np.float32)
        ys = np.zeros(self.T, dtype=np.float32)
        vs = np.zeros(self.T, dtype=np.float32)
        hs = np.zeros(self.T, dtype=np.float32)

        turn_rad = math.radians(cfg.heading_turn_deg)

        for t in range(self.T):
            xs[t], ys[t], vs[t], hs[t] = x, y, v, heading

            # Update position
            x += v * dt * math.cos(heading)
            y += v * dt * math.sin(heading)

            # Reflect at boundaries
            if x < 0 or x > L:
                heading = math.pi - heading
                x = max(0, min(L, 2 * L - x if x > L else -x))
            if y < 0 or y > L:
                heading = -heading
                y = max(0, min(L, 2 * L - y if y > L else -y))

            # Random turn
            heading += rng.uniform(-turn_rad, turn_rad)
            heading = (heading + math.pi) % (2 * math.pi) - math.pi

            # Speed update if Markov
            if cfg.speed_mode == "markov":
                r = rng.random()
                dv = -1.0 if r < 0.25 else (1.0 if r >= 0.75 else 0.0)
                v = np.clip(v + dv, cfg.speed_min_mps, cfg.speed_max_mps)

        return {"xs": xs, "ys": ys, "vs": vs, "hs": hs}
    
    def _compute_aods(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Compute AoDs from BS to user positions"""
        aods = np.zeros(self.T, dtype=np.float32)
        for t in range(self.T):
            user_pos = np.array([xs[t], ys[t]])
            aods[t] = aod_from_positions(self.bs_pos, user_pos)
        return aods
    
    def _compute_optimal_beams(
        self,
        aods: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute optimal beams and gains from AoDs"""
        cfg = self.cfg
        
        qs = np.zeros(self.T, dtype=np.int64)
        gains = np.zeros(self.T, dtype=np.float32)
        
        for t in range(self.T):
            # Simulate channel
            seed = int(self.rng.integers(0, 2**31 - 1))
            rng_local = np.random.default_rng(seed)
            
            h = simulate_sv_channel(
                cfg.M,
                cfg.d_over_lam,
                float(aods[t]),
                L_paths=cfg.L_paths,
                aod_spread_deg=cfg.aod_spread_deg,
                rng=rng_local
            )
            
            # Find optimal beam
            q_opt, gain_opt = optimal_beam_index(self.W, h)
            qs[t] = q_opt
            gains[t] = gain_opt
        
        return qs, gains
    
    def _compute_ctrv_baseline(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        hs: np.ndarray,
        vs: np.ndarray
    ) -> np.ndarray:
        """Compute CTRV baseline predictions for future H steps"""
        cfg = self.cfg
        
        # Use last state from past
        x_last = float(xs[self.U - 1])
        y_last = float(ys[self.U - 1])
        h_last = float(hs[self.U - 1])
        v_last = float(vs[self.U - 1])
        
        # Estimate turning rate from last two headings
        if self.U >= 2:
            dh = (hs[self.U-1] - hs[self.U-2] + math.pi) % (2*math.pi) - math.pi
            omega = dh / cfg.delta_t_s
        else:
            omega = 0.0
        
        # Rollout CTRV model
        xs_pred, ys_pred, _ = ctrv_rollout(
            x_last, y_last, h_last, v_last, omega,
            self.H, cfg.delta_t_s, cfg.area_size_m,
            use_reflection=True
        )
        
        # Compute predicted AoDs
        a_baseline = np.zeros(self.H, dtype=np.float32)
        for h in range(self.H):
            user_pos_pred = np.array([xs_pred[h], ys_pred[h]])
            a_baseline[h] = aod_from_positions(self.bs_pos, user_pos_pred)
        
        return a_baseline
    
    def _build_features(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        hs: np.ndarray,
        vs: np.ndarray,
        qs: np.ndarray,
        aods: np.ndarray
    ) -> np.ndarray:
        """Build input feature matrix X: [C, U]
        
        Features (per time step):
            0: q_norm - normalized beam index [-1, 1]
            1: x_norm - normalized x position [0, 1]
            2: y_norm - normalized y position [0, 1]
            3: sin(heading)
            4: cos(heading)
            5: v_norm - normalized speed [0, 1]
            6: r_norm - normalized distance to BS [0, 1]
            7: omega_norm - normalized turning rate [-1, 1]
            [8, 9]: sin(aod), cos(aod) - if include_aod_in_features
        """
        cfg = self.cfg
        U = self.U
        
        # Extract past
        q_past = qs[:U]
        x_past = xs[:U]
        y_past = ys[:U]
        h_past = hs[:U]
        v_past = vs[:U]
        a_past = aods[:U]
        
        # Normalize beam indices to [-1, 1]
        q_norm = (q_past.astype(np.float32) / (self.Q - 1)) * 2 - 1.0
        
        # Normalize positions to [0, 1]
        x_norm = x_past / cfg.area_size_m
        y_norm = y_past / cfg.area_size_m
        
        # Heading as (sin, cos)
        h_sin = np.sin(h_past)
        h_cos = np.cos(h_past)
        
        # Normalize speed to [0, 1]
        v_norm = (v_past - cfg.speed_min_mps) / \
                 (cfg.speed_max_mps - cfg.speed_min_mps + 1e-8)
        
        # Distance to BS, normalized
        dx = x_past - self.bs_pos[0]
        dy = y_past - self.bs_pos[1]
        r = np.sqrt(dx**2 + dy**2)
        r_max = cfg.area_size_m * math.sqrt(2)
        r_norm = r / (r_max + 1e-6)
        
        # Turning rate (angular velocity)
        # Use the utility function from `mobility` to compute angular
        # velocities rather than duplicating the logic here.  This
        # ensures consistency across the codebase and makes it easier
        # to modify the turning rate computation in one place.
        if U > 0:
            omega = compute_turning_rate(h_past, cfg.delta_t_s)
            # Replace the first element with the second to avoid using
            # an uninformative zero value at t=0 (for short sequences
            # compute_turning_rate returns 0 for the first element)
            if len(omega) >= 2:
                omega[0] = omega[1]
        else:
            omega = np.zeros(U, dtype=np.float32)
        omega_norm = np.clip(omega / math.pi, -1, 1)
        
        # Stack features
        features = [
            q_norm,      # 0
            x_norm,      # 1
            y_norm,      # 2
            h_sin,       # 3
            h_cos,       # 4
            v_norm,      # 5
            r_norm,      # 6
            omega_norm   # 7
        ]
        
        # Optional: add AoD features
        if cfg.include_aod_in_features:
            features.append(np.sin(a_past))  # 8
            features.append(np.cos(a_past))  # 9
        
        X = np.stack(features, axis=0).astype(np.float32)  # [C, U]
        
        return X
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """Get a sample
        
        Returns:
            X: [C, U]
            Y: [H, 2]
            a_past: [U]
            a_fut: [H]
            q_all: [U+H]
            gain_all: [U+H]
            traj: [U+H, 2]
            a_baseline: [H]
        """
        sample = self.data[idx]
        
        return (
            torch.from_numpy(sample["X"]),
            torch.from_numpy(sample["Y"]),
            torch.from_numpy(sample["a_past"]),
            torch.from_numpy(sample["a_fut"]),
            torch.from_numpy(sample["q_all"]),
            torch.from_numpy(sample["gain_all"]),
            torch.from_numpy(sample["traj"]),
            torch.from_numpy(sample["a_baseline"])
        )


def create_dataloaders(
    cfg: Config
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, val, test dataloaders - V2.0 Mac-Safe
    
    Args:
        cfg: configuration
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Create datasets
    train_dataset = BeamSeqDataset(cfg.n_train, cfg, "train", cfg.seed)
    val_dataset = BeamSeqDataset(cfg.n_val, cfg, "val", cfg.seed)
    test_dataset = BeamSeqDataset(cfg.n_test, cfg, "test", cfg.seed)
    
    # Mac-safe DataLoader settings
    loader_kwargs = {
        'batch_size': cfg.batch_size,
        'num_workers': 0,  # MUST be 0 for Mac
        'pin_memory': False,  # MUST be False for MPS
        'persistent_workers': False,  # Disabled
    }

    # Only add prefetch_factor when num_workers > 0
    if cfg.num_workers > 0:
        loader_kwargs['prefetch_factor'] = cfg.prefetch_factor
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers
    )
    
    return train_loader, val_loader, test_loader
