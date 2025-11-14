"""
Improved Markov Mobility Model
Features: reflection at boundaries, flexible speed modes, better state tracking
"""
import math
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class MarkovMobility:
    """Markov mobility model for user movement in square area"""
    
    area_size_m: float
    delta_t_s: float
    speed_min_mps: float
    speed_max_mps: float
    heading_turn_deg: float = 5.0
    reflect_at_boundary: bool = True
    speed_mode: str = "fixed"  # "fixed" or "markov"
    
    def __post_init__(self):
        """Validate parameters"""
        assert self.area_size_m > 0, "area_size_m must be positive"
        assert self.delta_t_s > 0, "delta_t_s must be positive"
        assert self.speed_min_mps >= 0, "speed_min_mps must be non-negative"
        assert self.speed_max_mps > self.speed_min_mps, \
            "speed_max_mps must be > speed_min_mps"
        assert self.heading_turn_deg >= 0, "heading_turn_deg must be non-negative"
        assert self.speed_mode in ["fixed", "markov"], \
            "speed_mode must be 'fixed' or 'markov'"
    
    def simulate(
        self,
        T: int,
        seed: int = 0,
        init_pos: Optional[Tuple[float, float]] = None,
        init_heading: Optional[float] = None,
        init_speed: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Simulate mobility trajectory
        
        Args:
            T: number of time steps
            seed: random seed
            init_pos: initial (x, y) position
            init_heading: initial heading in radians
            init_speed: initial speed in m/s
        
        Returns:
            xs: [T] x coordinates
            ys: [T] y coordinates
            vs: [T] speeds
            hs: [T] headings
        """
        rng = np.random.default_rng(seed)
        L = self.area_size_m
        
        # Initialize position
        if init_pos is not None:
            x, y = init_pos
        else:
            # Start away from edges
            margin = L * 0.1
            x = rng.uniform(margin, L - margin)
            y = rng.uniform(margin, L - margin)
        
        # Initialize heading
        if init_heading is not None:
            heading = init_heading
        else:
            heading = rng.uniform(-math.pi, math.pi)
        
        # Initialize speed
        if init_speed is not None:
            v = init_speed
        else:
            v = rng.uniform(self.speed_min_mps, self.speed_max_mps)
        
        # Storage
        xs = np.zeros(T, dtype=np.float32)
        ys = np.zeros(T, dtype=np.float32)
        vs = np.zeros(T, dtype=np.float32)
        hs = np.zeros(T, dtype=np.float32)
        
        turn_rad = math.radians(self.heading_turn_deg)
        
        for t in range(T):
            # Store current state
            xs[t] = x
            ys[t] = y
            vs[t] = v
            hs[t] = heading
            
            # Update position
            dx = v * self.delta_t_s * math.cos(heading)
            dy = v * self.delta_t_s * math.sin(heading)
            x_new = x + dx
            y_new = y + dy
            
            # Boundary handling with reflection
            if self.reflect_at_boundary:
                # X boundary
                if x_new < 0:
                    x_new = -x_new
                    heading = math.pi - heading
                elif x_new > L:
                    x_new = 2 * L - x_new
                    heading = math.pi - heading
                
                # Y boundary
                if y_new < 0:
                    y_new = -y_new
                    heading = -heading
                elif y_new > L:
                    y_new = 2 * L - y_new
                    heading = -heading
                
                # Clamp to ensure within bounds
                x_new = np.clip(x_new, 0, L)
                y_new = np.clip(y_new, 0, L)
            else:
                # Wrap around
                x_new = x_new % L
                y_new = y_new % L
            
            x = x_new
            y = y_new
            
            # Update heading with random turn
            turn = rng.uniform(-turn_rad, turn_rad)
            heading = (heading + turn + math.pi) % (2 * math.pi) - math.pi
            
            # Update speed
            if self.speed_mode == "markov":
                # Markov speed: +/- 1 m/s with probabilities
                r = rng.random()
                if r < 0.25:
                    dv = -1.0
                elif r < 0.75:
                    dv = 0.0
                else:
                    dv = 1.0
                v = np.clip(v + dv, self.speed_min_mps, self.speed_max_mps)
            # else: fixed speed mode, v stays constant
        
        return xs, ys, vs, hs
    
    def simulate_fixed(
        self,
        T: int,
        v: float,
        seed: int = 0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Simulate with fixed speed"""
        return self.simulate(T, seed=seed, init_speed=v)
    
    def simulate_markov(
        self,
        T: int,
        v0: float,
        v_min: Optional[float] = None,
        v_max: Optional[float] = None,
        seed: int = 0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Simulate with Markov speed model"""
        if v_min is not None:
            self.speed_min_mps = v_min
        if v_max is not None:
            self.speed_max_mps = v_max
        
        original_mode = self.speed_mode
        self.speed_mode = "markov"
        result = self.simulate(T, seed=seed, init_speed=v0)
        self.speed_mode = original_mode
        return result


def wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]"""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def compute_turning_rate(
    headings: np.ndarray,
    delta_t: float
) -> np.ndarray:
    """Compute turning rate (angular velocity) from headings
    
    Args:
        headings: [T] array of headings in radians
        delta_t: time step in seconds
    
    Returns:
        omega: [T] array of turning rates in rad/s
    """
    T = len(headings)
    omega = np.zeros(T, dtype=np.float32)
    
    if T < 2:
        return omega
    
    for t in range(1, T):
        dheading = wrap_angle(headings[t] - headings[t-1])
        omega[t] = dheading / delta_t
    
    # First time step: copy from second
    omega[0] = omega[1] if T > 1 else 0.0
    
    return omega


def ctrv_predict(
    x: float,
    y: float,
    heading: float,
    v: float,
    omega: float,
    dt: float,
    L: float,
    use_reflection: bool = True
) -> Tuple[float, float, float]:
    """CTRV (Constant Turn Rate and Velocity) motion model prediction
    
    Args:
        x, y: current position
        heading: current heading in radians
        v: speed in m/s
        omega: turning rate in rad/s
        dt: time step in seconds
        L: area size (square)
        use_reflection: whether to reflect at boundaries
    
    Returns:
        x_new, y_new, heading_new: predicted state
    """
    # Predict based on CTRV model
    if abs(omega) < 1e-6:
        # Straight line motion
        x_new = x + v * dt * math.cos(heading)
        y_new = y + v * dt * math.sin(heading)
        heading_new = heading
    else:
        # Circular motion
        R = v / omega
        heading_new = heading + omega * dt
        x_new = x + R * (math.sin(heading_new) - math.sin(heading))
        y_new = y - R * (math.cos(heading_new) - math.cos(heading))
    
    # Boundary handling
    if use_reflection:
        # X boundary
        if x_new < 0 or x_new > L:
            heading_new = math.pi - heading_new
            if x_new < 0:
                x_new = -x_new
            else:
                x_new = 2 * L - x_new
            x_new = np.clip(x_new, 0, L)
        
        # Y boundary
        if y_new < 0 or y_new > L:
            heading_new = -heading_new
            if y_new < 0:
                y_new = -y_new
            else:
                y_new = 2 * L - y_new
            y_new = np.clip(y_new, 0, L)
        
        # Wrap heading
        heading_new = wrap_angle(heading_new)
    else:
        # Wrap around
        x_new = x_new % L
        y_new = y_new % L
        heading_new = wrap_angle(heading_new)
    
    return x_new, y_new, heading_new


def ctrv_rollout(
    x0: float,
    y0: float,
    heading0: float,
    v: float,
    omega: float,
    steps: int,
    dt: float,
    L: float,
    use_reflection: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rollout CTRV model for multiple steps
    
    Args:
        x0, y0: initial position
        heading0: initial heading
        v: constant speed
        omega: constant turning rate
        steps: number of steps to predict
        dt: time step
        L: area size
        use_reflection: use reflection at boundaries
    
    Returns:
        xs, ys, headings: [steps] arrays
    """
    xs = np.zeros(steps, dtype=np.float32)
    ys = np.zeros(steps, dtype=np.float32)
    headings = np.zeros(steps, dtype=np.float32)
    
    x, y, heading = x0, y0, heading0
    
    for i in range(steps):
        x, y, heading = ctrv_predict(x, y, heading, v, omega, dt, L, use_reflection)
        xs[i] = x
        ys[i] = y
        headings[i] = heading
    
    return xs, ys, headings
