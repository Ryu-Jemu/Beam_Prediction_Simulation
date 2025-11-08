
import math, random
import numpy as np
from dataclasses import dataclass

@dataclass
class MarkovMobility:
    area_size_m: float
    delta_t_s: float
    speed_min_mps: float
    speed_max_mps: float
    heading_turn_deg: float = 20.0
    reflect_at_boundary: bool = True
    speed_mode: str = "fixed"  # "fixed" or "markov"

    def simulate(self, T: int, seed: int = 0):
        rng = random.Random(seed)
        # init
        L = self.area_size_m
        x = rng.uniform(0.1*L, 0.9*L)
        y = rng.uniform(0.1*L, 0.9*L)
        heading = rng.uniform(-math.pi, math.pi)
        v = rng.uniform(self.speed_min_mps, self.speed_max_mps)

        xs, ys, vs, hs = [], [], [], []
        for t in range(T):
            xs.append(x); ys.append(y); vs.append(v); hs.append(heading)
            # position update
            x += v * self.delta_t_s * math.cos(heading)
            y += v * self.delta_t_s * math.sin(heading)

            # boundary handling
            if self.reflect_at_boundary:
                if x < 0: x = -x; heading = math.pi - heading
                if x > L: x = 2*L - x; heading = math.pi - heading
                if y < 0: y = -y; heading = -heading
                if y > L: y = 2*L - y; heading = -heading
            else:
                x = x % L; y = y % L

            # heading is Markov with small random turn
            turn = rng.uniform(-1.0, 1.0) * (self.heading_turn_deg * math.pi/180.0)
            heading = (heading + turn + math.pi) % (2*math.pi) - math.pi

            # speed update
            if self.speed_mode == "fixed":
                pass  # keep v
            else:
                # Δv ∈ {-1, 0, +1} with p=(0.25, 0.5, 0.25)
                r = rng.random()
                if r < 0.25:
                    dv = -1.0
                elif r < 0.75:
                    dv = 0.0
                else:
                    dv = 1.0
                v = max(self.speed_min_mps, min(self.speed_max_mps, v + dv))

        return (
            np.array(xs, dtype=np.float32),
            np.array(ys, dtype=np.float32),
            np.array(vs, dtype=np.float32),
            np.array(hs, dtype=np.float32),
        )
