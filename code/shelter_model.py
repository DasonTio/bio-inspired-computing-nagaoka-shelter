"""Shared problem definition for the Nagaoka earthquake shelter assignment.

Both GA and PSO call `Problem.evaluate`, so they are scored by exactly the same
function. Every call is counted in `Problem.n_evals`, which is how the matched
evaluation budget is enforced.
"""
import numpy as np
import pandas as pd

ROAD_FACTOR = 1.3        # straight-line km -> approximate road km
M2_PER_PERSON = 3.5      # Nagaoka shelter operations manual (FY2026)
FLOOD_ONLY = "信濃川等のはんらん時避難場所"


def haversine_km(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = (np.sin((lat2 - lat1) * p / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 12742 * np.arcsin(np.sqrt(a))


def load_shelters(path):
    s = pd.read_csv(path)
    s["area_m2"] = pd.to_numeric(s["P20_006"], errors="coerce")
    s["cap_official"] = pd.to_numeric(s["P20_005"], errors="coerce")
    s = s[(s["P20_004"] != FLOOD_ONLY) & (s["area_m2"] > 0)].reset_index(drop=True)
    s["cap"] = np.floor(s["area_m2"] / M2_PER_PERSON).astype(int)
    return s


class Problem:
    def __init__(self, cells_csv, shelters_csv, evac_rate=0.3, k=5,
                 max_walk_km=3.0, penalty=10.0, capacity="area", objective="convex"):
        self.objective = objective      # "convex" (main model) or "linear" (exact-solver reference)
        c = pd.read_csv(cells_csv)
        c = c[c["pop"] > 0].reset_index(drop=True)
        s = load_shelters(shelters_csv)
        self.cells, self.shelters = c, s
        self.evac_rate, self.k, self.max_walk_km, self.penalty = evac_rate, k, max_walk_km, penalty

        self.demand = c["pop"].to_numpy(float) * evac_rate                    # e_i
        self.cap = (s["cap"] if capacity == "area" else s["cap_official"]).to_numpy(float)  # C_j
        D = haversine_km(c.lat.to_numpy()[:, None], c.lon.to_numpy()[:, None],
                         s.lat.to_numpy()[None], s.lon.to_numpy()[None]) * ROAD_FACTOR
        order = np.argsort(D, axis=1)[:, :k]                                   # k nearest
        dist = np.take_along_axis(D, order, axis=1)
        # walking limit: a candidate beyond the limit is replaced by the nearest one,
        # so every cell always has its nearest shelter available
        too_far = dist > max_walk_km
        too_far[:, 0] = False
        order = np.where(too_far, order[:, :1], order)
        dist = np.where(too_far, dist[:, :1], dist)
        self.cand, self.cand_dist = order, dist                                # (n, k)
        self.n_valid = (~too_far).sum(1)                                       # real choices per cell
        self.n, self.m = len(c), len(s)
        self.n_evals = 0

    # ---- encoding -------------------------------------------------------
    def decode_continuous(self, z):
        """PSO decoding: real value in [0, n_valid_i) -> integer candidate index for cell i."""
        return np.clip(np.floor(z), 0, self.n_valid - 1).astype(int)

    def random_solution(self, rng):
        return (rng.random(self.n) * self.n_valid).astype(int)

    # ---- evaluation -----------------------------------------------------
    def components(self, x):
        """x: integer vector of length n, each in [0, k). Returns the parts of the score."""
        j = self.cand[np.arange(self.n), x]
        d = self.cand_dist[np.arange(self.n), x]
        load = np.bincount(j, weights=self.demand, minlength=self.m)
        overflow = np.maximum(load - self.cap, 0)
        return {
            "walk_person_km": float(self.demand @ d),
            "crowding": float((overflow ** 2 / self.cap).sum()),
            "max_load_ratio": float((load / self.cap).max()),
            "overflow_people": float(overflow.sum()),
            "overloaded_shelters": int((overflow > 1e-9).sum()),
            "mean_walk_km": float(self.demand @ d / self.demand.sum()),
            "max_walk_km": float(d.max()),
            "load": load,
        }

    def evaluate(self, x):
        self.n_evals += 1
        c = self.components(x)
        if self.objective == "linear":
            return c["walk_person_km"] + self.penalty * c["overflow_people"]
        return c["walk_person_km"] + self.penalty * c["crowding"]
