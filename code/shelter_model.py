"""Shared problem definition for the Nagaoka earthquake shelter assignment.

Both GA and PSO call `Problem.evaluate`, so they are scored by exactly the same
function. Every call is counted in `Problem.n_evals`, which is how the matched
evaluation budget is enforced.
"""
import numpy as np
import pandas as pd

import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset")


class Problem:
    """Reads the files built by dataset/build_dataset.py: cells.csv (demand),
    shelters.csv (current GSI shelters with P20 floor areas), distance_road_km.npz.
    Road factor, m2 per person and the shelter selection live in the dataset build."""

    def __init__(self, evac_rate=0.3, k=5, max_walk_km=3.0, penalty=10.0,
                 capacity="area", objective="convex", data_dir=DATA_DIR):
        self.objective = objective      # "convex" (main model) or "linear" (exact-solver reference)
        c = pd.read_csv(os.path.join(data_dir, "cells.csv"))
        s = pd.read_csv(os.path.join(data_dir, "shelters.csv"))
        self.cells, self.shelters = c, s
        self.evac_rate, self.k, self.max_walk_km, self.penalty = evac_rate, k, max_walk_km, penalty

        self.demand = c["population"].to_numpy(float) * evac_rate              # e_i
        if capacity == "area":
            self.cap = s["capacity_3p5m2"].to_numpy(float)                       # C_j
        else:   # official 2012 capacity; shelters without one keep the 3.5 m2 value
            self.cap = s["capacity_official"].fillna(s["capacity_3p5m2"]).to_numpy(float)
        D = np.load(os.path.join(data_dir, "distance_road_km.npz"))["distance_road_km"].astype(float)
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
