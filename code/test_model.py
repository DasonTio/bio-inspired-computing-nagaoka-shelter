"""Checks that the shared evaluator scores solutions the way the formulation says."""
import numpy as np
from shelter_model import Problem


def test_hand_calculation():
    for obj in ["linear", "convex"]:
        p = Problem("nagaoka_cells_2020.csv", "nagaoka_shelters_raw.csv", evac_rate=0.3, objective=obj)
        x = np.zeros(p.n, dtype=int)
        j = p.cand[:, 0]
        load = np.zeros(p.m)
        for i in range(p.n):                      # slow loop on purpose: independent of bincount
            load[j[i]] += p.demand[i]
        walk = sum(p.demand[i] * p.cand_dist[i, 0] for i in range(p.n))
        if obj == "linear":
            crowd = sum(max(0, load[s] - p.cap[s]) for s in range(p.m))
        else:
            crowd = sum(max(0, load[s] - p.cap[s]) ** 2 / p.cap[s] for s in range(p.m))
        assert abs(p.evaluate(x) - (walk + p.penalty * crowd)) < 1e-6 * max(1, walk)
        assert p.n_evals == 1


def test_random_solutions_are_valid():
    p = Problem("nagaoka_cells_2020.csv", "nagaoka_shelters_raw.csv")
    x = p.random_solution(np.random.default_rng(0))
    assert (x >= 0).all() and (x < p.n_valid).all()


def test_decoding_and_bounds():
    p = Problem("nagaoka_cells_2020.csv", "nagaoka_shelters_raw.csv")
    z = np.full(p.n, 9.0)                          # far above any bound
    x = p.decode_continuous(z)
    assert (x == p.n_valid - 1).all()              # clipped to the last *valid* candidate
    assert (p.decode_continuous(np.full(p.n, -1.0)) == 0).all()
    # every valid candidate is within the walking limit, except a cell's own nearest shelter
    i, c = np.nonzero(np.arange(p.k)[None] < p.n_valid[:, None])
    far = p.cand_dist[i, c] > p.max_walk_km + 1e-9
    assert (c[far] == 0).all()


def test_totals_match_census():
    p = Problem("nagaoka_cells_2020.csv", "nagaoka_shelters_raw.csv", evac_rate=1.0)
    assert abs(p.demand.sum() - 266975) < 1       # official 2020 census: 266,936
    assert p.m == 246


if __name__ == "__main__":
    for t in [test_hand_calculation, test_random_solutions_are_valid,
              test_decoding_and_bounds, test_totals_match_census]:
        t(); print("ok", t.__name__)
