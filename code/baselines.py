"""Reference solutions: current practice, a simple heuristic, and exact references."""
import numpy as np
from scipy.optimize import linprog, milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix, csr_matrix


def nearest(p):
    """Current practice: everyone goes to the nearest shelter."""
    return np.zeros(p.n, dtype=int)


def greedy(p):
    """Simple heuristic: biggest cells pick first, each choosing the candidate
    that adds the least walking plus overcrowding given the loads so far."""
    x = np.zeros(p.n, dtype=int)
    load = np.zeros(p.m)
    for i in np.argsort(-p.demand):
        best, best_cost = 0, np.inf
        for c in range(p.n_valid[i]):
            j = p.cand[i, c]
            before = max(0.0, load[j] - p.cap[j])
            after = max(0.0, load[j] + p.demand[i] - p.cap[j])
            if p.objective == "linear":
                extra = after - before
            else:
                extra = (after ** 2 - before ** 2) / p.cap[j]
            cost = p.demand[i] * p.cand_dist[i, c] + p.penalty * extra
            if cost < best_cost:
                best, best_cost = c, cost
        x[i] = best
        load[p.cand[i, best]] += p.demand[i]
    return x


def _model(p):
    """Variables: y[i,c] (cell i uses candidate c) then o[j] (overflow at shelter j)."""
    n, k, m = p.n, p.k, p.m
    nv = n * k + m
    cost = np.concatenate([(p.demand[:, None] * p.cand_dist).ravel(), np.full(m, p.penalty)])
    # each cell assigned exactly once (unused duplicate slots allowed; they repeat the nearest)
    A_eq = lil_matrix((n, nv))
    for i in range(n):
        A_eq[i, i * k:(i + 1) * k] = 1
    # load_j - o_j <= cap_j
    A_ub = lil_matrix((m, nv))
    for i in range(n):
        for c in range(k):
            A_ub[p.cand[i, c], i * k + c] += p.demand[i]
    for j in range(m):
        A_ub[j, n * k + j] = -1
    return cost, csr_matrix(A_eq), csr_matrix(A_ub), nv


def lp_bound(p):
    """Relaxation: cells may be split across shelters. No real plan can score lower."""
    cost, A_eq, A_ub, nv = _model(p)
    r = linprog(cost, A_ub=A_ub, b_ub=p.cap, A_eq=A_eq, b_eq=np.ones(p.n),
                bounds=[(0, None)] * nv, method="highs")
    return r.fun


def exact_ilp(p, time_limit=120):
    """Exact unsplit assignment with HiGHS (for checking, not part of the comparison)."""
    cost, A_eq, A_ub, nv = _model(p)
    integrality = np.r_[np.ones(p.n * p.k), np.zeros(p.m)]
    ub = np.r_[np.ones(p.n * p.k), np.full(p.m, np.inf)]
    r = milp(cost, integrality=integrality, bounds=Bounds(0, ub),
             constraints=[LinearConstraint(A_eq, 1, 1), LinearConstraint(A_ub, -np.inf, p.cap)],
             options={"time_limit": time_limit, "disp": False})
    x = None
    if r.x is not None:
        x = r.x[:p.n * p.k].reshape(p.n, p.k).argmax(1)
    return r.fun, x, r.status, getattr(r, "mip_gap", None)
