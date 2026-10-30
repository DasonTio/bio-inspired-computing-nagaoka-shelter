"""GA and PSO for the shelter assignment problem.

Both algorithms:
  * call the same `Problem.evaluate`,
  * stop after exactly `budget` evaluations (counted by the problem),
  * start from uniformly random valid solutions,
  * return the best solution and its best-so-far curve per evaluation.
"""
import numpy as np


class _Tracker:
    def __init__(self, p, budget):
        self.p, self.budget = p, budget
        self.best_f, self.best_x = np.inf, None
        self.curve = []                       # (evaluations used, best-so-far)
        self.start = p.n_evals

    def used(self):
        return self.p.n_evals - self.start

    def left(self):
        return self.budget - self.used()

    def eval(self, x):
        f = self.p.evaluate(x)
        if f < self.best_f:
            self.best_f, self.best_x = f, x.copy()
        self.curve.append((self.used(), self.best_f))
        return f


def genetic_algorithm(p, budget, seed, pop_size=50, tournament=3,
                      crossover_rate=0.9, mutation_rate=None, elite=2):
    """Integer GA: tournament selection, uniform crossover, reset mutation, elitism."""
    rng = np.random.default_rng(seed)
    pm = mutation_rate if mutation_rate is not None else 2.0 / p.n
    t = _Tracker(p, budget)
    pop = np.array([p.random_solution(rng) for _ in range(pop_size)])
    fit = np.array([t.eval(x) for x in pop])
    while t.left() > 0:
        order = np.argsort(fit)
        children = [pop[i].copy() for i in order[:elite]]
        child_fit = [fit[i] for i in order[:elite]]
        while len(children) < pop_size and t.left() > 0:
            a = min(rng.integers(pop_size, size=tournament), key=lambda i: fit[i])
            b = min(rng.integers(pop_size, size=tournament), key=lambda i: fit[i])
            child = pop[a].copy()
            if rng.random() < crossover_rate:            # uniform crossover
                mask = rng.random(p.n) < 0.5
                child[mask] = pop[b][mask]
            mut = rng.random(p.n) < pm                   # reset mutation to any valid candidate
            child[mut] = (rng.random(mut.sum()) * p.n_valid[mut]).astype(int)
            children.append(child)
            child_fit.append(t.eval(child))
        pop, fit = np.array(children), np.array(child_fit)
    return t.best_x, t.best_f, t.curve


def particle_swarm(p, budget, seed, swarm_size=50, w=0.729, c1=1.49445, c2=1.49445,
                   vmax_frac=0.5):
    """Continuous global-best PSO; positions in [0, n_valid_i) decoded by floor()."""
    rng = np.random.default_rng(seed)
    lo, hi = np.zeros(p.n), p.n_valid.astype(float) - 1e-9
    vmax = vmax_frac * (hi - lo)
    t = _Tracker(p, budget)
    pos = rng.random((swarm_size, p.n)) * p.n_valid
    vel = (rng.random((swarm_size, p.n)) * 2 - 1) * vmax
    pbest = pos.copy()
    pbest_f = np.array([t.eval(p.decode_continuous(z)) for z in pos])
    g = pbest[pbest_f.argmin()].copy()
    while t.left() > 0:
        for s in range(swarm_size):
            if t.left() <= 0:
                break
            r1, r2 = rng.random(p.n), rng.random(p.n)
            vel[s] = w * vel[s] + c1 * r1 * (pbest[s] - pos[s]) + c2 * r2 * (g - pos[s])
            vel[s] = np.clip(vel[s], -vmax, vmax)
            pos[s] = np.clip(pos[s] + vel[s], lo, hi)
            f = t.eval(p.decode_continuous(pos[s]))
            if f < pbest_f[s]:
                pbest[s], pbest_f[s] = pos[s].copy(), f
                if f < pbest_f.min() + 1e-12:
                    g = pos[s].copy()
    return t.best_x, t.best_f, t.curve
