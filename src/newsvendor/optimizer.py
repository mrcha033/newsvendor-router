import math
from collections import Counter
from functools import lru_cache

from .io import require


def validate(theta):
    require(all(math.isfinite(theta[k]) for k in ("c", "p", "v", "b")), "Non-finite cost")
    require(
        theta["c"] >= theta["v"] and theta["p"] - theta["c"] + theta["b"] >= 0,
        "Negative Newsvendor loss coefficient",
    )
    require(bool(theta["F"]), "Empty demand distribution")
    require(
        all(math.isfinite(d) and d >= 0 and math.isfinite(p) and p >= 0 for d, p in theta["F"]),
        "Invalid demand support",
    )
    require(abs(sum(p for _, p in theta["F"]) - 1) < 1e-8, "Probabilities must sum to one")


def risk(q, theta):
    return sum(
        p
        * (
            (theta["c"] - theta["v"]) * max(q - d, 0)
            + (theta["p"] - theta["c"] + theta["b"]) * max(d - q, 0)
        )
        for d, p in theta["F"]
    )


def orders(theta, bounds, integer):
    lo, hi = bounds
    require(math.isfinite(lo) and math.isfinite(hi) and 0 <= lo <= hi, "Invalid order bounds")
    if integer:
        qs = list(range(math.ceil(lo), math.floor(hi) + 1))
    else:
        qs = sorted({lo, hi, *(min(hi, max(lo, d)) for d, _ in theta["F"])})
    require(bool(qs), "No feasible order")
    return qs


def optimal(theta, bounds=(0, 100), integer=False):
    validate(theta)
    q = min(orders(theta, bounds, integer), key=lambda q: (risk(q, theta), q))
    return {"q": q, "cost": risk(q, theta)}


def regret(q, theta, bounds=(0, 100), integer=False):
    return max(0, risk(q, theta) - optimal(theta, bounds, integer)["cost"])


def minimax(omega, bounds=(0, 100), integer=False):
    """Exact upper-envelope minimum for finite, piecewise-linear expected regrets."""
    require(bool(omega), "Cannot optimize an empty parameter set")
    costs = [optimal(t, bounds, integer)["cost"] for t in omega]
    knots = sorted(set().union(*(orders(t, bounds, False) for t in omega)))
    qs = set(knots)

    def r(q, j):
        return risk(q, omega[j]) - costs[j]

    if integer:
        qs = set(orders(omega[0], bounds, True))
    else:
        for lo, hi in zip(knots, knots[1:], strict=False):
            slopes = [(r(hi, j) - r(lo, j)) / (hi - lo) for j in range(len(omega))]
            for i in range(len(omega)):
                for j in range(i + 1, len(omega)):
                    den = slopes[i] - slopes[j]
                    if abs(den) < 1e-12:
                        continue
                    q = lo + (r(lo, j) - r(lo, i)) / den
                    if lo - 1e-8 <= q <= hi + 1e-8:
                        qs.add(min(hi, max(lo, q)))
    scored = [(max(0, *(r(q, j) for j in range(len(omega)))), q) for q in sorted(qs)]
    gamma, q = min(scored, key=lambda item: (round(item[0], 8), item[1]))
    return {"q": q, "gamma": gamma}


def empirical(rows):
    require(bool(rows), "No observations")
    require(
        all(r["complete"] and not r["stockout"] for r in rows),
        "Censored sales are not observed demand",
    )
    require(
        all(math.isfinite(r["demand"]) and r["demand"] >= 0 for r in rows),
        "Invalid demand observations",
    )
    return [[d, n / len(rows)] for d, n in sorted(Counter(r["demand"] for r in rows).items())]


def finite(omega, budget, costs=None, bounds=(0, 100), rho=None, hold=math.inf):
    """Small analytical planner used to verify the proposal's numerical example."""
    costs, rho = costs or {"v": 10, "b": 20}, rho or {"v": 1, "b": 1}

    @lru_cache(None)
    def solve(indices, left, used):
        subset = [omega[i] for i in indices]
        terminal = minimax(subset, bounds)
        values = {"handoff": terminal["gamma"], "hold": hold}
        for slot in costs if left > 0 else ():
            if slot in used:
                continue
            keys = {t[slot] for t in subset}
            if len(keys) < 2:
                continue
            attempted = tuple(sorted((*used, slot)))
            worst = max(
                solve(tuple(i for i in indices if omega[i][slot] == value), left - 1, attempted)[
                    "value"
                ]
                for value in keys
            )
            absent = solve(indices, left - 1, attempted)["value"]
            prob = rho.get(slot, 1)
            values[slot] = costs[slot] + prob * worst + (0 if prob == 1 else (1 - prob) * absent)
        action = min(values, key=lambda a: (values[a], costs.get(a, 0), a))
        return {**terminal, "values": values, "action": action, "value": values[action]}

    return solve(tuple(range(len(omega))), budget, ())
