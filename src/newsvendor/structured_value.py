"""Exact observable action costs plus learned remaining rollout losses."""

import math

from .io import require


def known_costs(actions, *, hold, costs, remaining, history_length, deadline, retrieval_cost):
    """Only public costs/budgets; no response model or hidden outcomes."""
    require(math.isfinite(hold) and hold > 0, "Invalid hold cost")
    future = remaining > 1 and history_length + 1 < deadline
    fixed, learned = [], []
    for action in actions:
        require(
            action in ("hold", "handoff", "c", "p", "v", "b", "demand", "retrieve"),
            "Unknown economic action",
        )
        terminal = action in ("hold", "handoff")
        cost = 0.0 if terminal else retrieval_cost if action == "retrieve" else costs[action]
        require(math.isfinite(cost) and cost >= 0, "Invalid request cost")
        fixed.append([float(action == "hold"), cost / hold])
        learned.append([float(action != "hold"), float(not terminal and future)])
    return {"fixed": fixed, "learned": learned}


def constrain(values, costs=None):
    """Values are positive predictions, normalized by the observed hold cost."""
    if costs is None:
        return values
    fixed = values.new_tensor(costs["fixed"])
    learned = values.new_tensor(costs["learned"])
    require(values.shape == fixed.shape == learned.shape, "Action cost dimensions differ")
    return fixed + learned * values
