"""Exact observable action costs plus learned remaining rollout losses."""

import math

from .corpus import KINDS, SLOTS, STATUSES
from .io import require

ACTIONS = ("hold", "handoff", "c", "p", "v", "b", "demand", "retrieve")
FIELDS = (*SLOTS, "F")
STATE_FEATURES = (
    "valid",
    "expected_loss",
    "remaining",
    "history",
    "errors",
    "candidate_count",
    "q_present",
    "q",
    "gamma_present",
    "gamma",
    "tolerance_present",
    "tolerance",
    "expected_cost_present",
    "expected_cost",
    "demand_present",
    "demand_mean_min",
    "demand_mean_max",
    "demand_std_min",
    "demand_std_max",
    *(f"{slot}_state_{status}" for slot in FIELDS for status in STATUSES),
    *(f"{slot}_type_{kind}" for slot in FIELDS for kind in KINDS),
    *(f"{slot}_{part}" for slot in SLOTS for part in ("present", "value")),
    *(f"{action}_{part}" for action in ACTIONS[2:] for part in ("available", "cost")),
    *("action_" + action for action in ACTIONS),
    "action_cost",
    "previous_requests",
)
VALUE_FEATURES = STATE_FEATURES + tuple(
    f"{slot}_range_{part}" for slot in SLOTS for part in ("present", "min", "max")
)


def state_features(value, state, actions, retrieval_cost=1.0):
    """Numeric own-state/public-cost inputs, independent of labels and response probabilities.

    Quantity values use the public order bound, monetary losses use the hold
    cost, and unit prices use hold/bound. Presence and state masks distinguish
    absent estimates from zero. No environment parameter or oracle is consulted.
    """
    task = value["task"]
    hold = task["hold"]
    require(math.isfinite(hold) and hold > 0, "Invalid hold cost")
    quantity = max(1.0, *(abs(v) for v in task["bounds"]))
    deadline = max(1, task["deadline"])

    def scalar(number, scale):
        if number is None or not isinstance(number, (int, float)) or not math.isfinite(number):
            return [0.0, 0.0]
        scaled = number / scale
        return [1.0, math.copysign(math.log1p(abs(scaled)), scaled)]

    distributions = [t["F"] for t in state.get("omega", []) if "F" in t]
    forecast = state.get("forecast")
    if forecast is not None:
        distributions.append(forecast["F"])
    means, deviations = [], []
    for distribution in distributions:
        mean = sum(d * p for d, p in distribution)
        variance = sum(p * (d - mean) ** 2 for d, p in distribution)
        require(
            math.isfinite(mean) and math.isfinite(variance) and variance >= 0,
            "Invalid own-state demand",
        )
        means.append(mean / quantity)
        deviations.append(math.sqrt(variance) / quantity)
    shared = [
        float(state["valid"]),
        float(task["decision"] == "expected_loss"),
        value["remaining"] / deadline,
        len(value["history"]) / deadline,
        math.log1p(len(state["errors"])),
        math.log1p(len(state.get("omega", []))),
        *scalar(state["q"], quantity),
        *scalar(state["gamma"], hold),
        *scalar(task["tolerance"], hold),
        *scalar(state.get("expectedCost"), hold),
        float(bool(means)),
        *(
            [math.log1p(x) for x in (min(means), max(means), min(deviations), max(deviations))]
            if means
            else [0.0] * 4
        ),
    ]
    shared += [float(state["state"].get(s) == status) for s in FIELDS for status in STATUSES]
    shared += [float(state["types"].get(s) == kind) for s in FIELDS for kind in KINDS]
    for slot in SLOTS:
        shared += scalar(state["values"].get(slot), hold / quantity)
    for action in ACTIONS[2:]:
        available = action in actions
        cost = retrieval_cost if action == "retrieve" else task["costs"].get(action, 0.0)
        shared += [float(available), scalar(cost, hold)[1]]
    result = []
    for action in actions:
        require(action in ACTIONS, "Unknown numeric-state action")
        cost = (
            0.0
            if action in ACTIONS[:2]
            else retrieval_cost
            if action == "retrieve"
            else task["costs"][action]
        )
        row = (
            shared
            + [float(action == a) for a in ACTIONS]
            + [
                scalar(cost, hold)[1],
                sum(h["action"] == action for h in value["history"]) / deadline,
            ]
        )
        require(
            len(row) == len(STATE_FEATURES) and all(math.isfinite(v) for v in row),
            "Invalid economic state features",
        )
        result.append(row)
    return result


def value_features(value, state, actions, retrieval_cost=1.0):
    """Own-state values and feasible ranges, without encoder activations or outcomes."""
    rows = state_features(value, state, actions, retrieval_cost)
    unit = value["task"]["hold"] / max(1.0, *(abs(v) for v in value["task"]["bounds"]))
    ranges = []
    for slot in SLOTS:
        numbers = [theta[slot] for theta in state.get("omega", []) if slot in theta]
        if slot in state["values"]:
            numbers.append(state["values"][slot])
        require(all(math.isfinite(v) for v in numbers), "Invalid own-state parameter range")
        ranges += (
            [
                1.0,
                *(
                    math.copysign(math.log1p(abs(v / unit)), v)
                    for v in (min(numbers), max(numbers))
                ),
            ]
            if numbers
            else [0.0, 0.0, 0.0]
        )
    return [row + ranges for row in rows]


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
