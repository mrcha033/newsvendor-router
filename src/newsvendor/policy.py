import copy
import math

import numpy as np

from .construction import construct, reference
from .corpus import SLOTS, STATUSES, outcome, possible
from .encoder import ACTIONS, EXTRA
from .io import digest
from .optimizer import optimal, regret


def actions(input, state):
    allowed = ["hold"]
    tolerance = input["task"]["tolerance"]
    if (
        state["valid"]
        and (tolerance is None or state["gamma"] <= tolerance)
        and input["task"]["decision"] == "minimax"
    ):
        allowed.insert(0, "handoff")
    if input["remaining"] > 0 and len(input["history"]) < input["task"]["deadline"]:
        for action in ("v", "b", "demand"):
            missing = (
                "censored-demand" in state["errors"]
                if action == "demand"
                else (action not in state["values"] or state["state"][action] == "conflict")
            )
            if (
                missing
                and not any(h["action"] == action for h in input["history"])
                and input["task"]["rho"].get(action, 1) > 0
            ):
                allowed.append(action)
    return allowed


def continuation(input, action, solve):
    rho = input["task"]["rho"].get(action, 1)
    partial = input["task"]["partial"].get(action, 0)
    worst = max(solve(outcome(input, action, value)) for value in possible(input, action))
    absent = solve(outcome(input, action, None))
    incomplete = solve(outcome(input, action, "partial")) if partial else 0
    return rho * ((1 - partial) * worst + partial * incomplete) + (1 - rho) * absent


def planner(input, builder=reference, memo=None):
    """Finite budgeted minimax rollout over declared public answer sets and response probabilities."""
    memo = {} if memo is None else memo
    key = digest(input)
    if key in memo:
        return memo[key]
    state, ref, values = builder(input), reference(input), {}
    for action in actions(input, state):
        if action == "hold":
            values[action] = input["task"]["hold"]
        elif action == "handoff":
            values[action] = (
                max(regret(state["q"], theta, input["task"]["bounds"]) for theta in ref["omega"])
                if ref["omega"]
                else input["task"]["hold"]
            )
        else:
            values[action] = input["task"]["costs"][action] + continuation(
                input, action, lambda next: planner(next, builder, memo)["value"]
            )
    action = min(values, key=lambda a: (values[a], input["task"]["costs"].get(a, 0), a))
    result = {"state": state, "values": values, "action": action, "value": values[action]}
    memo[key] = result
    return result


def features(input, state, action, cache, ablation=None):
    x = np.zeros(EXTRA, dtype=np.float32)
    scale = input["task"]["hold"]
    x[:8] = [
        min(2, (state["gamma"] if state["gamma"] is not None else scale) / scale),
        (state["q"] or 0) / max(1, input["task"]["bounds"][1]),
        input["remaining"] / 3,
        input["task"]["costs"].get(action, 0) / scale,
        input["task"]["rho"].get(action, 1),
        state["valid"],
        len(state["omega"]) / 8,
        len(input["history"]) / 3,
    ]
    for i, slot in enumerate(SLOTS):
        x[8 + i], x[12 + i] = slot not in state["values"], state["state"][slot] == "conflict"
        x[24 + i] = STATUSES.index(state["state"][slot]) / (len(STATUSES) - 1)
        x[28 + i] = state["types"][slot] == "preference"
    x[16:18] = ["censored-demand" in state["errors"], input["task"]["deadline"] / 3]
    x[18:23] = [action == a for a in ACTIONS]
    x[23] = input["task"]["partial"].get(action, 0)
    if ablation == "impact":
        x[[0, 1, 6]] = 0
    if ablation == "type":
        x[11] = 0
    return np.concatenate((cache.pool(input), cache.vector(ACTIONS[action]), x))


def choose(input, state, method, model=None, cache=None, ablation=None, constructor=None):
    allowed = actions(input, state)
    requests = [a for a in allowed if a not in ("handoff", "hold")]
    builder = constructor or (
        reference if model is None else lambda next: construct(next, model, cache, ablation)
    )
    if method == "reference":
        return planner(input, builder)["action"]
    if method == "checklist" and "handoff" in allowed:
        return "handoff"
    if method in ("checklist", "ask_all"):
        return requests[0] if requests else "handoff" if "handoff" in allowed else "hold"
    if method == "uncertainty":
        return (
            min(requests, key=lambda a: (-len(possible(input, a)), input["task"]["costs"][a]))
            if requests
            else "handoff"
            if "handoff" in allowed
            else "hold"
        )
    if method == "one_step":

        def immediate(next):
            prediction = builder(next)
            return prediction["gamma"] if prediction["valid"] else next["task"]["hold"]

        values = {
            a: input["task"]["hold"]
            if a == "hold"
            else state["gamma"]
            if a == "handoff"
            else input["task"]["costs"][a] + continuation(input, a, immediate)
            for a in allowed
        }
    elif method == "learned":
        if not state["omega"]:
            scores = model["repair"].scores([features(input, state, "hold", cache)])[0]
            return max(allowed, key=lambda a: scores[list(ACTIONS).index(a)])
        scores = (
            model["value"]
            .scores([features(input, state, a, cache, ablation) for a in allowed])
            .ravel()
        )
        values = dict(zip(allowed, scores, strict=True))
    else:
        raise ValueError(f"Unknown policy {method}")
    return min(values, key=lambda a: (values[a], input["task"]["costs"].get(a, 0), a))


def states(input, builder=reference):
    visited = {}

    def visit(next):
        key = digest(next)
        if key in visited:
            return
        visited[key] = next
        for action in actions(next, builder(next)):
            if action in ("handoff", "hold"):
                continue
            answers = [*possible(next, action), None]
            if next["task"]["partial"].get(action, 0):
                answers.append("partial")
            for value in answers:
                visit(outcome(next, action, value))

    visit(input)
    return list(visited.values())


def response(episode, input, action, noise):
    # Hidden truth is accessed only by the environment after the policy selected an action.
    chance = int(digest(episode["id"] + ":" + action)[:8], 16) / 2**32
    quality = int(digest(episode["id"] + ":quality:" + action)[:8], 16) / 2**32
    if chance >= input["task"]["rho"].get(action, 1):
        return None
    if quality < input["task"]["partial"].get(action, 0):
        return "partial"
    answer = episode["gold"]["theta"]["F" if action == "demand" else action]
    if quality < noise:
        answer = next((v for v in possible(input, action) if v != answer), answer)
    return answer


def trajectory(
    episode,
    method,
    model=None,
    cache=None,
    construction="learned",
    ablation=None,
    noise=0,
    constructor=None,
):
    if method == "oracle":
        q = optimal(episode["gold"]["theta"], episode["input"]["task"]["bounds"])["q"]
        return {
            "id": episode["id"],
            "family": episode["family"],
            "scenario": episode["scenario"],
            "method": method,
            "construction": "full-information-bound",
            "ablation": None,
            "noise": noise,
            "result": "handoff",
            "q": q,
            "regret": 0,
            "total": 0,
            "requestCost": 0,
            "requests": 0,
            "coverage": True,
            "falseHandoff": False,
            "events": [{"action": "oracle-bound", "q": q}],
        }
    input = copy.deepcopy(episode["input"])
    original = copy.deepcopy(input)
    events, cost, requests, result = [], 0, 0, "hold"
    for step in range(original["remaining"] + 2):
        view = (
            {**original, "remaining": input["remaining"], "history": input["history"]}
            if ablation == "update" and step > 0
            else input
        )
        builder = None if construction == "rules" else model
        override = constructor if construction != "rules" else None
        state = override(view) if override else construct(view, builder, cache, ablation)
        action = choose(
            view,
            state,
            method,
            model if method == "learned" else builder,
            cache,
            ablation,
            override,
        )
        event = {
            "step": step,
            "action": action,
            "q": state["q"],
            "gamma": state["gamma"],
            "values": state["values"],
            "links": state["links"],
            "errors": state["errors"],
        }
        events.append(event)
        if action in ("handoff", "hold"):
            result = action
            break
        requests += 1
        cost += input["task"]["costs"][action]
        answer = response(episode, input, action, noise)
        event["answer"] = answer
        input = outcome(input, action, answer)
    coverage = episode["gold"]["theta"] in state["omega"]
    loss = (
        regret(state["q"], episode["gold"]["theta"], input["task"]["bounds"])
        if result == "handoff"
        else None
    )
    total = cost + (loss if loss is not None else input["task"]["hold"])
    if not math.isfinite(total):
        raise ValueError("Non-finite trajectory loss")
    return {
        "id": episode["id"],
        "family": episode["family"],
        "scenario": episode["scenario"],
        "method": method,
        "construction": construction,
        "ablation": ablation,
        "noise": noise,
        "result": result,
        "q": state["q"] if result == "handoff" else None,
        "regret": loss,
        "total": total,
        "requestCost": cost,
        "requests": requests,
        "coverage": coverage,
        "falseHandoff": result == "handoff" and not coverage,
        "events": events,
    }
