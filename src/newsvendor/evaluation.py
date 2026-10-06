"""Evaluation-only checks. Coverage does not imply permission to execute an order."""

import copy
import math

from .construction import reference
from .corpus import outcome
from .io import require
from .optimizer import regret

VERSION = 2


def covers(theta, omega):
    """Source arithmetic and serialized decimal amounts may differ by floating roundoff."""
    for candidate in omega:
        if not all(
            math.isclose(theta[k], candidate[k], rel_tol=1e-9, abs_tol=1e-8)
            for k in ("c", "p", "v", "b")
        ):
            continue
        first, second = sorted(theta["F"]), sorted(candidate["F"])
        if len(first) == len(second) and all(
            math.isclose(d, e, abs_tol=1e-8) and math.isclose(p, q, abs_tol=1e-8)
            for (d, p), (e, q) in zip(first, second, strict=True)
        ):
            return True
    return False


def authorization(input, q):
    """Check the observed legacy task independently of the predictor's claimed status."""
    observed = reference(input)
    reasons = []
    if any(s == "conflict" for s in observed["state"].values()):
        reasons.append("unresolved-conflict")
    if "censored-demand" in observed["errors"]:
        reasons.append("censored-demand")
    if not observed["omega"]:
        reasons.append("unsupported-parameters")
    if input["task"].get("preferenceRequired", False) and "b" not in observed["values"]:
        reasons.append("unselected-preference")
    if len(input["history"]) > input["task"]["deadline"]:
        reasons.append("deadline")
    lo, hi = input["task"]["bounds"]
    quantity_valid = isinstance(q, (int, float)) and math.isfinite(q) and lo <= q <= hi
    if not quantity_valid:
        reasons.append("quantity")
    elif input["task"]["tolerance"] is not None and observed["omega"]:
        worst = max(regret(q, t, (lo, hi)) for t in observed["omega"])
        if worst > input["task"]["tolerance"] + 1e-8:
            reasons.append("tolerance")
    if input["task"]["decision"] != "minimax":
        reasons.append("unsupported-objective")
    return {"authorized": not reasons, "handoffReasons": reasons}


def score(episode, input, state, result):
    """Only terminal evaluation accesses hidden theta; action selection never calls this."""
    coverage = covers(episode["gold"]["theta"], state["omega"])
    checks = (
        authorization(input, state["q"])
        if result == "handoff"
        else {
            "authorized": None,
            "handoffReasons": [],
        }
    )
    return {
        "evaluationVersion": VERSION,
        "coverage": coverage,
        **checks,
        "coverageErrorHandoff": result == "handoff" and not coverage,
        "authorizationErrorHandoff": result == "handoff" and not checks["authorized"],
        "falseHandoff": result == "handoff" and (not coverage or not checks["authorized"]),
    }


def rescore(rows, episodes):
    """Replay recorded replies, without re-sampling responses or changing old economic losses.

    Old traces do not contain their complete predicted omega, so their recorded coverage is
    retained. Independent authorization checks are added and provenance must identify the
    original corpus. Oracle rows remain evaluation bounds rather than executable orders.
    """
    indexed = {e["id"]: e for e in episodes}
    updated = []
    for row in rows:
        require(row["id"] in indexed, "Trajectory ID absent from original corpus")
        episode = indexed[row["id"]]
        require(row["family"] == episode["family"], "Trajectory source family differs")
        current = copy.deepcopy(row)
        current["legacyFalseHandoff"] = row["falseHandoff"]
        current["evaluationVersion"] = VERSION
        if row["method"] == "oracle":
            current.update(
                authorized=None,
                authorizationErrorHandoff=False,
                coverageErrorHandoff=False,
                handoffReasons=[],
            )
            updated.append(current)
            continue
        input = copy.deepcopy(episode["input"])
        terminal = None
        for event in row["events"]:
            action = event["action"]
            if action in ("hold", "handoff"):
                terminal = action
                break
            require("answer" in event, "Recorded response required; cannot re-sample history")
            input = outcome(input, action, event["answer"])
        require(terminal == row["result"], "Terminal action differs from trajectory result")
        checks = (
            authorization(input, row["q"])
            if terminal == "handoff"
            else {
                "authorized": None,
                "handoffReasons": [],
            }
        )
        current.update(checks)
        current["coverageErrorHandoff"] = terminal == "handoff" and not row["coverage"]
        current["authorizationErrorHandoff"] = terminal == "handoff" and not checks["authorized"]
        current["falseHandoff"] = (
            current["coverageErrorHandoff"] or current["authorizationErrorHandoff"]
        )
        updated.append(current)
    return updated
