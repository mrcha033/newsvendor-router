"""Select a supported parameter expression from observed, applicable evidence."""

import copy
import math

from .construction import candidates, matches
from .corpus import SLOTS
from .io import require
from .structured_forecast import compatible, current_candidates, source_issue
from .structured_inputs import OPS


def expression_key(expression):
    return expression["op"], tuple(
        (a["kind"], a["id"], a["start"], a["end"]) for a in expression["operands"]
    )


def visible_candidates(expressions, view):
    """Locate every operand in the tokens actually encoded for this decision."""
    atoms = {
        (
            a["location"]["kind"],
            a["location"]["id"],
            a["location"]["start"],
            a["location"]["end"],
        ): i
        for i, a in enumerate(view["atoms"])
    }
    result = []
    for expression in expressions:
        locations = [
            {
                "kind": "document",
                "id": expression["doc"]["id"],
                "start": a["span"][0],
                "end": a["span"][1],
            }
            for a in expression["args"]
        ]
        key = expression_key({"op": expression["op"], "operands": locations})
        if not all(location in atoms for location in key[1]):
            continue
        result.append(
            {
                "id": expression["id"],
                "key": key,
                "atoms": [atoms[location] for location in key[1]],
                "value": expression["value"],
                "expression": {"op": expression["op"], "operands": locations},
            }
        )
    return result


def parameter_candidates(value, view, slot):
    """No labels or unread source tokens; retain the existing supported document scope."""
    return visible_candidates(
        [
            e
            for e in candidates(value, slot)
            if matches(e["doc"], slot) and compatible(e, slot) and not source_issue(value, slot, e)
        ],
        view,
    )


def decode_parameters(value, view, output, fields, method="scoped"):
    """Retry rejected expressions and retain conflicts present in observed source tokens."""
    require(method in ("scoped", "first"), "Unknown parameter decoder")
    decoded, diagnostics = copy.deepcopy(fields), []
    for i, field in enumerate(decoded):
        slot = field["name"]
        if slot not in SLOTS:
            continue
        computable = (
            field["mode"] == "compute"
            and field["state"] == "verified"
            and (slot != "b" or field["type"] == "preference")
        )
        selected = field.get("expression")
        # Preserve every originally accepted expression, including supported
        # selected sources with arbitrary titles. Only rejected selections retry.
        if computable and selected:
            key = expression_key(selected)
            for expression in candidates(value, slot):
                if not compatible(expression, slot) or source_issue(value, slot, expression):
                    continue
                old = {
                    "op": expression["op"],
                    "operands": [
                        {
                            "kind": "document",
                            "id": expression["doc"]["id"],
                            "start": a["span"][0],
                            "end": a["span"][1],
                        }
                        for a in expression["args"]
                    ],
                }
                if expression_key(old) == key:
                    break
            else:
                expression = None
            if expression is not None:
                continue
        current = visible_candidates(current_candidates(value, slot), view)
        if current and any(
            not math.isclose(e["value"], current[0]["value"], rel_tol=1e-9, abs_tol=1e-8)
            for e in current
        ):
            # Absence of a selected value is not permission to ignore disagreeing
            # current sources. A failed manager reply does not resolve them.
            field.update(
                state="conflict",
                mode="conflict",
                value=None,
                expression=None,
                evidence=[loc for e in current for loc in e["expression"]["operands"]],
                reason="conflicting-source",
            )
            diagnostics.append(
                {
                    "field": slot,
                    "candidates": len(current),
                    "changed": True,
                    "kind": "conflict",
                    "sources": [e["id"] for e in current],
                }
            )
            continue
        if not computable:
            continue
        choices = parameter_candidates(value, view, slot)
        if not choices:
            diagnostics.append({"field": slot, "candidates": 0, "changed": False})
            continue
        relation = output["relation"][i].float().log_softmax(-1)
        left = output["operand1"][i].float().log_softmax(-1)
        right = output["operand2"][i].float().log_softmax(-1)
        scores = []
        for choice in choices:
            ids = choice["atoms"]
            score = relation[OPS.index(choice["expression"]["op"])] + left[ids[0]]
            if len(ids) == 2:
                score = score + right[ids[1]]
            scores.append(float(score))
        index = max(range(len(choices)), key=lambda j: scores[j]) if method == "scoped" else 0
        choice = choices[index]
        field.update(
            value=choice["value"],
            expression=choice["expression"],
            evidence=choice["expression"]["operands"],
        )
        field.pop("reason", None)
        diagnostics.append(
            {
                "field": slot,
                "candidates": len(choices),
                "changed": True,
                "selected": choice["id"],
                "score": scores[index],
                "method": method,
                "kind": "expression",
            }
        )
    return decoded, diagnostics
