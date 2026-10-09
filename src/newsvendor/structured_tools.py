"""Observed field memory and role-aware validation, without evaluation annotations."""

import json
import math
import re
from datetime import date

from .io import require
from .suite import canonical


def roles(names, definitions):
    # ABCD's ontology lists possible value vocabularies, not closed JSON Schema enums.
    # Each button can accept different semantic roles at the same argument position.
    return [
        {
            "name": name,
            "description": definitions.get(name, {}).get("description", name.replace("_", " ")),
            "valueType": definitions.get(name, {}).get("type", "string"),
            "choices": definitions.get(name, {}).get("enum", []),
            "closed": False,
        }
        for name in names
    ]


def validate(value, spec, role=None):
    effective = spec if role is None else role
    value_type = effective.get("valueType", "string")
    if value_type == "number":
        from .suite_score import number

        value = number(value)
        require(value is not None, "Invalid numeric argument")
    elif value_type == "integer":
        require(not isinstance(value, bool), "Invalid integer argument")
        value = float(value)
        require(value.is_integer(), "Fractional integer argument")
        value = int(value)
    elif value_type in ("array", "object", "boolean"):
        value = json.loads(value) if isinstance(value, str) else value
        require(
            isinstance(value, {"array": list, "object": dict, "boolean": bool}[value_type]),
            "Invalid JSON argument type",
        )
    elif value_type == "string":
        require(isinstance(value, str), "Invalid string argument")
    if effective.get("format") == "date":
        value = date.fromisoformat(value).isoformat()
    if isinstance(value, float):
        require(math.isfinite(value), "Nonfinite argument")
    # Named JSON-schema enums remain strict. Positional ontology vocabularies do not.
    if effective.get("choices") and effective.get("closed", "position" not in spec):
        require(value in effective["choices"], "Argument outside enum")
    json.dumps(value, allow_nan=False)
    return value


def memory(fields, previous=(), *, active_tool=None):
    """Keep model-selected values with their original evidence; never invent missing values."""
    result = {r["field"]: dict(r) for r in previous}
    for field in fields:
        if active_tool is not None and field.get("tool") != active_tool:
            continue
        if not field.get("use", True) or field.get("queryOnly"):
            continue
        if field.get("mode") == "conflict":
            result.pop(field["field"], None)
            continue
        if field.get("use", True) and field.get("evidence"):
            if field.get("value") is not None:
                result[field["field"]] = {
                    k: field[k]
                    for k in ("field", "name", "tool", "value", "evidence", "role", "expression", "state", "type")
                    if k in field
                }
    return list(result.values())


def remember(prediction, previous=()):
    """Only a selected tool supplies supervised argument fields; other queries are proposals."""
    tool = prediction.get("tool")
    return memory(prediction["fields"], previous, active_tool=tool) if tool else list(previous)


def copy_agrees(entity, text, role=None):
    """Copied strings must agree with current field evidence and its predicted role."""
    if entity["origin"] == "state" and role and entity.get("role") != role:
        return False
    return canonical(str(entity["value"])) == canonical(text.strip())


def entities(raw, state):
    """Candidates from observed strings and the caller's prior predictions only."""
    result, seen = [], set()
    by_source = {(r["kind"], r["id"], r.get("row"), r.get("column")): r for r in raw}

    def add(value, role, evidence, origin, expression=None):
        valid = []
        operands = []
        for loc in evidence:
            source = by_source.get((loc["kind"], loc["id"], loc.get("row"), loc.get("column")))
            if source and 0 <= loc["start"] < loc["end"] <= len(source["text"]):
                text = source["text"][loc["start"] : loc["end"]]
                if expression:
                    from .suite_score import number

                    operands.append(number(text))
                    valid.append(loc)
                elif canonical(str(value)) == canonical(text):
                    valid.append(loc)
        if expression:
            from .structured_model import compute

            if expression.get("operands") != valid or not operands or None in operands:
                return
            try:
                expected = compute(expression["op"], operands)
                if not math.isclose(float(value), expected, rel_tol=1e-9, abs_tol=1e-9):
                    return
            except (ValueError, TypeError, KeyError):
                return
        if not valid:
            return
        key = (str(value), role, json.dumps(valid, sort_keys=True))
        if key not in seen:
            seen.add(key)
            result.append(
                {"value": value, "role": role, "evidence": valid, "origin": origin}
                | ({"expression": expression} if expression else {})
            )

    for item in (state or {}).get("memory", []):
        add(
            item["value"],
            item.get("role", item["name"]),
            item.get("evidence", []),
            "state",
            item.get("expression"),
        )
    for source in raw:
        if source["kind"] not in ("history", "request"):
            continue
        for pattern, role in (
            (r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "email"),
            (r"\b(?=[A-Za-z0-9-]*[A-Za-z])(?=[A-Za-z0-9-]*\d)[A-Za-z0-9-]{4,}\b", "identifier"),
        ):
            for match in re.finditer(pattern, source["text"]):
                loc = {k: source[k] for k in ("kind", "id", "row", "column") if k in source}
                add(
                    match[0], role, [loc | {"start": match.start(), "end": match.end()}], "observed"
                )
    return result


def role_targets(spec, value):
    """Ambiguous source roles stay marginal alternatives, not fabricated exact labels."""
    candidates = spec.get("roles", [])
    exact = [
        i
        for i, r in enumerate(candidates)
        if canonical(str(value)) in [canonical(str(v)) for v in r["choices"]]
    ]
    if exact:
        return exact
    if re.fullmatch(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", str(value)):
        return [i for i, r in enumerate(candidates) if r["name"] == "email"]
    # A single declared role is unambiguous. Otherwise no semantic label is available.
    return [0] if len(candidates) == 1 else []
