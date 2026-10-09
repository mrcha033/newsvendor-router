"""Observed sales forecasts joined to evidence-backed Newsvendor parameters."""

import copy
import math
import re
from datetime import date, timedelta

from . import sequence
from .construction import candidates, matches
from .corpus import SLOTS
from .io import digest, require
from .optimizer import optimal


def compatible(expression, slot):
    """Reject a pointer to an explicitly different financial quantity; never fill a value."""
    label = expression["args"][0]["label"].lower()
    roles = {
        "b": r"shortage|unmet|missed",
        "p": r"selling price|retail price",
        "v": r"refund|salvage|residual value",
        "c": r"purchase cost|purchase price|pack price",
    }
    role = next((key for key, pattern in roles.items() if re.search(pattern, label)), None)
    return role is None or role == slot


def current_candidates(value, slot, selected=None):
    """Current applicable expressions, including a selected source with an arbitrary title."""
    pool = [
        e
        for e in candidates(value, slot)
        if (e["id"] == selected or matches(e["doc"], slot)) and compatible(e, slot)
    ]
    version = max((e["doc"]["version"] for e in pool), default=0)
    return [e for e in pool if e["doc"]["version"] == version]


def source_issue(value, slot, expression):
    """Validate our selected evidence without substituting another expression or value."""
    current = current_candidates(value, slot, expression["id"])
    if current and any(
        not math.isclose(e["value"], current[0]["value"], rel_tol=1e-9, abs_tol=1e-8)
        for e in current
    ):
        return "conflicting-source"
    if all(e["id"] != expression["id"] for e in current):
        return "superseded-source"
    return None


def task_key(value):
    return digest({k: value["task"].get(k) for k in ("sku", "period", "quantityUnit")})


def remember(value, state):
    """Carry only our accepted expressions, bound to unchanged observed source documents."""
    if state.get("taskKey") != task_key(value):
        return []
    result = []
    for slot in SLOTS:
        if slot not in state["values"] or state["state"].get(slot) != "verified":
            continue
        expr = next(
            (
                e
                for e in candidates(value, slot)
                if e["id"] == state["expressions"].get(slot)
                and compatible(e, slot)
                and math.isclose(e["value"], state["values"][slot], rel_tol=1e-9, abs_tol=1e-8)
            ),
            None,
        )
        if expr is not None and state.get("sourceHashes", {}).get(slot) == digest(expr["doc"]):
            result.append(
                {
                    "slot": slot,
                    "value": expr["value"],
                    "type": state["types"][slot],
                    "expression": expr["id"],
                    "sourceHash": digest(expr["doc"]),
                    "taskKey": state["taskKey"],
                }
            )
    return result


def copy_memory(value, record):
    copied = {}
    for item in value.get("memory", []):
        slot = item.get("slot")
        if slot not in SLOTS or slot in record["values"] or record["state"][slot] == "conflict":
            continue
        if item.get("taskKey") != task_key(value) or (
            slot == "b" and item.get("type") != "preference"
        ):
            continue
        if type(item.get("value")) not in (int, float) or not math.isfinite(item["value"]):
            continue
        current = current_candidates(value, slot, item.get("expression"))
        if any(
            not math.isclose(e["value"], item["value"], rel_tol=1e-9, abs_tol=1e-8) for e in current
        ):
            continue
        expr = next(
            (
                e
                for e in current
                if e["id"] == item.get("expression") and digest(e["doc"]) == item.get("sourceHash")
            ),
            None,
        )
        if expr is None:
            continue
        record["values"][slot], record["links"][slot] = expr["value"], expr["doc"]["id"]
        record["expressions"][slot], record["state"][slot], record["types"][slot] = (
            expr["id"],
            "verified",
            item["type"],
        )
        record["errors"] = [e for e in record["errors"] if not e.endswith("-" + slot)]
        locations = [
            {
                "kind": "document",
                "id": expr["doc"]["id"],
                "start": a["span"][0],
                "end": a["span"][1],
            }
            for a in expr["args"]
        ]
        copied[slot] = {
            "expression": {"op": expr["op"], "operands": locations},
            "evidence": locations,
            "sourceHash": item["sourceHash"],
        }
    record["copiedMemory"] = copied


def copy_responses(value, record):
    """Copy a received typed reply only when its current source agrees in value and scope."""
    latest = {}
    for index, event in enumerate(value["history"]):
        if (
            event["action"] in SLOTS
            and type(event["answer"]) in (int, float)
            and math.isfinite(event["answer"])
        ):
            latest[event["action"]] = (index, event["answer"])
    copied = {}
    for slot, (index, answer) in latest.items():
        if type(answer) not in (int, float) or not math.isfinite(answer):
            continue
        current = current_candidates(value, slot)
        if not current or any(
            not math.isclose(e["value"], answer, rel_tol=1e-9, abs_tol=1e-8) for e in current
        ):
            continue
        reply = next((e for e in current if e["doc"]["id"] == "response-" + slot), None)
        if reply is None:
            continue
        record["values"][slot] = reply["value"]
        record["links"][slot] = reply["doc"]["id"]
        record["expressions"][slot] = reply["id"]
        record["state"][slot] = "verified"
        record["types"][slot] = "preference" if slot == "b" else "fact"
        record["errors"] = [
            e
            for e in record["errors"]
            if not e.endswith("-" + slot) and not (slot == "b" and e == "unselected-preference")
        ]
        locations = [
            {
                "kind": "document",
                "id": reply["doc"]["id"],
                "start": a["span"][0],
                "end": a["span"][1],
            }
            for a in reply["args"]
        ]
        copied[slot] = {
            "historyIndex": index,
            "responseHash": digest(value["history"][index]),
            "expression": {"op": reply["op"], "operands": locations},
            "evidence": locations,
        }
    record["copiedResponses"] = copied


def resolved_fields(record, forecast):
    """Keep raw head outputs, while publishing the values actually accepted for ordering."""
    raw = copy.deepcopy(record["fields"])
    record["rawFields"] = raw
    fields = []
    for field in raw:
        slot = field["name"]
        if slot == "F":
            continue
        current = copy.deepcopy(field)
        if slot not in record["values"] and record["state"][slot] == "verified":
            record["state"][slot] = "conflict" if field.get("mode") == "conflict" else "unconfirmed"
        current.update(value=record["values"].get(slot), state=record["state"][slot])
        if slot in record["values"]:
            current.update(scale="", source=record["links"][slot])
            if slot in record.get("copiedMemory", {}):
                entry = record["copiedMemory"][slot]
                current.update(
                    mode="compute",
                    type=record["types"][slot],
                    evidence=entry["evidence"],
                    expression=entry["expression"],
                    copiedFromState=entry["sourceHash"],
                )
            if slot in record.get("copiedResponses", {}):
                reply = record["copiedResponses"][slot]
                current.update(
                    mode="compute",
                    type=record["types"][slot],
                    evidence=reply["evidence"],
                    expression=reply["expression"],
                    copiedFromHistory=reply["historyIndex"],
                )
        else:
            current["reason"] = next(
                (e for e in record["errors"] if e.endswith("-" + slot)),
                "unselected-preference"
                if slot == "b" and field["type"] != "preference"
                else current["state"],
            )
        fields.append(current)
    fields.append(
        {
            "field": "F",
            "name": "F",
            "mode": "distribution",
            "type": "estimate",
            "state": "verified" if forecast is not None else "unconfirmed",
            "value": copy.deepcopy(forecast["distribution"]) if forecast is not None else None,
            "evidence": [{"kind": "series", **forecast["source"]}] if forecast is not None else [],
            "period": copy.deepcopy(forecast["period"]) if forecast is not None else None,
        }
    )
    record["fields"] = fields


def predict(model, value, min_history=28):
    """Validate the public SKU, dates and unit before applying the trained demand model."""
    task, rows = value["task"], value["observations"]
    spec, source = task["forecast"], value["historySource"]
    require(source["sku"] == task["sku"], "Demand history SKU differs from order SKU")
    require(bool(source["id"]), "Demand history has no source identity")
    # The current demand checkpoint was trained on FreshRetail's normalized sales.
    require(
        source["unit"] == spec["unit"] == task["quantityUnit"] == "globally-normalized-sales",
        "Demand history, forecast and order must use the trained sales unit",
    )
    days = spec["days"]
    require(type(days) is int and days in (1, 7), "Demand head was trained for horizons 1 and 7")
    start = date.fromisoformat(spec["start"])
    end = start + timedelta(days=days - 1)
    require(
        task["period"] == f"{start.isoformat()}/{end.isoformat()}",
        "Order and forecast periods differ",
    )
    require(len(rows) >= min_history, "Insufficient demand history")
    dates = [date.fromisoformat(row["date"]) for row in rows]
    require(
        dates == [start - timedelta(days=len(rows) - i) for i in range(len(rows))],
        "Demand history must end before the forecast and contain consecutive dates",
    )
    require(
        all(
            row.get("stockoutHours") is not None and 0 <= row["stockoutHours"] <= 24 for row in rows
        ),
        "Demand history needs observed stockout status",
    )
    forecast = sequence.predict(model.demand, {"observations": rows}, days)
    require(
        forecast["period"]
        == {"start": start.isoformat(), "end": end.isoformat(), "days": days, "unit": spec["unit"]},
        "Demand model returned a different period or unit",
    )
    forecast["source"] = {
        "id": source["id"],
        "sku": source["sku"],
        "unit": source["unit"],
        "historyHash": digest(rows),
        "start": rows[0]["date"],
        "end": rows[-1]["date"],
        "rows": len(rows),
    }
    return forecast


def finish(value, record, forecast, error=None):
    """Only the predicted distribution and accepted, observed parameters enter the solver."""
    copy_memory(value, record)
    copy_responses(value, record)
    resolved_fields(record, forecast)
    result = {
        **record,
        "taskKey": task_key(value),
        "sourceHashes": {
            slot: digest(next(d for d in value["docs"] if d["id"] == record["links"][slot]))
            for slot in SLOTS
            if slot in record["values"]
        },
        "forecast": forecast,
        "omega": [],
        "q": None,
        "gamma": None,
        "valid": False,
        "missing": [slot for slot in SLOTS if slot not in record["values"]],
    }
    result["types"]["F"] = "estimate"
    if forecast is None:
        result["state"]["F"] = "unconfirmed"
        result["missing"].append("F")
        result["errors"].append("demand-history: " + (error or "missing"))
        return result
    result["state"]["F"] = "verified"
    result["links"]["F"] = forecast["source"]
    # Keep an estimate separate from a fact: verified here means valid source/period/unit.
    if result["missing"] or "conflict" in result["state"].values() or record["errors"]:
        return result
    theta = {**record["values"], "F": forecast["F"]}
    try:
        order = optimal(theta, value["task"]["bounds"])
    except ValueError as exception:
        result["errors"].append(str(exception))
        return result
    result.update(omega=[theta], q=order["q"], expectedCost=order["cost"], valid=True)
    return result


def actions(value, state):
    """A small request set based only on public availability and our constructed state."""
    task = value["task"]
    ready = state["valid"]
    if "forecast" not in task:
        ready = (
            ready
            and task["decision"] == "minimax"
            and (task["tolerance"] is None or state["gamma"] <= task["tolerance"])
        )
    allowed = ["handoff", "hold"] if ready else ["hold"]
    if value["remaining"] <= 0 or len(value["history"]) >= task["deadline"]:
        return allowed
    for action in (*SLOTS, "demand"):
        slot = "F" if action == "demand" else action
        # Costs declare the available request channels; hidden response probabilities
        # and hypothetical parameter sets are never consulted by this path.
        if action not in task["costs"]:
            continue
        cost = task["costs"][action]
        require(math.isfinite(cost) and cost >= 0, "Invalid declared request cost")
        missing = (
            slot in state["missing"]
            if "missing" in state
            else (
                "censored-demand" in state["errors"]
                if slot == "F"
                else slot not in state["values"] or state["state"].get(slot) == "conflict"
            )
        )
        if missing and not any(h["action"] == action for h in value["history"]):
            allowed.append(action)
    return allowed
