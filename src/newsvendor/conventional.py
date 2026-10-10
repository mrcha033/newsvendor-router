"""Observed-input rules and a fixed censored lognormal model, without neural weights."""

import copy
import math
import re
from datetime import date, timedelta

import torch

from . import demand, structured_forecast
from .construction import atoms, expressions
from .corpus import SLOTS
from .io import digest, require


def role(expr, slot):
    labels = [a["label"].lower() for a in expr["args"]]
    first = labels[0]
    if slot == "c":
        return (expr["op"] == "divide" and "pack price" in first) or (
            expr["op"] == "copy" and bool(re.search(r"purchase (?:cost|price)", first))
        )
    if slot == "p":
        return expr["op"] == "copy" and bool(re.search(r"selling price|retail price", first))
    if slot == "v":
        fee = any("handling fee" in a["label"].lower() for a in atoms(expr["doc"]))
        return bool(re.search(r"refund|salvage|residual value", first)) and (
            (expr["op"] == "subtract" and "handling fee" in labels[1])
            if fee
            else expr["op"] == "copy"
        )
    return (
        expr["op"] == "copy"
        and bool(re.search(r"shortage|unmet|missed", first))
        and expr["doc"]["role"] == "manager"
        and bool(re.search(r"\bselected\b", expr["doc"]["text"], re.I))
    )


def rule_parameters(value):
    record = {k: {} for k in ("values", "links", "state", "types", "expressions")}
    record["errors"] = []
    candidates = [
        e
        for doc in value["docs"]
        if doc.get("complete", True) and all(doc[k] == value["task"][k] for k in ("sku", "period"))
        for e in expressions(doc)
        if e["unit"] == "currency/unit" and math.isfinite(e["value"]) and e["value"] >= 0
    ]
    for slot in SLOTS:
        pool = [e for e in candidates if role(e, slot)]
        newest = max((e["doc"]["version"] for e in pool), default=0)
        pool = sorted((e for e in pool if e["doc"]["version"] == newest), key=lambda e: e["id"])
        record["types"][slot] = "preference" if slot == "b" else "fact"
        if pool and all(math.isclose(e["value"], pool[0]["value"], abs_tol=1e-8) for e in pool):
            e = pool[0]
            record["values"][slot] = e["value"]
            record["links"][slot] = e["doc"]["id"]
            record["expressions"][slot] = e["id"]
            record["state"][slot] = "verified"
        elif pool:
            record["state"][slot] = "conflict"
            record["errors"].append("conflict-" + slot)
        else:
            attempted = any(h["action"] == slot for h in value["history"])
            record["state"][slot] = "unavailable" if attempted else "unconfirmed"
    record["fields"] = []
    for slot in SLOTS:
        selected = next((e for e in candidates if e["id"] == record["expressions"].get(slot)), None)
        locations = (
            [
                {
                    "kind": "document",
                    "id": selected["doc"]["id"],
                    "start": a["span"][0],
                    "end": a["span"][1],
                }
                for a in selected["args"]
            ]
            if selected
            else []
        )
        record["fields"].append(
            {
                "field": slot,
                "name": slot,
                "type": record["types"][slot],
                "state": record["state"][slot],
                "value": record["values"].get(slot),
                "mode": "compute" if selected else "missing",
                "evidence": locations,
                "expression": {"op": selected["op"], "operands": locations} if selected else None,
            }
        )
    return record


def statistical_forecast(value, settings):
    """Regularized censored likelihood on historical weekly blocks, not future labels.

    A fixed weak prior prevents unbounded extrapolation when all weeks are censored.
    Weekly censoring treats the sum as a lower bound, as in the prior aggregate
    experiment; this discards exact-day information and is explicitly reported.
    """
    rows, spec, source = value["observations"], value["task"]["forecast"], value["historySource"]
    days = spec["days"]
    start = date.fromisoformat(spec["start"])
    require(days == 7 and len(rows) >= settings["minHistory"], "Weekly history is required")
    require(source["sku"] == value["task"]["sku"], "History SKU mismatch")
    require(
        source["unit"]
        == spec["unit"]
        == value["task"]["quantityUnit"]
        == "globally-normalized-sales",
        "History unit mismatch",
    )
    require(
        [r["date"] for r in rows]
        == [(start - timedelta(days=len(rows) - i)).isoformat() for i in range(len(rows))],
        "Historical dates must precede the forecast",
    )
    require(
        all(
            math.isfinite(r["sales"]) and r["sales"] >= 0 and 0 <= r["stockoutHours"] <= 24
            for r in rows
        ),
        "Invalid observed history",
    )
    end = start + timedelta(days=6)
    require(value["task"]["period"] == f"{start.isoformat()}/{end.isoformat()}", "Period mismatch")
    past = rows[-settings["historyDays"] :]
    windows = [past[i - 7 : i] for i in range(len(past), 6, -7)]
    scale = max(sum(r["sales"] for r in past) / len(past) * 7, 0.7)
    y = torch.tensor([sum(r["sales"] for r in w) / scale for w in windows], dtype=torch.float64)
    censored = torch.tensor([any(r["stockoutHours"] > 0 for r in w) for w in windows])
    prior = torch.tensor(settings["priorRaw"], dtype=torch.float64)
    raw = prior.clone().requires_grad_()
    solver = torch.optim.LBFGS(
        [raw], max_iter=settings["iterations"], line_search_fn="strong_wolfe"
    )

    def closure():
        solver.zero_grad()
        objective = demand.nll("lognormal", raw, y, censored).sum()
        objective = objective + settings["priorWeight"] * (raw - prior).square().sum() / 2
        objective.backward()
        return objective

    with torch.enable_grad():
        solver.step(closure)
    first, second = demand.parameters("lognormal", raw.detach())
    distribution = {
        "family": "lognormal",
        "familyProbabilities": {f: float(f == "lognormal") for f in demand.FAMILIES},
        "selection": "fixed family; one-hot coding is not a probability of the true family",
        "parameters": {
            "zeroProbability": float(raw.detach()[0].sigmoid()),
            "logMean": float(first),
            "logStd": float(second),
        },
        "normalizationScale": scale,
    }
    return {
        "action": "answer",
        "F": demand.pmf(distribution),
        "distribution": distribution,
        "period": {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "days": 7,
            "unit": spec["unit"],
        },
        "source": {
            **source,
            "historyHash": digest(rows),
            "start": rows[0]["date"],
            "end": rows[-1]["date"],
            "rows": len(rows),
        },
        "fit": {
            "method": "fixed-family regularized censored likelihood",
            "blocks": len(windows),
            "censoredBlocks": int(censored.sum()),
            "objective": float(demand.nll("lognormal", raw.detach(), y, censored).sum()),
            "settingsHash": digest(settings),
        },
    }


class Conventional:
    def __init__(self, settings):
        self.settings, self.forecasts = copy.deepcopy(settings), {}

    def construct(self, value):
        require("forecast" in value["task"], "Conventional baseline is retail only")
        key = digest([value["observations"], value["task"]["forecast"], value["historySource"]])
        if key not in self.forecasts:
            self.forecasts[key] = statistical_forecast(value, self.settings)
        return structured_forecast.finish(
            value, rule_parameters(value), copy.deepcopy(self.forecasts[key])
        )


class CommonPolicy:
    """Identical decision rules and observed action channels in both AI conditions."""

    def __init__(self, constructor, questions):
        self.constructor, self.questions = constructor, questions
        self.config = {}

    def construct(self, value):
        return self.constructor.construct(value)

    def allowed(self, value, state):
        available = structured_forecast.actions(value, state)
        return available if self.questions else [a for a in available if a in ("hold", "handoff")]

    def choose(self, value, state):
        allowed = self.allowed(value, state)
        if "handoff" in allowed:
            return "handoff"
        pending = [s for s in SLOTS if s in allowed]
        return (
            min(pending, key=lambda s: (value["task"]["costs"][s], list(SLOTS).index(s)))
            if pending
            else "hold"
        )
