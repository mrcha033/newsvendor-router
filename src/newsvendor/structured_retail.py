"""Controlled Newsvendor contracts paired with real, source-separated sales histories.

Contracts, costs and manager replies are generated experimental conditions. They
are not claims about the retailer. Future sales are scorer/Train labels only.
"""

import copy
import math
import random
from datetime import date, timedelta

from . import corpus, demand
from .construction import parameter_record
from .io import digest, require
from .suite_score import number

VERSION = "retail-newsvendor-v2"
SCENARIOS = (
    "sufficient",
    "missing_c",
    "missing_p",
    "missing_v",
    "missing_b",
    "conflict_v",
    "missing_v_b",
)


def cases(rows, seed=53, stride=7):
    """Build observed inputs without accepting any future labels or target distributions."""
    require(stride > 0, "Invalid cutoff stride")
    result = []
    families = {}
    for row in rows:
        if row["component"] != "retail":
            continue
        require(
            families.setdefault(row["family"], row["split"]) == row["split"],
            "Retail source-group leakage",
        )
        past = row["input"]["observations"]
        require(len(past) >= 28, "Insufficient source sales history")
        rng = random.Random(digest([VERSION, seed, row["family"]]))
        c = round(rng.uniform(4, 25), 4)
        p = round(c + rng.uniform(3, 22), 4)
        v = round(c * rng.uniform(0.05, 0.8), 4)
        b = round(rng.uniform(0, 30), 4)
        cutoffs = sorted({*range(28, len(past) + 1, stride), len(past)})
        for cutoff in cutoffs:
            history = copy.deepcopy(past[:cutoff])
            start = date.fromisoformat(history[-1]["date"]) + timedelta(days=1)
            end = start + timedelta(days=6)
            scale = max(sum(r["sales"] for r in history) / len(history) * 7, 0.7)
            hold = 100 * scale  # Public experimental cost, independent of every hidden parameter.
            for scenario in SCENARIOS:
                value = {
                    "task": {
                        "sku": row["id"],
                        "period": f"{start.isoformat()}/{end.isoformat()}",
                        "forecast": {
                            "start": start.isoformat(),
                            "days": 7,
                            "unit": "globally-normalized-sales",
                        },
                        "quantityUnit": "globally-normalized-sales",
                        "bounds": [0, 10 * scale],
                        "hold": hold,
                        "costs": {
                            "c": 0.02 * hold,
                            "p": 0.02 * hold,
                            "v": 0.02 * hold,
                            "b": 0.03 * hold,
                        },
                        "tolerance": None,
                        "deadline": 3,
                        "decision": "expected_loss",
                    },
                    "docs": [],
                    "observations": copy.deepcopy(history),
                    "history": [],
                    "remaining": 3,
                    "historySource": {
                        "id": row["id"],
                        "sku": row["id"],
                        "unit": "globally-normalized-sales",
                    },
                }
                value["docs"] = [
                    corpus.document(
                        value,
                        "quote",
                        "Purchase quotation",
                        f"Pack price = {10 * c:.12g}; units per pack = 10.",
                        "buyer",
                    ),
                    corpus.document(
                        value,
                        "price",
                        "Sales price list",
                        f"Selling price per unit = {p:.12g}.",
                        "sales",
                    ),
                ]
                return_doc = corpus.answer_doc(value, "v", v, "return")
                manager_doc = corpus.answer_doc(value, "b", b, "manager")
                return_doc["version"] = manager_doc["version"] = 1
                if scenario in ("missing_c", "missing_p"):
                    hidden_title = (
                        "Purchase quotation" if scenario == "missing_c" else "Sales price list"
                    )
                    value["docs"] = [d for d in value["docs"] if d["title"] != hidden_title]
                if scenario not in ("missing_v", "missing_v_b"):
                    value["docs"].append(return_doc)
                if scenario not in ("missing_b", "missing_v_b"):
                    value["docs"].append(manager_doc)
                if scenario == "conflict_v":
                    conflicting = corpus.answer_doc(value, "v", max(0, v - 0.5), "other-return")
                    conflicting["version"] = 1
                    value["docs"].append(conflicting)
                result.append(
                    {
                        "id": f"{row['id']}:{cutoff}:{scenario}",
                        "family": row["family"],
                        "split": row["split"],
                        "scenario": scenario,
                        "benchmark": VERSION,
                        "sourceId": row["id"],
                        "cutoffIndex": cutoff,
                        "input": value,
                        "responses": {"c": c, "p": p, "v": v, "b": b},
                        "financialAnnotations": {"c": c, "p": p, "v": v, "b": b},
                        "provenance": {
                            "financialConditions": "generated",
                            "sales": "observed",
                            "sourceInputHash": digest(row["input"]),
                        },
                    }
                )
    return result


def attach_targets(episodes, rows, labels):
    """Scoring/training boundary. No observed input changes when future labels change."""
    sources = {r["id"]: r for r in rows}
    result = []
    for episode in episodes:
        # Independent numeric annotations catch calculator/annotation bugs before
        # they can become language targets or erroneous economic scores.
        record = parameter_record(episode["input"])
        for slot, amount in record["values"].items():
            require(
                math.isclose(
                    amount, episode["financialAnnotations"][slot], rel_tol=1e-9, abs_tol=1e-8
                ),
                f"Generated {slot} annotation disagrees with document arithmetic",
            )
        row = sources[episode["sourceId"]]
        label = labels[row["id"]]
        future = [
            {"date": d, "sales": y, "stockoutHours": 0 if complete else 1}
            for d, y, complete in zip(
                label["dates"], label["answer"], label["complete"], strict=True
            )
        ]
        measured = (row["input"]["observations"][episode["cutoffIndex"] :] + future)[:7]
        start = date.fromisoformat(episode["input"]["task"]["forecast"]["start"])
        require(
            [r["date"] for r in measured]
            == [(start + timedelta(days=i)).isoformat() for i in range(7)],
            "Wrong outcome period",
        )
        require(
            all(math.isfinite(r["sales"]) and r["sales"] >= 0 for r in measured),
            "Invalid observed outcome",
        )
        target = {
            "dates": [r["date"] for r in measured],
            "answer": [r["sales"] for r in measured],
            "complete": [r["stockoutHours"] == 0 for r in measured],
            "latentDemandLabels": False,
        }
        result.append({**episode, "target": target})
    return result


def response(episode, current, action, noise=0, *, seed=None):
    require(action in episode["responses"], "Request has no registered observed-response channel")
    require(0 <= noise <= 1, "Invalid missing-response probability")
    key = [episode["family"], episode["cutoffIndex"], action]
    if seed is None:
        key.append(len(current["history"]))
    else:
        require(episode["split"] == "train", "Response resampling accepts Train only")
        # One draw per request channel: changing question order cannot change
        # which manager answers. Default historical evaluation remains unchanged.
        key = ["train-response-v1", seed, *key]
    rng = random.Random(digest(key))
    return None if rng.random() < noise else episode["responses"][action]


def parameter_metrics(value, predicted):
    reference = parameter_record(value)
    raw = {f["name"]: f for f in predicted.get("rawFields", predicted.get("fields", []))}
    values, states, types, evidence, raw_values = [], [], [], [], []
    for slot in corpus.SLOTS:
        expected, actual = reference["values"].get(slot), predicted["values"].get(slot)
        correct = (
            (actual is None)
            if expected is None
            else (actual is not None and math.isclose(expected, actual, rel_tol=1e-7, abs_tol=1e-7))
        )
        values.append(correct)
        raw_value = number(raw.get(slot, {}).get("value"))
        raw_values.append(
            raw_value is None
            if expected is None
            else (
                raw_value is not None
                and math.isclose(expected, raw_value, rel_tol=1e-7, abs_tol=1e-7)
            )
        )
        states.append(reference["state"][slot] == predicted["state"].get(slot))
        types.append(reference["types"][slot] == predicted["types"].get(slot))
        if expected is not None:
            evidence.append(
                correct and reference["expressions"][slot] == predicted["expressions"].get(slot)
            )
    return {
        "parameterAccuracy": sum(values) / len(values),
        "allParametersCorrect": all(values),
        "stateAccuracy": sum(states) / len(states),
        "typeAccuracy": sum(types) / len(types),
        "evidenceCorrect": sum(evidence),
        "evidenceCount": len(evidence),
        "rawStateAccuracy": sum(
            raw.get(slot, {}).get("state") == reference["state"][slot] for slot in corpus.SLOTS
        )
        / len(corpus.SLOTS)
        if set(corpus.SLOTS) <= set(raw)
        else None,
        "rawParameterAccuracy": sum(raw_values) / len(raw_values)
        if set(corpus.SLOTS) <= set(raw)
        else None,
        "rawTypeAccuracy": sum(
            raw.get(slot, {}).get("type") == reference["types"][slot] for slot in corpus.SLOTS
        )
        / len(corpus.SLOTS)
        if set(corpus.SLOTS) <= set(raw)
        else None,
    }


def evaluate(episode, current, state, action):
    """Score the actual emitted quantity; censored demand yields a loss lower bound only."""
    target, task = episode["target"], current["task"]
    complete = all(target["complete"])
    reference = parameter_record(current)
    metrics = parameter_metrics(current, state)
    missing = [s for s in corpus.SLOTS if s not in reference["values"]]
    reasons = []
    if missing:
        reasons.append("unresolved-parameters")
    if not metrics["allParametersCorrect"]:
        reasons.append("incorrect-parameters")
    if state.get("forecast") is None:
        reasons.append("missing-forecast")
    q = state["q"]
    if q is None or not math.isfinite(q) or not task["bounds"][0] <= q <= task["bounds"][1]:
        reasons.append("quantity")
    if len(current["history"]) > task["deadline"]:
        reasons.append("deadline")
    valid = not reasons
    loss, lower = None, None
    if action == "handoff" and not missing and q is not None:
        values = reference["values"]
        under, over = values["p"] - values["c"] + values["b"], values["c"] - values["v"]
        observed = sum(target["answer"])
        if complete:
            loss = over * max(q - observed, 0) + under * max(observed - q, 0)
        else:
            lower = under * max(observed - q, 0)
    terminal = task["hold"] if action == "hold" else loss
    if action == "handoff" and not valid:
        terminal = max(loss or 0, task["hold"])
    forecast_metrics = {}
    if state.get("forecast") is not None:
        forecast_metrics = demand.metrics(
            {"request": "Forecast next 7 daily sales"}, target, state["forecast"]
        )
    return {
        "evaluationVersion": VERSION,
        "outcomeComplete": complete,
        "terminalLoss": terminal,
        "economicLoss": loss if action == "handoff" else task["hold"],
        "censoredOrderLossLowerBound": lower,
        "invalidHandoffPenalty": (terminal - (loss or 0))
        if action == "handoff" and not valid
        else 0.0,
        "falseHandoff": action == "handoff" and not valid,
        "authorized": valid if action == "handoff" else None,
        "handoffReasons": reasons if action == "handoff" else [],
        "unresolvedAtStop": missing,
        "lossDefinition": "Observed complete-period asymmetric order loss or declared hold cost, plus request costs; censored order losses are lower bounds only",
        "parameters": metrics,
        "forecastMetrics": forecast_metrics,
    }


def interactions(events):
    needed, unnecessary, resolved, failed = 0, 0, 0, 0
    for index, event in enumerate(events):
        action = event["action"]
        if action not in corpus.SLOTS:
            continue
        reference = parameter_record(event["input"])
        needed += int(action not in reference["values"])
        unnecessary += int(action in reference["values"])
        if index + 1 < len(events):
            after = events[index + 1]["state"]
            observed = parameter_record(events[index + 1]["input"])
            expected = observed["values"].get(action)
            actual = after["values"].get(action)
            correct = (
                expected is not None
                and actual is not None
                and math.isclose(expected, actual, rel_tol=1e-7, abs_tol=1e-7)
            )
            resolved += int(correct)
            failed += int(not correct)
    return {
        "necessaryRequests": needed,
        "unnecessaryRequests": unnecessary,
        "recoveredAfterRequest": resolved,
        "unresolvedAfterRequest": failed,
    }


def summarize(records):
    require(records, "No Newsvendor measurements")

    def mean(values):
        finite = [float(v) for v in values if v is not None]
        return sum(finite) / len(finite) if finite else None

    complete = [r for r in records if r["outcomeComplete"]]

    def periods(group):
        return {
            (r["events"][0]["input"]["task"]["sku"], r["events"][0]["input"]["task"]["period"])
            for r in group
        }

    forecast_keys = sorted({k for r in records for k in r["forecastMetrics"]})
    initial = [parameter_metrics(r["events"][0]["input"], r["events"][0]["state"]) for r in records]
    return {
        "episodes": len(records),
        "sourceFamilies": len({r["family"] for r in records}),
        "completePeriods": len(complete),
        "censoredPeriods": len(records) - len(complete),
        "uniqueObservedPeriods": len(periods(records)),
        "completeUniquePeriods": len(periods(complete)),
        "initialParameters": {
            k: mean(r[k] for r in initial)
            for k in (
                "parameterAccuracy",
                "stateAccuracy",
                "rawStateAccuracy",
                "rawParameterAccuracy",
                "typeAccuracy",
                "rawTypeAccuracy",
            )
        },
        "finalParameters": {
            k: mean(r["parameters"][k] for r in records)
            for k in (
                "parameterAccuracy",
                "allParametersCorrect",
                "stateAccuracy",
                "rawStateAccuracy",
                "rawParameterAccuracy",
                "typeAccuracy",
                "rawTypeAccuracy",
            )
        },
        "finalEvidenceAccuracy": sum(r["parameters"]["evidenceCorrect"] for r in records)
        / max(1, sum(r["parameters"]["evidenceCount"] for r in records)),
        "meanTotalOnCompletePeriods": mean(r["total"] for r in complete),
        "meanCensoredOrderLossLowerBound": mean(r["censoredOrderLossLowerBound"] for r in records),
        "falseHandoffs": sum(r["falseHandoff"] for r in records),
        "handoffs": sum(r["result"] == "handoff" for r in records),
        "meanInteractions": mean(r["interactions"] for r in records),
        "meanRequestCost": mean(r["requestCost"] for r in records),
        **{
            k: sum(r[k] for r in records)
            for k in (
                "necessaryRequests",
                "unnecessaryRequests",
                "recoveredAfterRequest",
                "unresolvedAfterRequest",
            )
        },
        "missingParametersAtStop": sum(len(r["unresolvedAtStop"]) for r in records),
        "forecast": {
            k: {
                "mean": mean(r["forecastMetrics"].get(k) for r in records),
                "count": sum(r["forecastMetrics"].get(k) is not None for r in records),
            }
            for k in forecast_keys
        },
    }
