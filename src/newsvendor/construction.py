import math
import re

import numpy as np

from .corpus import KINDS, SLOTS, STATUSES, outcome
from .encoder import CONSTRAINT, DEMAND, EXTRA
from .io import digest
from .numbers import atoms as prose_atoms
from .optimizer import empirical, minimax


def atoms(doc):
    result = []
    for i, match in enumerate(
        re.finditer(r"([A-Za-z][A-Za-z ]*?)\s*=\s*(-?\d+(?:\.\d+)?)", doc["text"])
    ):
        label, number = match.groups()
        unit = (
            "count"
            if "pack size" in label.lower()
            else "currency/pack"
            if "pack price" in label.lower()
            else "currency/unit"
        )
        result.append(
            {
                "id": f"{doc['id']}:{i}",
                "label": label.strip(),
                "value": float(number),
                "span": list(match.span(2)),
                "unit": unit,
            }
        )
    spans = {tuple(a["span"]) for a in result}
    result.extend(a for a in prose_atoms(doc) if tuple(a["span"]) not in spans)
    return result


def expressions(doc):
    args = atoms(doc)
    result = [
        {
            "id": f"{doc['id']}:copy:{a['id']}",
            "doc": doc,
            "op": "copy",
            "args": [a],
            "value": a["value"],
            "unit": a["unit"],
        }
        for a in args
    ]
    for a in args:
        for b in args:
            if a is b:
                continue
            op, value = None, None
            if a["unit"] == "currency/pack" and b["unit"] == "count" and b["value"] > 0:
                op, value = "divide", a["value"] / b["value"]
            elif a["unit"] == b["unit"] == "currency/unit":
                op, value = "subtract", a["value"] - b["value"]
            if op:
                result.append(
                    {
                        "id": f"{doc['id']}:{op}:{a['id']}:{b['id']}",
                        "doc": doc,
                        "op": op,
                        "args": [a, b],
                        "value": value,
                        "unit": "currency/unit",
                    }
                )
    return result


def validate(expr, slot, input):
    errors, doc = [], expr["doc"]
    for key in ("sku", "period"):
        if doc[key] != input["task"][key]:
            errors.append(key)
    if not doc.get("complete", True):
        errors.append("partial-source")
    if expr["unit"] != "currency/unit":
        errors.append("unit")
    if not math.isfinite(expr["value"]) or expr["value"] < 0:
        errors.append("value")
    if slot == "b" and (doc["role"] != "manager" or "Selected" not in doc["text"]):
        errors.append("unselected-preference")
    if (
        slot == "v"
        and any("handling fee" in a["label"] for a in atoms(doc))
        and expr["op"] != "subtract"
    ):
        errors.append("omitted-fee")
    if (
        slot == "v"
        and expr["op"] == "subtract"
        and "refund" not in expr["args"][0]["label"].lower()
    ):
        errors.append("refund-order")
    return {"accepted": not errors, "errors": errors}


def candidates(input, slot):
    return [
        e for d in input["docs"] for e in expressions(d) if validate(e, slot, input)["accepted"]
    ]


def matches(doc, slot):
    pattern = {
        "c": "purchase quotation",
        "p": "sales price list",
        "v": "return contract",
        "b": "manager decision",
    }.get(slot)
    return bool(pattern and pattern in doc["title"].lower())


def demand(input, record):
    try:
        Fs = [empirical(input["observations"])]
        record["state"]["F"] = "verified"
    except ValueError:
        Fs = input["task"]["allowed"]["F"]
        record["state"]["F"] = "unconfirmed"
        record["errors"].append("censored-demand")
        # A forecast is an estimate, not replacement observations of lost sales.
        forecasts = [
            d
            for d in input["docs"]
            if d["title"] == "Demand forecast"
            and d.get("complete", True)
            and d["sku"] == input["task"]["sku"]
            and d["period"] == input["task"]["period"]
        ]
        if forecasts:
            pairs = re.findall(
                r"^\s*(\d+(?:\.\d+)?)\s*\|\s*(\d+(?:\.\d+)?)\s*$",
                forecasts[-1]["text"],
                re.MULTILINE,
            )
            F = [[float(d), float(p)] for d, p in pairs]
            if F and abs(sum(p for _, p in F) - 1) < 1e-8:
                Fs = [F]
                record["state"]["F"] = "verified"
                record["errors"].remove("censored-demand")
    record["types"]["F"] = "estimate"
    return Fs


def finish(input, record, Fs):
    values, omega = record["values"], []
    if "c" in values and "p" in values:
        for v in [values["v"]] if "v" in values else input["task"]["allowed"]["v"]:
            for b in [values["b"]] if "b" in values else input["task"]["allowed"]["b"]:
                for F in Fs:
                    theta = {"c": values["c"], "p": values["p"], "v": v, "b": b, "F": F}
                    if theta["c"] >= v and theta["p"] - theta["c"] + b >= 0:
                        omega.append(theta)
    fit = minimax(omega, input["task"]["bounds"]) if omega else {"q": None, "gamma": None}
    valid = (
        bool(omega)
        and "censored-demand" not in record["errors"]
        and "conflict" not in record["state"].values()
    )
    return {**record, "omega": omega, **fit, "valid": valid}


def reference(input):
    """Public-evidence rules; never reads hidden episode truth or future observations."""
    record = {k: {} for k in ("values", "links", "state", "types", "expressions")}
    record["errors"] = []
    for slot in SLOTS:
        pool = [e for e in candidates(input, slot) if matches(e["doc"], slot)]
        version = max((e["doc"]["version"] for e in pool), default=0)
        applicable = [e for e in pool if e["doc"]["version"] == version]
        distinct = {e["value"] for e in applicable}
        record["types"][slot] = "preference" if slot == "b" else "fact"
        if len(distinct) == 1:
            expr = applicable[0]
            record["values"][slot], record["links"][slot] = expr["value"], expr["doc"]["id"]
            record["expressions"][slot], record["state"][slot] = expr["id"], "verified"
        elif len(distinct) > 1:
            record["state"][slot] = "conflict"
            record["errors"].append("conflict-" + slot)
        else:
            unavailable = input["task"]["rho"].get(slot, 1) == 0 or any(
                h["action"] == slot and h["answer"] in (None, "no_response", "partial")
                for h in input["history"]
            )
            partial = any(matches(d, slot) and not d.get("complete", True) for d in input["docs"])
            record["state"][slot] = (
                "candidate" if partial else "unavailable" if unavailable else "unconfirmed"
            )
    return finish(input, record, demand(input, record))


def features(cache, input, slot, doc=None, expr=None):
    first = cache.vector(doc["title"] + ": " + doc["text"]) if doc else cache.pool(input)
    second = cache.vector(DEMAND if slot == "F" else CONSTRAINT if slot == "C" else SLOTS[slot])
    x = np.zeros(EXTRA, dtype=np.float32)
    if doc:
        x[:4] = [
            doc["sku"] == input["task"]["sku"],
            doc["period"] == input["task"]["period"],
            doc["version"] / 3,
            doc["role"] == "manager",
        ]
        x[21:24] = [doc["role"] == role for role in ("buyer", "sales", "supplier")]
    if expr:
        x[4:9] = [expr["op"] == op for op in ("divide", "subtract", "copy")] + [
            math.log1p(abs(expr["value"])) / 10,
            expr["unit"] == "currency/unit",
        ]
    x[9] = (
        min(
            3,
            sum(matches(d, slot) and d["period"] == input["task"]["period"] for d in input["docs"]),
        )
        / 3
    )
    x[10] = any(h["action"] == slot for h in input["history"])
    x[24] = first @ second
    x[25 + list((*SLOTS, "F", "C")).index(slot)] = 1
    return np.concatenate((first, second, x))


def training_rows(episodes, cache, transitions=True):
    # Imported here because the public transition traversal also uses construction.
    from .policy import states

    rows = {k: [] for k in ("evidence", "type", "state", "relation")}
    inputs = []
    for episode in episodes:
        if transitions:
            snapshots = states(episode["input"], reference)
        else:
            snapshots = [episode["input"]]
            for action in ("v", "b"):
                if episode["input"]["task"]["partial"].get(action, 0):
                    snapshots.append(outcome(episode["input"], action, "partial"))
        inputs.extend((episode, snapshot) for snapshot in snapshots)
    for episode, input in inputs:
        source = {"id": episode["id"], "family": episode["family"], "stateHash": digest(input)}
        ref = reference(input)
        ref["types"]["C"], ref["state"]["C"] = "assumption", "verified"
        for slot in (*SLOTS, "F", "C"):
            x = features(cache, input, slot)
            rows["type"].append({**source, "x": x, "y": KINDS.index(ref["types"][slot])})
            rows["state"].append({**source, "x": x, "y": STATUSES.index(ref["state"][slot])})
            if slot in ("F", "C"):
                continue
            for doc in input["docs"]:
                rows["evidence"].append(
                    {
                        **source,
                        "x": features(cache, input, slot, doc),
                        "y": int(doc["id"] == ref["links"].get(slot)),
                    }
                )
            cs = candidates(input, slot)
            target = next(
                (i for i, c in enumerate(cs) if c["id"] == ref["expressions"].get(slot)), None
            )
            if target is not None:
                rows["relation"].append(
                    {
                        **source,
                        "xs": [features(cache, input, slot, c["doc"], c) for c in cs],
                        "y": target,
                        "values": [c["value"] for c in cs],
                        "value": ref["values"][slot],
                        "scale": max(1, ref["values"][slot]),
                    }
                )
    return rows


def construct(input, model=None, cache=None, ablation=None):
    if model is None:
        return reference(input)
    record = {k: {} for k in ("values", "links", "state", "types", "expressions", "probabilities")}
    record["errors"] = []
    ref = reference(input)
    for slot in SLOTS:
        x = features(cache, input, slot)
        kind = model["type"].probabilities(x, model["temperatures"]["type"]["temperature"])
        state = model["state"].probabilities(x, model["temperatures"]["state"]["temperature"])
        record["types"][slot] = "fact" if ablation == "type" else KINDS[int(kind.argmax())]
        record["state"][slot] = STATUSES[int(state.argmax())]
        record["probabilities"][slot] = {"type": kind.tolist(), "state": state.tolist()}
        # A known contradiction must survive low confidence or rejected evidence.
        if ref["state"][slot] == "conflict":
            record["state"][slot] = "conflict"
            record["errors"].append("conflict-" + slot)
            continue
        cs = candidates(input, slot)
        docs = {c["doc"]["id"]: c["doc"] for c in cs}
        scores = [
            (
                0
                if ablation == "evidence"
                else model["evidence"].probabilities(
                    features(cache, input, slot, doc),
                    model["temperatures"]["evidence"]["temperature"],
                )[1],
                doc,
            )
            for doc in docs.values()
        ]
        scores.sort(key=lambda item: (-item[0], item[1]["id"]))
        if not scores or ablation != "evidence" and scores[0][0] < model.get("threshold", 0.5):
            record["state"][slot] = (
                ref["state"][slot]
                if ref["state"][slot] in ("candidate", "unavailable")
                else "unconfirmed"
            )
            continue
        choices = [c for c in cs if c["doc"]["id"] == scores[0][1]["id"]]
        logits = (
            model["relation"]
            .scores([features(cache, input, slot, c["doc"], c) for c in choices])
            .ravel()
        )
        selected = choices[int(logits.argmax())]
        if slot == "b" and record["types"][slot] != "preference":
            record["state"][slot] = "unconfirmed"
            record["errors"].append("preference-type")
            continue
        record["values"][slot], record["links"][slot] = selected["value"], selected["doc"]["id"]
        record["expressions"][slot] = selected["id"]
        record["state"][slot] = "verified"
    return finish(input, record, demand(input, record))
