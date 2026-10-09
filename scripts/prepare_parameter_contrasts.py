"""Build Train-only document contrasts without importing hidden outcomes or replies."""

import argparse
import copy
import math
import random
import re
import sys
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from newsvendor.construction import parameter_record
from newsvendor.corpus import answer_doc
from newsvendor.io import digest, jsonl, lines, read, require, write
from newsvendor.structured_inputs import research_input

CONDITIONS = ("wrong_sku", "wrong_period", "older_version", "newer_version", "refund_order")


def alternate_scope(current, key, rng):
    """Use ordinary identifiers/dates, without an 'other' or 'distractor' marker."""
    if key == "sku":
        if re.fullmatch(r"retail:\d+:\d+", current):
            parts = current.split(":")
            parts[-1] = str(int(parts[-1]) + rng.choice([1, 3, 7, 11]))
            return ":".join(parts)
        require(re.fullmatch(r"SKU-[0-9a-f]{10}", current), "Unknown SKU format")
        return "SKU-" + digest([current, rng.random()])[:10]
    require(key == "period", "Unknown applicability field")
    if "/" in current:
        start, end = [date.fromisoformat(s) for s in current.split("/")]
        delta = timedelta(days=rng.choice([-28, -14, -7, 7, 14, 28]))
        return (start + delta).isoformat() + "/" + (end + delta).isoformat()
    require(current == "next-day", "Unknown relative period")
    return rng.choice(["previous-day", "next-week", "previous-week"])


def visible(value):
    """Retain constructor inputs; strip environment response probabilities and cached guesses."""
    keys = ("task", "docs", "observations", "history", "remaining", "historySource")
    result = {k: copy.deepcopy(value[k]) for k in keys if k in value}
    result["task"] = {
        k: result["task"][k]
        for k in (
            "sku",
            "period",
            "bounds",
            "hold",
            "costs",
            "tolerance",
            "deadline",
            "decision",
            "forecast",
            "quantityUnit",
            "allowed",
        )
        if k in result["task"]
    }
    return result


def contrast(row, condition, seed):
    require(row["split"] == "train", "Document contrasts accept Train only")
    require(condition in CONDITIONS, "Unknown document contrast")
    value = visible(row["input"])
    require(not value["history"], "Initial observed inputs only; no invented response history")
    original = parameter_record(value)
    expected = copy.deepcopy(original)
    rng = random.Random(digest([seed, row["family"], research_input(value), condition]))
    if condition == "refund_order":
        if "v" not in original["values"]:
            return None
        doc = next(d for d in value["docs"] if d["id"] == original["links"]["v"])
        fee = round(rng.uniform(0.3, 3), 4)
        refund = original["values"]["v"] + fee
        doc["text"] = (
            f"handling fee per unit = {fee:.12g}; Refund per unit = {refund:.12g}. "
            "Unlimited returns."
        )
    else:
        factor, offset = rng.uniform(0.65, 1.4), rng.uniform(0.2, 2)
        scope_key = {"wrong_sku": "sku", "wrong_period": "period"}.get(condition)
        foreign = alternate_scope(value["task"][scope_key], scope_key, rng) if scope_key else None
        if scope_key:
            require(foreign != value["task"][scope_key], "Foreign scope collision")
        for slot, amount in original["values"].items():
            current = next(d for d in value["docs"] if d["id"] == original["links"][slot])
            amount = round(amount * factor + offset, 4)
            added = answer_doc(value, slot, amount, "contrast-" + slot)
            added["version"] = current["version"] + 1
            if scope_key:
                added[scope_key] = foreign
            elif condition == "older_version":
                added["version"] = 1
                current["version"] = max(2, current["version"])
            else:
                expected["values"][slot] = amount
            value["docs"].append(added)
    # Every document gets an opaque ID and a shuffled position. Neither original IDs
    # nor a fixed distractor prefix may reveal which source supports the answer.
    for index, doc in enumerate(value["docs"]):
        doc["id"] = digest([seed, row["family"], condition, index])[:16]
    rng.shuffle(value["docs"])
    measured = parameter_record(value)
    require(measured["state"] == expected["state"], "Contrast unexpectedly changed field states")
    require(measured["types"] == expected["types"], "Contrast unexpectedly changed field types")
    require(set(measured["values"]) == set(expected["values"]), "Contrast changed observed fields")
    for slot, amount in expected["values"].items():
        require(
            math.isclose(measured["values"][slot], amount, rel_tol=1e-7, abs_tol=1e-7),
            "Document arithmetic differs from independent contrast amounts: " + slot,
        )
    return {k: row[k] for k in ("id", "family", "split")} | {
        "input": value,
        "contrast": condition,
        "originHash": digest(research_input(row["input"])),
        "inputHash": digest(research_input(value)),
    }


def partition(rows, fraction, seed):
    require(0 < fraction < 1, "Invalid held-source fraction")
    require(all(r["split"] == "train" for r in rows), "Document contrasts accept Train only")
    held = set()
    for linked in (False, True):
        families = sorted(
            {r["family"] for r in rows if ("forecast" in r["input"]["task"]) == linked},
            key=lambda f: digest([seed, linked, f]),
        )
        if not families:
            continue
        require(len(families) >= 2, "Need separate contrast fit and held sources")
        count = max(1, min(len(families) - 1, round(fraction * len(families))))
        held.update(families[:count])
    return held


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    plan = read(args.config)
    require(plan.get("version") == 2, "Use contrast config v2; v1 generation code is archived")
    root = Path(plan["output"])
    require(not root.exists(), "Preserve prior contrast captures")
    rows, seen, splits = [], set(), {}
    for path in plan["sources"]:
        require(digest(Path(path).read_bytes()) == plan["files"][path], "Source file changed")
        for row in lines(path):
            require(
                splits.setdefault(row["family"], row["split"]) == row["split"], "Source overlap"
            )
            if row["split"] != "train":
                continue
            key = digest([row["family"], research_input(row["input"])])
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    held = partition(rows, plan["heldFraction"], plan["splitSeed"])
    groups = {"fit": [], "held": []}
    for row in rows:
        destination = groups["held" if row["family"] in held else "fit"]
        for condition in CONDITIONS:
            item = contrast(row, condition, plan["seed"])
            if item is not None:
                destination.append(item)
    require(all(groups.values()), "Empty contrast partition")
    root.mkdir(parents=True)
    report = {
        "scope": "Generated document contrasts on previously observed Train sources; held sources withhold contrast augmentation only, not past encoder training",
        "testUsed": False,
        "devUsed": False,
        "plan": plan,
        "scriptHash": digest(Path(__file__).read_bytes()),
        "sourceFamilies": {
            split: sorted(f for f, s in splits.items() if s == split)
            for split in set(splits.values())
        },
        "partitions": {},
    }
    for name, group in groups.items():
        path = root / (name + ".jsonl")
        jsonl(path, group)
        report["partitions"][name] = {
            "path": str(path),
            "sha256": digest(path.read_bytes()),
            "cases": len(group),
            "families": sorted({r["family"] for r in group}),
            "conditions": dict(Counter(r["contrast"] for r in group)),
        }
    write(root / "manifest.json", report)
    print(
        {
            k: {"cases": v["cases"], "families": len(v["families"])}
            for k, v in report["partitions"].items()
        }
    )


if __name__ == "__main__":
    main()
