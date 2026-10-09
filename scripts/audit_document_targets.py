"""Audit scalar document supervision against source answers without model inference."""

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    from newsvendor.cli import provenance
    from newsvendor.io import digest, jsonl, lines, read, require, write
    from newsvendor.structured_inputs import OPS
    from newsvendor.structured_model import compute
    from newsvendor.structured_train import language_view
    from newsvendor.suite import check, public_input
    from newsvendor.suite_score import exact, rounded_numeric_exact

    config = read(args.config)
    directory = Path(args.output)
    require(not directory.exists(), "Use a new audit directory")
    check(config["dataset"])
    source = Path(config["dataset"])
    rows = [r for r in lines(source / "inputs.jsonl") if r["component"] == "tatqa" and r["split"] in ("train", "dev")]
    observed_hash = digest([public_input(r) for r in rows])
    ids = {r["id"] for r in rows}
    labels = {r["id"]: r["target"] for r in lines(source / "labels.jsonl") if r["id"] in ids}
    tokenizer = AutoTokenizer.from_pretrained(config["encoder"]["model"], revision=config["encoder"]["revision"], cache_dir=".cache/torch-models", local_files_only=True)
    counts = Counter()
    records = []
    for row in rows:
        label = labels[row["id"]]
        counts[row["split"] + ":" + label["answerType"]] += 1
        if label["answerType"] != "arithmetic":
            continue
        view, targets = language_view(("public", row), tokenizer, config, labels, [])
        target = targets["fields"][0]
        item = {"id": row["id"], "split": row["split"], "family": row["family"],
                "inputHash": view["inputHash"], "derivation": label["derivation"], "answer": label["answer"],
                "scale": label["scale"], "supervised": "relation" in target}
        if "relation" in target:
            operation = OPS[target["relation"]]
            values = [view["atoms"][target[k][0]]["value"] for k in ("operand1", "operand2") if k in target]
            calculation = compute(operation, values)
            item.update(operation=operation, values=values, calculation=calculation,
                        strictMatch=exact(calculation, label["answer"]),
                        roundedMatch=rounded_numeric_exact(calculation, label["answer"]))
            counts[row["split"] + ":supervisedArithmetic"] += 1
            counts[row["split"] + ":op:" + operation] += 1
            counts[row["split"] + ":inconsistentArithmetic"] += int(not item["roundedMatch"])
            counts[row["split"] + ":requiresRounding"] += int(not item["strictMatch"])
        else:
            counts[row["split"] + ":maskedArithmetic"] += 1
        records.append(item)
    require(digest([public_input(r) for r in rows]) == observed_hash, "Targets changed observed inputs")
    require(not any(not r["roundedMatch"] for r in records if r["supervised"]), "Inconsistent supervised arithmetic")
    jsonl(directory / "raw.jsonl", records)
    report = {"testUsed": False, "scope": "Source-label consistency, not prediction accuracy",
              "rounding": "Two decimal scalar consistency; not the full official TAT-QA scorer",
              "counts": dict(counts), "observedInputHash": observed_hash, "labelHash": digest(labels),
              "rawHash": digest((directory / "raw.jsonl").read_bytes()), "provenance": provenance(config),
              "masked": [r for r in records if not r["supervised"]]}
    write(directory / "report.json", report)
    print({"counts": dict(counts), "sourceHash": report["provenance"]["sourceHash"]}, flush=True)


if __name__ == "__main__":
    main()
