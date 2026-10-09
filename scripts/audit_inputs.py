"""Compare Train evidence coverage and token cost without opening Test annotations."""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", required=True)
    parser.add_argument("--new", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    from newsvendor import corpus
    from newsvendor.cli import provenance
    from newsvendor.io import lines, read, write
    from newsvendor.structured_inputs import prepare
    from newsvendor.structured_labels import suite_targets
    from newsvendor.structured_train import language_view

    old, new = read(args.old), read(args.new)
    backbone = new["backbones"]["base"]
    tokenizer = AutoTokenizer.from_pretrained(
        backbone["model"], revision=backbone["revision"], cache_dir=".cache/torch-models", token=False
    )
    tokenizer.model_max_length = 10**9
    rows = [r for r in lines(Path(new["dataset"]) / "inputs.jsonl") if r["split"] == "train" and r["component"] != "retail"]
    allowed = {r["id"] for r in rows}
    labels = {r["id"]: r["target"] for r in lines(Path(new["dataset"]) / "labels.jsonl") if r["id"] in allowed}
    collection = lines(Path(new["dataset"]) / "collection.jsonl")
    records = []
    groups = defaultdict(list)
    for row in rows:
        measured = {"id": row["id"], "component": row["component"], "family": row["family"], "settings": {}}
        for name, config in (("old", old), ("new", new)):
            rounds = [0] if name == "old" else range(config["maxRetrievals"] + 1)
            for round in rounds:
                view = prepare(row, tokenizer, config["encoder"], collection=collection, round=round)
                target = suite_targets(view, row["component"], labels[row["id"]])
                fields = target["fields"]
                result = {
                    "tokens": view["encodedTokens"],
                    "needsRetrieval": bool(target.get("needsRetrieval")),
                    "spans": sum("spans" in field for field in fields),
                    "operands": sum("operand1" in field for field in fields),
                    "argumentValues": sum("mode" in field for field in fields[1:]),
                }
                measured["settings"][name + ":" + str(round)] = result
                if not result["needsRetrieval"]:
                    break
        records.append(measured)
        groups[row["component"]].append(measured)
    summary = {}
    for component, cases in groups.items():
        summary[component] = {
            "cases": len(cases),
            "oldMeanTokens": sum(c["settings"]["old:0"]["tokens"] for c in cases) / len(cases),
            "newMeanTokens": sum(c["settings"]["new:0"]["tokens"] for c in cases) / len(cases),
            "oldMissingSupport": sum(c["settings"]["old:0"]["needsRetrieval"] for c in cases),
            "newInitialMissingSupport": sum(c["settings"]["new:0"]["needsRetrieval"] for c in cases),
            "newFinalMissingSupport": sum(list(c["settings"].values())[-1]["needsRetrieval"] for c in cases),
            "oldArgumentValues": sum(c["settings"]["old:0"]["argumentValues"] for c in cases),
            "newArgumentValues": sum(c["settings"]["new:0"]["argumentValues"] for c in cases),
            "newFinalArgumentValues": sum(list(c["settings"].values())[-1]["argumentValues"] for c in cases),
        }
    episodes = corpus.generate(read(new["researchConfig"]))
    research = []
    for episode in episodes:
        if episode["split"] != "train":
            continue
        measured = {"id": episode["id"], "family": episode["family"], "settings": {}}
        for name, config in (("old", old), ("new", new)):
            view, target = language_view(("research", episode), tokenizer, config, {}, [], 0)
            measured["settings"][name] = {
                "tokens": view["encodedTokens"],
                "operandFields": [i for i, field in enumerate(target["fields"]) if "operand1" in field],
            }
        research.append(measured)
    missing = [r["id"] for r in research if set(r["settings"]["old"]["operandFields"]) - set(r["settings"]["new"]["operandFields"])]
    write(args.output, {
        "split": "train", "oldConfig": old, "newConfig": new,
        "sourceHash": provenance(new)["sourceHash"], "summary": summary,
        "records": records, "scope": "coverage and compute audit, not effectiveness evidence",
        "researchSplits": corpus.audit(episodes),
        "researchInputs": {"records": research, "missingPreviouslyVisibleOperands": missing},
    })
    print(summary, flush=True)
    if missing:
        raise ValueError("Research operands lost from initial inputs: " + str(missing))


if __name__ == "__main__":
    main()
