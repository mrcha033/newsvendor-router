"""Compare a registered state-head candidate with its parent on the original Dev only."""

import argparse
import copy
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from newsvendor import corpus, structured_retail, structured_train
from newsvendor.cli import provenance
from newsvendor.io import digest, jsonl, lines, read, require, write
from newsvendor.structured_metrics import generated_metrics
from newsvendor.structured_policy import public_validation
from newsvendor.structured_rollout import ResearchRouter, rollout
from newsvendor.structured_tool_eval import weights_hash
from newsvendor.suite import check
from newsvendor.train import seed


def state_metrics(records):
    groups, seen = {}, set()
    for row in records:
        for event in row["events"]:
            value = event["input"]
            key = digest([row["family"], value])
            if key in seen:
                continue
            seen.add(key)
            phase = "response" if value["history"] else "initial"
            metrics = structured_retail.parameter_metrics(value, event["state"])
            group = groups.setdefault(phase, {"states": 0, "correctFields": 0, "allCorrect": 0})
            group["states"] += 1
            group["correctFields"] += round(4 * metrics["stateAccuracy"])
            group["allCorrect"] += metrics["stateAccuracy"] == 1
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True)
    args = parser.parse_args()
    plan = read(args.registration)
    require(plan["split"] == "dev" and plan["testUsed"] is False, "Original Dev only")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == plan["gpu"], "Assigned GPU only")
    require(torch.cuda.device_count() == 1 and "L40S" in torch.cuda.get_device_name(0), "L40S only")
    require(provenance(plan)["sourceHash"] == plan["sourceHash"], "Registered source changed")
    for path, expected in plan["files"].items():
        require(digest(Path(path).read_bytes()) == expected, "Registered input changed: " + path)
    root = Path(plan["output"])
    require(not root.exists(), "Preserve prior evaluations")
    root.mkdir(parents=True)
    started = time.perf_counter()
    seed(plan["seed"])
    model, tokenizer, config, _ = structured_train.load(plan["checkpoint"], "cuda")
    patch = torch.load(plan["head"], weights_only=True, map_location="cpu")
    require(patch["parentHash"] == plan["files"][plan["checkpoint"]], "Head parent differs")
    parent_head = copy.deepcopy(model.heads["state"].state_dict())
    fixed = weights_hash(
        {k: v for k, v in model.state_dict().items() if not k.startswith("heads.state.")}
    )
    check(config["dataset"])
    source = [r for r in lines(Path(config["dataset"]) / "inputs.jsonl") if r["split"] == "dev"]
    rows = [r for r in source if r["component"] == "retail"]
    documents = [r for r in source if r["component"] in config["documentComponents"]]
    collection = lines(Path(config["dataset"]) / "collection.jsonl")
    retail = structured_retail.cases(rows, config["dataSeed"], config["cutoffStride"])
    generated = [e for e in corpus.generate(read(config["researchConfig"])) if e["split"] == "dev"]
    fit_families = set(read(plan["studyReport"])["partitions"]["fit"]["families"])
    fit_families.update(read(plan["studyReport"])["partitions"]["held"]["families"])
    require(
        not fit_families & {e["family"] for e in retail + generated}, "Train/Dev source overlap"
    )
    for name, episodes in (("retail", retail), ("generated", generated)):
        jsonl(
            root / f"{name}-inputs.jsonl",
            [{k: e[k] for k in ("id", "family", "split", "input")} for e in episodes],
        )
    ids = {r["id"] for r in rows + documents}
    labels = {
        r["id"]: r["target"]
        for r in lines(Path(config["dataset"]) / "labels.jsonl")
        if r["id"] in ids
    }
    retail = structured_retail.attach_targets(retail, rows, labels)
    report = {
        "testUsed": False,
        "trainingPerformed": False,
        "provenance": provenance(plan),
        "registration": plan,
        "fixedWeightsHash": fixed,
        "models": {},
        "results": {},
    }
    write(root / "registered.json", report)
    for name, head in (("parent", parent_head), ("candidate", patch["weights"])):
        model.heads["state"].load_state_dict(head, strict=True)
        require(
            weights_hash(
                {k: v for k, v in model.state_dict().items() if not k.startswith("heads.state.")}
            )
            == fixed,
            "Non-state weights changed",
        )
        report["models"][name] = {
            "stateHash": weights_hash(model.heads["state"].state_dict()),
            "fullHash": weights_hash(model.state_dict()),
        }
        router = ResearchRouter(model, tokenizer, config["encoder"], config["noValue"])
        router.cache_states()
        measured = {}
        for benchmark, episodes in (("generated", generated), ("retail", retail)):
            for policy in ("learned", "checklist"):
                records = []
                for i, episode in enumerate(episodes):
                    records.append(rollout(episode, router, explore=policy == "checklist"))
                    if (i + 1) % 64 == 0:
                        print(
                            {
                                "model": name,
                                "benchmark": benchmark,
                                "policy": policy,
                                "processed": i + 1,
                            },
                            flush=True,
                        )
                path = root / f"{name}-{benchmark}-{policy}.jsonl"
                jsonl(path, records)
                metrics = (
                    generated_metrics(records)
                    if benchmark == "generated"
                    else structured_retail.summarize(records)
                )
                measured[benchmark + "-" + policy] = {
                    "metrics": metrics,
                    "states": state_metrics(records),
                    "rawHash": digest(path.read_bytes()),
                }
        measured["documents"] = public_validation(
            model,
            tokenizer,
            config,
            [("public", r) for r in documents],
            labels,
            collection,
            root / f"{name}-documents.jsonl",
        )
        report["results"][name] = measured
        write(root / "report.json", report)
    left, right = [lines(root / f"{name}-documents.jsonl") for name in ("parent", "candidate")]
    require([r["id"] for r in left] == [r["id"] for r in right], "Public Dev identities differ")
    report["publicPredictionsIdentical"] = all(
        a["prediction"] == b["prediction"] for a, b in zip(left, right, strict=True)
    )
    report["seconds"] = time.perf_counter() - started
    report["peakGpuBytes"] = torch.cuda.max_memory_allocated()
    write(root / "report.json", report)
    print(
        {
            "stage": "complete",
            "seconds": report["seconds"],
            "publicPredictionsIdentical": report["publicPredictionsIdentical"],
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
