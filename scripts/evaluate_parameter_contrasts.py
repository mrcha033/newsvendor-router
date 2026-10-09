"""Measure a fixed constructor on held Train document contrasts, separate from Dev selection."""

import argparse
import hashlib
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from newsvendor.cli import provenance
from newsvendor.io import digest, jsonl, lines, read, require, write
from newsvendor.structured_inputs import research_input
from newsvendor.structured_retail import parameter_metrics
from newsvendor.structured_rollout import ResearchRouter
from newsvendor.structured_tool_eval import weights_hash
from newsvendor.structured_train import load


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    plan = read(args.config)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == plan["gpu"], "Assigned GPU only")
    require(torch.cuda.device_count() == 1 and "L40S" in torch.cuda.get_device_name(0), "L40S only")
    require(provenance({})["sourceHash"] == plan["sourceHash"], "Registered source changed")
    root = Path(plan["output"])
    require(not root.exists(), "Preserve prior measurements")
    for path, expected in plan["files"].items():
        require(file_hash(path) == expected, "Registered file changed: " + path)
    manifest = read(plan["manifest"])
    fit = set(manifest["partitions"]["fit"]["families"])
    held = set(manifest["partitions"]["held"]["families"])
    require(not fit & held, "Contrast augmentation source overlap")
    rows = lines(manifest["partitions"]["held"]["path"])
    require(
        all(r["split"] == "train" and r["family"] in held for r in rows),
        "Held Train sources only",
    )
    require(
        all(digest(research_input(r["input"])) == r["inputHash"] for r in rows),
        "Contrast input changed",
    )
    root.mkdir(parents=True)
    write(root / "registered.json", plan)
    torch.set_num_threads(8)
    torch.manual_seed(42)
    started = time.perf_counter()
    report = {
        "scope": "Source-held contrast augmentation diagnostic on previously observed Train; no independent effectiveness or economic comparison",
        "testUsed": False,
        "devUsed": False,
        "cases": len(rows),
        "families": len(held),
        "provenance": provenance({}),
        "registration": plan,
        "models": {},
    }
    seen = {}
    for name, spec in plan["models"].items():
        model, tokenizer, config, _ = load(spec["checkpoint"], "cuda")
        if spec.get("weights"):
            payload = torch.load(spec["weights"], map_location="cpu", weights_only=True, mmap=True)
            model.load_state_dict(payload["weights"], strict=True)
            del payload
        full_hash = weights_hash(model.state_dict())
        key = digest([full_hash, config["encoder"], config["noValue"]])
        if key in seen:
            previous = seen[key]
            shutil.copy2(root / (previous + ".jsonl"), root / (name + ".jsonl"))
            report["models"][name] = report["models"][previous] | {
                "checkpoint": spec,
                "predictionReusedFrom": previous,
            }
            write(root / "report.json", report)
            print({"model": name, "identicalWeightsAndSettings": previous}, flush=True)
            del model, tokenizer
            torch.cuda.empty_cache()
            continue
        seen[key] = name
        router = ResearchRouter(model, tokenizer, config["encoder"], config["noValue"])
        raw, groups = [], defaultdict(lambda: defaultdict(int))
        for index, row in enumerate(rows):
            state = router.construct(row["input"])
            metrics = parameter_metrics(row["input"], state)
            raw.append(
                {k: row[k] for k in ("id", "family", "contrast", "inputHash")}
                | {"state": state, "metrics": metrics}
            )
            for key in ("all", row["contrast"]):
                group = groups[key]
                group["states"] += 1
                group["fields"] += 4
                for output, source in (
                    ("valueErrors", "parameterAccuracy"),
                    ("rawValueErrors", "rawParameterAccuracy"),
                    ("stateErrors", "stateAccuracy"),
                ):
                    group[output] += round(4 * (1 - metrics[source]))
                group["evidenceFields"] += metrics["evidenceCount"]
                group["evidenceErrors"] += metrics["evidenceCount"] - metrics["evidenceCorrect"]
                group["unreadCases"] += state["retrieval"]["unreadChunks"] > 0
            if (index + 1) % 256 == 0:
                print({"model": name, "processed": index + 1, "total": len(rows)}, flush=True)
        path = root / (name + ".jsonl")
        jsonl(path, raw)
        report["models"][name] = {
            "weightsHash": full_hash,
            "groups": dict(groups),
            "rawHash": file_hash(path),
            "checkpoint": spec,
        }
        write(root / "report.json", report)
        print({"model": name, "groups": dict(groups)}, flush=True)
        del model, tokenizer, router
        torch.cuda.empty_cache()
    report["seconds"] = time.perf_counter() - started
    write(root / "report.json", report)


if __name__ == "__main__":
    main()
