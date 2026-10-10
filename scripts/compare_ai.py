"""Execute the fixed AI x question pilot on published retail Train/Dev inputs."""

import argparse
import copy
import gzip
import hashlib
import json
import os
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from evaluate_parameters import documents

from newsvendor import structured_retail
from newsvendor.bundle import load_bundle
from newsvendor.cli import provenance
from newsvendor.conventional import CommonPolicy, Conventional
from newsvendor.io import digest, jsonl, read, require, write
from newsvendor.structured_rollout import ResearchRouter, rollout
from newsvendor.structured_tool_eval import weights_hash


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_cases(plan):
    require(sha(plan["inputArchive"]) == plan["archiveSha256"], "Input archive changed")
    evidence = read("docs/evidence/research-parameter-decoding-results.json")["artifacts"][
        "records"
    ]
    result = {}
    with tarfile.open(plan["inputArchive"], "r:xz") as archive:
        for split in ("train", "dev"):
            paths = [
                f"results/parameter-decoding-v2/{split}/original-inputs.jsonl",
                f"results/parameter-decoding-v2/prepared/{split}-environment.jsonl",
            ]
            rows = []
            for path in paths:
                data = archive.extractfile(path).read()
                require(
                    hashlib.sha256(data).hexdigest() == evidence["files"][path]["sha256"],
                    "Member changed",
                )
                rows.append([json.loads(line) for line in data.splitlines()])
            environment = {r["id"]: r for r in rows[1]}
            result[split] = []
            for r in rows[0]:
                if r["kind"] != "retail":
                    continue
                require(r["split"] == split, "Protected split in pilot")
                episode = copy.deepcopy(r) | environment[r["id"]]
                require(episode["input"] == r["input"], "Environment changed observed inputs")
                episode.update(
                    benchmark=structured_retail.VERSION, cutoffIndex=len(r["input"]["observations"])
                )
                result[split].append(episode)
    require(
        not {r["family"] for r in result["train"]} & {r["family"] for r in result["dev"]},
        "Source-group leakage",
    )
    return result


def observation(row, condition, arm, record):
    first, last = record["events"][0], record["events"][-1]
    parameters = structured_retail.parameter_metrics(first["input"], first["state"])
    final = record["parameters"]
    # This observable source condition is defined before either treatment acts.
    selected = any(
        d["role"] == "manager" and "Selected" in d["text"] and d.get("complete", True)
        for d in row["input"]["docs"]
    )
    return {
        "id": row["id"],
        "family": row["family"],
        "split": row["split"],
        "period": row["id"].rsplit(":", 1)[0],
        "condition": condition,
        "stratum": "fixed_preference" if selected else "unselected_preference",
        "arm": arm,
        "A": int(arm[1]),
        "H": int(arm[3]),
        "complete": record["outcomeComplete"],
        "metrics": {
            "total": record["total"] if record["outcomeComplete"] else None,
            "terminalLoss": record["terminalLoss"] if record["outcomeComplete"] else None,
            "requestCost": record["requestCost"],
            "interactions": record["interactions"],
            "parameterAccuracy": final["parameterAccuracy"],
            "initialParameterAccuracy": parameters["parameterAccuracy"],
            "evidenceAccuracy": final["evidenceCorrect"] / max(1, final["evidenceCount"]),
            "hold": int(record["result"] == "hold"),
            "falseHandoff": int(record["falseHandoff"]),
            "unnecessaryRequests": record["unnecessaryRequests"],
        },
        "q": record["q"],
        "initialStateHash": digest(first["state"]),
        "finalStateHash": digest(last["state"]),
        "inputHash": digest(row["input"]),
        "elapsedMs": record["elapsedMs"],
    }


def summarize(rows):
    from newsvendor.factorial import factorial

    return {
        stratum: {
            condition: {
                metric: factorial(
                    [r for r in rows if r["stratum"] == stratum and r["condition"] == condition],
                    metric,
                )
                for metric in rows[0]["metrics"]
            }
            for condition in sorted({r["condition"] for r in rows})
        }
        for stratum in sorted({r["stratum"] for r in rows})
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ai-comparison-v1.json")
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage", choices=("train", "dev"), required=True)
    parser.add_argument("--train-report")
    args = parser.parse_args()
    plan, root = read(args.config), Path(args.output)
    require(not root.exists(), "Preserve previous runs")
    source, script_hash = provenance(plan), sha(__file__)
    if args.stage == "dev":
        pilot = read(args.train_report)
        require(pilot["passed"] and pilot["split"] == "train", "Train validation must pass first")
        require(
            pilot["source"]["sourceHash"] == source["sourceHash"]
            and pilot["configHash"] == sha(args.config),
            "Baseline changed after Train validation",
        )
    cases = load_cases(plan)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    require(bool(os.environ.get("CUDA_VISIBLE_DEVICES")), "Select the L40S explicitly")
    require(
        torch.cuda.device_count() == 1 and "L40S" in torch.cuda.get_device_name(),
        "One L40S required",
    )
    model, tokenizer, config, manifest = load_bundle(args.bundle, "cuda")
    require(manifest["weightsHash"] == plan["weightsHash"], "AI weights changed")
    model.eval().requires_grad_(False)
    ai = ResearchRouter(model, tokenizer, config["encoder"] | {"parameterDecoding": "scoped"})
    conventional = Conventional(plan["baseline"])
    constructors = {0: conventional, 1: ai}
    root.mkdir(parents=True)
    write(root / "registered.json", {"plan": plan, "source": source, "scriptHash": script_hash})
    chosen = cases[args.stage]
    jsonl(
        root / "inputs.jsonl",
        [{k: r[k] for k in ("id", "family", "split", "input")} for r in chosen],
    )
    jsonl(
        root / "environment.jsonl", [{k: v for k, v in r.items() if k != "input"} for r in chosen]
    )
    results, started = [], time.perf_counter()
    conditions = ["original"] if args.stage == "train" else plan["conditions"]
    with gzip.open(root / "trajectories.jsonl.gz", "wt", compresslevel=3) as stream:
        for condition in conditions:
            for index, episode in enumerate(chosen):
                row = copy.deepcopy(episode)
                if condition != "original":
                    row["input"] = documents(row["input"], condition, plan["seed"])
                ai.cache_states()
                for a in (0, 1):
                    for h in (0, 1):
                        arm = f"A{a}H{h}"
                        record = rollout(
                            row,
                            CommonPolicy(constructors[a], bool(h)),
                            missing_responses=frozenset(),
                        )
                        record.update(condition=condition, arm=arm)
                        stream.write(
                            json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n"
                        )
                        results.append(observation(row, condition, arm, record))
                if (index + 1) % 20 == 0:
                    write(
                        root / "progress.json",
                        {
                            "condition": condition,
                            "cases": index + 1,
                            "total": len(chosen),
                            "seconds": time.perf_counter() - started,
                        },
                    )
                    print(json.dumps({"condition": condition, "cases": index + 1}), flush=True)
    require(weights_hash(model.state_dict()) == plan["weightsHash"], "Pilot changed AI weights")
    jsonl(root / "outcomes.jsonl", results)
    report = {
        "status": "exploratory-controlled-pilot",
        "split": args.stage,
        "source": source,
        "scriptHash": script_hash,
        "configHash": sha(args.config),
        "weightsHash": plan["weightsHash"],
        "gpu": torch.cuda.get_device_name(),
        "rows": len(results),
        "cases": len(chosen),
        "families": len({r["family"] for r in chosen}),
        "conditions": conditions,
        "analysis": summarize(results),
        "seconds": time.perf_counter() - started,
        "peakGpuGiB": torch.cuda.max_memory_allocated() / 2**30,
        "rawHashes": {p.name: sha(p) for p in root.iterdir() if p.is_file()},
        "testUsed": False,
        "newSourceHoldoutUsed": False,
        "humanParticipants": 0,
        "trainingPerformed": False,
        "passed": True,
        "validationMeaning": "Inputs, finite outputs, balanced treatments and provenance validated; does not require AI to outperform rules.",
    }
    write(root / "report.json", report)
    print(json.dumps({"rows": len(results), "seconds": report["seconds"], "output": str(root)}))


if __name__ == "__main__":
    main()
