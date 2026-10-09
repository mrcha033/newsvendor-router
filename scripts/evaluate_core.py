"""Evaluate frozen core checkpoints once on registered Test inputs; no training."""

import argparse
import os
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from study_demand import evaluate, summarize

from newsvendor import corpus, sequence, structured_retail, structured_train
from newsvendor.cli import provenance
from newsvendor.io import digest, jsonl, lines, read, require, write
from newsvendor.structured_metrics import generated_metrics
from newsvendor.structured_rollout import ResearchRouter, rollout
from newsvendor.structured_tool_eval import weights_hash
from newsvendor.suite import check

ROOT = Path(__file__).resolve().parents[1]


def main():
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True)
    args = parser.parse_args()
    plan = read(args.registration)
    require(
        plan["split"] == "test" and plan["trainingAllowed"] is False,
        "Frozen Test registration required",
    )
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == plan["gpu"], "Assigned GPU only")
    require(
        torch.cuda.is_available()
        and torch.cuda.device_count() == 1
        and "L40S" in torch.cuda.get_device_name(0),
        "Single L40S required",
    )
    require(
        provenance(plan)["sourceHash"] == plan["sourceHash"],
        "Inference source changed after registration",
    )
    for path, expected in plan["scripts"].items():
        require(
            digest(Path(path).read_bytes()) == expected,
            "Evaluation code changed after registration",
        )
    for entry in plan["models"].values():
        require(
            digest(Path(entry["path"]).read_bytes()) == entry["hash"],
            "Registered checkpoint changed",
        )
    directory = Path(plan["output"])
    require(not directory.exists(), "Preserve prior evaluations")
    torch.set_num_threads(4)
    check(plan["dataset"])
    require(
        digest(read(Path(plan["dataset"]) / "manifest.json")) == plan["dataManifestHash"],
        "Test source changed",
    )
    rows = [
        r
        for r in lines(Path(plan["dataset"]) / "inputs.jsonl")
        if r["split"] == "test" and r["component"] == "retail"
    ]
    retail = structured_retail.cases(rows, plan["dataSeed"], plan["cutoffStride"])
    original = corpus.generate(read(plan["researchConfig"]))
    corpus_sources = corpus.audit(original)
    generated = [e for e in original if e["split"] == "test"]
    directory.mkdir(parents=True)
    for name, episodes in (("retail", retail), ("generated", generated)):
        jsonl(
            directory / f"{name}-inputs.jsonl",
            [{k: e[k] for k in ("id", "family", "split", "input")} for e in episodes],
        )
    input_hashes = {
        name: digest((directory / f"{name}-inputs.jsonl").read_bytes())
        for name in ("retail", "generated")
    }
    # Outcome annotations are opened only after the observed inputs are frozen.
    ids = {r["id"] for r in rows}
    labels = {
        r["id"]: r["target"]
        for r in lines(Path(plan["dataset"]) / "labels.jsonl")
        if r["id"] in ids
    }
    retail = structured_retail.attach_targets(retail, rows, labels)
    demand_rows = sequence.samples(rows, labels, plan["minHistory"])
    require(all(r["split"] == "test" for r in demand_rows), "Non-Test forecast in final evaluation")
    report = {
        "scope": "Frozen core Test evaluation after Train/Dev selection; original generated Test was used in earlier legacy work and is not a new blind Test",
        "testUsed": True,
        "trainingPerformed": False,
        "registration": plan,
        "registrationHash": digest(Path(args.registration).read_bytes()),
        "provenance": provenance(plan),
        "corpusSources": corpus_sources,
        "inputHashes": input_hashes,
        "cases": {
            name: {"episodes": len(group), "families": sorted({e["family"] for e in group})}
            for name, group in (("retail", retail), ("generated", generated))
        },
        "results": {},
        "models": {},
    }
    write(directory / "registered.json", report)
    with tarfile.open(directory / "source.tar.gz", "w:gz") as archive:
        for path in sorted(Path("src").rglob("*.py")):
            archive.add(path, arcname=str(path))
        for path in (*map(Path, plan["scripts"]), Path(args.registration)):
            archive.add(path, arcname=str(path))
    started = time.perf_counter()
    write(directory / "job.json", {"stage": "loading", "pid": os.getpid()})
    frozen_hash = None
    try:
        for name, entry in plan["models"].items():
            model, tokenizer, config, _ = structured_train.load(entry["path"], "cuda")
            require(config["noValue"] == (name == "no_value"), "Wrong checkpoint policy mode")
            fixed = weights_hash(
                {
                    k: v
                    for k, v in model.state_dict().items()
                    if not k.startswith(("heads.value.", "heads.recovery."))
                }
            )
            require(
                frozen_hash is None or fixed == frozen_hash,
                "Policy comparison changed the constructor/GRU",
            )
            frozen_hash = fixed
            report["models"][name] = {
                **entry,
                "fixedWeightsHash": fixed,
                "parameters": sum(p.numel() for p in model.parameters()),
            }
            router = ResearchRouter(model, tokenizer, config["encoder"], config["noValue"])
            router.cache_states()
            for benchmark, episodes in (("generated", generated), ("retail", retail)):
                for policy in (name, "checklist") if name == "value" else (name,):
                    records = []
                    for index, episode in enumerate(episodes):
                        records.append(
                            rollout(episode, router, explore=policy == "checklist", noise=0)
                        )
                        if (index + 1) % 16 == 0:
                            write(
                                directory / "progress.json",
                                {
                                    "benchmark": benchmark,
                                    "policy": policy,
                                    "processed": index + 1,
                                    "total": len(episodes),
                                    "seconds": time.perf_counter() - started,
                                },
                            )
                    path = directory / f"{benchmark}-{policy}.jsonl"
                    jsonl(path, records)
                    metrics = (
                        generated_metrics(records)
                        if benchmark == "generated"
                        else structured_retail.summarize(records)
                    )
                    report["results"][f"{benchmark}-{policy}"] = {
                        **metrics,
                        "rawHash": digest(path.read_bytes()),
                    }
                    write(directory / "partial.json", report)
                    print(
                        {"benchmark": benchmark, "policy": policy, "metrics": metrics}, flush=True
                    )
            if name == "value":
                forecasts = evaluate(model.demand, demand_rows, 64)
                jsonl(directory / "demand.jsonl", forecasts)
                report["demand"] = {
                    "metrics": summarize(forecasts),
                    "rawHash": digest((directory / "demand.jsonl").read_bytes()),
                    "families": len({r["family"] for r in demand_rows}),
                }
                write(directory / "partial.json", report)
            router.cache_states(False)
            del model, tokenizer, router
            torch.cuda.empty_cache()
        for name, expected in input_hashes.items():
            require(
                digest((directory / f"{name}-inputs.jsonl").read_bytes()) == expected,
                "Labels changed observed inputs",
            )
        for entry in plan["models"].values():
            require(
                digest(Path(entry["path"]).read_bytes()) == entry["hash"],
                "Evaluation mutated checkpoint",
            )
        report["seconds"] = time.perf_counter() - started
        report["peakGpuBytes"] = torch.cuda.max_memory_allocated()
        report["legacyGeneratedCriterion"] = {
            "requiredBelow": plan["generatedLossBelow"],
            "observed": report["results"]["generated-value"]["meanTotal"],
            "passed": report["results"]["generated-value"]["meanTotal"]
            < plan["generatedLossBelow"],
        }
        write(directory / "report.json", report)
        write(directory / "job.json", {"stage": "complete", "seconds": report["seconds"]})
    except BaseException as error:
        write(directory / "job.json", {"stage": "failed", "error": str(error)})
        raise


if __name__ == "__main__":
    main()
