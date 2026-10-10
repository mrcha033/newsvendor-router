"""Reproduce recorded parameter-decoder trajectories using the public recovery bundle."""

import argparse
import copy
import gzip
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def signature(record):
    result = {
        k: copy.deepcopy(v)
        for k, v in record.items()
        if k not in ("elapsedMs", "documentCondition", "missingResponses", "kind")
    }
    for event in result["events"]:
        # Compare the actual state, inputs, response, action, q and costs. Cached
        # and direct model decision diagnostics need not be serialized twice.
        event.pop("decision", None)
    return result


def main():
    import torch
    from evaluate_parameters import CONDITIONS, METHODS
    from evaluate_responses import CachedRouter, conditions

    import newsvendor
    from newsvendor import structured_retail
    from newsvendor.bundle import file_hash, load_bundle
    from newsvendor.cli import provenance
    from newsvendor.io import lines, read, require, write
    from newsvendor.structured_rollout import rollout
    from newsvendor.structured_tool_eval import weights_hash

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "records", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--split", choices=("train", "dev"), default="dev")
    args = parser.parse_args()
    require(
        Path(newsvendor.__file__).resolve().is_relative_to(ROOT / "src"),
        "Runtime imported a different source checkout",
    )
    require(not Path(args.output).exists(), "Preserve previous reproduction")
    require(bool(os.environ.get("CUDA_VISIBLE_DEVICES")), "Choose one L40S")
    require(
        torch.cuda.device_count() == 1 and "L40S" in torch.cuda.get_device_name(), "L40S required"
    )
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    root = Path(args.records)
    directory = root / args.split
    report, plan = read(directory / "report.json"), read(directory / "registered.json")
    require(
        provenance({})["sourceHash"] == report["source"]["sourceHash"], "Runtime source changed"
    )
    environment = root / "prepared" / f"{args.split}-environment.jsonl"
    expected_hash = plan["files"][plan["cohorts"][args.split]["environment"]]
    require(file_hash(environment) == expected_hash, "Scoring environment changed")
    targets = {r["id"]: r for r in lines(environment)}
    model, tokenizer, config, manifest = load_bundle(args.bundle, "cuda")
    require(manifest["weightsHash"] == report["weightsHash"], "Parent weights differ")
    model.eval().requires_grad_(False)
    started = time.perf_counter()
    result = {
        "scope": __doc__,
        "split": args.split,
        "testUsed": False,
        "trainingPerformed": False,
        "runtimeSource": str(Path(newsvendor.__file__).resolve().relative_to(ROOT)),
        "selection": "Fixed stride over observed IDs, independent of outcomes; all original retail response masks.",
        "weightsHash": manifest["weightsHash"],
        "conditions": {},
    }
    for condition in ("original", *CONDITIONS):
        path = directory / f"{condition}-inputs.jsonl"
        require(file_hash(path) == report["rawHashes"][path.name], "Recorded inputs changed")
        inputs = lines(path)
        selected = []
        for kind in ("generated", "retail"):
            group = [r for r in inputs if r["kind"] == kind]
            selected.extend(group[:: max(1, len(group) // 7)])
        require(all(r["split"] == args.split for r in selected), "Wrong evaluation partition")
        result["conditions"][condition] = {"ids": [r["id"] for r in selected], "verified": {}}
        ids = {r["id"] for r in selected}
        for name, method in METHODS.items():
            path = directory / f"{condition}-{name}.jsonl.gz"
            expected = {}
            if path.exists():
                require(
                    file_hash(path) == report["rawHashes"][path.name], "Raw measurement changed"
                )
                stream = gzip.open(path, "rt")
            else:
                metadata = read(directory / "raw-format.json")[path.name]
                require(
                    metadata["sourceGzipSha256"] == report["rawHashes"][path.name],
                    "Original measurement hash differs",
                )
                path = directory / metadata["path"]
                require(
                    path.stat().st_size == metadata["bytes"]
                    and file_hash(path) == metadata["sha256"],
                    "Uncompressed raw bytes changed",
                )
                stream = path.open()
            with stream:
                for line in stream:
                    row = json.loads(line)
                    if row["id"] in ids:
                        expected[row["id"], tuple(row["missingResponses"])] = row
            router = CachedRouter(
                model, tokenizer, config["encoder"] | {"parameterDecoding": method}
            )
            count = 0
            for observed in selected:
                episode = copy.deepcopy(observed) | targets[observed["id"]]
                require(episode["input"] == observed["input"], "Scoring changed observed input")
                if observed["kind"] == "retail":
                    episode.update(
                        benchmark=structured_retail.VERSION,
                        cutoffIndex=len(episode["input"]["observations"]),
                    )
                router.reset()
                masks = (
                    conditions(0)
                    if observed["kind"] == "retail" and condition == "original"
                    else [()]
                )
                for missing in masks:
                    options = (
                        {"missing_responses": frozenset(missing)}
                        if observed["kind"] == "retail"
                        else {}
                    )
                    actual = rollout(episode, router, **options)
                    reference = expected[episode["id"], tuple(missing)]
                    require(
                        signature(actual) == signature(reference),
                        f"Trajectory differs: {condition}/{name}/{episode['id']}/{missing}",
                    )
                    count += 1
            result["conditions"][condition]["verified"][name] = count
            write(args.output, result)
    require(
        weights_hash(model.state_dict()) == manifest["weightsHash"], "Reproduction updated weights"
    )
    result.update(seconds=time.perf_counter() - started, allMatch=True)
    write(args.output, result)
    print(
        {
            "allMatch": True,
            "seconds": result["seconds"],
            "paths": sum(sum(v["verified"].values()) for v in result["conditions"].values()),
        }
    )


if __name__ == "__main__":
    main()
