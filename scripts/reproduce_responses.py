"""Reproduce paired Dev response measurements using only published inference assets."""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from evaluate_responses import CachedRouter, conditions, file_hash, measurement

from newsvendor import structured_retail
from newsvendor.bundle import load_bundle
from newsvendor.cli import provenance
from newsvendor.io import jsonl, lines, read, require, write
from newsvendor.structured_rollout import rollout
from newsvendor.structured_tool_eval import weights_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--evidence", default="docs/evidence/research-response-state-results.json")
    parser.add_argument("--data", default="results/l40s-response-state-retail-v1")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Check the first fixed case of each scenario; reproduction only",
    )
    args = parser.parse_args()
    output, data = Path(args.output), Path(args.data)
    require(not output.exists(), "Preserve previous reproduction")
    require(
        bool(os.environ.get("CUDA_VISIBLE_DEVICES"))
        and torch.cuda.device_count() == 1
        and "L40S" in torch.cuda.get_device_name(0),
        "Expose one L40S for same-device reproduction",
    )
    evidence = read(args.evidence)
    require(
        provenance({})["sourceHash"] == evidence["source"]["sourceHash"],
        "Use the recorded inference source",
    )
    members = {
        Path(k).name: v
        for k, v in evidence["artifact"]["files"].items()
        if k.startswith(evidence.get("dataPrefix", "results/l40s-response-eval-v1") + "/")
        and Path(k).name
        in {
            "inputs.jsonl",
            "environment.jsonl",
            "policy-heads.pt",
            *[f"{p}-measurements.jsonl" for p in ("value", "checklist", "no_value")],
        }
    }
    require(len(members) == 6, "Missing published reproduction assets")
    for name, entry in members.items():
        require(
            file_hash(data / name) == entry["sha256"], "Published input or policy delta changed"
        )
    inputs = lines(data / "inputs.jsonl")
    require(all(r["split"] == "dev" for r in inputs), "Dev reproduction only")
    if args.smoke:
        selected = {}
        for row in inputs:
            selected.setdefault(row["id"].rsplit(":", 1)[-1], row)
        require(set(selected) == set(structured_retail.SCENARIOS), "Missing reproduction scenario")
        inputs = list(selected.values())
    environment = {r["id"]: r for r in lines(data / "environment.jsonl")}
    episodes = [
        r
        | environment[r["id"]]
        | {"benchmark": structured_retail.VERSION, "cutoffIndex": len(r["input"]["observations"])}
        for r in inputs
    ]
    heads = torch.load(data / "policy-heads.pt", weights_only=True, map_location="cpu")
    require(heads["schema"] == "newsvendor-policy-delta-v1", "Wrong policy delta")
    require(
        all(k.startswith(("heads.value.", "heads.recovery.")) for k in heads["weights"]),
        "Delta changed the constructor or GRU",
    )
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    model, tokenizer, config, manifest = load_bundle(args.bundle, "cuda")
    require(manifest["weightsHash"] == heads["baseWeightsHash"], "Wrong released base model")
    require(config["encoder"] == heads["config"]["encoder"], "Policy architectures differ")
    started = time.perf_counter()
    report = {
        "scope": "Published-assets same-device reproduction, not a new effectiveness evaluation",
        "testUsed": False,
        "trainingPerformed": False,
        "smokeOnly": args.smoke,
        "episodes": [r["id"] for r in inputs],
        "results": {},
    }
    for policy in ("value", "checklist", "no_value"):
        if policy == "no_value":
            model.load_state_dict(model.state_dict() | heads["weights"], strict=True)
            require(
                weights_hash(model.state_dict()) == heads["weightsHash"],
                "Restored no-value tensors differ",
            )
        router = CachedRouter(model, tokenizer, config["encoder"], policy == "no_value")
        expected = {
            (r["key"], tuple(r["missingResponses"])): r
            for r in lines(data / f"{policy}-measurements.jsonl")
        }
        actual, mismatches = [], []
        for episode in episodes:
            router.reset()
            for missing in conditions(0):
                row = rollout(
                    episode,
                    router,
                    explore=policy == "checklist",
                    missing_responses=frozenset(missing),
                )
                measured = measurement(row) | {"missingResponses": list(missing)}
                actual.append(measured)
                if measured != expected[(episode["id"], missing)]:
                    mismatches.append(
                        {
                            "id": episode["id"],
                            "missingResponses": list(missing),
                            "expected": expected[(episode["id"], missing)],
                            "actual": measured,
                        }
                    )
        jsonl(output / f"{policy}-measurements.jsonl", actual)
        report["results"][policy] = {
            "measurements": len(actual),
            "allExactlyEqual": not mismatches,
            "mismatches": mismatches,
        }
        write(output / "partial.json", report)
        require(not mismatches, "Published response measurements differ; raw results preserved")
    report["seconds"] = time.perf_counter() - started
    report["source"] = provenance({})
    report["scriptHash"] = file_hash(__file__)
    write(output / "report.json", report)
    print({"reproduced": True, "episodes": len(episodes), "seconds": report["seconds"]}, flush=True)


if __name__ == "__main__":
    main()
