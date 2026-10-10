"""Reproduce observed-source-stratified paths from the four-condition AI pilot."""

import argparse
import copy
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from compare_ai import load_cases, observation, sha
from evaluate_parameters import documents

from newsvendor.bundle import load_bundle
from newsvendor.conventional import CommonPolicy, Conventional
from newsvendor.io import lines, read, require, write
from newsvendor.structured_rollout import ResearchRouter, rollout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ai-comparison-v1.json")
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    require(not Path(args.output).exists(), "Preserve earlier reproduction")
    require(bool(os.environ.get("CUDA_VISIBLE_DEVICES")), "Select the L40S explicitly")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    require(
        torch.cuda.device_count() == 1 and "L40S" in torch.cuda.get_device_name(),
        "One L40S required",
    )
    plan = read(args.config)
    root = Path(args.records)
    report = read(root / "report.json")
    require(
        sha(root / "outcomes.jsonl") == report["rawHashes"]["outcomes.jsonl"],
        "Reference measurements changed",
    )
    expected = {(r["id"], r["condition"], r["arm"]): r for r in lines(root / "outcomes.jsonl")}
    cases = load_cases(plan)["dev"]
    # First observed ID per source, independent of q, loss, missingness or accuracy.
    selected = {}
    for case in sorted(cases, key=lambda r: r["id"]):
        selected.setdefault(case["family"], case)
    model, tokenizer, config, manifest = load_bundle(args.bundle, "cuda")
    require(manifest["weightsHash"] == plan["weightsHash"], "Weights differ")
    model.eval().requires_grad_(False)
    constructors = {
        0: Conventional(plan["baseline"]),
        1: ResearchRouter(model, tokenizer, config["encoder"] | {"parameterDecoding": "scoped"}),
    }
    verified = []
    for condition in plan["conditions"]:
        for case in selected.values():
            row = copy.deepcopy(case)
            if condition != "original":
                row["input"] = documents(row["input"], condition, plan["seed"])
            constructors[1].cache_states()
            for a in (0, 1):
                for h in (0, 1):
                    arm = f"A{a}H{h}"
                    record = rollout(
                        row, CommonPolicy(constructors[a], bool(h)), missing_responses=frozenset()
                    )
                    actual = observation(row, condition, arm, record)
                    previous = expected[row["id"], condition, arm]
                    require(
                        {k: v for k, v in actual.items() if k != "elapsedMs"}
                        == {k: v for k, v in previous.items() if k != "elapsedMs"},
                        "Reproduced path differs",
                    )
                    verified.append([row["id"], condition, arm])
    result = {
        "allMatch": True,
        "paths": len(verified),
        "families": len(selected),
        "verified": verified,
        "weightsHash": manifest["weightsHash"],
        "configHash": sha(args.config),
        "scriptHash": sha(__file__),
        "testUsed": False,
        "humanParticipants": 0,
        "comparison": "Every outcome, quantity, initial/final constructed state hash and input hash; wall-clock time excluded",
    }
    write(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "verified"}))


if __name__ == "__main__":
    main()
