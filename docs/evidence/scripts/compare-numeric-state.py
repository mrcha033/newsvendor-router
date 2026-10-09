"""Audit the registered Train/Dev policy comparison; never load Test examples."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from newsvendor.io import digest, lines, read, require, write
from newsvendor.structured_tool_eval import weights_hash


def paired(left, right):
    require([r["id"] for r in left] == [r["id"] for r in right], "Different Dev cases")
    groups = defaultdict(list)
    changed = []
    for a, b in zip(left, right, strict=True):
        require(a["split"] == b["split"] == "dev", "Expected Dev")
        require(a["family"] == b["family"], "Different source family")
        groups[a["family"]].append(a["total"] - b["total"])
        if [e["action"] for e in a["events"]] != [e["action"] for e in b["events"]]:
            changed.append(
                {
                    "id": a["id"],
                    "family": a["family"],
                    "leftTotal": a["total"],
                    "rightTotal": b["total"],
                    "leftActions": [e["action"] for e in a["events"]],
                    "rightActions": [e["action"] for e in b["events"]],
                }
            )
    values = np.array([np.mean(v) for v in groups.values()])
    require(len({len(v) for v in groups.values()}) == 1, "Unequal family sizes")
    rng = np.random.default_rng(53)
    boot = values[rng.integers(len(values), size=(5000, len(values)))].mean(1)
    return {
        "meanLeftMinusRight": float(values.mean()),
        "familyBootstrap95CI": np.quantile(boot, [0.025, 0.975]).tolist(),
        "families": len(values),
        "resamples": 5000,
        "seed": 53,
        "byFamily": {k: float(np.mean(v)) for k, v in groups.items()},
        "changedActionCases": changed,
        "scope": "Descriptive paired Dev interval after model selection, not confirmatory evidence",
    }


def decisions(rows):
    return [
        {
            **{k: r[k] for k in ("id", "result", "q", "total", "requestCost")},
            "actions": [e["action"] for e in r["events"]],
        }
        for r in rows
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/l40s-research-numeric-v1")
    parser.add_argument("--output", default="results/research-checks/numeric-state-results.json")
    args = parser.parse_args()
    root = Path(args.root)
    require(read(root / "queue.json")["stage"] == "complete", "Incomplete registered conditions")
    registered = read(root / "registered.json")
    parent_path = Path(registered["conditions"]["control"]["warmStart"])
    parent = torch.load(parent_path, map_location="cpu", weights_only=True)["weights"]

    def fixed(weights):
        return {
            k: v
            for k, v in weights.items()
            if not k.startswith(("heads.value.", "heads.recovery."))
        }

    fixed_hash = weights_hash(fixed(parent))
    result = {
        "scope": "Same pretrained constructor/GRU, policy-only continuation with or without numeric state",
        "testUsed": False,
        "registration": registered,
        "parentCheckpoint": {"path": str(parent_path), "sha256": digest(parent_path.read_bytes())},
        "fixedWeightsHash": fixed_hash,
        "scriptHash": digest(Path(__file__).read_bytes()),
        "conditions": {},
    }
    raw = {}
    reference_manifest = read(root / "control/42/manifest.json")
    for name in ("control", "value", "no_value"):
        directory = root / name / "42"
        report, run = read(directory / "report.json"), read(directory / "run.json")
        expected_config = {
            **registered["conditions"][name],
            "linkedManifest": str(directory / "manifest.json"),
        }
        require(report["config"] == expected_config, "Registered config changed")
        require(
            run["provenance"]["sourceHash"] == registered["source"]["sourceHash"], "Source changed"
        )
        require(read(directory / "manifest.json") == reference_manifest, "Data changed")
        require(
            report["parent"]["fileHash"] == result["parentCheckpoint"]["sha256"], "Parent changed"
        )
        weights = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)[
            "weights"
        ]
        initial = torch.load(directory / "policy-base.pt", map_location="cpu", weights_only=True)[
            "weights"
        ]
        require(
            weights_hash(fixed(weights)) == weights_hash(fixed(initial)) == fixed_hash,
            "Fixed weights changed",
        )
        require(
            all(torch.equal(fixed(weights)[k], v) for k, v in fixed(parent).items()),
            "Constructor changed",
        )
        for key, weight in initial.items():
            old = parent[key]
            if old.shape == weight.shape:
                require(torch.equal(old, weight), "Warm start changed existing weights")
            else:
                require(
                    key in ("heads.value.layers.0.weight", "heads.recovery.layers.0.weight"),
                    "Unexpected new shape",
                )
                require(
                    torch.equal(old, weight[:, :256]) and weight[:, 256:].count_nonzero() == 0,
                    "Numeric initialization changed legacy function",
                )
        head = "recovery" if name == "no_value" else "value"
        changed = [k for k in weights if not torch.equal(weights[k], initial[k])]
        require(all(k.startswith(f"heads.{head}.") for k in changed), "Unexpected trained weights")
        columns = {
            h: float(weights[f"heads.{h}.layers.0.weight"][:, 256:].norm())
            for h in ("value", "recovery")
        }
        raw[name] = lines(directory / "final-generated-dev-learned.jsonl")
        artifacts = {}
        for pattern in ("*.json", "*.jsonl", "model.pt", "policy-*.pt", "source.tar.gz"):
            for path in directory.glob(pattern):
                artifacts[path.name] = {
                    "sha256": digest(path.read_bytes()),
                    "bytes": path.stat().st_size,
                }
        candidates = []
        for path in sorted(directory.glob("policy-candidate-*.pt")):
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            candidates.append(
                {
                    "path": path.name,
                    "head": checkpoint["state"]["head"],
                    "dev": checkpoint["state"]["dev"],
                    "numericColumnNorm": float(
                        checkpoint["weights"]["layers.0.weight"][:, 256:].norm()
                    ),
                    "optimizerMaxStep": max(
                        float(v["step"]) for v in checkpoint["optimizer"]["state"].values()
                    ),
                }
            )
        result["conditions"][name] = {
            "parameters": sum(v.numel() for v in weights.values()),
            "seconds": report["seconds"],
            "peakGpuBytes": report["peakGpuBytes"],
            "hardware": run["hardware"],
            "runProvenance": {
                key: run["provenance"][key]
                for key in (
                    "timestamp",
                    "packages",
                    "sourceHash",
                    "configHash",
                    "lockHash",
                    "python",
                    "commit",
                )
            },
            "initialGeneratedDev": report["initialGeneratedDev"],
            "initialRetailDev": report["initialDev"],
            "generatedDev": report["finalGeneratedDev"],
            "retailDev": report["finalDev"],
            "policy": report["policy"],
            "candidates": candidates,
            "changedParameters": changed,
            "numericColumnNorm": columns,
            "fixedWeightsUnchanged": True,
            "initialWeightsMatchLegacyWithZeroExtension": True,
            "artifacts": artifacts,
        }
        del weights, initial
    result["manifestsIdentical"] = True
    result["manifest"] = reference_manifest
    result["pairedGeneratedDev"] = {
        "numericMinusControl": paired(raw["value"], raw["control"]),
        "numericMinusNoValue": paired(raw["value"], raw["no_value"]),
    }
    result["initialNumericAndControlDecisionsIdentical"] = all(
        decisions(lines(root / "control/42" / name)) == decisions(lines(root / "value/42" / name))
        for name in ("initial-dev-learned.jsonl", "initial-generated-dev-learned.jsonl")
    )
    require(result["initialNumericAndControlDecisionsIdentical"], "Initial policy changed")
    target_files = [root / name / "42/value-targets-0.jsonl" for name in ("control", "value")]
    result["initialValueTrainingTargetsIdentical"] = lines(target_files[0]) == lines(
        target_files[1]
    )
    control, candidate = (result["conditions"][n] for n in ("control", "value"))
    a, b = (c["generatedDev"]["learned"] for c in (control, candidate))
    c, d = (c["retailDev"]["learned"] for c in (control, candidate))
    result["promotionGates"] = {
        "generatedRawNotWorse": b["meanTotal"] <= 147.6812873 + 1e-6,
        "retailRawNotWorse": d["meanTotalOnCompletePeriods"] <= 44.3274213 + 1e-6,
        "noFalseHandoffs": b["falseHandoffs"] == d["falseHandoffs"] == 0,
        "noMoreUnnecessaryRequests": b["resolvedFieldRequests"] <= a["resolvedFieldRequests"]
        and d["unnecessaryRequests"] <= c["unnecessaryRequests"],
        "constructorAndDemandUnchanged": True,
        "meaningfulDevImprovement": b["meanTotal"] < a["meanTotal"] - 1e-6
        or d["meanTotalOnCompletePeriods"] < c["meanTotalOnCompletePeriods"] - 1e-6,
    }
    result["eligibleForFurtherValidation"] = all(result["promotionGates"].values())
    result["limitations"] = [
        "One fixed training seed; repeated Dev selection is not independent confirmation.",
        "Generated Dev has 12 families. Retail exact loss uses only 5 unique complete periods across 7 controlled conditions each.",
        "This changes action inputs, not document extraction or demand forecasting. It does not establish appropriate demand calibration or final Test goal attainment.",
        "Raw files remain immutable; model weights and underlying expanded training data are not supplied by this report.",
    ]
    write(args.output, result)
    print(
        {
            "output": args.output,
            "eligibleForFurtherValidation": result["eligibleForFurtherValidation"],
            "promotionGates": result["promotionGates"],
        }
    )


if __name__ == "__main__":
    main()
