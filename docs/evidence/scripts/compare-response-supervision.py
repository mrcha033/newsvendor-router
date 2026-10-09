"""Audit the registered Train-only sampling study and its fixed Dev comparisons."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from newsvendor.io import digest, lines, read, require, write
from newsvendor.structured_tool_eval import weights_hash


def paired(left, right):
    require([r["id"] for r in left] == [r["id"] for r in right], "Different Dev cases")
    families = defaultdict(list)
    changed = []
    for a, b in zip(left, right, strict=True):
        require(a["split"] == b["split"] == "dev", "Dev only")
        require(a["family"] == b["family"], "Different families")
        families[a["family"]].append(a["total"] - b["total"])
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
    require(len({len(v) for v in families.values()}) == 1, "Unequal family sizes")
    values = np.array([np.mean(v) for v in families.values()])
    random = np.random.default_rng(53)
    boot = values[random.integers(len(values), size=(5000, len(values)))].mean(1)
    return {
        "meanLeftMinusRight": float(values.mean()),
        "familyBootstrap95CI": np.quantile(boot, [0.025, 0.975]).tolist(),
        "families": len(values),
        "resamples": 5000,
        "seed": 53,
        "byFamily": {k: float(np.mean(v)) for k, v in families.items()},
        "changedActionCases": changed,
        "scope": "Descriptive selected Dev comparison, not independent confirmation",
    }


def target_audit(directory, iteration, expected_samples):
    rows = lines(directory / f"value-targets-{iteration}.jsonl")
    measured = defaultdict(list)
    raw_path = directory / f"rollout-train-{iteration}.jsonl"
    count = 0
    with raw_path.open() as stream:
        for line in stream:
            row = json.loads(line)
            require(row["split"] == "train", "Train target contained held-out data")
            measured[row["id"], row["stateHash"], row["forcedAction"]].append(
                (
                    row["responseSeed"] if expected_samples > 1 else None,
                    [row["terminalLoss"], row["requestCost"]],
                )
            )
            count += 1
    varying = 0
    max_error = 0.0
    for row in rows:
        require(row["split"] == "train", "Train target contained held-out data")
        require("responseSeed" not in row["input"], "Draw seed entered observation")
        for action, target in zip(row["actions"], row["values"], strict=True):
            samples = measured.pop((row["id"], digest(row["input"]), action))
            require(len(samples) == expected_samples, "Wrong sample count")
            if expected_samples > 1:
                require([s[0] for s in samples] == row["responseSeeds"], "Wrong draw identities")
                require(
                    row["observedResponseSeed"] not in row["responseSeeds"], "Behavior draw reused"
                )
            returns = np.array([s[1] for s in samples])
            expected = returns.mean(0) / row["scale"]
            error = float(np.abs(expected - target).max())
            require(error < 1e-12, "Target is not the mean measured return")
            max_error = max(error, max_error)
            varying += int(np.ptp(returns, axis=0).max() > 1e-9)
    require(not measured, "Unmatched measured return")
    return {
        "states": len(rows),
        "episodes": len({r["id"] for r in rows}),
        "families": len({r["family"] for r in rows}),
        "rawRollouts": count,
        "samplesPerAction": expected_samples,
        "actionsWithVariableReturns": varying,
        "meanTargetMaxError": max_error,
        "targetHash": digest((directory / f"value-targets-{iteration}.jsonl").read_bytes()),
        "rawHash": digest(raw_path.read_bytes()),
        "responseSeeds": rows[0].get("responseSeeds"),
    }


def decisions(path):
    return [
        {
            "id": r["id"],
            "total": r["total"],
            "q": r["q"],
            "actions": [e["action"] for e in r["events"]],
        }
        for r in lines(path)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/l40s-research-responses-v1")
    parser.add_argument("--output", default="results/research-checks/train-loss-audit/results.json")
    args = parser.parse_args()
    root = Path(args.root)
    require(read(root / "queue.json")["stage"] == "complete", "Incomplete study")
    registered = read(root / "registered.json")
    parent_path = Path(registered["conditions"]["aligned"]["warmStart"])
    parent = torch.load(parent_path, map_location="cpu", weights_only=True)["weights"]
    fixed = {
        k: v for k, v in parent.items() if not k.startswith(("heads.value.", "heads.recovery."))
    }
    fixed_hash = weights_hash(fixed)
    historical = read("docs/evidence/research-action-precision-results.json")["conditions"]["value"]
    prior_dir = Path(registered["historicalReference"])
    reference = lines(prior_dir / "final-generated-dev-learned.jsonl")
    manifest = read(prior_dir / "manifest.json")
    result = {
        "scope": "Train response-noise alignment and mean sampled return supervision",
        "testUsed": False,
        "registration": registered,
        "parentCheckpoint": {"path": str(parent_path), "sha256": digest(parent_path.read_bytes())},
        "fixedWeightsHash": fixed_hash,
        "scriptHash": digest(Path(__file__).read_bytes()),
        "manifest": manifest,
        "conditions": {},
        "promotionGates": {},
    }
    outputs = {}
    for name in ("aligned", "sampled"):
        directory = root / name / "42"
        report, run = read(directory / "report.json"), read(directory / "run.json")
        require(
            report["config"]
            == registered["conditions"][name]
            | {"linkedManifest": str(directory / "manifest.json")},
            "Config changed",
        )
        require(
            run["provenance"]["sourceHash"] == registered["source"]["sourceHash"], "Source changed"
        )
        require(read(directory / "manifest.json") == manifest, "Data changed")
        require(
            report["parent"]["fileHash"] == result["parentCheckpoint"]["sha256"], "Parent changed"
        )
        weights = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)[
            "weights"
        ]
        initial = torch.load(directory / "policy-base.pt", map_location="cpu", weights_only=True)[
            "weights"
        ]
        for candidate in (weights, initial):
            require(
                all(torch.equal(candidate[k], v) for k, v in fixed.items()),
                "Constructor/GRU changed",
            )
        for key, weight in initial.items():
            old = parent[key]
            if old.shape == weight.shape:
                require(torch.equal(weight, old), "Initial weights changed")
            else:
                require(
                    key in ("heads.value.layers.0.weight", "heads.recovery.layers.0.weight"),
                    "Unexpected shape",
                )
                require(
                    torch.equal(weight[:, :256], old) and weight[:, 256:].count_nonzero() == 0,
                    "Invalid zero extension",
                )
        changed = [k for k in weights if not torch.equal(weights[k], initial[k])]
        require(all(k.startswith("heads.value.") for k in changed), "Unexpected trained parameter")
        require(
            decisions(directory / "policy-dev-initial.jsonl")
            == decisions(prior_dir / "policy-dev-initial.jsonl"),
            "Initial Dev policy changed",
        )
        audits = [
            target_audit(directory, i, report["config"]["policy"]["responseSamples"])
            for i in range(2)
        ]
        if name == "sampled":
            require(
                set(audits[0]["responseSeeds"]).isdisjoint(audits[1]["responseSeeds"]),
                "Iterations reused response draws",
            )
            require(audits[0]["targetHash"] != audits[1]["targetHash"], "Iterations reused targets")
        artifacts = {}
        for pattern in ("*.json", "*.jsonl", "*.pt", "*.log", "source.tar.gz"):
            for path in directory.glob(pattern):
                artifacts[path.name] = {
                    "sha256": digest(path.read_bytes()),
                    "bytes": path.stat().st_size,
                }
        outputs[name] = lines(directory / "final-generated-dev-learned.jsonl")
        condition = {
            "parameters": sum(v.numel() for v in weights.values()),
            "seconds": report["seconds"],
            "peakGpuBytes": report["peakGpuBytes"],
            "hardware": run["hardware"],
            "runProvenance": run["provenance"],
            "generatedDev": report["finalGeneratedDev"],
            "retailDev": report["finalDev"],
            "policy": report["policy"],
            "changedParameters": changed,
            "fixedWeightsUnchanged": True,
            "initialPolicyUnchanged": True,
            "targetAudits": audits,
            "artifacts": artifacts,
            "versusPriorNoise": paired(outputs[name], reference),
        }
        result["conditions"][name] = condition
        a, b = (v["generatedDev"]["learned"] for v in (historical, condition))
        c, d = (v["retailDev"]["learned"] for v in (historical, condition))
        result["promotionGates"][name] = {
            "generatedRawNotWorse": b["meanTotal"] <= a["meanTotal"] + 1e-6,
            "retailRawNotWorse": d["meanTotalOnCompletePeriods"]
            <= c["meanTotalOnCompletePeriods"] + 1e-6,
            "noFalseHandoffs": b["falseHandoffs"] == d["falseHandoffs"] == 0,
            "noUnnecessaryRequests": b["resolvedFieldRequests"] == d["unnecessaryRequests"] == 0,
            "parametersMaintained": b["finalParameters"]["parameterAccuracy"]
            >= a["finalParameters"]["parameterAccuracy"]
            and d["finalParameters"]["parameterAccuracy"]
            >= c["finalParameters"]["parameterAccuracy"],
            "evidenceMaintained": b["finalEvidenceAccuracy"] >= a["finalEvidenceAccuracy"]
            and d["finalEvidenceAccuracy"] >= c["finalEvidenceAccuracy"],
            "meaningfulDevImprovement": b["meanTotal"] < a["meanTotal"] - 1e-6
            or d["meanTotalOnCompletePeriods"] < c["meanTotalOnCompletePeriods"] - 1e-6,
            "constructorAndDemandUnchanged": True,
        }
        del weights, initial
    result["pairedSampledMinusAligned"] = paired(outputs["sampled"], outputs["aligned"])
    result["eligibleForFurtherValidation"] = [
        name for name, gates in result["promotionGates"].items() if all(gates.values())
    ]
    result["limitations"] = [
        "One training seed; repeated Dev selection is not confirmatory evidence.",
        "Generated Dev has 12 families, exact retail Dev losses use five unique observed periods with seven generated financial conditions each.",
        "Eight response samples increase environment work; optimizer update budgets remain fixed. Behavior draws also refresh, so this does not isolate averaging from state resampling.",
        "The optional training mode preserves default historical Dev/Test response behavior. No Test reevaluation was performed.",
        "The study does not fix demand calibration, document generalization, or publication of historical full checkpoints.",
    ]
    write(args.output, result)
    print(
        {
            "eligibleForFurtherValidation": result["eligibleForFurtherValidation"],
            "promotionGates": result["promotionGates"],
        }
    )


if __name__ == "__main__":
    main()
