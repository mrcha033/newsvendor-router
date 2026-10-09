"""Summarize all registered observation-model repeats without selecting a seed."""

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from compare_demand import interval, key

from newsvendor.io import digest, lines, read, require, write

METRICS = (
    "observationNLL",
    "dailyAllocationNLL",
    "uncensoredCRPS",
    "uncensoredOrderLoss_u1",
    "uncensoredOrderLoss_u3",
    "uncensoredOrderLoss_u9",
    "uncensoredMeanOrderLoss",
)


def records(path):
    rows = lines(path)
    for row in rows:
        values = [row["metrics"][f"uncensoredOrderLoss_u{u}"] for u in (1, 3, 9)]
        row["metrics"]["uncensoredMeanOrderLoss"] = (
            None if values[0] is None else float(np.mean(values))
        )
    indexed = {key(r): r for r in rows}
    require(len(indexed) == len(rows), "Duplicate origins")
    return indexed


def observed_intervals(rows):
    hit, unresolved, width = [], [], []
    for row in rows:
        F = np.asarray(row["prediction"]["F"])
        ids = np.minimum(np.searchsorted(F[:, 1].cumsum(), [0.1, 0.9]), len(F) - 1)
        lower, upper = F[ids, 0]
        observed = row["y"] * row["scale"]
        hit.append(not row["censored"] and lower <= observed <= upper)
        unresolved.append(row["censored"] and observed <= upper)
        width.append(upper - lower)
    return {
        "identifiedCoverageBounds": [
            float(np.mean(hit)),
            float(np.mean(np.asarray(hit) | unresolved)),
        ],
        "meanIntervalWidth": float(np.mean(width)),
        "origins": len(rows),
        "sourceFamilies": len({r["family"] for r in rows}),
        "selectedFamilies": dict(Counter(r["prediction"]["distribution"]["family"] for r in rows)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.study)
    require(not Path(args.output).exists(), "Preserve previous comparisons")
    report = read(root / "report.json")
    require(not report["testUsed"], "Dev-only study required")
    seeds = report["config"]["seeds"]
    require(sorted(map(int, report["runs"])) == sorted(seeds), "Missing registered repeats")
    deployed = records(root / "deployed.jsonl")
    require(
        digest((root / "deployed.jsonl").read_bytes()) == report["deployed"]["rawHash"],
        "Deployed artifact changed",
    )
    data = {}
    for seed in seeds:
        run = report["runs"][str(seed)]
        require(
            run["initialHash"] == report["controls"][str(seed)]["initialHash"],
            "Unmatched initialization",
        )
        for name, expected in run["rawHashes"].items():
            require(
                digest((root / str(seed) / name).read_bytes()) == expected, "Changed raw artifact"
            )
        for name in ("candidate", "control"):
            rows = records(root / str(seed) / f"{name}.jsonl")
            require(rows.keys() == deployed.keys(), "Different origins")
            for identity, row in rows.items():
                require(
                    all(
                        row[k] == deployed[identity][k]
                        for k in (
                            "family",
                            "historyHash",
                            "censored",
                            "y",
                            "scale",
                            "dailySales",
                            "dailyCensored",
                        )
                    ),
                    "Different observed targets",
                )
            data[name, seed] = rows
    comparisons = {}
    for name in ("control", "deployed"):
        comparisons[name] = {}
        for horizon in (1, 7):
            comparisons[name][str(horizon)] = {}
            for metric in METRICS:
                groups, differences = defaultdict(list), defaultdict(list)
                for identity, old in deployed.items():
                    if old["horizon"] != horizon or old["metrics"][metric] is None:
                        continue
                    repeated = []
                    for seed in seeds:
                        candidate = data["candidate", seed][identity]["metrics"][metric]
                        reference = (old if name == "deployed" else data[name, seed][identity])[
                            "metrics"
                        ][metric]
                        repeated.append(candidate - reference)
                        differences[seed].append(candidate - reference)
                    groups[old["family"]].append(float(np.mean(repeated)))
                per_seed = {str(s): float(np.mean(v)) for s, v in differences.items()}
                comparisons[name][str(horizon)][metric] = {
                    **interval(groups),
                    "seedMeanDifferences": per_seed,
                    "improvingSeeds": sum(v < 0 for v in per_seed.values()),
                }
    fixed = str(report["config"]["study"]["candidateSeed"])
    candidate = report["runs"][fixed]["candidate"]
    old = report["deployed"]["metrics"]
    gates = {
        "fixedSeedWeeklyCRPSAndAllOrderRatiosImprove": all(
            candidate["7"][m]["mean"] < old["7"][m]["mean"]
            for m in (
                "uncensoredCRPS",
                "uncensoredOrderLoss_u1",
                "uncensoredOrderLoss_u3",
                "uncensoredOrderLoss_u9",
            )
        ),
        "atLeastTwoSeedsImproveWeeklyCRPSAndMeanOrderLossAgainstBothControls": all(
            comparisons[name]["7"][m]["improvingSeeds"] >= 2
            for name in ("control", "deployed")
            for m in ("uncensoredCRPS", "uncensoredMeanOrderLoss")
        ),
        "fixedSeedDailyCRPSWithinTenPercent": candidate["1"]["uncensoredCRPS"]["mean"]
        <= 1.1 * old["1"]["uncensoredCRPS"]["mean"],
        "fixedSeedWeeklyObservedCoverageDoesNotDecrease": candidate["7"][
            "uncensoredInterval80Coverage"
        ]["mean"]
        >= old["7"]["uncensoredInterval80Coverage"]["mean"],
    }
    result = {
        "scope": "Dev observation-model study, all registered seeds; historical controls reevaluated, not retrained",
        "testUsed": False,
        "study": str(root),
        "reportHash": digest((root / "report.json").read_bytes()),
        "registeredHash": digest((root / "registered.json").read_bytes()),
        "scriptHash": digest(Path(__file__).read_bytes()),
        "sourceHash": report["provenance"]["sourceHash"],
        "seeds": seeds,
        "candidateSeed": int(fixed),
        "perSeed": {
            s: {k: v for k, v in r.items() if k != "training"} for s, r in report["runs"].items()
        },
        "deployed": report["deployed"],
        "comparisons": comparisons,
        "gates": gates,
        "eligibleForCoreRollout": all(gates.values()),
        "coverage": {
            name: {
                str(h): observed_intervals([r for r in group.values() if r["horizon"] == h])
                for h in (1, 7)
            }
            for name, group in (
                {"deployed": deployed} | {f"{n}-{s}": r for (n, s), r in data.items()}
            ).items()
        },
        "limitations": [
            "Rolling periods overlap. Seeds and repeated forecasts are not new independent observations.",
            "Uncensored CRPS/order loss and conditional coverage do not establish full latent-demand calibration.",
            "Aggregate lower-bound and joint daily NLL are different scores and are never subtracted from each other.",
            "Joint daily likelihood assumes uniform positive allocation and noninformative daily right censoring; raw neural parameters are preserved.",
            "This comparison measures emitted F and orders, not a full manager-request rollout.",
        ],
    }
    write(args.output, result)
    print(
        {
            "gates": gates,
            "eligibleForCoreRollout": result["eligibleForCoreRollout"],
            "weeklyDifferencesFromDeployed": {
                m: {k: v[k] for k in ("mean", "improvingSeeds")}
                for m, v in comparisons["deployed"]["7"].items()
            },
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
