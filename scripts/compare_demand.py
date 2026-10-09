"""Compare registered demand repeats on identical Dev origins, including every seed."""

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from newsvendor.io import digest, lines, read, require, write

FAMILIES = ("truncated_normal", "lognormal", "weibull")
METRICS = ("observationNLL", "uncensoredCRPS", "uncensoredOrderLoss_u1",
           "uncensoredOrderLoss_u3", "uncensoredOrderLoss_u9")


def key(row):
    return row["id"], row["cutoff"], row["horizon"]


def interval(groups):
    sums = np.array([sum(groups[k]) for k in sorted(groups)])
    counts = np.array([len(groups[k]) for k in sorted(groups)])
    ids = np.random.default_rng(42).integers(len(sums), size=(5000, len(sums)))
    boot = sums[ids].sum(-1) / counts[ids].sum(-1)
    return {"mean": float(sums.sum() / counts.sum()),
            "familyBootstrap95": np.quantile(boot, [.025, .975]).tolist(),
            "uniqueOrigins": int(counts.sum()), "sourceFamilies": len(sums)}


def diagnosis(rows):
    result = {"familyCounts": dict(Counter(r["prediction"]["distribution"]["family"] for r in rows))}
    for censored in (False, True):
        selected = [r for r in rows if r["censored"] == censored]
        values = np.array([r["normalizedFamilyNLL"] for r in selected])
        result["censored" if censored else "complete"] = {
            "origins": len(selected),
            "meanNormalizedFamilyNLL": dict(zip(FAMILIES, values.mean(0).tolist(), strict=True)),
            "oracleFamilyNLL": float(values.min(-1).mean()),
            "selectedFamilyNLL": float(np.mean([
                r["normalizedFamilyNLL"][FAMILIES.index(r["prediction"]["distribution"]["family"])]
                for r in selected])),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", required=True, help="SEED=run directory")
    parser.add_argument("--reference", action="append", required=True, help="SEED=run directory")
    parser.add_argument("--deployed", required=True, help="Previous model's Dev raw predictions")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    require(not output.exists(), "Preserve prior comparisons")
    paths = {name: {int(s): Path(p) for s, p in (item.split("=", 1) for item in items)}
             for name, items in (("candidate", args.candidate), ("reference", args.reference))}
    require(paths["candidate"].keys() == paths["reference"].keys(), "Unmatched seed sets")
    seeds = sorted(paths["candidate"])
    previous = {key(r): r for r in lines(args.deployed)}
    data, reports, measured = {}, {}, {}
    for name, cohort in paths.items():
        measured[name] = {}
        for seed, path in cohort.items():
            report = read(path / "report.json")
            require(not report["testUsed"], "Only Dev results may enter this analysis")
            condition = report["conditions"]["daily10"]
            for file, expected in condition["rawHashes"].items():
                require(digest((path / "daily10" / file).read_bytes()) == expected, "Changed artifact: " + file)
            rows = lines(path / "daily10/dev.jsonl")
            indexed = {key(r): r for r in rows}
            require(len(indexed) == len(rows) and indexed.keys() == previous.keys(), "Different Dev origins")
            for identity, row in indexed.items():
                require(all(row[k] == previous[identity][k] for k in ("family", "historyHash", "censored", "y", "scale")), "Different observations")
            data[name, seed], reports[name, seed] = indexed, report
            measured[name][str(seed)] = {
                "path": str(path), "reportHash": digest((path / "report.json").read_bytes()),
                "seconds": report["seconds"], "settings": condition["settings"],
                "initialHash": condition["initialHash"], "samples": report["samples"],
                "sourceHash": report["provenance"]["sourceHash"], "metrics": condition["dev"],
                "diagnosis": {str(h): diagnosis([r for r in rows if r["horizon"] == h]) for h in (1, 7)},
            }
    for seed in seeds:
        a, b = (reports[n, seed] for n in ("candidate", "reference"))
        require(a["conditions"]["daily10"]["initialHash"] == b["conditions"]["daily10"]["initialHash"], "Initial GRU weights differ")
        for source in ("src/newsvendor/sequence.py", "src/newsvendor/demand.py"):
            require(a["provenance"]["sources"][source] == b["provenance"]["sources"][source], "Demand implementation changed")
    comparisons = {}
    for name in ("reference", "deployed"):
        comparisons[name] = {}
        for horizon in (1, 7):
            comparisons[name][str(horizon)] = {}
            for metric in METRICS:
                groups, differences = defaultdict(list), defaultdict(list)
                for identity, old in previous.items():
                    if old["horizon"] != horizon or old["metrics"].get(metric) is None:
                        continue
                    repeated = []
                    for seed in seeds:
                        candidate = data["candidate", seed][identity]["metrics"][metric]
                        reference = (old if name == "deployed" else data[name, seed][identity])["metrics"][metric]
                        repeated.append(candidate - reference)
                        differences[seed].append(candidate - reference)
                    groups[old["family"]].append(float(np.mean(repeated)))
                per_seed = {str(s): float(np.mean(v)) for s, v in differences.items()}
                comparisons[name][str(horizon)][metric] = {
                    **interval(groups), "seedMeanDifferences": per_seed,
                    "improvingSeeds": sum(v < 0 for v in per_seed.values()),
                }
    result = {
        "scope": "Dev comparison across all registered seeds on identical observed origins; settings record experimental differences",
        "testUsed": False, "seeds": seeds, "perSeed": measured, "comparisons": comparisons,
        "deployed": {"path": args.deployed, "hash": digest(Path(args.deployed).read_bytes())},
        "limitations": [
            "Losses are averaged across seeds; this is not an ensemble forecast.",
            "Overlapping origins and repeated seeds are not additional independent observations.",
            "Exact losses on uncensored origins do not establish full latent-demand calibration.",
            "Oracle family losses use outcomes for diagnosis only, never for predictions or model inputs.",
            "No new full manager-request rollout is measured here.",
        ],
        "scriptHash": digest(Path(__file__).read_bytes()),
    }
    write(output, result)
    print({name: {h: {m: {k: v[k] for k in ("mean", "improvingSeeds")} for m, v in ms.items()}
                  for h, ms in values.items()} for name, values in comparisons.items()}, flush=True)


if __name__ == "__main__":
    main()
