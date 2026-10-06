"""Run pinned-data, independently initialized ablations without external provider calls."""

import argparse
import random
import statistics
from pathlib import Path

import numpy as np

from newsvendor.cli import experiment, provenance
from newsvendor.io import digest, lines, read, require, write
from newsvendor.report import label


def family_losses(path, policy):
    groups = {}
    for row in lines(path):
        if label(row) == policy:
            groups.setdefault(row["family"], []).append(row["total"])
    require(groups, f"Missing trajectories for {policy}")
    return {family: statistics.mean(values) for family, values in groups.items()}


def contrast(first, second, repetitions=2000):
    require(first.keys() == second.keys(), "Seed sets differ between paired arms")
    seeds = sorted(first)
    families = sorted(first[seeds[0]])
    require(
        all(set(first[s]) == set(second[s]) == set(families) for s in seeds),
        "Source families differ between paired arms",
    )
    differences = [statistics.mean(first[s][f] - second[s][f] for s in seeds) for f in families]
    rand = random.Random(42)
    samples = [
        statistics.mean(rand.choices(differences, k=len(families))) for _ in range(repetitions)
    ]
    return {
        "seeds": seeds,
        "families": len(families),
        "difference": statistics.mean(differences),
        "ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
        "unit": "Source family after averaging paired initialization seeds; seeds are not new sources",
        "seedDifferences": {
            s: statistics.mean(first[s][f] - second[s][f] for f in families) for s in seeds
        },
    }


def summarize(schedule):
    arms, reports, hashes = {}, [], {}
    source_hash = None
    for arm in schedule["arms"]:
        fitted, reference, seeds = {}, {}, []
        for seed in schedule["seeds"]:
            directory = Path(schedule["output"]) / arm["name"] / str(seed)
            if not (directory / "metrics.json").exists():
                continue
            metrics = read(directory / "metrics.json")
            measured_hash = metrics["provenance"]["sourceHash"]
            if source_hash is None:
                source_hash = measured_hash
            require(source_hash == measured_hash, "Study runs have different code snapshots")
            hashes[arm["name"], seed] = (
                metrics["provenance"]["inputHash"],
                metrics["provenance"]["labelHash"],
            )
            fitted[seed] = family_losses(directory / "trajectories.jsonl", "learned/learned")
            reference[seed] = family_losses(directory / "trajectories.jsonl", "reference/rules")
            seeds.append(seed)
        if seeds:
            require(
                len({hashes[arm["name"], s] for s in seeds}) == 1,
                "Initialization seeds changed study data",
            )
            arms[arm["name"]] = fitted
            reports.append(
                {"a": arm["name"], "b": "reference/rules", **contrast(fitted, reference)}
            )
    for a, b in schedule["comparisons"]:
        if a not in arms or b not in arms:
            continue
        shared = sorted(arms[a].keys() & arms[b].keys())
        if not shared:
            continue
        require(
            all(hashes[a, s] == hashes[b, s] for s in shared), "Paired arms have different inputs"
        )
        reports.append(
            {
                "a": a,
                "b": b,
                **contrast({s: arms[a][s] for s in shared}, {s: arms[b][s] for s in shared}),
            }
        )
    write(
        Path(schedule["output"]) / "comparisons.json",
        {"plannedSeeds": schedule["seeds"], "sourceHash": source_hash, "comparisons": reports},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", default="configs/repair-study.json")
    parser.add_argument("--arms", nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    schedule = read(args.schedule)
    names = {arm["name"] for arm in schedule["arms"]}
    require(not args.arms or set(args.arms) <= names, "Unknown study arm")
    require(not args.seeds or set(args.seeds) <= set(schedule["seeds"]), "Unplanned study seed")
    if not args.summarize_only:
        for arm in schedule["arms"]:
            if args.arms and arm["name"] not in args.arms:
                continue
            for seed in args.seeds or schedule["seeds"]:
                config = read(arm["config"])
                config.update(arm.get("overrides", {}))
                require("dataSeed" in config, "Pin dataSeed separately from initialization seed")
                config.update(seed=seed, output=f"{schedule['output']}/{arm['name']}/{seed}")
                config["studyDriverHash"] = digest(Path(__file__).read_bytes())
                directory = Path(config["output"])
                if (directory / "metrics.json").exists():
                    require(
                        args.resume, "Run already exists; use --resume or a new output directory"
                    )
                    saved = read(directory / "metrics.json")
                    require(
                        saved["provenance"]["configHash"] == digest(config), "Resume config changed"
                    )
                    require(
                        saved["provenance"]["sourceHash"] == provenance(config)["sourceHash"],
                        "Resume code changed",
                    )
                    continue
                write(directory / "config.json", config)
                experiment(config)
    summarize(schedule)


if __name__ == "__main__":
    main()
