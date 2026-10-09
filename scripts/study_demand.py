"""Compare a registered observation model with preserved, matched GRU controls."""

import argparse
import os
import sys
import tarfile
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from newsvendor import demand, observations, sequence
from newsvendor.cli import provenance
from newsvendor.io import digest, jsonl, lines, read, require, write
from newsvendor.structured_tool_eval import weights_hash
from newsvendor.structured_train import DEMAND_SCHEMA, dataset_hashes
from newsvendor.suite import check
from newsvendor.train import seed

ROOT = Path(__file__).resolve().parents[1]


def identity(row):
    return row["id"], row["cutoff"], row["horizon"]


@torch.inference_mode()
def evaluate(model, rows, nodes, concentration=1):
    records = []
    model.eval()
    for start in range(0, len(rows), 64):
        batch = rows[start : start + 64]
        output = model([r["sequence"] for r in batch], [r["horizon"] for r in batch], daily=True)
        aggregate = sequence.losses(output, batch)
        joint = sequence.losses(
            output,
            batch,
            {
                "observation": "daily_allocation",
                "observationNodes": nodes,
                "allocationConcentration": concentration,
            },
        )
        for i, row in enumerate(batch):
            scores = output["scores"][i]
            selected = int(scores.argmin())
            family = demand.FAMILIES[selected]
            raw = output["raw"][family][i]
            first, second = demand.parameters(family, raw)
            distribution = {
                "family": family,
                "familyScores": dict(zip(demand.FAMILIES, scores.tolist(), strict=True)),
                "normalizationScale": row["scale"],
                "parameters": {
                    "zeroProbability": min(max(float(raw[0].sigmoid()), 1e-8), 1 - 1e-8),
                    demand.PARAMETERS[family][0]: float(first),
                    demand.PARAMETERS[family][1]: float(second),
                },
            }
            F = demand.pmf(distribution)
            days = row["horizon"]
            prediction = {
                "action": "answer",
                "F": F,
                "distribution": distribution,
                "period": {
                    "start": row["dates"][0],
                    "end": row["dates"][-1],
                    "days": days,
                    "unit": "globally-normalized-sales",
                },
                "orders": [
                    {"underage": u, "overage": 1, **demand.order(F, underage=u)}
                    for u in demand.COST_RATIOS
                ],
            }
            target = {
                "answer": [v * row["scale"] for v in row["dailySales"]],
                "complete": [not v for v in row["dailyCensored"]],
                "dates": row["dates"],
            }
            metrics = demand.metrics(
                {"request": f"Forecast next {days} daily sales"}, target, prediction
            )
            require(
                metrics["distributionValid"] == metrics["orderValid"] == 1, "Invalid demand output"
            )
            # This density uses each exact positive day, so its Jacobian is k*log(scale).
            # It is a different score from the aggregate lower-bound score above.
            exact_positive = sum(
                v > 0 and not c
                for v, c in zip(row["dailySales"], row["dailyCensored"], strict=True)
            )
            metrics["dailyAllocationNLL"] = float(joint[i, selected]) + exact_positive * np.log(
                row["scale"]
            )
            records.append(
                {
                    **{k: v for k, v in row.items() if k != "sequence"},
                    "prediction": prediction,
                    "metrics": metrics,
                    "normalizedFamilyNLL": aggregate[i].tolist(),
                    "normalizedJointFamilyNLL": joint[i].tolist(),
                    "rawParameters": {f: output["raw"][f][i].tolist() for f in demand.FAMILIES},
                    "dailyZeroLogits": {
                        f: float(output["dailyZero"][f][i]) for f in demand.FAMILIES
                    },
                    "allocationConcentration": concentration,
                }
            )
    return records


def summarize(records):
    result = {}
    for horizon in (1, 7):
        values = defaultdict(list)
        for row in records:
            if row["horizon"] == horizon:
                for metric, value in row["metrics"].items():
                    if value is not None:
                        values[metric].append(value)
        result[str(horizon)] = {
            k: {"mean": float(np.mean(v)), "count": len(v)} for k, v in values.items()
        }
    return result


def check_reference(records, previous):
    old = {identity(r): r for r in previous}
    require(len(old) == len(records), "Changed Dev origins")
    for row in records:
        reference = old[identity(row)]
        require(
            all(
                row[k] == reference[k] for k in ("family", "historyHash", "censored", "y", "scale")
            ),
            "Changed Dev observations",
        )
        for key in ("F", "orders", "distribution"):
            require(
                row["prediction"][key] == reference["prediction"][key],
                "Changed reference prediction: " + key,
            )


def control(config, number):
    """Resolve and verify a preserved same-seed control, without retraining it."""
    root = Path(config["reference"])
    uniform = config.get("referenceKind") == "daily_allocation"
    report_path = root / "report.json" if uniform else root / str(number) / "report.json"
    previous = read(report_path)
    entry = previous["runs"][str(number)] if uniform else previous["conditions"]["daily10"]
    path = root / str(number) if uniform else root / str(number) / "daily10"
    settings = previous["config"]["demand"] if uniform else entry["settings"]
    require(
        all(config["demand"][k] == v for k, v in settings.items()),
        "Unmatched training settings",
    )
    for name, expected in entry["rawHashes"].items():
        require(digest((path / name).read_bytes()) == expected, "Changed control artifact")
    seed(number)
    initial_hash = weights_hash(sequence.DemandEncoder().state_dict())
    require(initial_hash == entry["initialHash"], "Control initialization differs")
    return {
        "path": str(path),
        "checkpoint": str(path / ("model.pt" if uniform else "demand.pt")),
        "predictions": str(path / ("candidate.jsonl" if uniform else "dev.jsonl")),
        "dataManifestHash": previous["dataManifestHash"],
        "reportHash": digest(report_path.read_bytes()),
        "sourceHash": previous["provenance"]["sourceHash"],
        "initialHash": initial_hash,
        "reuse": "Historical trained control, reevaluated on unchanged observations; no new baseline training",
    }


def main():
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = read(args.config)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == config["gpu"], "Use the assigned L40S only")
    require(
        torch.cuda.is_available()
        and torch.cuda.device_count() == 1
        and "L40S" in torch.cuda.get_device_name(0),
        "Single visible L40S required",
    )
    torch.set_num_threads(4)
    root = Path(config["output"])
    require(not root.exists(), "Preserve prior studies")
    check(config["dataset"])
    rows = [
        r
        for r in lines(Path(config["dataset"]) / "inputs.jsonl")
        if r["component"] == "retail" and r["split"] in ("train", "dev")
    ]
    ids = {r["id"] for r in rows}
    labels = {
        r["id"]: r["target"]
        for r in lines(Path(config["dataset"]) / "labels.jsonl")
        if r["id"] in ids
    }
    samples = sequence.samples(rows, labels, config["demand"]["minHistory"])
    training = [r for r in samples if r["split"] == "train"]
    development = [r for r in samples if r["split"] == "dev"]
    require(
        not ({r["family"] for r in training} & {r["family"] for r in development}), "Source leakage"
    )
    numerical = read(config["numericalCheck"])
    require(
        not numerical["devUsed"] and not numerical["testUsed"],
        "Train-only quadrature check required",
    )
    require(
        all(v["aggregateLossAndGradientEqual"] for v in numerical["conditions"].values()),
        "Changed control objective",
    )
    allocation_fit = (
        observations.fit_concentration(training)
        if config["demand"].get("allocationConcentration") == "fit"
        else None
    )
    concentration = (
        allocation_fit["concentration"]
        if allocation_fit
        else config["demand"].get("allocationConcentration", 1)
    )
    if concentration > 1:
        require(
            numerical["concentration"] == concentration
            and all(v["uniformLossAndGradientEqual"] for v in numerical["conditions"].values()),
            "Train-fitted concentration and unchanged uniform control must be checked",
        )
    root.mkdir(parents=True)
    started = time.perf_counter()
    report = {
        "config": config,
        "testUsed": False,
        "provenance": provenance(config),
        "scriptHash": digest(Path(__file__).read_bytes()),
        "numericalCheckHash": digest(Path(config["numericalCheck"]).read_bytes()),
        "dataManifestHash": digest(read(Path(config["dataset"]) / "manifest.json")),
        "samples": {
            name: {"forecasts": len(group), "families": sorted({r["family"] for r in group})}
            for name, group in (("train", training), ("dev", development))
        },
        "controls": {},
        "runs": {},
        "scoreConcentration": concentration,
        "allocationFit": allocation_fit,
    }
    for number in config["seeds"]:
        reference = control(config, number)
        require(reference["dataManifestHash"] == report["dataManifestHash"], "Control data changed")
        report["controls"][str(number)] = reference
    write(root / "registered.json", report)
    with tarfile.open(root / "source.tar.gz", "w:gz") as archive:
        for path in sorted(Path("src").rglob("*.py")):
            archive.add(path, arcname=str(path))
        for path in (Path(__file__).relative_to(ROOT), Path(args.config)):
            archive.add(path, arcname=str(path))
    jsonl(
        root / "origins.jsonl", [{k: v for k, v in r.items() if k != "sequence"} for r in samples]
    )
    report["originsHash"] = digest((root / "origins.jsonl").read_bytes())

    class Progress:
        def update(self, stage, **values):
            state = {
                "stage": stage,
                "seed": number,
                "seconds": time.perf_counter() - started,
                **values,
            }
            write(root / "progress.json", state)
            print(state, flush=True)

    try:
        nodes = config["demand"]["observationNodes"]
        for number in config["seeds"]:
            seed(number)
            model = sequence.DemandEncoder().to("cuda")
            initial_hash = weights_hash(model.state_dict())
            stamp = time.perf_counter()
            path = root / str(number)
            path.mkdir()
            trained, records = sequence.cross_fit(
                model, training, development, config["demand"], number, Progress()
            )
            seconds = time.perf_counter() - stamp
            jsonl(path / "crossfit.jsonl", records)
            torch.save(
                {
                    "schema": DEMAND_SCHEMA,
                    "config": {**config, "seed": number},
                    "report": trained,
                    "datasetHashes": dataset_hashes(config),
                    "initialHash": initial_hash,
                    "weights": {
                        "demand." + k: v.detach().cpu() for k, v in model.state_dict().items()
                    },
                },
                path / "model.pt",
            )
            require(trained["allocationConcentration"] == concentration, "Changed full-Train fit")
            candidates = evaluate(model, development, nodes, concentration)
            jsonl(path / "candidate.jsonl", candidates)
            reference = report["controls"][str(number)]
            model.load_state_dict(
                {
                    k.removeprefix("demand."): v
                    for k, v in torch.load(
                        reference["checkpoint"], weights_only=True, map_location="cpu"
                    )["weights"].items()
                }
            )
            controls = evaluate(model, development, nodes, concentration)
            check_reference(controls, lines(reference["predictions"]))
            jsonl(path / "control.jsonl", controls)
            report["runs"][str(number)] = {
                "initialHash": initial_hash,
                "training": trained,
                "trainingSeconds": seconds,
                "candidate": summarize(candidates),
                "control": summarize(controls),
                "controlPredictionsIdentical": True,
                "rawHashes": {
                    name: digest((path / name).read_bytes())
                    for name in ("model.pt", "crossfit.jsonl", "candidate.jsonl", "control.jsonl")
                },
            }
            write(root / "partial.json", report)
            Progress().update("seed_complete", metrics=report["runs"][str(number)]["candidate"])
        model.load_state_dict(
            {
                k.removeprefix("demand."): v
                for k, v in torch.load(
                    config["deployedCheckpoint"], weights_only=True, map_location="cpu"
                )["weights"].items()
                if k.startswith("demand.")
            }
        )
        # The retained reference was measured on CPU. Preserve its inference
        # device and thread count as well as its weights. seed() sets one CPU
        # thread for training, but the historical reference used four.
        torch.set_num_threads(config.get("referenceThreads", 4))
        deployed = evaluate(model.to("cpu"), development, nodes, concentration)
        check_reference(deployed, lines(config["deployedPredictions"]))
        jsonl(root / "deployed.jsonl", deployed)
        report["deployed"] = {
            "metrics": summarize(deployed),
            "device": "cpu",
            "threads": torch.get_num_threads(),
            "predictionsIdentical": True,
            "rawHash": digest((root / "deployed.jsonl").read_bytes()),
            "checkpointHash": digest(Path(config["deployedCheckpoint"]).read_bytes()),
        }
        report["seconds"] = time.perf_counter() - started
        report["peakGpuBytes"] = torch.cuda.max_memory_allocated()
        write(root / "report.json", report)
        write(
            root / "job.json",
            {"stage": "complete", "seconds": report["seconds"], "testUsed": False},
        )
    except BaseException as error:
        write(root / "job.json", {"stage": "failed", "error": str(error)})
        raise


if __name__ == "__main__":
    main()
