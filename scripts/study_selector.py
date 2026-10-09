"""Compare selector objectives on identical frozen demand representations."""

import argparse
import copy
import os
import sys
import tarfile
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from study_demand import check_reference, evaluate, identity, summarize

from newsvendor import demand, sequence
from newsvendor.cli import provenance
from newsvendor.io import digest, jsonl, lines, read, require, write
from newsvendor.structured_tool_eval import weights_hash
from newsvendor.train import seed


@torch.no_grad()
def cache(model, rows, settings, *, measure=False):
    states, losses = [], []
    model.eval()
    for start in range(0, len(rows), 64):
        batch = rows[start : start + 64]
        out = model([r["sequence"] for r in batch], [r["horizon"] for r in batch], daily=measure)
        states.append(out["state"].detach())
        if measure:
            losses.append(sequence.losses(out, batch, settings).detach())
    return torch.cat(states), torch.cat(losses) if measure else None


def score_summary(scores, targets, rows):
    values, scores = targets.cpu().numpy(), scores.detach().cpu().numpy()
    result = {}
    for horizon in (1, 7):
        mask = np.array([r["horizon"] == horizon for r in rows])
        selected = scores[mask].argmin(-1)
        losses = values[mask]
        residual = scores[mask] - losses
        centered = residual - residual.mean(-1, keepdims=True)
        result[str(horizon)] = {
            "forecasts": int(mask.sum()),
            "meanSelectedNLL": float(losses[np.arange(len(losses)), selected].mean()),
            "familyCounts": dict(Counter(demand.FAMILIES[i] for i in selected)),
            "meanFamilyNLL": losses.mean(0).tolist(),
            "huberSaturatedFraction": float((abs(residual) > 1).mean()),
            "centeredMSE": float(np.square(centered).mean()),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = read(args.config)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == config["gpu"], "Use assigned L40S only")
    require(
        torch.cuda.is_available()
        and torch.cuda.device_count() == 1
        and "L40S" in torch.cuda.get_device_name(0),
        "Single visible L40S required",
    )
    output, parent = Path(config["output"]), Path(config["parent"])
    require(not output.exists(), "Preserve previous experiments")
    previous = read(parent / "report.json")
    require(not previous["testUsed"], "Expected Train/Dev parent")
    manifest = read(Path(config["dataset"]) / "manifest.json")
    require(digest(manifest) == previous["dataManifestHash"], "Changed dataset")
    report = {
        "config": config,
        "testUsed": False,
        "provenance": provenance(config),
        "scriptHash": digest(Path(__file__).read_bytes()),
        "parentReportHash": digest((parent / "report.json").read_bytes()),
        "dataManifestHash": digest(manifest),
        "trainDiagnosis": read(config["study"]["trainDiagnosis"]),
        "runs": {},
    }
    output.mkdir(parents=True)
    write(output / "registered.json", report)
    with tarfile.open(output / "source.tar.gz", "w:gz") as archive:
        for path in sorted(Path("src/newsvendor").glob("*.py")):
            archive.add(path, arcname=str(path))
        for path in (Path(__file__), Path("scripts/study_demand.py"), Path(args.config)):
            archive.add(
                path, arcname=str(path.relative_to(Path.cwd()) if path.is_absolute() else path)
            )
    started = time.perf_counter()
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
    samples = sequence.samples(rows, labels, previous["config"]["demand"]["minHistory"])
    training, development = ([r for r in samples if r["split"] == s] for s in ("train", "dev"))
    require(
        not ({r["family"] for r in training} & {r["family"] for r in development}), "Source leakage"
    )
    origins = [{k: v for k, v in r.items() if k != "sequence"} for r in samples]
    require(origins == lines(parent / "origins.jsonl"), "Changed forecast origins")
    jsonl(output / "origins.jsonl", origins)
    report["originsHash"] = digest((output / "origins.jsonl").read_bytes())
    report["samples"] = previous["samples"]
    for number in config["seeds"]:
        seed(number)
        path = output / str(number)
        path.mkdir()
        model = sequence.DemandEncoder().to("cuda")
        require(
            weights_hash(model.state_dict()) == previous["runs"][str(number)]["initialHash"],
            "Changed initialization",
        )
        initial = copy.deepcopy(model.family.state_dict())
        artifact_hashes = previous["runs"][str(number)]["rawHashes"]
        for name in ("model.pt", "crossfit.jsonl", "candidate.jsonl"):
            require(
                digest((parent / str(number) / name).read_bytes()) == artifact_hashes[name],
                "Changed parent artifact: " + name,
            )
        checkpoint = torch.load(
            parent / str(number) / "model.pt", map_location="cpu", weights_only=True
        )
        model.load_state_dict(
            {k.removeprefix("demand."): v for k, v in checkpoint["weights"].items()}
        )
        frozen = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if not k.startswith("family.")
        }
        settings = {
            **previous["config"]["demand"],
            "allocationConcentration": checkpoint["report"]["allocationConcentration"],
        }
        old_predictions = evaluate(
            model, development, settings["observationNodes"], settings["allocationConcentration"]
        )
        check_reference(old_predictions, lines(parent / str(number) / "candidate.jsonl"))
        jsonl(path / "parent.jsonl", old_predictions)
        records = {identity(r): r for r in lines(parent / str(number) / "crossfit.jsonl")}
        require(len(records) == len(training), "Different Train targets")
        folds = checkpoint["report"]["folds"]
        for row in training:
            record = records[identity(row)]
            require(
                all(record[k] == row[k] for k in record if k not in ("fold", "familyNLL")),
                "Changed Train observation",
            )
            fold = folds[record["fold"]]
            require(
                row["family"] in fold["heldFamilies"] and row["family"] not in fold["fitFamilies"],
                "Cross-fit source leakage",
            )
        target = torch.tensor([records[identity(r)]["familyNLL"] for r in training], device="cuda")
        train_states, _ = cache(model, training, settings)
        dev_states, dev_losses = cache(model, development, settings, measure=True)
        weekly = torch.tensor([r["horizon"] == 7 for r in development], device="cuda")
        torch.save(
            {"train": train_states.cpu(), "dev": dev_states.cpu(), "devNLL": dev_losses.cpu()},
            path / "cache.pt",
        )
        with torch.no_grad():
            parent_scores = model.family(train_states)
        item = {
            "parentHashes": artifact_hashes,
            "frozenWeightsHash": weights_hash(frozen),
            "initialSelectorHash": weights_hash(initial),
            "parentReproduced": True,
            "cacheHash": digest((path / "cache.pt").read_bytes()),
            "parentTrainScores": score_summary(parent_scores, target, training),
            "conditions": {},
        }
        torch.save(
            {"scores": parent_scores.cpu(), "targets": target.cpu()}, path / "train-parent.pt"
        )
        for kind in config["conditions"]:
            destination = path / kind
            destination.mkdir()
            model.family.load_state_dict(initial)
            optimizer = torch.optim.AdamW(model.family.parameters(), lr=config["lr"])
            generator = torch.Generator().manual_seed(number)
            history, raw_scores, weights = [], [], []
            best, selected, saved = float("inf"), 0, None
            stamp = time.perf_counter()
            for epoch in range(1, config["epochs"] + 1):
                objectives = []
                for indexes in torch.randperm(len(training), generator=generator).split(
                    config["batchSize"]
                ):
                    indexes = indexes.to("cuda")
                    loss = sequence.selector_loss(
                        model.family(train_states[indexes]), target[indexes], kind
                    )
                    require(torch.isfinite(loss).item(), "Nonfinite selector loss")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    objectives.append(float(loss.detach()))
                with torch.no_grad():
                    scores = model.family(dev_states)
                    chosen = dev_losses.gather(1, scores.argmin(-1, keepdim=True))[weekly].flatten()
                    objective = float(chosen.double().mean())
                history.append(
                    {
                        "epoch": epoch,
                        "trainObjective": float(np.mean(objectives)),
                        "devWeeklyNLL": objective,
                    }
                )
                raw_scores.append(scores.cpu())
                weights.append(
                    {k: v.detach().cpu().clone() for k, v in model.family.state_dict().items()}
                )
                if objective < best:
                    best, selected, saved = (
                        objective,
                        epoch,
                        copy.deepcopy(model.family.state_dict()),
                    )
            torch.cuda.synchronize()
            seconds = time.perf_counter() - stamp
            model.family.load_state_dict(saved)
            with torch.no_grad():
                scores = model.family(train_states)
            require(
                weights_hash(
                    {k: v for k, v in model.state_dict().items() if not k.startswith("family.")}
                )
                == item["frozenWeightsHash"],
                "Frozen demand parameters changed",
            )
            predictions = evaluate(
                model,
                development,
                settings["observationNodes"],
                settings["allocationConcentration"],
            )
            jsonl(destination / "dev.jsonl", predictions)
            torch.save(
                {
                    "scores": scores.cpu(),
                    "devEpochScores": torch.stack(raw_scores),
                    "epochWeights": weights,
                },
                destination / "raw.pt",
            )
            result = {
                "selectedEpoch": selected,
                "devSelectedNLL": best,
                "epochs": history,
                "trainingSeconds": seconds,
                "trainScores": score_summary(scores, target, training),
                "dev": summarize(predictions),
                "selectorLoss": kind,
                "frozenWeightsIdentical": True,
            }
            torch.save(
                {
                    **checkpoint,
                    "config": {
                        **checkpoint["config"],
                        "output": str(destination),
                        "demand": {**checkpoint["config"]["demand"], "selectorLoss": kind},
                    },
                    "report": {
                        **checkpoint["report"],
                        "selectorLoss": kind,
                        "selectedEpoch": selected,
                        "devSelectedNLL": best,
                        "selectorLosses": [row["trainObjective"] for row in history],
                        "selectorRefit": result,
                    },
                    "weights": {
                        "demand." + k: v.detach().cpu() for k, v in model.state_dict().items()
                    },
                },
                destination / "model.pt",
            )
            result["rawHashes"] = {
                name: digest((destination / name).read_bytes())
                for name in ("raw.pt", "model.pt", "dev.jsonl")
            }
            write(destination / "report.json", result)
            item["conditions"][kind] = result
            print(
                {
                    "seed": number,
                    "condition": kind,
                    "seconds": seconds,
                    "selectedEpoch": selected,
                    "dev": result["dev"],
                },
                flush=True,
            )
            write(
                output / "progress.json",
                {"seed": number, "condition": kind, "seconds": time.perf_counter() - started},
            )
        report["runs"][str(number)] = item
        write(output / "partial.json", report)
    report["seconds"] = time.perf_counter() - started
    report["peakGpuBytes"] = torch.cuda.max_memory_allocated()
    write(output / "report.json", report)
    print({"complete": True, "seconds": report["seconds"]}, flush=True)


if __name__ == "__main__":
    main()
