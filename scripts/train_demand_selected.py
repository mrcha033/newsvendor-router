"""Registered expanded-source GRU training on the assigned L40S; Train/Dev only."""

import os
import tarfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from newsvendor import demand, sequence
from newsvendor.cli import provenance
from newsvendor.io import digest, jsonl, lines, read, require, write
from newsvendor.structured_tool_eval import weights_hash
from newsvendor.suite import check
from newsvendor.train import seed

torch.set_num_threads(4)
study_seed = int(os.environ["DEMAND_STUDY_SEED"])
require(study_seed in (42, 43, 44), "Registered seed required")
require(os.environ.get("CUDA_VISIBLE_DEVICES") == "GPU-674d64b8-4bdf-7006-1791-5dc7f7245409", "Assigned L40S only")
require(torch.cuda.is_available() and torch.cuda.device_count() == 1 and "L40S" in torch.cuda.get_device_name(0), "Single visible L40S required")
root = Path("results/l40s-demand-selected-v1") / str(study_seed)
require(not root.exists(), "Use a new study directory")
root.mkdir()
started = time.perf_counter()
config = read("configs/l40s-demand-selected-v1.json")
# Fail before feature preparation/training if runtime serialization is unavailable.
probe = root / "runtime-probe.pt"
torch.save({"value": torch.tensor([1.0], device="cuda")}, probe)
require(torch.load(probe, map_location="cpu", weights_only=True)["value"].item() == 1.0, "GPU checkpoint preflight failed")
write(root / "config.json", config)
with tarfile.open(root / "source.tar.gz", "w:gz") as archive:
    for path in sorted(Path("src").rglob("*.py")):
        archive.add(path, arcname=str(path))
    archive.add(Path(__file__), arcname=str(Path(__file__)))
check(config["dataset"])
rows = [r for r in lines(Path(config["dataset"]) / "inputs.jsonl") if r["component"] == "retail" and r["split"] in ("train", "dev")]
ids = {r["id"] for r in rows}
labels = {r["id"]: r["target"] for r in lines(Path(config["dataset"]) / "labels.jsonl") if r["id"] in ids}
samples = sequence.samples(rows, labels)
training = [r for r in samples if r["split"] == "train"]
development = [r for r in samples if r["split"] == "dev"]
require(not ({r["family"] for r in training} & {r["family"] for r in development}), "Source leakage")
report = {"scope": "Same expanded Train data and GRU; nested source-group parameter epoch selection; no outer held labels select their own fit",
          "testUsed": False, "device": "cuda", "pid": os.getpid(), "registeredWeights": [1.0], "studySeed": study_seed,
          "dataManifestHash": digest(read(Path(config["dataset"]) / "manifest.json")),
          "provenance": provenance(config), "scriptHash": digest(Path(__file__).read_bytes()),
          "samples": {s: {"forecasts": sum(r["split"] == s for r in samples), "families": len({r["family"] for r in samples if r["split"] == s})} for s in ("train", "dev")},
          "conditions": {}}
write(root / "registered.json", report)


class Progress:
    def __init__(self, name):
        self.name = name

    def update(self, stage, **values):
        status = {"stage": stage, "condition": self.name, "pid": os.getpid(), "seconds": time.perf_counter() - started, **values}
        write(root / "progress.json", status)
        print(status, flush=True)


@torch.inference_mode()
def evaluate(model):
    records = []
    for start in range(0, len(development), 64):
        batch = development[start:start + 64]
        output = model([r["sequence"] for r in batch], [r["horizon"] for r in batch])
        likelihoods = sequence.losses(output, batch)
        for i, row in enumerate(batch):
            scores = output["scores"][i]
            family = demand.FAMILIES[int(scores.argmin())]
            raw = output["raw"][family][i]
            first, second = demand.parameters(family, raw)
            distribution = {"family": family, "familyScores": dict(zip(demand.FAMILIES, scores.tolist(), strict=True)),
                            "normalizationScale": row["scale"], "parameters": {"zeroProbability": min(max(float(raw[0].sigmoid()), 1e-8), 1 - 1e-8),
                            demand.PARAMETERS[family][0]: float(first), demand.PARAMETERS[family][1]: float(second)}}
            F = demand.pmf(distribution)
            days = row["horizon"]
            target = {"answer": [row["y"] * row["scale"]] + [0.] * (days - 1), "complete": [not row["censored"]] * days, "dates": row["dates"]}
            prediction = {"action": "answer", "F": F, "distribution": distribution,
                          "period": {"start": row["dates"][0], "end": row["dates"][-1], "days": days, "unit": "globally-normalized-sales"},
                          "orders": [{"underage": u, "overage": 1, **demand.order(F, underage=u)} for u in demand.COST_RATIOS]}
            measured = demand.metrics({"request": f"Forecast next {days} daily sales"}, target, prediction)
            require(measured["distributionValid"] == measured["orderValid"] == 1, "Invalid demand output")
            records.append({**{k: row[k] for k in ("id", "family", "cutoff", "dates", "horizon", "historyHash", "censored", "y", "scale")},
                            "prediction": prediction, "metrics": measured, "normalizedFamilyNLL": likelihoods[i].tolist()})
    return records


def fit_with_validation(model, training, validation, settings, random_seed):
    """Select a parameter epoch using only an inner set of Train source groups."""
    require(training and validation, "Empty inner source split")
    require(all(r["split"] == "train" for r in training + validation), "Inner selection is Train only")
    require(not ({r["family"] for r in training} & {r["family"] for r in validation}), "Inner source leakage")
    optimizer = torch.optim.AdamW([p for name, p in model.named_parameters() if not name.startswith("family.")], lr=settings["lr"])
    random = torch.Generator().manual_seed(random_seed)
    curves, best, selected = [], float("inf"), None
    weekly = [r for r in validation if r["horizon"] == 7]
    for epoch in range(settings["epochs"]):
        model.train()
        batches = []
        for ids in torch.randperm(len(training), generator=random).split(settings["batchSize"]):
            batch = [training[i] for i in ids.tolist()]
            out = model([r["sequence"] for r in batch], [r["horizon"] for r in batch])
            values = sequence.losses(out, batch)
            weights = values.new_tensor([settings["dailyWeight"] if r["horizon"] == 1 else 1 for r in batch])
            loss = (values.mean(-1) * weights).sum() / weights.sum()
            require(torch.isfinite(loss).item(), "Nonfinite inner parameter objective")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            batches.append(float(loss.detach()))
        measured = sequence.measure(model, weekly, settings["batchSize"])
        objective = float(measured.mean())
        accepted = objective < best
        if accepted:
            best, selected = objective, epoch + 1
        by_group = defaultdict(list)
        for row, losses in zip(weekly, measured.tolist(), strict=True):
            by_group[row["family"]].append(losses)
        curves.append({"epoch": epoch + 1, "trainLoss": float(np.mean(batches)),
                       "innerWeeklyFamilyNLL": measured.mean(0).tolist(), "innerObjective": objective,
                       "accepted": accepted, "families": {
                           f: {"origins": len(v), "familyNLL": np.mean(v, axis=0).tolist()}
                           for f, v in by_group.items()}})
    require(selected is not None, "No finite inner selection")
    return {"selectedEpoch": selected, "weeklyMeanFamilyNLL": best, "curves": curves,
            "fitFamilies": sorted({r["family"] for r in training}),
            "validationFamilies": sorted({r["family"] for r in validation})}


def nested_params(model, rows, settings, random_seed, initial, progress, label):
    groups = sorted({r["family"] for r in rows}, key=lambda f: digest([random_seed, "inner", f]))
    count = max(1, round(len(groups) * settings["parameterValidationFraction"]))
    require(count < len(groups), "Insufficient inner Train groups")
    held = set(groups[:count])
    model.load_state_dict(initial)
    progress.update("parameter_selection", fold=label, innerFitFamilies=len(groups) - count, innerHeldFamilies=count)
    selection = fit_with_validation(model, [r for r in rows if r["family"] not in held],
                                    [r for r in rows if r["family"] in held], settings, random_seed)
    model.load_state_dict(initial)
    progress.update("parameter_refit", fold=label, selectedEpoch=selection["selectedEpoch"])
    history = sequence.fit_params(model, rows, {**settings, "epochs": selection["selectedEpoch"]}, random_seed)
    return selection, history


@torch.inference_mode()
def states_and_losses(model, rows, batch_size):
    states, values = [], []
    model.eval()
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        output = model([r["sequence"] for r in batch], [r["horizon"] for r in batch])
        states.append(output["state"])
        values.append(sequence.losses(output, batch))
    # Return ordinary tensors so the family head can save them for backward.
    return torch.cat(states).clone(), torch.cat(values).clone()


def cross_fit(model, training, development, settings, random_seed, progress):
    import copy

    from torch.nn import functional as fn

    require(all(r["split"] == "train" for r in training), "Outer fitting requires Train")
    require(all(r["split"] == "dev" for r in development), "Final selection requires Dev")
    train_groups, dev_groups = [{r["family"] for r in group} for group in (training, development)]
    require(not train_groups & dev_groups, "Train/Dev source overlap")
    groups = sorted(train_groups, key=lambda f: digest([random_seed, f]))
    count = settings["folds"]
    folds = {g: i % count for i, g in enumerate(groups)}
    initial = copy.deepcopy(model.state_dict())
    target = torch.zeros(len(training), 3)
    records, reports = [], []
    for fold in range(count):
        fit = [r for r in training if folds[r["family"]] != fold]
        ids = [i for i, r in enumerate(training) if folds[r["family"]] == fold]
        held = [training[i] for i in ids]
        selection, history = nested_params(model, fit, settings, random_seed + fold, initial, progress, fold)
        require(not set(selection["validationFamilies"]) & {r["family"] for r in held}, "Outer labels entered epoch selection")
        measured = sequence.measure(model, held, settings["batchSize"])
        target[ids] = measured
        reports.append({"fold": fold, "fitFamilies": sorted({r["family"] for r in fit}),
                        "heldFamilies": sorted({r["family"] for r in held}), "losses": history,
                        "innerSelection": selection})
        for row, losses in zip(held, measured.tolist(), strict=True):
            records.append({k: row[k] for k in ("id", "family", "cutoff", "dates", "horizon", "historyHash", "censored", "y", "scale")} |
                           {"fold": fold, "familyNLL": losses})
    selection, full = nested_params(model, training, settings, random_seed + count, initial, progress, "full")
    # The encoder and parameter heads are frozen here. Cache their exact states once.
    frozen = weights_hash({k: v for k, v in model.state_dict().items() if not k.startswith("family.")})
    train_states, _ = states_and_losses(model, training, settings["batchSize"])
    dev_states, dev_losses = states_and_losses(model, development, settings["batchSize"])
    train_states = train_states.clone()
    target = target.to(train_states.device)
    weekly = torch.tensor([r["horizon"] == 7 for r in development], device=dev_states.device)
    optimizer = torch.optim.AdamW(model.family.parameters(), lr=settings["lr"])
    random = torch.Generator().manual_seed(random_seed)
    best, saved, selected = float("inf"), None, None
    curves = []
    for epoch in range(settings["selectorEpochs"]):
        current = []
        for ids in torch.randperm(len(training), generator=random).split(settings["batchSize"]):
            objective = fn.smooth_l1_loss(model.family(train_states[ids]), target[ids])
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            optimizer.step()
            current.append(float(objective.detach()))
        with torch.no_grad():
            scores = model.family(dev_states)
            loss = float(dev_losses.gather(1, scores.argmin(-1, keepdim=True)).flatten()[weekly].mean())
        accepted = loss < best
        if accepted:
            best, selected, saved = loss, epoch + 1, copy.deepcopy(model.state_dict())
        curves.append({"epoch": epoch + 1, "trainLoss": float(np.mean(current)), "devWeeklyNLL": loss, "accepted": accepted})
    require(saved is not None, "No selector checkpoint")
    model.load_state_dict(saved)
    model.eval()
    require(weights_hash({k: v for k, v in model.state_dict().items() if not k.startswith("family.")}) == frozen, "Selector changed parameter model")
    return {"folds": reports, "parameterLosses": full, "parameterSelection": selection,
            "selectorLosses": curves, "selectedEpoch": selected, "devSelectedNLL": best,
            "objective": "Nested Train-source epoch selection for each parameter fit; honest outer cross-fit family NLL targets; Dev selects selector epoch only",
            "targetHash": digest(records), "cachedFrozenStates": True}, records


try:
    initial_hash = None
    for name, weight in (("daily10", 1.0),):
        seed(study_seed)
        model = sequence.DemandEncoder().to("cuda")
        initial = weights_hash(model.state_dict())
        require(initial_hash is None or initial == initial_hash, "Initial weights differ")
        initial_hash = initial
        settings = {**config["demand"], "dailyWeight": weight}
        progress = Progress(name)
        progress.update("initialize", settings=settings)
        stamp = time.perf_counter()
        trained, oof = cross_fit(model, training, development, settings, study_seed, progress)
        path = root / name
        path.mkdir()
        jsonl(path / "crossfit.jsonl", oof)
        torch.save({"weights": model.state_dict(), "config": settings, "report": trained,
                    "initialHash": initial, "sourceHash": report["provenance"]["sourceHash"], "dataManifestHash": report["dataManifestHash"]}, path / "demand.pt")
        records = evaluate(model.eval())
        jsonl(path / "dev.jsonl", records)
        summary = {}
        for horizon in (1, 7):
            selected = [r for r in records if r["horizon"] == horizon]
            values = defaultdict(list)
            for r in selected:
                for metric, value in r["metrics"].items():
                    if value is not None:
                        values[metric].append(value)
            summary[str(horizon)] = {k: {"mean": float(np.mean(v)), "count": len(v)} for k, v in values.items()}
        report["conditions"][name] = {"settings": settings, "initialHash": initial, "training": trained,
                                      "dev": summary, "seconds": time.perf_counter() - stamp,
                                      "rawHashes": {file: digest((path / file).read_bytes()) for file in ("crossfit.jsonl", "dev.jsonl", "demand.pt")}}
        write(root / "partial.json", report)
        progress.update("condition_complete", metrics=summary)
    report["seconds"] = time.perf_counter() - started
    report["peakGpuBytes"] = torch.cuda.max_memory_allocated()
    write(root / "report.json", report)
    write(root / "job.json", {"stage": "complete", "pid": os.getpid(), "seconds": report["seconds"], "testUsed": False})
except BaseException as error:
    write(root / "job.json", {"stage": "failed", "pid": os.getpid(), "error": str(error)})
    raise
