"""Fit the existing state head on observed Train rollout states with source-held selection."""

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
from torch.nn import functional as fn

from newsvendor.cli import provenance
from newsvendor.construction import parameter_record
from newsvendor.corpus import SLOTS, STATUSES
from newsvendor.io import digest, jsonl, lines, read, require, write
from newsvendor.structured_inputs import prepare, research_input
from newsvendor.structured_rollout import FIELDS, TEXT
from newsvendor.structured_tool_eval import weights_hash
from newsvendor.structured_train import load
from newsvendor.train import seed


def observed_states(paths, allowed):
    """Deduplicate model-visible states; never use the rollout loss to select examples."""
    result, seen = [], set()
    for path in paths:
        for row in lines(path):
            require(
                row["split"] == "train" and allowed.get(row["id"]) == row["family"],
                "State replay requires a registered Train source",
            )
            for step, event in enumerate(row["events"]):
                value = event["input"]
                observed = research_input(value)
                key = digest([row["family"], observed])
                if key in seen:
                    continue
                seen.add(key)
                reference = parameter_record(value)
                result.append(
                    {
                        "id": row["id"],
                        "family": row["family"],
                        "split": "train",
                        "benchmark": "retail" if "forecast" in value["task"] else "generated",
                        "phase": "response" if value["history"] else "initial",
                        "input": observed,
                        "inputHash": key,
                        "round": sum(h["action"] == "retrieve" for h in value["history"]),
                        "target": [STATUSES.index(reference["state"][s]) for s in SLOTS],
                        "source": {"path": str(path), "step": step},
                    }
                )
    require(result, "No observed Train states")
    return result


def source_partition(rows, fraction, split_seed):
    require(0 < fraction < 1, "Invalid held-source fraction")
    require(all(r["split"] == "train" for r in rows), "Head selection accepts Train only")
    held = set()
    for benchmark in sorted({r["benchmark"] for r in rows}):
        families = sorted(
            {r["family"] for r in rows if r["benchmark"] == benchmark},
            key=lambda f: digest([split_seed, benchmark, f]),
        )
        require(len(families) >= 2, "Head selection needs distinct source families")
        count = max(1, min(len(families) - 1, round(len(families) * fraction)))
        held.update(families[:count])
    fit = [i for i, row in enumerate(rows) if row["family"] not in held]
    valid = [i for i, row in enumerate(rows) if row["family"] in held]
    require(fit and valid, "Empty head selection partition")
    return fit, valid


@torch.inference_mode()
def measure(head, x, y, indices):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = head(x[indices])[:, : len(SLOTS)]
    predictions = logits.argmax(-1)
    truth = y[indices]
    confusion = torch.bincount(
        (truth * len(STATUSES) + predictions).flatten(), minlength=len(STATUSES) ** 2
    ).reshape(len(STATUSES), len(STATUSES))
    return {
        "states": len(indices),
        "fields": truth.numel(),
        "correct": int((predictions == truth).sum()),
        "allCorrect": int((predictions == truth).all(-1).sum()),
        "loss": float(fn.cross_entropy(logits.float().flatten(0, 1), truth.flatten())),
        "confusion": confusion.cpu().tolist(),
    }


def fit(head, x, y, indices, epochs, config):
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=config["lr"], weight_decay=config["weightDecay"], fused=True
    )
    random = np.random.default_rng(config["seed"])
    for epoch in range(epochs):
        order = random.permutation(indices)
        for start in range(0, len(order), config["batchSize"]):
            batch = order[start : start + config["batchSize"]]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = head(x[batch])[:, : len(SLOTS)]
                loss = fn.cross_entropy(logits.flatten(0, 1), y[batch].flatten())
            require(torch.isfinite(loss).item(), "Nonfinite state loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
        yield epoch + 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    plan = read(args.config)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == plan["gpu"], "Assigned GPU only")
    require(torch.cuda.device_count() == 1 and "L40S" in torch.cuda.get_device_name(0), "L40S only")
    require(provenance(plan)["sourceHash"] == plan["sourceHash"], "Registered source changed")
    for path, expected in plan["files"].items():
        require(digest(Path(path).read_bytes()) == expected, "Registered input changed: " + path)
    root = Path(plan["output"])
    require(not root.exists(), "Preserve earlier studies")
    root.mkdir(parents=True)
    started = time.perf_counter()
    seed(plan["seed"])
    allowed = {}
    for path in plan["sourceInputs"]:
        for row in lines(path):
            if row["split"] == "train":
                require(row["id"] not in allowed, "Duplicate registered episode")
                allowed[row["id"]] = row["family"]
    rows = observed_states(plan["rollouts"], allowed)
    fit_ids, held_ids = source_partition(rows, plan["heldFraction"], plan["splitSeed"])
    jsonl(root / "states.jsonl", rows)
    report = {
        "scope": "Train-only state-head diagnosis. The inherited encoder has already seen Train; source-held rows select this head only, not an independent model effectiveness test.",
        "testUsed": False,
        "devUsed": False,
        "plan": plan,
        "provenance": provenance(plan),
        "scriptHash": digest(Path(__file__).read_bytes()),
        "statesHash": digest((root / "states.jsonl").read_bytes()),
        "statuses": STATUSES,
        "groups": dict(Counter(r["benchmark"] + ":" + r["phase"] for r in rows)),
        "partitions": {
            key: {
                "states": len(ids),
                "families": sorted({rows[i]["family"] for i in ids}),
                "inputHashes": [rows[i]["inputHash"] for i in ids],
            }
            for key, ids in (("fit", fit_ids), ("held", held_ids))
        },
    }
    write(root / "registered.json", report)
    with tarfile.open(root / "source.tar.gz", "w:gz") as archive:
        for path in [
            *sorted(Path("src/newsvendor").glob("*.py")),
            Path(__file__),
            Path(args.config),
        ]:
            archive.add(
                path, arcname=str(path.relative_to(Path.cwd()) if path.is_absolute() else path)
            )
    model, tokenizer, config, _ = load(plan["checkpoint"], "cuda")
    model.requires_grad_(False)
    initial = copy.deepcopy(model.heads["state"]).requires_grad_(True)
    report["parentWeightsHash"] = weights_hash(model.state_dict())
    report["parentStateHash"] = weights_hash(initial.state_dict())
    features, predictions = [], []
    with torch.inference_mode():
        for i, row in enumerate(rows):
            view = prepare(
                row["input"],
                tokenizer,
                config["encoder"],
                fields=FIELDS,
                actions=[{"id": "hold", "text": TEXT["hold"]}],
                round=row["round"],
            )
            output = model(view)
            features.append(output["fieldState"].cpu())
            predictions.append(output["state"][: len(SLOTS)].argmax(-1).cpu())
            if (i + 1) % 128 == 0:
                print({"stage": "features", "processed": i + 1, "total": len(rows)}, flush=True)
    x = torch.stack(features).to("cuda").clone()
    y = torch.tensor([r["target"] for r in rows], device="cuda")
    predicted = torch.stack(predictions).to("cuda")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        cached = initial(x)[:, : len(SLOTS)].argmax(-1)
    require(torch.equal(cached, predicted), "Cached state predictions differ from actual model")
    torch.save(
        {"features": x.cpu(), "targets": y.cpu(), "predictions": predicted.cpu()},
        root / "features.pt",
    )
    torch.save(initial.state_dict(), root / "initial-head.pt")
    all_ids = list(range(len(rows)))
    report["initial"] = {
        name: measure(initial, x, y, ids)
        for name, ids in (("fit", fit_ids), ("held", held_ids), ("all", all_ids))
    }
    head = copy.deepcopy(initial)
    baseline = report["initial"]["held"]
    best = (baseline["correct"], -baseline["loss"])
    selected, curve = 0, []
    for epoch in fit(head, x, y, fit_ids, plan["epochs"], plan):
        measured = {
            "epoch": epoch,
            "fit": measure(head, x, y, fit_ids),
            "held": measure(head, x, y, held_ids),
        }
        key = (measured["held"]["correct"], -measured["held"]["loss"])
        if key > best:
            best, selected = key, epoch
            torch.save(head.state_dict(), root / "selected-held-head.pt")
        curve.append(measured)
    head = copy.deepcopy(initial)
    for _ in fit(head, x, y, all_ids, selected, plan):
        pass
    report.update(selectedEpoch=selected, curve=curve, finalTrain=measure(head, x, y, all_ids))
    report["headWeightsHash"] = weights_hash(head.state_dict())
    torch.save(
        {
            "parent": plan["checkpoint"],
            "parentHash": plan["files"][plan["checkpoint"]],
            "weights": {k: v.detach().cpu() for k, v in head.state_dict().items()},
            "selectedEpoch": selected,
        },
        root / "state-head.pt",
    )
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        fitted = head(x)[:, : len(SLOTS)].argmax(-1).cpu().tolist()
    jsonl(
        root / "predictions.jsonl",
        [
            {
                "inputHash": r["inputHash"],
                "family": r["family"],
                "target": r["target"],
                "initial": p.tolist(),
                "fitted": q,
            }
            for r, p, q in zip(rows, predictions, fitted, strict=True)
        ],
    )
    report["seconds"] = time.perf_counter() - started
    report["peakGpuBytes"] = torch.cuda.max_memory_allocated()
    write(root / "report.json", report)
    print(
        {
            "selectedEpoch": selected,
            "initial": report["initial"]["all"],
            "final": report["finalTrain"],
            "seconds": report["seconds"],
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
