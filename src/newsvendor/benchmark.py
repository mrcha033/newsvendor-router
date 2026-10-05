import time
from pathlib import Path

import torch

from . import corpus
from .cli import provenance
from .encoder import EXTRA, embed
from .heads import Head, train
from .io import digest, jsonl, require, write
from .policy import trajectory
from .providers import settings
from .report import save
from .train import HEADS, seed, targets
from .typed import TypedBuilder


def run(config, provider, budget):
    """Fit shared value/recovery heads on the frozen external predictor's actual constructions."""
    require(budget >= 0, "Provider call budget cannot be negative")
    configured = settings(provider)
    started = time.perf_counter()
    episodes = corpus.generate(config)
    corpus.audit(episodes)
    cache = embed(episodes, config["encoder"])
    builder = TypedBuilder(configured, budget)
    partitions = {
        k: [e for e in episodes if e["split"] == k] for k in ("train", "dev", "cal", "test")
    }
    # This backbone is fixed before the study and never fitted on these families; no in-sample construction fit occurs.
    values, repairs = targets(partitions["train"], None, cache, constructor=builder)
    valid, repairdev = targets(partitions["dev"], None, cache, constructor=builder)
    seed(config["seed"] + 104)
    model, losses = {}, {}
    for name, rows, dev, mode in (
        ("value", values, valid, "value"),
        ("repair", repairs, repairdev, "repair"),
    ):
        require(rows and dev, "Missing external-construction training or development targets")
        model[name] = Head(cache.config["dim"] * 2 + EXTRA, *HEADS[name])
        losses[name] = train(
            model[name],
            rows,
            config["epochs"] * (2 if name == "value" else 1),
            config["seed"] + 104,
            mode,
            valid=dev,
        )
    calibration = builder.calibrate(partitions["cal"])
    rows = []
    for episode in partitions["test"]:
        for construction in ("rules", "typed"):
            for method in (
                "checklist",
                "ask_all",
                "uncertainty",
                "one_step",
                "reference",
                "learned",
            ):
                rows.append(
                    trajectory(
                        episode,
                        method,
                        model,
                        cache,
                        construction=construction,
                        constructor=builder,
                    )
                )
    directory = f"results/benchmarks/{provider}/{config['seed']}"
    Path(directory).mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "weights": {k: v.state_dict() for k, v in model.items()},
            "config": config,
            "calibration": calibration,
            "provider": builder.identity,
            "encoderHash": cache.id,
        },
        directory + "/policy.pt",
    )
    records = list(builder.snapshots.values())
    jsonl(
        directory + "/calls.jsonl",
        [{"cacheKey": r["requestHash"], **r["trace"]} for r in records],
    )
    write(
        directory + "/training.json",
        {
            "scope": "Fixed external constructor with shared learned value/recovery heads",
            "config": config,
            "losses": losses,
            "calibration": calibration,
            "fitFamilies": sorted({e["family"] for e in partitions["train"]}),
            "developmentFamilies": sorted({e["family"] for e in partitions["dev"]}),
            "calibrationFamilies": sorted({e["family"] for e in partitions["cal"]}),
            "testFamilies": sorted({e["family"] for e in partitions["test"]}),
            "targetHash": digest([{k: v for k, v in r.items() if k != "x"} for r in values]),
            "provider": builder.identity,
            "newCalls": builder.calls,
            "cachedStatesUsed": len(records) - builder.calls,
        },
    )
    summary = save(
        rows,
        directory,
        config,
        {
            **provenance(config),
            "provider": builder.identity,
            "wallSeconds": time.perf_counter() - started,
            "newCalls": builder.calls,
            "split": corpus.audit(episodes),
        },
    )
    print(
        {"directory": directory, "newCalls": builder.calls, "typed": summary["learned/typed"]},
        flush=True,
    )
