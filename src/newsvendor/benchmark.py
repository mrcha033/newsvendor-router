import time
from pathlib import Path

import torch

from . import corpus
from .cli import provenance
from .encoder import embed
from .io import digest, jsonl, require, write
from .policy import trajectory
from .providers import settings
from .report import save
from .train import HEADS, policy_model, targets
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
    calibration = builder.calibrate(partitions["cal"])
    # This backbone is fixed before the study and never fitted on these families; no in-sample construction fit occurs.
    context = config.get("policyContext", True)
    values, repairs = targets(
        partitions["train"], None, cache, constructor=builder, context=context
    )
    valid, repairdev = targets(partitions["dev"], None, cache, constructor=builder, context=context)
    require(values and valid, "Missing external-construction training or development targets")
    model, losses = policy_model(config, cache, values, repairs, valid, repairdev)
    rulevalues, rulerepairs = targets(partitions["train"], None, cache, context=context)
    rulevalid, rulerepairdev = targets(partitions["dev"], None, cache, context=context)
    rulemodel, rulelosses = policy_model(
        config, cache, rulevalues, rulerepairs, rulevalid, rulerepairdev
    )
    model.update({"rule_" + name: rulemodel[name] for name in ("value", "repair")})
    rows = []
    for episode in partitions["test"]:
        for construction in ("rules", "typed"):
            for method in (
                "checklist",
                "ask_all",
                "uncertainty",
                "one_step",
                "reference",
                "planner",
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
            "weights": {k: v.state_dict() for k, v in model.items() if k in HEADS},
            "policyContext": context,
            "config": config,
            "calibration": calibration,
            "provider": builder.identity,
            "encoderHash": cache.id,
        },
        directory + "/policy.pt",
    )
    records = list(builder.snapshots.values())
    jsonl(directory + "/predictions.jsonl", records)
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
            "rulePolicyLosses": rulelosses,
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
            "inputHash": digest([e["input"] for e in episodes]),
            "labelHash": digest([e["gold"] for e in episodes]),
        },
    )
    print(
        {"directory": directory, "newCalls": builder.calls, "typed": summary["learned/typed"]},
        flush=True,
    )
