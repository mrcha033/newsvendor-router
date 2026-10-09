"""Compare value-head fitting with and without measured Train continuation states."""

import argparse
import copy
import gzip
import json
import os
import sys
import tarfile
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from evaluate_responses import CachedRouter, file_hash
from study_parameter_state import source_partition

from newsvendor import corpus, structured_retail, structured_train
from newsvendor.cli import provenance
from newsvendor.io import digest, jsonl, lines, read, require, write
from newsvendor.structured_critic import value_loss
from newsvendor.structured_rollout import rollout
from newsvendor.structured_tool_eval import weights_hash
from newsvendor.structured_value import constrain
from newsvendor.train import seed


def replay_states(paths, episodes):
    """Select observed states by identity, never by hidden outcome or recorded loss."""
    allowed = {e["id"]: e for e in episodes}
    require(all(e["split"] == "train" for e in episodes), "Train episodes required")
    result = {}
    for path in paths:
        with Path(path).open() as stream:
            for line, row in enumerate(map(json.loads, stream)):
                require(row["split"] == "train", "Train rollouts required")
                episode = allowed.get(row["id"])
                require(
                    episode is not None and episode["family"] == row["family"],
                    "Train source differs",
                )
                for step, event in enumerate(row["events"]):
                    value = event["input"]
                    require(value["task"] == episode["input"]["task"], "Observed task changed")
                    key = digest([row["id"], row["family"], value])
                    if key not in result:
                        result[key] = {
                            "id": row["id"],
                            "family": row["family"],
                            "split": "train",
                            "benchmark": "retail" if "forecast" in value["task"] else "generated",
                            "input": copy.deepcopy(value),
                            "inputHash": key,
                            "supervised": False,
                            "source": {"path": str(path), "line": line, "step": step},
                        }
                    result[key]["supervised"] |= step == 0
    require(result, "No observed states")
    return list(result.values())


def scores(head, feature):
    return constrain(torch.nn.functional.softplus(head(feature[0]).float()), feature[2])


@torch.no_grad()
def measure(head, features, rows, indices):
    grouped = defaultdict(lambda: defaultdict(list))
    selected = []
    for index in indices:
        prediction = scores(head, features[index]).sum(-1)
        target = features[index][1].sum(-1)
        action = int(prediction.argmin())
        selected.append(action)
        grouped[rows[index]["benchmark"]][rows[index]["family"]].append(
            [
                float(target[action]),
                float(target[action] - target.min()),
                rows[index]["falseHandoffs"][action],
            ]
        )
    metrics = {}
    for name, families in grouped.items():
        values = np.mean([np.mean(items, axis=0) for items in families.values()], axis=0)
        metrics[name] = dict(
            zip(("targetLoss", "targetRegret", "falseHandoff"), map(float, values), strict=True)
        )
        metrics[name]["families"] = len(families)
    return {
        "meanTargetLoss": float(np.mean([v["targetLoss"] for v in metrics.values()])),
        "benchmarks": metrics,
        "selectedActions": selected,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True)
    args = parser.parse_args()
    plan = read(args.registration)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == plan["gpu"], "Assigned L40S only")
    require(torch.cuda.device_count() == 1 and "L40S" in torch.cuda.get_device_name(0), "L40S only")
    require(provenance({})["sourceHash"] == plan["sourceHash"], "Package source changed")
    require(file_hash(__file__) == plan["scriptHash"], "Study source changed")
    for path, expected in plan["files"].items():
        require(file_hash(path) == expected, "Registered asset changed: " + path)
    root = Path(plan["output"])
    require(not root.exists(), "Preserve previous runs")
    root.mkdir()
    seed(42)
    torch.set_num_threads(4)
    started = time.perf_counter()
    originals = [e for e in corpus.generate(read("configs/full.json")) if e["split"] == "train"]
    source = [r for r in lines(plan["retailInputs"]) if r["split"] == "train"]
    observed = [*originals, *source]
    jsonl(
        root / "initial-inputs.jsonl",
        [{k: e[k] for k in ("id", "family", "split", "input")} for e in observed],
    )
    input_hash = file_hash(root / "initial-inputs.jsonl")
    environment = {
        r["id"]: r for r in lines(plan["retailEnvironment"]) if r["id"] in {e["id"] for e in source}
    }
    retail = [
        e
        | environment[e["id"]]
        | {"benchmark": structured_retail.VERSION, "cutoffIndex": len(e["input"]["observations"])}
        for e in source
    ]
    episodes = originals + [e for e in retail if all(e["target"]["complete"])]
    indexed = {e["id"]: e for e in episodes}
    require(len(episodes) == 514, "Original complete Train cohort changed")
    rows = replay_states(plan["rollouts"], episodes)
    fit, held = source_partition(rows, plan["heldFraction"], plan["splitSeed"])
    jsonl(root / "states.jsonl", rows)
    jsonl(
        root / "environment.jsonl", [{k: v for k, v in e.items() if k != "input"} for e in episodes]
    )
    report = {
        "scope": "Source-held Train screening of measured continuation replay, with frozen parent follow-up actions. Not an independent policy effectiveness evaluation.",
        "testUsed": False,
        "devUsed": False,
        "plan": plan,
        "source": provenance({}),
        "inputHash": input_hash,
        "statesHash": file_hash(root / "states.jsonl"),
        "states": len(rows),
        "previouslySupervised": sum(r["supervised"] for r in rows),
        "partitions": {
            name: {"states": len(ids), "families": sorted({rows[i]["family"] for i in ids})}
            for name, ids in (("fit", fit), ("held", held))
        },
        "results": {},
    }
    write(root / "registered.json", report)
    with tarfile.open(root / "source.tar.gz", "w:gz") as archive:
        for path in [
            *sorted(Path("src/newsvendor").glob("*.py")),
            Path(__file__),
            Path(args.registration),
            Path("scripts/evaluate_responses.py"),
            Path("scripts/study_parameter_state.py"),
        ]:
            archive.add(path, arcname=str(path))
    model, tokenizer, config, _ = structured_train.load(plan["checkpoint"], "cuda")
    require(weights_hash(model.state_dict()) == plan["weightsHash"], "Wrong parent weights")
    model.requires_grad_(False)
    # Use the existing explicit FP32 action-head path for matching cached fitting/inference.
    config["encoder"]["actionPrecision"] = "float32"
    model.config["actionPrecision"] = "float32"
    router = CachedRouter(model, tokenizer, config["encoder"])
    router.reset()
    head = model.heads["value"]
    initial = copy.deepcopy(head.state_dict())
    features, targets = [], []
    with gzip.open(root / "train-rollouts.jsonl.gz", "wt", compresslevel=3) as stream:
        for index, row in enumerate(rows):
            episode, value = indexed[row["id"]], row["input"]
            state = router.construct(value)
            view = router.view(value, state)
            with torch.inference_mode():
                output = model(view)
                feature = output["valueState"].float().detach().clone()
                direct = output["value"].clone()
                cached = constrain(
                    torch.nn.functional.softplus(head(feature).float()), view.get("valueCosts")
                )
                require(
                    torch.equal(cached, direct), "Cached action values differ from actual model"
                )
            actions = [a["id"] for a in view["actions"]]
            values, false = [], []
            for action in actions:
                samples = []
                for sample in range(plan["responseSamples"]):
                    draw = digest([plan["responseSeed"], "target", sample])
                    result = rollout(
                        episode,
                        router,
                        value=value,
                        first=action,
                        noise=plan["noise"][row["benchmark"]],
                        response_seed=draw,
                    )
                    stream.write(
                        json.dumps(
                            {
                                "stateKey": row["inputHash"],
                                "forcedAction": action,
                                "responseSeed": draw,
                                **result,
                            },
                            allow_nan=False,
                        )
                        + "\n"
                    )
                    samples.append(result)
                values.append(
                    [
                        float(np.mean([s[k] for s in samples])) / value["task"]["hold"]
                        for k in ("terminalLoss", "requestCost")
                    ]
                )
                false.append(float(np.mean([s["falseHandoff"] for s in samples])))
            row.update(actions=actions, values=values, falseHandoffs=false)
            targets.append({**row, "state": state, "valueCosts": view.get("valueCosts")})
            # Clone outside inference mode so gradients through the small head are valid.
            features.append(
                (feature.clone(), torch.tensor(values, device="cuda"), view.get("valueCosts"))
            )
            if (index + 1) % 64 == 0:
                write(
                    root / "progress.json",
                    {
                        "stage": "collect",
                        "states": index + 1,
                        "total": len(rows),
                        "seconds": time.perf_counter() - started,
                    },
                )
                print(read(root / "progress.json"), flush=True)
    jsonl(root / "targets.jsonl", targets)
    torch.save(
        {"features": [(a.cpu(), b.cpu(), c) for a, b, c in features], "initial": initial},
        root / "features.pt",
    )
    report["collectionSeconds"] = time.perf_counter() - started
    report["initial"] = measure(head, features, rows, held)
    head.requires_grad_(True)
    for condition in ("observed", "continuations"):
        fit_ids = [i for i in fit if condition == "continuations" or rows[i]["supervised"]]
        for random_seed in plan["seeds"]:
            head.load_state_dict(initial)
            random = np.random.default_rng(random_seed)
            optimizer = torch.optim.AdamW(
                head.parameters(), lr=plan["lr"], weight_decay=0.01, fused=True
            )
            best, selected = report["initial"]["meanTargetLoss"], 0
            saved = copy.deepcopy(initial)
            saved_optimizer = None
            history = []
            updates = (len(fit) + plan["batchSize"] - 1) // plan["batchSize"]
            for epoch in range(plan["epochs"]):
                for _ in range(updates):
                    order = random.choice(fit_ids, size=plan["batchSize"], replace=True)
                    batch = [features[int(i)] for i in order]
                    optimizer.zero_grad(set_to_none=True)
                    loss = value_loss(head, batch, plan["rankWeight"])
                    require(torch.isfinite(loss).item(), "Nonfinite value loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(head.parameters(), 5, error_if_nonfinite=True)
                    optimizer.step()
                measured = measure(head, features, rows, held)
                if measured["meanTargetLoss"] < best - 1e-10:
                    best, selected, saved = (
                        measured["meanTargetLoss"],
                        epoch + 1,
                        copy.deepcopy(head.state_dict()),
                    )
                    saved_optimizer = copy.deepcopy(optimizer.state_dict())
                history.append({"epoch": epoch + 1, **measured})
            final_weights = copy.deepcopy(head.state_dict())
            head.load_state_dict(saved)
            result = {
                "fitStates": len(fit_ids),
                "updatesPerEpoch": updates,
                "selectedEpoch": selected,
                "selected": measure(head, features, rows, held),
                "epochs": history,
            }
            key = f"{condition}-{random_seed}"
            report["results"][key] = result
            torch.save(
                {
                    "weights": saved,
                    "optimizer": saved_optimizer,
                    "finalWeights": final_weights,
                    "finalOptimizer": optimizer.state_dict(),
                    "finalEpoch": plan["epochs"],
                    "seed": random_seed,
                    "config": config,
                    "parentWeightsHash": plan["weightsHash"],
                    "selectedEpoch": selected,
                },
                root / f"{key}.pt",
            )
            write(root / "partial.json", report)
            print({"condition": key, "selectedEpoch": selected, "heldLoss": best}, flush=True)
    report["gates"] = {}
    for random_seed in plan["seeds"]:
        a = report["results"][f"observed-{random_seed}"]["selected"]
        b = report["results"][f"continuations-{random_seed}"]["selected"]
        report["gates"][str(random_seed)] = {
            "betterThanObserved": b["meanTargetLoss"] < a["meanTargetLoss"] - 1e-10,
            "bothBenchmarksNotWorseThanParent": all(
                b["benchmarks"][k]["targetLoss"]
                <= report["initial"]["benchmarks"][k]["targetLoss"] + 1e-10
                for k in b["benchmarks"]
            ),
            "noAdditionalExpectedFalseHandoff": all(
                b["benchmarks"][k]["falseHandoff"]
                <= report["initial"]["benchmarks"][k]["falseHandoff"] + 1e-10
                for k in b["benchmarks"]
            ),
        }
    report["eligibleForSequentialTrainCheck"] = all(
        all(g.values()) for g in report["gates"].values()
    )
    head.load_state_dict(initial)
    require(weights_hash(model.state_dict()) == plan["weightsHash"], "Frozen parent changed")
    report["allParentTensorsPreserved"] = True
    report["seconds"] = time.perf_counter() - started
    report["peakGpuBytes"] = torch.cuda.max_memory_allocated()
    report["rawHashes"] = {p.name: file_hash(p) for p in root.iterdir() if p.is_file()}
    write(root / "report.json", report)
    print({"completed": True, "seconds": report["seconds"], "gates": report["gates"]}, flush=True)


if __name__ == "__main__":
    main()
