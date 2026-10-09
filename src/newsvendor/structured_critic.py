"""Fit value or recovery decisions on own-state rollouts with a fixed constructor."""

import copy
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as fn

from .io import digest, jsonl, require
from .structured_rollout import ResearchRouter, collect, rollout
from .structured_value import constrain


def value_loss(head, examples, rank_weight=0.25):
    losses = []
    for features, target, constraints in examples:
        prediction = constrain(fn.softplus(head(features).float()), constraints)
        regression = fn.smooth_l1_loss(prediction, target)
        costs, estimates = target.sum(-1), prediction.sum(-1)
        gap = costs[:, None] - costs[None, :]
        pairs = gap > 1e-5
        ranking = (
            fn.relu(gap.clamp(max=1) - (estimates[:, None] - estimates[None, :]))[pairs].mean()
            if pairs.any()
            else regression * 0
        )
        losses.append(regression + rank_weight * ranking)
    return torch.stack(losses).mean()


def recovery_loss(head, examples):
    return torch.stack(
        [
            fn.cross_entropy(head(features).flatten()[None], target.reshape(1))
            for features, target, _ in examples
        ]
    ).mean()


def selection_score(episodes, rows, metric="mean_total"):
    require(metric in ("mean_total", "benchmark_normalized"), "Unknown policy selection metric")
    groups = {}
    for episode, row in zip(episodes, rows, strict=True):
        task = episode["input"]["task"]
        name = "retail" if "forecast" in task else "generated"
        require(np.isfinite(row["total"]) and task["hold"] > 0, "Invalid policy evaluation loss")
        groups.setdefault(name, []).append((row["total"], row["total"] / task["hold"]))
    require(groups, "No policy evaluation cases")
    measured = {
        name: {
            "episodes": len(values),
            "meanTotal": float(np.mean([v[0] for v in values])),
            "meanNormalizedTotal": float(np.mean([v[1] for v in values])),
        }
        for name, values in groups.items()
    }
    if metric == "mean_total":
        require(len(groups) == 1, "Mixed benchmarks require explicit normalized selection")
        return float(np.mean([r["total"] for r in rows])), measured
    return float(np.mean([m["meanNormalizedTotal"] for m in measured.values()])), measured


def fit(model, tokenizer, config, episodes, progress, public_dev, labels, collection):
    from .structured_policy import public_validation, retained
    from .structured_tool_eval import weights_hash

    no_value = config["noValue"]
    name = "recovery" if no_value else "value"
    policy, directory = config["policy"], Path(config["output"])
    selection_metric = policy.get("selection", "mean_total")
    loss_key = "devActualTotalLoss" if selection_metric == "mean_total" else "devSelectionLoss"
    directory.mkdir(parents=True, exist_ok=True)
    model.eval()
    head = model.heads[name]

    def frozen():
        return weights_hash(
            {k: v for k, v in model.state_dict().items() if not k.startswith(f"heads.{name}.")}
        )

    frozen_hash = frozen()
    checkpoint_base = None
    if progress:
        # Only this head changes during policy fitting. Preserve the full model
        # once, then save each candidate with its optimizer and an exact base hash.
        path = directory / "policy-base.pt"
        if path.exists():
            base = torch.load(path, map_location="cpu", weights_only=True)
            require(base["identity"] == progress.identity, "Policy base identity changed")
            require(
                weights_hash(
                    {k: v for k, v in base["weights"].items() if not k.startswith(f"heads.{name}.")}
                )
                == frozen_hash,
                "Policy base frozen weights changed",
            )
        else:
            progress.checkpoint(
                "policy-base", model, {"config": config, "frozenWeightsHash": frozen_hash}
            )
        checkpoint_base = {"path": path.name, "hash": digest(path.read_bytes())}
    router = ResearchRouter(model, tokenizer, config["encoder"], no_value)
    train = [e for e in episodes if e["split"] == "train"]
    dev = [e for e in episodes if e["split"] == "dev"]
    require(train and dev, "Value fitting requires disjoint Train and Dev episodes")
    baseline = public_validation(
        model,
        tokenizer,
        config,
        public_dev,
        labels,
        collection,
        directory / "retention-initial.jsonl",
    )

    def evaluate(name):
        router.cache_states()
        scores = [rollout(e, router) for e in dev]
        router.cache_states(False)
        jsonl(directory / name, scores)
        return selection_score(dev, scores, selection_metric)

    best, selected_metrics = evaluate("policy-dev-initial.jsonl")
    selected = {
        "iteration": None,
        "epoch": None,
        "selectionMetric": selection_metric,
        "economicLoss" if selection_metric == "mean_total" else "selectionLoss": best,
    }
    saved = copy.deepcopy(head.state_dict())
    random = np.random.default_rng(config["seed"])
    reports = []
    for iteration in range(policy["iterations"]):
        if progress:
            progress.update(
                "policy",
                iteration=iteration + 1,
                activity="collect_train",
                trainable=name + "_head",
            )
        training, raw = collect(
            train,
            router,
            noise=policy.get("noise", 0.1),
            progress=progress,
            policy=policy.get("collectionPolicy", "mixed"),
            with_values=not no_value,
        )
        jsonl(directory / f"rollout-train-{iteration}.jsonl", raw)
        jsonl(directory / f"{name}-targets-{iteration}.jsonl", training)
        features = []
        # The encoder was fine-tuned in the preceding stage. Only this phase freezes it.
        # Detaching permits reuse across critic epochs with identical representations.
        with torch.no_grad():
            for start in range(0, len(training), policy["batchSize"]):
                group = training[start : start + policy["batchSize"]]
                views = [router.view(row["input"], row["state"]) for row in group]
                for lo, hi in model.training_batches(views):
                    for output, row, view in zip(
                        model(views[lo:hi]), group[lo:hi], views[lo:hi], strict=True
                    ):
                        state = output["actionState"].detach()
                        require(
                            len(state) == len(row["actions"]),
                            "Action targets differ from encoded actions",
                        )
                        require(
                            no_value or len(state) == len(row["values"]),
                            "Value targets differ from encoded actions",
                        )
                        target = torch.tensor(
                            row["recovery"] if no_value else row["values"],
                            device=state.device,
                            dtype=torch.long if no_value else state.dtype,
                        )
                        features.append((state, target, view.get("valueCosts")))
        require(features, "No own-state value targets")
        optimizer = torch.optim.AdamW(
            head.parameters(), lr=policy.get("valueLr", 1e-3), weight_decay=0.01
        )
        epochs, stale = [], 0
        for epoch in range(policy["epochs"]):
            losses = []
            order = random.permutation(len(features))
            for start in range(0, len(order), policy["batchSize"]):
                group = [features[int(i)] for i in order[start : start + policy["batchSize"]]]
                optimizer.zero_grad(set_to_none=True)
                loss = (
                    recovery_loss(head, group)
                    if no_value
                    else value_loss(head, group, policy.get("rankWeight", 0.25))
                )
                require(torch.isfinite(loss).item(), "Nonfinite critic loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 5, error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach()))
            if (epoch + 1) % policy.get("validationEvery", 10) and epoch + 1 != policy["epochs"]:
                continue
            dev_total, dev_metrics = evaluate(f"policy-dev-{iteration}-{epoch}.jsonl")
            accepted = dev_total < best
            if accepted:
                best, stale = dev_total, 0
                selected_metrics = dev_metrics
                saved = copy.deepcopy(head.state_dict())
                selected = {
                    "iteration": iteration,
                    "epoch": epoch,
                    "selectionMetric": selection_metric,
                    "economicLoss" if selection_metric == "mean_total" else "selectionLoss": best,
                }
            else:
                stale += 1
            record = {
                "epoch": epoch + 1,
                "trainLoss": float(np.mean(losses)),
                loss_key: dev_total,
                "selectionMetric": selection_metric,
                "devBenchmarks": dev_metrics,
                "accepted": accepted,
                "retentionFailures": [],
            }
            epochs.append(record)
            if progress:
                progress.checkpoint(
                    f"policy-candidate-{iteration}-{epoch}",
                    head,
                    {
                        "format": "policy-head-v1",
                        "head": name,
                        "base": checkpoint_base,
                        "frozenWeightsHash": frozen_hash,
                        "config": config,
                        "iteration": iteration,
                        "epoch": epoch,
                        "dev": record,
                    },
                    optimizer,
                )
                progress.update("policy_dev", iteration=iteration + 1, **record)
            if stale >= policy.get("patience", 3):
                break
        head.load_state_dict(saved)
        require(frozen() == frozen_hash, "Policy fitting changed frozen parameters")
        candidate = public_validation(
            model,
            tokenizer,
            config,
            public_dev,
            labels,
            collection,
            directory / f"retention-{iteration}.jsonl",
        )
        require(not retained(baseline, candidate, 0.0), "Frozen critic changed public decisions")
        reports.append(
            {
                "iteration": iteration,
                "ownStateTargets": len(training),
                "trainRawHash": digest(raw),
                "epochs": epochs,
                "selected": dict(selected),
                "publicDevBaseline": baseline,
                "frozenWeightsHash": frozen_hash,
                "frozenWeightsUnchanged": True,
                "checkpointBase": checkpoint_base,
                "trainable": f"heads.{name} only; encoder fine-tuned in preceding language stage",
                "target": "Observed missing-parameter request/checklist, no economic targets"
                if no_value
                else "Measured terminal loss and request costs after forced first action",
                "selectionMetric": selection_metric,
                "devBenchmarks": selected_metrics,
                loss_key: best,
            }
        )
    return reports
