"""Mixed policy learning with source-annotated replay and component Dev retention."""

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as fn

from .io import digest, jsonl, require
from .structured_labels import objective, research_targets
from .structured_rollout import ResearchRouter, collect, rollout
from .suite import public_input
from .suite_score import metrics

RETAIN = {
    "abcd": ("actionAccuracy", "toolExact", "observableToolAndArgumentsExact"),
    "cuad": ("actionAccuracy", "answerabilityAccuracy", "evidenceF1", "groundedScore"),
    "contractnli": ("actionAccuracy", "stateAccuracy", "evidenceF1"),
    "orsharc": ("actionAccuracy", "decisionAccuracy", "ruleRecallAt5"),
    "tatqa": ("actionAccuracy", "answerExact", "scaleAccuracy"),
}

ANCHOR = {
    "abcd": (
        "mode",
        "use",
        "start",
        "end",
        "choice",
        "evidence",
        "recovery",
        "question",
        "role",
        "entity",
        "procedure",
        "stage",
        "baseRecovery",
    ),
    "cuad": ("mode", "start", "end", "evidence", "recovery"),
    "contractnli": ("mode", "start", "end", "evidence", "decision", "recovery"),
    "orsharc": ("decision", "question", "recovery"),
    "tatqa": (
        "mode",
        "start",
        "end",
        "evidence",
        "relation",
        "operand1",
        "operand2",
        "scale",
        "recovery",
    ),
}


def distillation(student, teacher, view, component):
    losses = []
    for key in ANCHOR[component]:
        if key not in student:
            continue
        current, parent = student[key], teacher[key].detach()
        if key == "question":
            current, parent = current.flatten(), parent.flatten()
        if key == "choice":
            rows = [i for i, field in enumerate(view["fields"]) if field.get("choices")]
            if not rows:
                continue
            current, parent = current[rows], parent[rows]
        losses.append(
            fn.kl_div(current.log_softmax(-1), parent.softmax(-1), reduction="none").sum(-1).mean()
        )
    return torch.stack(losses).mean().clamp_min(0) if losses else student["recovery"].sum() * 0


class Replay:
    def __init__(self, cases, labels, seed):
        self.random = np.random.default_rng(seed)
        self.pools = defaultdict(lambda: defaultdict(list))
        for mode, row in cases:
            if mode != "public":
                continue
            require(row["split"] == "train", "Replay accepts Train only")
            action = labels[row["id"]]["action"] if row["component"] == "abcd" else "any"
            self.pools[row["component"], action][row["family"]].append((mode, row))
        schedule = [("abcd", "call_tool"), ("abcd", "speak")] * 2 + [
            (name, "any") for name in ("contractnli", "cuad", "orsharc", "tatqa")
        ]
        self.schedule = [key for key in schedule if key in self.pools]
        self.cursor = 0

    def sample(self, size):
        result = []
        for _ in range(size if self.schedule else 0):
            key = self.schedule[self.cursor % len(self.schedule)]
            self.cursor += 1
            families = list(self.pools[key])
            family = families[int(self.random.integers(len(families)))]
            pool = self.pools[key][family]
            result.append(pool[int(self.random.integers(len(pool)))])
        return result


def retained(baseline, candidate, tolerance=0.0):
    failures = []
    for component, values in baseline.items():
        for metric in RETAIN[component]:
            if metric not in values:
                continue
            current = candidate.get(component, {}).get(metric)
            if current is None or current + tolerance + 1e-12 < values[metric]:
                failures.append(
                    {
                        "component": component,
                        "metric": metric,
                        "before": values[metric],
                        "after": current,
                    }
                )
    return failures


@torch.inference_mode()
def public_validation(model, tokenizer, config, cases, labels, collection, path):
    from .structured_metrics import tool_metrics
    from .structured_train import infer

    model.eval()
    groups = defaultdict(lambda: defaultdict(list))
    records = []
    conversations = {}
    if config["training"].get("conversationValidation"):
        from .structured_tool_eval import conversation_replay

        rows = [row for mode, row in cases if mode == "public"]
        require(all(row["split"] == "dev" for row in rows), "Policy retention accepts Dev only")
        conversations = {
            r["key"]: r
            for r in conversation_replay(
                model, tokenizer, config, rows, labels, collection, split="dev"
            )
        }
    for mode, row in cases:
        if mode != "public":
            continue
        require(row["split"] == "dev", "Policy retention accepts Dev only")
        if row["id"] in conversations:
            record = conversations[row["id"]]
            prediction, values = record["prediction"], record["metrics"]
        else:
            prediction, _ = infer(model, tokenizer, public_input(row), config, collection)
            values = (tool_metrics if row["component"] == "abcd" else metrics)(
                row, labels[row["id"]], prediction
            )
        records.append(
            {
                "id": row["id"],
                "family": row["family"],
                "component": row["component"],
                "prediction": prediction,
                "metrics": values,
            }
        )
        for name, value in values.items():
            if value is not None:
                groups[row["component"]][name].append(float(value))
    jsonl(path, records)
    return {c: {k: float(np.mean(v)) for k, v in values.items()} for c, values in groups.items()}


def fit(
    model,
    tokenizer,
    config,
    episodes,
    progress=None,
    public_train=(),
    public_dev=(),
    labels=None,
    collection=(),
):
    if config["policy"].get("trainable") in ("value", "recovery"):
        from .structured_critic import fit as fit_value

        return fit_value(
            model, tokenizer, config, episodes, progress, public_dev, labels or {}, collection
        )
    from .structured_model import Router, load_backbone
    from .structured_train import language_backward, language_view

    labels = labels or {}
    router = ResearchRouter(model, tokenizer, config["encoder"], config["noValue"])
    train = [e for e in episodes if e["split"] == "train"]
    dev = [e for e in episodes if e["split"] == "dev"]
    directory = Path(config["output"])
    directory.mkdir(parents=True, exist_ok=True)
    policy = config["policy"]
    size, replay_size = policy.get("batchSize", 8), policy.get("replaySize", 8)
    weight, tolerance = policy.get("replayWeight", 2.0), policy.get("retentionTolerance", 0.0)
    require(
        size > 0 and replay_size > 0 and weight > 0 and tolerance >= 0,
        "Invalid mixed policy configuration",
    )
    replay = Replay(public_train, labels, config["seed"])
    distill_weight = policy.get("distillWeight", 0.0)
    require(distill_weight >= 0, "Invalid replay distillation weight")
    teacher = None
    if distill_weight and replay.schedule:
        # These are frozen training targets, never cached features or inference representations.
        _, encoder = load_backbone(config["encoder"])
        teacher = Router(encoder, config["encoder"]).to(next(model.parameters()).device)
        teacher.load_state_dict(model.state_dict(), strict=True)
        teacher.requires_grad_(False).eval()
    optimizer = model.optimizer(config["training"])
    reports = []
    model.eval()
    baseline = public_validation(
        model,
        tokenizer,
        config,
        public_dev,
        labels,
        collection,
        directory / "retention-initial.jsonl",
    )
    initial_rollouts = [rollout(e, router) for e in dev]
    jsonl(directory / "policy-dev-initial.jsonl", initial_rollouts)
    best = float(np.mean([r["total"] for r in initial_rollouts]))
    selected = {
        "iteration": None,
        "epoch": None,
        "economicLoss": best,
        "reason": "initial language/demand checkpoint",
    }
    saved = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    for iteration in range(policy["iterations"]):
        if progress:
            progress.update("policy", iteration=iteration + 1, activity="collect_train")
        training, raw = collect(
            train,
            router,
            noise=policy.get("noise", 0.1),
            progress=progress,
            policy=policy.get("collectionPolicy", "behavior"),
            response_samples=policy.get("responseSamples", 1),
            response_seed=digest([policy["responseSeed"], iteration])
            if policy.get("responseSeed") is not None
            else None,
        )
        jsonl(directory / f"rollout-train-{iteration}.jsonl", raw)
        details, epochs, replay_ids = defaultdict(list), [], []
        for epoch in range(policy["epochs"]):
            model.train()
            order = replay.random.permutation(len(training))
            for start in range(0, len(order), size):
                optimizer.zero_grad(set_to_none=True)
                group = [training[int(i)] for i in order[start : start + size]]
                views = [router.view(row["input"], row["state"]) for row in group]
                for lo, hi in model.training_batches(views):
                    output = model(views[lo:hi])
                    losses = []
                    for row, view, prediction in zip(
                        group[lo:hi], views[lo:hi], output, strict=True
                    ):
                        target = research_targets(view, row["input"]) | {
                            k: row[k] for k in ("recovery", "values", "valueIndices")
                        }
                        if row["actions"][row["recovery"]] in ("v", "b"):
                            target["question"] = [f["name"] for f in view["fields"]].index(
                                row["actions"][row["recovery"]]
                            )
                        loss, measured = objective(prediction, target, no_value=config["noValue"])
                        require(torch.isfinite(loss).item(), "Nonfinite policy objective")
                        losses.append(loss)
                        for name, value in measured.items():
                            details[name].append(value)
                    (sum(losses) / len(group)).backward()
                sampled = replay.sample(replay_size)
                if sampled:
                    prepared = [
                        language_view(case, tokenizer, config, labels, collection)
                        for case in sampled
                    ]
                    for i, (_, target) in enumerate(prepared):
                        target.update(
                            caseIndex=i,
                            retrievalRound=int(
                                replay.random.integers(1, config.get("maxRetrievals", 2) + 1)
                            ),
                        )
                    replay_views = [view for view, _ in prepared]
                    for lo, hi in model.training_batches(replay_views):
                        _, measurements, _ = language_backward(
                            model,
                            prepared[lo:hi],
                            tokenizer,
                            config,
                            sampled,
                            labels,
                            collection,
                            scale=weight / len(sampled),
                            teacher=teacher,
                            distill_weight=distill_weight,
                        )
                        for measured in measurements:
                            for name, value in measured.items():
                                details["replay:" + name].append(value)
                    replay_ids.extend(case[1]["id"] for case in sampled)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
                optimizer.step()
                if progress and (start + len(group)) // 128 > start // 128:
                    progress.update(
                        "policy",
                        iteration=iteration + 1,
                        epoch=epoch + 1,
                        processed=start + len(group),
                        total=len(training),
                        researchBatch=size,
                        replayBatch=len(sampled),
                    )
            router.cache_states(False)
            model.eval()
            scores = [rollout(e, router) for e in dev]
            dev_total = float(np.mean([r["total"] for r in scores]))
            jsonl(directory / f"policy-dev-{iteration}-{epoch}.jsonl", scores)
            candidate = public_validation(
                model,
                tokenizer,
                config,
                public_dev,
                labels,
                collection,
                directory / f"retention-{iteration}-{epoch}.jsonl",
            )
            failures = retained(baseline, candidate, tolerance)
            accepted = not failures and dev_total < best
            if accepted:
                best = dev_total
                saved = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                selected = {
                    "iteration": iteration,
                    "epoch": epoch,
                    "economicLoss": best,
                    "reason": "economic improvement with public Dev retention",
                }
                if policy.get("retainBest"):
                    baseline = candidate
            epochs.append(
                {
                    "epoch": epoch + 1,
                    "devActualTotalLoss": dev_total,
                    "publicDev": candidate,
                    "retentionFailures": failures,
                    "accepted": accepted,
                }
            )
            print(
                {
                    "policyIteration": iteration + 1,
                    "epoch": epoch + 1,
                    "devActualTotalLoss": dev_total,
                    "retentionFailures": failures,
                    "accepted": accepted,
                },
                flush=True,
            )
        reports.append(
            {
                "iteration": iteration,
                "ownStateTargets": len(training),
                "trainRawHash": digest(raw),
                "replayCases": len(replay_ids),
                "replayIdsHash": digest(replay_ids),
                "replayIds": replay_ids,
                "epochs": epochs,
                "distillWeight": distill_weight,
                "teacher": "initial common checkpoint" if teacher else None,
                "devActualTotalLoss": epochs[-1]["devActualTotalLoss"],
                "headLosses": {k: float(np.mean(v)) for k, v in details.items()},
                "selected": selected,
                "publicDevBaseline": baseline,
                "target": "Measured terminal loss and request costs after forced first action",
                "recoveryTarget": "Observed request/checklist action, no reward-derived labels",
            }
        )
    model.load_state_dict(saved)
    return reports
