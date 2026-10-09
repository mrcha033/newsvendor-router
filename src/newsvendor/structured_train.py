"""Three-stage end-to-end training, portable checkpoints and structured evaluation."""

import copy
import math
import re
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from . import corpus, demand, sequence, structured_data
from .io import digest, jsonl, lines, read, require, write
from .native_inputs import kind
from .structured_batches import case_batch, prepared_batches
from .structured_inputs import prepare, research_input
from .structured_labels import objective, research_targets, suite_targets
from .structured_model import (
    SCHEMA,
    Router,
    assemble,
    configure_optimizer,
    load_backbone,
)
from .structured_progress import Progress
from .structured_rollout import FIELDS, TEXT, ResearchRouter, rollout
from .suite import public_input
from .suite_score import score
from .train import seed

DEMAND_SCHEMA = "newsvendor-demand-gru-v1"
RESEARCH_METRICS = ("parameterAccuracy", "rawStateAccuracy", "rawTypeAccuracy", "evidenceAccuracy")


def variant(config, name, value=None):
    result = copy.deepcopy(config)
    require(name in ("base", "large", "no_value"), "Unknown structured comparison")
    result["variant"] = name
    result["noValue"] = name == "no_value"
    result["encoder"].update(result["backbones"]["large" if name == "large" else "base"])
    if value is not None:
        result["seed"] = value
    result["output"] = str(Path(config["output"]) / name / str(result["seed"]))
    return result


def dataset_hashes(config):
    result = {"snapshot": digest(read(Path(config["dataset"]) / "manifest.json"))}
    if config.get("expansion"):
        result["expansion"] = digest(read(Path(config["expansion"]) / "manifest.json"))
    if config.get("encoder", {}).get("toolSchema"):
        result["toolSchema"] = digest(read(config["encoder"]["toolSchema"]))
    result["researchConfig"] = digest(read(config["researchConfig"]))
    if config.get("constructionReplay"):
        result["constructionReplay"] = {
            path: digest(Path(path).read_bytes()) for path in config["constructionReplay"]
        }
    if config.get("linkedManifest"):
        result["linkedManifest"] = digest(read(config["linkedManifest"]))
    if config.get("training", {}).get("distillWeight", 0):
        require(config.get("warmStart"), "Language distillation requires a frozen warm start")
        result["languageTeacher"] = digest(Path(config["warmStart"]).read_bytes())
    return result


def save(path, model, config, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "schema": SCHEMA,
            "config": config,
            "report": report,
            "datasetHashes": dataset_hashes(config),
            "weights": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        },
        temporary,
    )
    temporary.replace(path)


def load(path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    require(payload["schema"] == SCHEMA, "Wrong structured checkpoint schema")
    config = payload["config"]
    require(dataset_hashes(config) == payload["datasetHashes"], "Checkpoint data changed")
    tokenizer, encoder = load_backbone(config["encoder"])
    model = Router(encoder, config["encoder"])
    model.load_state_dict(payload["weights"], strict=True)
    return model.to(device).eval(), tokenizer, config, payload["report"]


def import_core(model, weights):
    """Only an explicitly enabled numeric-state head may extend legacy input columns."""
    weights = dict(weights)
    if model.config.get("numericState"):
        for name in ("value", "recovery"):
            key = f"heads.{name}.layers.0.weight"
            expected = model.heads[name].layers[0].weight
            if weights[key].shape != expected.shape:
                require(
                    weights[key].shape == (expected.shape[0], 256), "Unexpected action-head shape"
                )
                expanded = weights[key].new_zeros(expected.shape)
                expanded[:, :256] = weights[key]
                weights[key] = expanded
    model.load_state_dict(weights, strict=True)


def initialize(config):
    """Warm-start a core, or pair pretrained encoders with identical fresh functional heads."""
    from .structured_tool_eval import weights_hash

    warm = config.get("warmStart")
    require(
        bool(warm) != bool(config.get("demandCheckpoint")),
        "Specify a core warm start or a shared demand checkpoint",
    )
    path = Path(warm or config["demandCheckpoint"])
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    require(
        dataset_hashes(payload["config"]) == payload["datasetHashes"],
        "Parent checkpoint data changed",
    )
    if payload.get("schema") == DEMAND_SCHEMA:
        require(not warm, "A demand-only checkpoint cannot warm-start the full core")
        require(
            all(k.startswith("demand.") for k in payload["weights"]),
            "Demand-only checkpoint contains unrelated weights",
        )
    if warm:
        require(
            all(
                config["encoder"][k] == payload["config"]["encoder"][k]
                for k in ("model", "revision")
            ),
            "Warm-start backbone identity differs",
        )
    tokenizer, encoder = load_backbone(config["encoder"])
    # Backbone loading consumes different amounts of RNG across model sizes.
    seed(config["seed"])
    model = Router(encoder, config["encoder"])
    if warm:
        weights = {
            k: v for k, v in payload["weights"].items() if not k.startswith(("tools.", "control."))
        }
        import_core(model, weights)
    else:
        weights = {
            k.removeprefix("demand."): v
            for k, v in payload["weights"].items()
            if k.startswith("demand.")
        }
        model.demand.load_state_dict(weights, strict=True)
    parent = {
        "mode": "warm_start" if warm else "pretrained_encoder_and_fresh_heads",
        "checkpoint" if warm else "demandCheckpoint": str(path),
        "fileHash": digest(path.read_bytes()),
        "encoder": {k: config["encoder"][k] for k in ("model", "revision")},
        "coreWeightsHash": weights_hash(model.state_dict()),
        "sharedHeadsHash": weights_hash(
            {
                k: v
                for k, v in model.state_dict().items()
                if not k.startswith(("encoder.", "project."))
            }
        ),
        "demandWeightsHash": weights_hash(model.demand.state_dict()),
        "numericStateAdded": bool(config["encoder"].get("numericState"))
        and not payload["config"].get("encoder", {}).get("numericState", False),
        "removedAuxiliaryParameters": sum(
            v.numel() for k, v in payload["weights"].items() if k.startswith(("tools.", "control."))
        )
        if warm
        else 0,
    }
    return model, tokenizer, parent


def cases(rows, episodes, split, limit=None):
    public = [r for r in rows if r["split"] == split and r["component"] != "retail"]
    research = [e for e in episodes if e["split"] == split]
    if limit:
        # Deterministic per-component/per-scenario diagnostics, recorded as a limited run.
        public = [
            r
            for component in sorted({r["component"] for r in public})
            for r in [v for v in public if v["component"] == component][:limit]
        ]
        research = [
            e
            for scenario in corpus.SCENARIOS
            for e in [v for v in research if v["scenario"] == scenario][:limit]
        ]
    return [("public", r) for r in public] + [("research", e) for e in research]


def research_cases(episodes, *, replay=()):
    """Initial inputs and actual replies, without hypothetical response-availability labels."""
    from . import structured_retail
    from .construction import demand as observed_demand
    from .construction import parameter_record
    from .policy import response

    result = []
    for episode in episodes:
        require(episode["split"] in ("train", "dev"), "Research supervision is Train/Dev only")
        result.append(("research", episode))
        value = episode["input"]
        linked = "forecast" in value["task"]
        for action in value["task"]["costs"]:
            if value["remaining"] <= 0 or len(value["history"]) >= value["task"]["deadline"]:
                break
            record = parameter_record(value)
            if not linked:
                observed_demand(value, record)
            missing = (
                record["state"].get("F") == "unconfirmed"
                if action == "demand"
                else action not in record["values"]
            )
            if not missing:
                continue
            received = (
                structured_retail.response(episode, value, action)
                if linked
                else response(episode, value, action, 0)
            )
            value = corpus.outcome(value, action, received)
            result.append(
                (
                    "research",
                    {**episode, "id": episode["id"] + ":response-" + action, "input": value},
                )
            )
    if replay:
        seen = {digest(research_input(row["input"])) for _, row in result}
        for mode, row in replay_construction(replay, episodes):
            key = digest(research_input(row["input"]))
            if key not in seen:
                result.append((mode, row))
                seen.add(key)
    return result


def language_view(case, tokenizer, config, labels, collection, round=0, state=None):
    mode, row = case
    if mode == "research":
        view = prepare(
            research_input(row["input"], config["encoder"]),
            tokenizer,
            config["encoder"],
            fields=FIELDS,
            # This loss supervises fields only. Match the constructor's query set:
            # packed encoding and fusion self-attention otherwise change field states.
            actions=[{"id": "hold", "text": TEXT["hold"]}],
            round=round,
        )
        target = research_targets(view, row["input"])
    else:
        view = prepare(
            row, tokenizer, config["encoder"], collection=collection, round=round, state=state
        )
        target = suite_targets(view, row["component"], labels[row["id"]])
        if row["component"] == "abcd" and row.get("split") == "train":
            label = labels[row["id"]]
            key = label["tool"] if label["action"] == "call_tool" else "speak"
            target["weight"] = config["training"].get("actionWeights", {}).get(key, 1.0)
    return view, target


def replay_construction(paths, episodes):
    allowed = {e["id"]: e for e in episodes if e["split"] == "train"}
    seen = {digest(e["input"]) for e in allowed.values()}
    result = []
    for path in paths:
        for row in lines(path):
            require(
                row["split"] == "train" and row["id"] in allowed,
                "Construction replay accepts Train only",
            )
            episode = allowed[row["id"]]
            require(row["family"] == episode["family"], "Construction replay source family differs")
            key = digest(row["input"])
            if key in seen:
                continue
            seen.add(key)
            result.append(
                (
                    "research",
                    {
                        "id": row["id"] + ":state:" + key[:16],
                        "family": row["family"],
                        "split": "train",
                        "input": row["input"],
                        "scenario": episode["scenario"],
                    },
                )
            )
    return result


def balance_cases(cases, config):
    fractions = config.get("mix")
    if not fractions:
        return cases
    require(
        set(fractions) <= {"tools", "research", "public"}
        and math.isclose(sum(fractions.values()), 1),
        "Invalid language task mixture",
    )
    pools = {name: [] for name in fractions}
    for case in cases:
        mode, row = case
        name = (
            "research"
            if mode == "research"
            else "tools"
            if row["component"] == "abcd"
            else "public"
        )
        require(name in pools, "Training component omitted from language mixture")
        pools[name].append(case)
    require(
        all(pools.values()) and all(v > 0 for v in fractions.values()),
        "Empty language mixture component",
    )
    total = max(math.ceil(len(pool) / fractions[name]) for name, pool in pools.items())
    # Every original case remains present; only Train examples are repeated.
    return [
        pool[i % len(pool)]
        for name, pool in pools.items()
        for i in range(math.ceil(total * fractions[name]))
    ]


def action_weights(counts, preserve_prior=False):
    selected = {k: n for k, n in counts.items() if not preserve_prior or k != "speak"}
    weights = {
        k: min(3.0, max(0.25, (sum(selected.values()) / len(selected) / n) ** 0.5))
        for k, n in selected.items()
    }
    normalizer = (
        sum(weights[k] * n for k, n in selected.items()) / sum(selected.values())
        if selected
        else 1.0
    )
    result = {k: w / normalizer for k, w in weights.items()}
    if preserve_prior and "speak" in counts:
        result["speak"] = 1.0
    return result


def conversation_successors(cases):
    groups = defaultdict(list)
    for index, (mode, row) in enumerate(cases):
        if mode == "public" and row["component"] == "abcd":
            require(row["split"] == "train", "Conversation supervision accepts Train only")
            groups[row["id"].rsplit(":", 1)[0]].append((index, row["input"]["history"]))
    result = {}
    for group in groups.values():
        ordered = sorted(group, key=lambda pair: len(pair[1]))
        for (index, history), (next_index, next_history) in zip(ordered, ordered[1:], strict=False):
            if len(next_history) > len(history) and next_history[: len(history)] == history:
                result[index] = next_index
    return result


def language_teacher(model, config, train):
    weight = config["training"].get("distillWeight", 0.0)
    require(math.isfinite(weight) and weight >= 0, "Invalid language distillation weight")
    if not weight:
        return None, None
    public = [row for mode, row in train if mode == "public"]
    require(
        public and all(row.get("split") == "train" for row in public), "Teacher accepts Train only"
    )
    require(config.get("warmStart"), "Language distillation requires a frozen warm start")
    from .structured_progress import restore_rng, rng_state

    random, threads = rng_state(), torch.get_num_threads()
    try:
        # Reload the immutable parent on resume, never the partly trained student.
        teacher, _, parent = initialize(config)
    finally:
        restore_rng(random)
        torch.set_num_threads(threads)
    teacher.to(next(model.parameters()).device).requires_grad_(False).eval()
    return teacher, {"weight": weight, "parent": parent, "scope": "public Train inputs only"}


def teacher_predictions(teacher, views, targets, train):
    if teacher is None:
        return {}, 0
    selected = []
    for index, target in enumerate(targets):
        mode, row = train[target["caseIndex"]]
        if mode == "public":
            require(row.get("split") == "train", "Teacher accepts Train only")
            selected.append((index, row["component"]))
    if not selected:
        return {}, 0
    with torch.no_grad():
        predictions = teacher([views[index] for index, _ in selected])
    return {
        index: (component, prediction)
        for (index, component), prediction in zip(selected, predictions, strict=True)
    }, teacher.last_batch["tokens"]


def language_backward(
    model,
    group,
    tokenizer,
    config,
    train,
    labels,
    collection,
    scale=1.0,
    teacher=None,
    distill_weight=0.0,
):
    """One primary case loss; optional Train-supervised retrieval is averaged into that case."""
    views, targets = zip(*group, strict=True)
    teacher_output, teacher_tokens = teacher_predictions(teacher, views, targets, train)
    output = model(list(views))
    objectives = [
        objective(o, t, no_value=config["noValue"]) for o, t in zip(output, targets, strict=True)
    ]
    expand = [
        "retrievalRound" in t
        and (
            (bool(t.get("needsRetrieval")) and v["selectedChunks"] < v["indexedChunks"])
            or (
                config["encoder"].get("structuredTools", False)
                and t["caseIndex"] % 4 == 0
                and (not config["training"].get("nextTurnTraining") or "nextIndex" in t)
            )
        )
        for v, t in zip(views, targets, strict=True)
    ]
    loss = sum(
        o[0] * (0.5 if follow else 1) * t.get("weight", 1.0)
        for o, follow, t in zip(objectives, expand, targets, strict=True)
    )
    if teacher_output:
        from .structured_policy import distillation

        for index, (component, parent) in teacher_output.items():
            anchor = distillation(output[index], parent, views[index], component)
            loss += distill_weight * anchor * (0.5 if expand[index] else 1)
            objectives[index][1]["distillation"] = float(anchor.detach())
    require(torch.isfinite(loss).item(), "Nonfinite language loss")
    (loss * scale).backward()
    values = [float(loss.detach()) for loss, _ in objectives]
    measured = [m for _, m in objectives]
    tokens = model.last_batch["tokens"]
    followups = []
    for i, follow in enumerate(expand):
        if not follow:
            continue
        state = None
        if config["encoder"].get("structuredTools"):
            from .structured_tools import remember

            current = output[i]
            ids = (
                current["procedure"].topk(min(3, len(current["procedure"]))).indices.tolist()
                if "procedure" in current
                else []
            )
            prediction = assemble(views[i], current, no_value=config["noValue"], use_value=False)
            state = {
                "memory": remember(prediction),
                "procedureIds": [views[i]["procedures"][j]["id"] for j in ids],
            }
        next_index = targets[i].get("nextIndex")
        retrieval = (
            bool(targets[i].get("needsRetrieval"))
            and views[i]["selectedChunks"] < views[i]["indexedChunks"]
        )
        case_index = (
            next_index if next_index is not None and not retrieval else targets[i]["caseIndex"]
        )
        view, target = language_view(
            train[case_index],
            tokenizer,
            config,
            labels,
            collection,
            0 if next_index is not None and not retrieval else targets[i]["retrievalRound"],
            state=state,
        )
        target["caseIndex"] = case_index
        followups.append((i, view, target))
    if followups:
        follow_views = [v for _, v, _ in followups]
        for lo, hi in model.training_batches(follow_views):
            parents, cost = teacher_predictions(
                teacher, follow_views[lo:hi], [t for _, _, t in followups[lo:hi]], train
            )
            teacher_tokens += cost
            follow_outputs = model(follow_views[lo:hi])
            extra_losses = []
            for (i, _, target), out in zip(followups[lo:hi], follow_outputs, strict=True):
                loss, extra = objective(out, target, no_value=config["noValue"])
                require(torch.isfinite(loss).item(), "Nonfinite retrieval loss")
                extra_losses.append(loss * target.get("weight", 1.0))
                values[i] = (values[i] + float(loss.detach())) * 0.5
                measured.append(extra)
            if parents:
                from .structured_policy import distillation

                for index, (component, parent) in parents.items():
                    anchor = distillation(
                        follow_outputs[index], parent, follow_views[lo + index], component
                    )
                    extra_losses[index] += distill_weight * anchor
                    measured[-len(extra_losses) + index]["distillation"] = float(anchor.detach())
            (sum(extra_losses) * 0.5 * scale).backward()
            tokens += model.last_batch["tokens"]
    cost = {"tokens": tokens, "retrievals": sum(expand)}
    if teacher is not None:
        cost["teacherTokens"] = teacher_tokens
    return values, measured, cost


def language_validation(model, case, tokenizer, config, labels, collection):
    needed = config["training"].get("retrievalTraining") == "needed"
    for round in range(config.get("maxRetrievals", 2) + 1 if needed else 1):
        view, target = language_view(case, tokenizer, config, labels, collection, round)
        output = model(view)
        selected = view["actions"][int(output["recovery"].argmax())]["id"]
        if selected != "retrieve" or view["selectedChunks"] == view["indexedChunks"]:
            break
    loss, _ = objective(output, target, no_value=config["noValue"])
    # Missing annotated support cannot improve Dev selection merely by masking span loss.
    return float(loss) + float(bool(target.get("needsRetrieval")))


def train_language(model, tokenizer, config, train, dev, labels, collection, progress=None):
    teacher, distillation = language_teacher(model, config, train)
    optimizer = model.optimizer(config["training"])
    rand = np.random.default_rng(config["seed"])
    report = []
    select_tools = config["training"].get("selection") in ("tools", "calls", "goal", "research")
    best, selected, saved = (-math.inf,) if select_tools else math.inf, 0, None
    resumed = progress.restore("language-progress", model, optimizer) if progress else None
    # State restoration preserves moments and steps; execution kernels follow this run's config.
    configure_optimizer(optimizer, config["training"])
    first = resumed["epoch"] if resumed else 0
    if resumed:
        rand.bit_generator.state = resumed["random"]
        report, best, selected, saved = [
            resumed[k] for k in ("report", "best", "selected", "saved")
        ]
    checks = resumed.get("checks", []) if resumed else []
    best_metrics = resumed.get("bestMetrics") if resumed else None
    if resumed and best_metrics is None and config["training"].get("selection") == "research":
        # Older checkpoints did not store the selected metrics separately.
        for row in [*report, *checks]:
            if (
                row.get("epoch") == selected
                and row.get("accepted", True)
                and row.get("tools")
                and tool_selection_key(row["tools"], row["devLoss"], config) == best
            ):
                best_metrics = row["tools"]
    stale = resumed.get("stale", 0) if resumed else 0
    validation_every = config["training"].get("validationEvery", 0) if select_tools else 0
    maximum = config["training"].get("maxCases", 0)
    require(
        not maximum or (validation_every > 0 and maximum > 0 and maximum % validation_every == 0),
        "Case budget must end at a scheduled Dev check",
    )
    stopped = resumed.get("stopped", False) if resumed else False
    next_turn = conversation_successors(train) if config["training"].get("nextTurnTraining") else {}
    if not resumed and config["training"].get("validateInitial") and select_tools:
        best, dev_loss, tool_metrics = tool_selection(
            model,
            tokenizer,
            config,
            dev,
            labels,
            collection,
            Path(config["output"]) / "language-dev-initial.jsonl",
        )
        best_metrics = tool_metrics
        saved = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        checks.append(
            {
                "epoch": 0,
                "processed": 0,
                "tools": tool_metrics,
                "devLoss": dev_loss,
                "accepted": True,
                "initial": True,
            }
        )
        if progress:
            progress.update("language_dev", **checks[-1])
    if resumed and resumed.get("validateBest") and select_tools:
        from .structured_progress import restore_rng, rng_state

        current = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        random_state = rng_state()
        model.load_state_dict(saved)
        best, dev_loss, tool_metrics = tool_selection(
            model,
            tokenizer,
            config,
            dev,
            labels,
            collection,
            Path(config["output"]) / "language-dev-parent-revalidated.jsonl",
        )
        best_metrics = tool_metrics
        model.load_state_dict(current)
        restore_rng(random_state)
        del current
        stale, stopped = 0, False
        checks.append(
            {
                "epoch": selected,
                "processed": resumed["offset"],
                "tools": tool_metrics,
                "devLoss": dev_loss,
                "accepted": True,
                "revalidatedParent": True,
                "sourceHash": progress.identity["sourceHash"],
            }
        )
        progress.update("language_dev_revalidated", **checks[-1])
    if stopped:
        model.load_state_dict(saved)
        result = {
            "epochs": report,
            "selectedEpoch": selected,
            "checks": checks,
            "earlyStopped": True,
        }
        if distillation:
            result["distillation"] = distillation
        return result
    accumulation = config["training"].get("accumulation", 4)
    batch_size = case_batch(config)
    require(
        batch_size > 0 and accumulation % batch_size == 0, "Case batch must divide accumulation"
    )
    checkpoint_every = config["training"].get("checkpointEvery", 1000)
    require(checkpoint_every % accumulation == 0, "Checkpoint must follow a full optimizer step")
    require(
        not validation_every or validation_every % accumulation == 0,
        "Dev check must follow optimizer step",
    )
    needed = config["training"].get("retrievalTraining") == "needed"
    require(not needed or config.get("maxRetrievals", 2) > 0, "Needed retrieval requires a budget")
    for epoch in range(first, config["training"]["epochs"]):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        batches, details = [], defaultdict(list)
        order, offset = rand.permutation(len(train)).tolist(), 0
        if resumed and epoch == first and resumed["offset"]:
            order, offset, batches = [resumed[k] for k in ("order", "offset", "batches")]
            details.update(resumed["details"])
            # The checkpoint generator state precedes only the remaining retrieval draws.
            rand.bit_generator.state = resumed["random"]
        if maximum:
            order = order[: max(0, maximum - epoch * len(train))]
        epoch_cases = len(order)
        require(epoch_cases > offset, "Training budget has no remaining cases")
        stamp = time.perf_counter()
        pending = 0
        workload = (
            dict(resumed.get("workload", {"tokens": 0, "retrievals": 0}))
            if resumed and epoch == first and offset
            else {"tokens": 0, "retrievals": 0}
        )

        def make(index, round):
            view, target = language_view(
                train[index], tokenizer, config, labels, collection, 0 if needed else round
            )
            if needed:
                target |= {"caseIndex": int(index), "retrievalRound": round + 1}
                if index in next_turn:
                    target["nextIndex"] = next_turn[index]
            if teacher is not None:
                target["caseIndex"] = int(index)
            return view, target

        prepared = prepared_batches(
            order,
            offset,
            batch_size,
            rand,
            make,
            max_round=config.get("maxRetrievals", 2) - 1
            if needed
            else config.get("maxRetrievals", 2),
            workers=config["training"].get("prefetchWorkers", 0),
        )
        for number, group, random_state in prepared:
            views, targets = zip(*group, strict=True)
            segments = model.training_batches(views)
            for lo, hi in segments:
                values, measurements, cost = language_backward(
                    model,
                    list(zip(views[lo:hi], targets[lo:hi], strict=True)),
                    tokenizer,
                    config,
                    train,
                    labels,
                    collection,
                    teacher=teacher,
                    distill_weight=config["training"].get("distillWeight", 0.0),
                )
                batches.extend(values)
                for item in measurements:
                    for name, value in item.items():
                        details[name].append(value)
                for key, value in cost.items():
                    workload[key] = workload.get(key, 0) + value
            pending += len(group)
            processed = number + len(group)
            if pending == accumulation or processed == epoch_cases:
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                torch._foreach_div_(gradients, pending)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                pending = 0
            rand.bit_generator.state = random_state
            progress_every = config["training"].get("progressEvery", 100)
            if progress and processed // progress_every > number // progress_every:
                elapsed = time.perf_counter() - stamp
                rate = (processed - offset) / elapsed
                progress.update(
                    "language",
                    epoch=epoch + 1,
                    processed=processed,
                    total=epoch_cases,
                    casesPerSecond=rate,
                    remainingEpochSeconds=(epoch_cases - processed) / rate,
                    remainingBudgetSeconds=max(0, maximum - epoch * len(train) - processed) / rate
                    if maximum
                    else None,
                    meanLoss=float(np.mean(batches)),
                    encoderBatch=getattr(model, "last_batch", None),
                    plannedCases=len(group),
                    backwardBatches=len(segments),
                    encodedTokens=workload["tokens"],
                    retrievalExamples=workload["retrievals"],
                )
            if validation_every and (processed % validation_every == 0 or processed == epoch_cases):
                candidate_key, dev_loss, tool_metrics = tool_selection(
                    model,
                    tokenizer,
                    config,
                    dev,
                    labels,
                    collection,
                    Path(config["output"]) / f"language-dev-{epoch + 1}-{processed}.jsonl",
                )
                failures = selection_regressions(best_metrics, tool_metrics, config)
                accepted = candidate_key > best and not failures
                meaningful = accepted and candidate_key[:3] > best[:3]
                stale = 0 if meaningful else stale + 1
                if accepted:
                    best, selected = candidate_key, epoch + 1
                    best_metrics = tool_metrics
                    saved = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                seen = epoch * len(train) + processed
                budget_stop = bool(
                    config["training"].get("maxCases") and seen >= config["training"]["maxCases"]
                )
                goal_stop = False
                goal_checks = config["training"].get("stopAtGoalChecks", 0)
                if config["training"].get("selection") == "goal" and goal_checks:
                    from .structured_metrics import performance_goal

                    passed = performance_goal(tool_metrics, 0.0, config["performanceGoal"])[
                        "passed"
                    ]
                    previous = checks[-(goal_checks - 1) :] if goal_checks > 1 else []
                    goal_stop = (
                        passed
                        and len(previous) == goal_checks - 1
                        and all(c.get("languageGoalPassed", False) for c in previous)
                        and seen >= config["training"].get("minGoalCases", 0)
                    )
                else:
                    passed = False
                stopped = (
                    budget_stop
                    or goal_stop
                    or (
                        seen >= config["training"].get("minCases", 0)
                        and stale >= config["training"].get("patience", 3)
                    )
                )
                checks.append(
                    {
                        "epoch": epoch + 1,
                        "processed": processed,
                        "seenCases": seen,
                        "tools": tool_metrics,
                        "devLoss": dev_loss,
                        "accepted": accepted,
                        "retentionFailures": failures,
                        "staleChecks": stale,
                        "earlyStopped": stopped,
                        "budgetStopped": budget_stop,
                        "languageGoalPassed": passed,
                        "goalStopped": goal_stop,
                    }
                )
                if progress:
                    progress.update("language_dev", **checks[-1])
                model.train()
            if progress and processed % checkpoint_every == 0:
                require(pending == 0, "Checkpoint has unapplied gradients")
                progress.checkpoint(
                    "language-progress",
                    model,
                    {
                        "epoch": epoch,
                        "offset": processed,
                        "order": order,
                        "random": rand.bit_generator.state,
                        "batches": batches,
                        "details": dict(details),
                        "report": report,
                        "best": best,
                        "bestMetrics": best_metrics,
                        "selected": selected,
                        "saved": saved,
                        "workload": workload,
                        "checks": checks,
                        "stale": stale,
                        "stopped": stopped,
                    },
                    optimizer,
                )
            if stopped:
                break
        model.eval()
        validation = []
        with torch.inference_mode():
            for case in dev:
                validation.append(
                    language_validation(model, case, tokenizer, config, labels, collection)
                )
        dev_loss = float(np.mean(validation))
        report.append(
            {
                "epoch": epoch + 1,
                "trainLoss": float(np.mean(batches)),
                "devLoss": dev_loss,
                "supervisedHeads": dict(Counter({k: len(v) for k, v in details.items()})),
                "headLosses": {k: float(np.mean(v)) for k, v in details.items()},
                "workload": workload,
            }
        )
        candidate_key, failures = None, []
        if select_tools and not validation_every:
            candidate_key, _, tool_metrics = tool_selection(
                model,
                tokenizer,
                config,
                dev,
                labels,
                collection,
                Path(config["output"]) / f"language-dev-{epoch + 1}.jsonl",
                loss=dev_loss,
            )
            report[-1]["tools"] = tool_metrics
            failures = selection_regressions(best_metrics, tool_metrics, config)
            report[-1]["retentionFailures"] = failures
        accepted = (
            (candidate_key is not None and candidate_key > best and not failures)
            if select_tools
            else (dev_loss < best)
        )
        if not validation_every:
            report[-1]["accepted"] = accepted
        if accepted:
            best, selected = candidate_key if select_tools else dev_loss, epoch + 1
            best_metrics = tool_metrics if select_tools else None
            # CPU copies avoid doubling GPU memory during checkpoint selection.
            saved = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(
            f"Language epoch {epoch + 1}: train={np.mean(batches):.4f}, dev={dev_loss:.4f}",
            flush=True,
        )
        if progress:
            progress.checkpoint(
                "language-progress",
                model,
                {
                    "epoch": epoch + 1,
                    "offset": 0,
                    "order": [],
                    "random": rand.bit_generator.state,
                    "batches": [],
                    "details": {},
                    "report": report,
                    "best": best,
                    "bestMetrics": best_metrics,
                    "selected": selected,
                    "saved": saved,
                    "workload": {"tokens": 0, "retrievals": 0},
                    "checks": checks,
                    "stale": stale,
                    "stopped": stopped,
                },
                optimizer,
            )
        resumed = None
        if stopped or (
            not validation_every
            and config["training"].get("patience")
            and epoch + 1 - selected >= config["training"]["patience"]
        ):
            break
    require(saved is not None, "Language training requires epochs and Dev cases")
    model.load_state_dict(saved)
    result = {
        "epochs": report,
        "selectedEpoch": selected,
        "checks": checks,
        "earlyStopped": stopped,
    }
    if distillation:
        result["distillation"] = distillation
    return result


@torch.inference_mode()
def tool_selection(model, tokenizer, config, dev, labels, collection, path, *, loss=None):
    from .structured_policy import public_validation

    model.eval()
    if config["training"].get("selection") == "research":
        from .structured_retail import parameter_metrics

        router = ResearchRouter(model, tokenizer, config["encoder"])
        selected = [row for mode, row in dev if mode == "research"]
        require(
            selected and all(row["split"] == "dev" for row in selected),
            "Research selection requires Dev cases",
        )
        records = []
        for row in selected:
            state = router.construct(row["input"])
            records.append(
                {
                    "id": row["id"],
                    "family": row["family"],
                    "state": state,
                    "metrics": parameter_metrics(row["input"], state),
                }
            )
        jsonl(path, records)
        metrics = {
            key: float(np.mean([r["metrics"][key] for r in records]))
            for key in (
                "parameterAccuracy",
                "allParametersCorrect",
                "stateAccuracy",
                "rawStateAccuracy",
                "rawParameterAccuracy",
                "typeAccuracy",
                "rawTypeAccuracy",
            )
        }
        metrics["evidenceAccuracy"] = sum(r["metrics"]["evidenceCorrect"] for r in records) / max(
            1, sum(r["metrics"]["evidenceCount"] for r in records)
        )
        if loss is None:
            loss = float(
                np.mean(
                    [
                        language_validation(model, c, tokenizer, config, labels, collection)
                        for c in dev
                    ]
                )
            )
        return tool_selection_key(metrics, loss, config), loss, metrics
    selected = [c for c in dev if c[0] == "public" and c[1]["component"] == "abcd"]
    metrics = public_validation(model, tokenizer, config, selected, labels, collection, path)[
        "abcd"
    ]
    if loss is None:
        loss = float(
            np.mean(
                [language_validation(model, c, tokenizer, config, labels, collection) for c in dev]
            )
        )
    return tool_selection_key(metrics, loss, config), loss, metrics


def selection_regressions(previous, candidate, config):
    """Do not trade grounded parameter correctness for a better aggregate state score."""
    if not previous or config["training"].get("selection") != "research":
        return []
    return [
        {"metric": key, "before": previous[key], "after": candidate.get(key)}
        for key in RESEARCH_METRICS
        if candidate.get(key) is None or candidate[key] + 1e-12 < previous[key]
    ]


def tool_selection_key(metrics, loss, config):
    if config["training"].get("selection") == "research":
        rates = [metrics[k] for k in RESEARCH_METRICS]
        return (min(rates), sum(rates), metrics["allParametersCorrect"], -loss)
    if config["training"].get("selection") == "goal":
        rates = [
            metrics["toolExact"],
            metrics["observableToolAndArgumentsExact"],
            metrics["correctCall"] / max(metrics["predictedCall"], 1e-12),
        ]
        return (min(rates), sum(rates), rates[1], -loss)
    names = (
        ("argumentsDecisionAccuracy", "toolDecisionAccuracy", "actionAccuracy")
        if config["training"].get("selection") == "calls"
        else ("observableToolAndArgumentsExact", "toolExact", "actionAccuracy")
    )
    return tuple(metrics[k] for k in names) + (-loss,)


def train_policy(
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
    from .structured_policy import fit

    return fit(
        model, tokenizer, config, episodes, progress, public_train, public_dev, labels, collection
    )


@torch.inference_mode()
def infer(
    model,
    tokenizer,
    value,
    config,
    collection=(),
    *,
    adapt=True,
    linked=False,
    confirmations=(),
    state=None,
):
    model.eval()
    if linked or "forecast" in value.get("task", {}):
        require(
            "forecast" in value.get("task", {}) and "docs" in value and "historySource" in value,
            "Linked inference requires an explicit Newsvendor task, docs and historySource",
        )
        if state is not None:
            from .structured_forecast import remember

            value = {**value, "memory": remember(value, state)}
        router = ResearchRouter(model, tokenizer, config["encoder"], config.get("noValue", False))
        result = router.decision(value)
        return result, [
            {
                "linkedSources": True,
                "inputHash": digest(research_input(value)),
                "historyHash": digest(value["observations"]),
                "memoryHash": digest(value.get("memory", [])),
                "prediction": result,
            }
        ]
    if value["observations"]:
        return sequence.predict(model.demand, value, demand.horizon(value)), []
    traces = []
    memory_state = list((state or {}).get("memory", []))
    for round in range(config.get("maxRetrievals", 2) + 1):
        view = prepare(
            value,
            tokenizer,
            config["encoder"],
            round=round,
            collection=collection,
            state=state,
            confirmations=confirmations,
        )
        # Public component datasets have no economic costs; their observed-action head is used.
        result = assemble(view, model(view), no_value=config["noValue"], use_value=False)
        if config["encoder"].get("structuredTools"):
            from .structured_tools import remember

            memory_state = remember(result, memory_state)
            result["memory"] = memory_state
        traces.append(
            {
                "round": round,
                "indexedChunks": view["indexedChunks"],
                "selectedChunks": view["selectedChunks"],
                "queryTruncated": view["queryTruncated"],
                "prediction": result,
            }
        )
        if result["action"] != "retrieve":
            break
        state = {
            "memory": memory_state,
            "procedureIds": result.get("procedureIds", []),
            "unresolved": [
                {"field": f["field"], "state": f["state"]}
                for f in result["fields"]
                if f["value"] is None
            ],
            "question": result.get("question", {}),
        }
    if result["action"] == "retrieve":
        result["action"] = "hold"
        result["reason"] = "retrieval_budget_exhausted"
    if not adapt:
        return result, traces
    task = kind(value)
    if task == "contractnli" and result["action"] == "answer":
        result["answer"] = result["decision"]
    elif task == "orsharc":
        if result["action"] == "answer":
            result["answer"] = result["decision"]
        elif result["action"] == "ask":
            result["answer"] = result["question"]["text"]
    elif task == "abcd":
        if result["action"] == "call_tool":
            tool = next(t for t in value["tools"] if t["id"] == result["tool"])
            if isinstance(result["arguments"], dict):
                result["argumentFields"] = result["arguments"]
                result["arguments"] = [result["arguments"].get(s) for s in tool["argumentSlots"]]
        elif result["action"] in ("respond", "hold"):
            result["action"] = "speak"
            result["answer"] = "Please review the observed policy information."
        elif result["action"] in ("ask", "confirm"):
            result["nextAction"] = result["action"]
            result["action"] = "speak"
            result["answer"] = result["question"].get(
                "text", "Please provide or confirm " + result["question"]["field"] + "."
            )
    elif task == "cuad" and result["action"] == "hold":
        result["action"] = "abstain"
    return result, traces


def synchronize(model):
    if next(model.parameters()).is_cuda:
        torch.cuda.synchronize(next(model.parameters()).device)


def evaluate(
    model,
    tokenizer,
    config,
    rows,
    labels,
    collection,
    episodes,
    split="test",
    limit=None,
    progress=None,
):
    model.eval()
    if next(model.parameters()).is_cuda:
        torch.cuda.reset_peak_memory_stats(next(model.parameters()).device)
    selected = [r for r in rows if r["split"] == split]
    if limit:
        selected = [
            r
            for component in sorted({r["component"] for r in selected})
            for r in [v for v in selected if v["component"] == component][:limit]
        ]
    predictions, traces, timings = [], [], []
    for number, row in enumerate(selected):
        synchronize(model)
        start = time.perf_counter()
        output, trace = infer(model, tokenizer, public_input(row), config, collection)
        synchronize(model)
        elapsed = (time.perf_counter() - start) * 1000
        predictions.append({"id": row["id"], "prediction": output, "elapsedMs": elapsed})
        traces.append({"id": row["id"], "trace": trace})
        timings.append(elapsed)
        if progress and (number + 1) % 20 == 0:
            progress.update("evaluate", split=split, processed=number + 1, total=len(selected))
    directory = Path(config["output"])
    jsonl(directory / f"predictions-{split}.jsonl", predictions)
    jsonl(directory / f"traces-{split}.jsonl", traces)
    # Full snapshot scorer counts every case; limited diagnostics retain this denominator.
    metrics = score(
        config["dataset"],
        [{"id": p["id"], "prediction": p["prediction"]} for p in predictions],
        split,
        output=directory / f"scores-{split}",
    )
    retail = [r for r in selected if r["component"] == "retail"]
    periods = sequence.samples(retail, labels, config["demand"]["minHistory"])
    period_records = []
    by_id = {r["id"]: r for r in retail}
    for sample in periods:
        if sample["horizon"] != 7:
            continue
        value = copy.deepcopy(by_id[sample["id"]]["input"])
        history = value["observations"]
        value["observations"] = [r for r in history if r["date"] <= sample["cutoff"]]
        value["request"] = re.sub(
            r"after \d{4}-\d{2}-\d{2}", "after " + sample["cutoff"], value["request"]
        )
        output = sequence.predict(model.demand, value, 7)
        future = {r["date"]: r for r in history}
        label = labels[sample["id"]]
        future.update(
            {
                d: {"sales": s, "stockoutHours": 0 if c else 1}
                for d, s, c in zip(label["dates"], label["answer"], label["complete"], strict=True)
            }
        )
        target = {
            "answer": [future[d]["sales"] for d in sample["dates"]],
            "complete": [future[d]["stockoutHours"] == 0 for d in sample["dates"]],
            "dates": sample["dates"],
        }
        period_records.append(
            {k: sample[k] for k in ("id", "family", "cutoff", "dates", "censored")}
            | {"metrics": demand.metrics(value, target, output), "prediction": output}
        )
    jsonl(directory / f"demand-{split}.jsonl", period_records)
    router = ResearchRouter(model, tokenizer, config["encoder"], config["noValue"])
    controlled = [e for e in episodes if e["split"] == split]
    if limit:
        controlled = [
            e
            for scenario in corpus.SCENARIOS
            for e in [v for v in controlled if v["scenario"] == scenario][:limit]
        ]
    rollouts = []
    for number, episode in enumerate(controlled):
        rollouts.append(rollout(episode, router))
        if progress and (number + 1) % 10 == 0:
            progress.update("evaluate_research", processed=number + 1, total=len(controlled))
    jsonl(directory / f"research-{split}.jsonl", rollouts)
    return {
        "componentMetrics": metrics,
        "evaluatedCases": len(selected),
        "researchTotalLoss": float(np.mean([r["total"] for r in rollouts])),
        "researchInteractions": float(np.mean([r["interactions"] for r in rollouts])),
        "inferenceMs": {
            "median": float(np.median(timings)),
            "p95": float(np.quantile(timings, 0.95)),
        },
        "peakGpuBytes": torch.cuda.max_memory_allocated(next(model.parameters()).device)
        if next(model.parameters()).is_cuda
        else None,
        "scope": "Public component diagnostics and generated controlled rollouts; "
        "no effectiveness claim on real organizational work",
    }


def run(config, device="cpu", limit=None, resume=False, reuse_common=None, continuation=None):
    seed(config["seed"])
    rows, labels, collection = structured_data.load(config)
    procedure_targets = {}
    if config["encoder"].get("structuredTools"):
        from .structured_procedures import annotations

        procedure_targets = annotations(
            rows,
            config["encoder"]["toolSchema"],
            config["encoder"].get("workflowProgress", False),
            config["training"].get("sourceRoleSupervision", False),
            config["encoder"].get("dialogueState", False),
        )
        for key, annotation in procedure_targets.items():
            labels[key] = labels[key] | annotation
        counts = Counter(
            labels[r["id"]]["tool"] if labels[r["id"]]["action"] == "call_tool" else "speak"
            for r in rows
            if r["split"] == "train" and r["component"] == "abcd"
        )
        config["training"]["actionWeights"] = action_weights(
            counts, config["training"].get("preserveCallPrior", False)
        )
    progress = Progress(config, dataset_hashes(config), device, limit, resume)
    if config["encoder"].get("workflowProgress"):
        jsonl(
            progress.directory / "procedure-targets.jsonl",
            [{"id": key, **target} for key, target in sorted(procedure_targets.items())],
        )
        write(
            progress.directory / "procedure-targets.json",
            {
                "scope": "Train targets only; future tools never enter model inputs",
                "hash": digest(procedure_targets),
                "cases": len(procedure_targets),
                "nodeCases": sum("workflowNodes" in t for t in procedure_targets.values()),
                "sourceRoleCases": sum("argumentRoles" in t for t in procedure_targets.values()),
                "pastToolCases": sum("pastTools" in t for t in procedure_targets.values()),
                "pastToolTargets": sum(
                    len(t.get("pastTools", [])) for t in procedure_targets.values()
                ),
                "sourceHashes": progress.identity["datasetHashes"],
            },
        )
    if continuation and not (progress.directory / "language-progress.pt").exists():
        progress.import_language(continuation)
    progress.update("loading")
    research_config = read(config["researchConfig"])
    episodes = corpus.generate(research_config)
    split_report = corpus.audit(episodes)
    tokenizer, encoder = load_backbone(config["encoder"])
    seed(config["seed"])
    model = Router(encoder, config["encoder"]).to(device)
    if str(device).startswith("cuda"):
        require(torch.cuda.is_bf16_supported(), "Configured GPU does not support BF16")
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    training = cases(rows, episodes, "train", limit)
    replay = replay_construction(config.get("constructionReplay", []), episodes)
    if limit:
        replay = [
            case
            for scenario in corpus.SCENARIOS
            for case in [c for c in replay if c[1]["scenario"] == scenario][:limit]
        ]
    training = balance_cases(training + replay, config["training"])
    if replay or config["training"].get("mix"):
        progress.update(
            "language_mixture",
            constructionStates=len(replay),
            cases=len(training),
            components=dict(
                Counter(
                    "research" if mode == "research" else row["component"] for mode, row in training
                )
            ),
        )
    development = cases(rows, episodes, "dev", limit)
    complete = progress.restore("policy-done", model)
    common = complete or progress.restore("common", model)
    if not common and reuse_common:
        path = Path(reuse_common)
        common = progress.import_common(path, model)
        shutil.copyfile(
            path.parent / "demand-crossfit.jsonl", Path(config["output"]) / "demand-crossfit.jsonl"
        )
    if common:
        language, demand_report = common["language"], common["demand"]
    else:
        warm = (
            progress.import_tools(config["warmStart"], model) if config.get("warmStart") else None
        )
        language_done = progress.restore("language-done", model)
        if language_done:
            language = language_done["language"]
        else:
            language = train_language(
                model, tokenizer, config, training, development, labels, collection, progress
            )
            progress.checkpoint("language-done", model, {"language": language})
        progress.update("demand")
        if warm:
            demand_report = warm["demand"]
            shutil.copyfile(
                Path(config["warmStart"]).parent / "demand-crossfit.jsonl",
                Path(config["output"]) / "demand-crossfit.jsonl",
            )
        else:
            forecasts = sequence.samples(rows, labels, config["demand"]["minHistory"])
            fit = [r for r in forecasts if r["split"] == "train"]
            dev = [r for r in forecasts if r["split"] == "dev"]
            demand_report, measurements = sequence.cross_fit(
                model.demand, fit, dev, config["demand"], config["seed"], progress=progress
            )
            jsonl(Path(config["output"]) / "demand-crossfit.jsonl", measurements)
        common = {
            "language": language,
            "demand": demand_report,
            "config": config,
            "continuation": read(progress.directory / "continuation.json")
            if (progress.directory / "continuation.json").exists()
            else None,
            "toolWarmStart": warm["lineage"] if warm else None,
        }
        progress.checkpoint("common", model, common)
    allowed_ids = {e[1]["id"] for e in training + development if e[0] == "research"}
    policies = (
        complete["policy"]
        if complete
        else train_policy(
            model,
            tokenizer,
            config,
            [e for e in episodes if e["id"] in allowed_ids],
            progress,
            training,
            development,
            labels,
            collection,
        )
    )
    progress.checkpoint("policy-done", model, common | {"policy": policies})
    report = {
        "schema": SCHEMA,
        "config": config,
        "limitedCasesPerComponent": limit,
        "language": language,
        "demand": demand_report,
        "policy": policies,
        "trainableParameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "encoderParameters": sum(p.numel() for p in model.encoder.parameters()),
        "trainIdsHash": digest([r[1]["id"] for r in training]),
        "developmentIdsHash": digest([r[1]["id"] for r in development]),
        "researchSplits": split_report,
        "provenance": progress.provenance,
        "continuation": common.get("continuation"),
        "hardware": progress.hardware,
        "sharedCommonTraining": {
            k: common[k] for k in ("reusedCommon", "commonHash", "commonLineage") if k in common
        },
        "toolWarmStart": common.get("toolWarmStart"),
        "trainingSeconds": time.perf_counter()
        - started
        + (
            read(progress.directory / "continuation.json")["previousSeconds"]
            if (progress.directory / "continuation.json").exists()
            else 0
        ),
        "trainingPeakGpuBytes": torch.cuda.max_memory_allocated(device)
        if str(device).startswith("cuda")
        else None,
    }
    evaluation_split = config.get("evaluationSplit", "test")
    if config.get("performanceGoal"):
        from .structured_metrics import performance_goal

        public = policies[-1]["publicDevBaseline"]["abcd"]
        economic_loss = policies[-1]["devActualTotalLoss"]
        report["goalDevelopment"] = performance_goal(
            public, economic_loss, config["performanceGoal"]
        )
        if not report["goalDevelopment"]["passed"]:
            evaluation_split = "dev"
    save(Path(config["output"]) / "model.pt", model, config, report)
    report["evaluation"] = evaluate(
        model,
        tokenizer,
        config,
        rows,
        labels,
        collection,
        episodes,
        split=evaluation_split,
        limit=limit,
        progress=progress,
    )
    write(Path(config["output"]) / "training.json", report)
    progress.update(
        "complete", trainingSeconds=report["trainingSeconds"], evaluation=report["evaluation"]
    )
    return report
