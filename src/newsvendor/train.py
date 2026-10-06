import random
from pathlib import Path

import numpy as np
import torch

from .construction import construct, reference, training_rows
from .encoder import ACTIONS, EXTRA, POLICY_EXTRA
from .evaluation import covers
from .heads import Head, calibrate, metrics, train
from .io import digest, require, write
from .policy import features, planner, states

HEADS = {
    "evidence": (32, 2),
    "type": (32, 4),
    "state": (32, 5),
    "relation": (32, 1),
    "value": (32, 1),
    "repair": (32, 5),
    "rule_value": (32, 1),
    "rule_repair": (32, 5),
}


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def head_dim(name, cache):
    extra = POLICY_EXTRA if name in ("value", "repair", "rule_value", "rule_repair") else EXTRA
    return cache.config["dim"] * 2 + extra


def construction_model(config, episodes, dev, cache, value):
    seed(value)
    dim = cache.config["dim"] * 2 + EXTRA
    transitions = config.get("constructionTransitions", True)
    rows = training_rows(episodes, cache, transitions)
    validation = training_rows(dev, cache, transitions)
    model, losses = {"temperatures": {}}, {}
    for name in ("evidence", "type", "state", "relation"):
        model[name] = Head(dim, *HEADS[name])
        losses[name] = train(
            model[name],
            rows[name],
            config["epochs"],
            value,
            mode="choice" if name == "relation" else "class",
            valid=validation[name],
        )
        model["temperatures"][name] = {"temperature": 1.0}
    return model, losses, {k: len(v) for k, v in rows.items()}


def targets(episodes, model, cache, fold=None, constructor=None, context=True):
    rows, repairs, memo, built = [], [], {}, {}

    def builder(input):
        key = digest(input)
        if key not in built:
            built[key] = constructor(input) if constructor else construct(input, model, cache)
        return built[key]

    for episode in episodes:
        for input in states(episode["input"], builder):
            planned = planner(input, builder, memo, scoring="constructed")
            if not planned["state"]["valid"]:
                repairs.append(
                    {
                        "x": features(input, planned["state"], "hold", cache, context=context),
                        "y": list(ACTIONS).index(planned["action"]),
                        "mask": [a in planned["values"] for a in ACTIONS],
                        "id": episode["id"],
                        "family": episode["family"],
                        "stateHash": digest(input),
                        "teacherScoring": "constructed",
                    }
                )
            for action, value in planned["values"].items():
                rows.append(
                    {
                        "x": features(input, planned["state"], action, cache, context=context),
                        "y": value / input["task"]["hold"],
                        "id": episode["id"],
                        "family": episode["family"],
                        "fold": fold,
                        "stateHash": digest(input),
                        "action": action,
                        "teacherScoring": "constructed",
                    }
                )
    return rows, repairs


def policy_model(config, cache, values, repairs, validation, repairdev):
    """Use the same fit procedure for each constructor's own predicted states."""
    seed(config["seed"] + 104)
    model, losses = {"policyContext": config.get("policyContext", True)}, {}
    for name, rows, valid, mode in (
        ("value", values, validation, "value"),
        ("repair", repairs, repairdev, "repair"),
    ):
        require(rows, f"Missing {name} training targets")
        model[name] = Head(head_dim(name, cache), *HEADS[name])
        losses[name] = train(
            model[name],
            rows,
            config["epochs"] * (2 if name == "value" else 1),
            config["seed"] + (104 if name == "value" else 105),
            mode=mode,
            valid=valid,
        )
    return model, losses


def checkpoint(path, model, config, cache, episodes):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": 4,
        "teacherScoring": "constructed",
        "weights": {k: model[k].state_dict() for k in HEADS},
        "temperatures": model["temperatures"],
        "threshold": model["threshold"],
        "policyContext": model.get("policyContext", True),
        "config": config,
        "encoderHash": cache.id,
        "trainingSplitHash": digest([e["id"] for e in episodes if e["split"] == "train"]),
    }
    torch.save(payload, path)


def load(path, cache):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    require(payload.get("schema") == 4, "Checkpoint predates own-state policy targets; retrain it")
    require(payload.get("teacherScoring") == "constructed", "Checkpoint policy teacher differs")
    require(
        payload["encoderHash"] == cache.id, "Checkpoint encoder differs from configured encoder"
    )
    model = {
        "temperatures": payload["temperatures"],
        "threshold": payload["threshold"],
        "policyContext": payload["policyContext"],
    }
    for name, shape in HEADS.items():
        head = Head(head_dim(name, cache), *shape)
        head.load_state_dict(payload["weights"][name])
        model[name] = head.eval()
    return model, payload["config"]


def fit(config, episodes, cache, directory):
    partitions = {
        k: [e for e in episodes if e["split"] == k] for k in ("train", "dev", "cal", "test")
    }
    require(all(partitions.values()), "Missing train/dev/cal/test partition")
    trainset, dev, cal, test = (partitions[k] for k in ("train", "dev", "cal", "test"))
    families = sorted({e["family"] for e in trainset})
    folds = {f: i % config["folds"] for i, f in enumerate(families)}
    values, repairs, reports = [], [], []
    context = config.get("policyContext", True)
    for fold in range(config["folds"]):
        held = [e for e in trainset if folds[e["family"]] == fold]
        rest = [e for e in trainset if folds[e["family"]] != fold]
        model, losses, counts = construction_model(
            config, rest, dev, cache, config["seed"] + fold * 10
        )
        foldrows, foldrepairs = targets(held, model, cache, fold, context=context)
        values.extend(foldrows)
        repairs.extend(foldrepairs)
        reports.append(
            {
                "fold": fold,
                "fitFamilies": sorted({e["family"] for e in rest}),
                "targetFamilies": sorted({e["family"] for e in held}),
                "targets": len(foldrows),
                "targetHash": digest([{k: v for k, v in r.items() if k != "x"} for r in foldrows]),
                "counts": counts,
                "losses": losses,
            }
        )
        print(
            f"Cross-fit {fold + 1}/{config['folds']}: {len(held)} held-out training episodes",
            flush=True,
        )
    model, losses, counts = construction_model(config, trainset, dev, cache, config["seed"] + 100)
    validation, repairdev = targets(dev, model, cache, context=context)
    policies, policy_losses = policy_model(config, cache, values, repairs, validation, repairdev)
    model.update(policies)
    rulevalues, rulerepairs = targets(trainset, None, cache, context=context)
    rulevalid, rulerepairdev = targets(dev, None, cache, context=context)
    rulepolicies, rulelosses = policy_model(
        config, cache, rulevalues, rulerepairs, rulevalid, rulerepairdev
    )
    model.update({"rule_" + name: rulepolicies[name] for name in ("value", "repair")})
    # Calibration is independent of both gradient updates and development checkpoint selection.
    calrows = training_rows(cal, cache)
    testrows = training_rows(test, cache, transitions=False)
    transitionrows = training_rows(test, cache)
    scores = {}
    for name in ("evidence", "type", "state"):
        model["temperatures"][name] = calibrate(model[name], calrows[name])
        scores[name] = metrics(
            model[name], testrows[name], model["temperatures"][name]["temperature"]
        )
    transition_scores = {
        name: metrics(model[name], transitionrows[name], model["temperatures"][name]["temperature"])
        for name in ("evidence", "type", "state")
    }
    logits = model["evidence"].scores([r["x"] for r in calrows["evidence"]])
    temp = model["temperatures"]["evidence"]["temperature"]
    probabilities = torch.tensor(logits / temp).softmax(-1)[:, 1].numpy()
    truth = np.asarray([r["y"] for r in calrows["evidence"]])
    best = {"threshold": 0.5, "f1": -1.0}
    for threshold in np.linspace(0.1, 0.9, 17):
        predicted = probabilities >= threshold
        tp = np.sum(predicted & (truth == 1))
        f1 = 2 * tp / max(1, np.sum(predicted) + np.sum(truth))
        if f1 > best["f1"]:
            best = {"threshold": float(threshold), "f1": float(f1)}
    model["threshold"] = best["threshold"]
    links, expressions, numbers, count, coverage = 0, 0, 0, 0, 0
    for e in test:
        pred, ref = construct(e["input"], model, cache), reference(e["input"])
        for slot, actual in ref["values"].items():
            count += 1
            links += pred["links"].get(slot) == ref["links"][slot]
            expressions += pred["expressions"].get(slot) == ref["expressions"][slot]
            numbers += abs(pred["values"].get(slot, float("inf")) - actual) < 1e-8
        coverage += covers(e["gold"]["theta"], pred["omega"])
    scores["construction"] = {
        "n": count,
        "evidenceAccuracy": links / max(1, count),
        "expressionAccuracy": expressions / max(1, count),
        "valueAccuracy": numbers / max(1, count),
        "initialCoverage": coverage / len(test),
    }
    checkpoint(directory + "/model.pt", model, config, cache, episodes)
    write(
        directory + "/training.json",
        {
            "config": config,
            "counts": counts,
            "losses": losses,
            "constructionTransitions": config.get("constructionTransitions", True),
            "valueLoss": policy_losses["value"],
            "valueTargets": len(values),
            "repairLoss": policy_losses["repair"],
            "repairTargets": len(repairs),
            "rulePolicyLosses": rulelosses,
            "ruleValueTargets": len(rulevalues),
            "ruleRepairTargets": len(rulerepairs),
            "foldReports": reports,
            "thresholdSelection": {**best, "source": "calibration only"},
            "calibrationFamilies": sorted({e["family"] for e in cal}),
            "developmentFamilies": sorted({e["family"] for e in dev}),
            "testFamilies": sorted({e["family"] for e in test}),
            "metrics": scores,
            "transitionMetrics": transition_scores,
            "parameters": {k: sum(p.numel() for p in model[k].parameters()) for k in HEADS},
        },
    )
    return model
