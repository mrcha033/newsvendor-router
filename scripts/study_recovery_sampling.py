"""Compare observed-response sampling with a preserved uniform Train control."""

import argparse
import copy
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from evaluate_responses import file_hash
from study_recovery_replay import measure

from newsvendor.cli import provenance
from newsvendor.heads import Head
from newsvendor.io import jsonl, lines, read, require, write
from newsvendor.structured_critic import value_loss
from newsvendor.structured_tool_eval import weights_hash
from newsvendor.train import seed


def response_group(value):
    """Only responses already observed by the model determine a sampling stratum."""
    answers = [
        h.get("answer") for h in value["history"] if h["action"] in ("c", "p", "v", "b", "demand")
    ]
    if any(a is None or (isinstance(a, str) and a == "no_response") for a in answers):
        return "unanswered"
    if any(isinstance(a, str) and a == "partial" for a in answers):
        return "partial"
    return "answered" if answers else "initial"


def sampling_probabilities(rows, indices, mixture):
    """Mix uniform state sampling with equal benchmark/response/family sampling."""
    require(0 <= mixture <= 1, "Invalid sampling mixture")
    require(indices and len(set(indices)) == len(indices), "Unique fitting indices required")
    require(all(rows[i]["split"] == "train" for i in indices), "Train sources required")
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for position, index in enumerate(indices):
        row = rows[index]
        groups[row["benchmark"]][response_group(row["input"])][row["family"]].append(position)
    weights = np.full(len(indices), (1 - mixture) / len(indices))
    for strata in groups.values():
        for families in strata.values():
            for positions in families.values():
                weights[positions] += mixture / (
                    len(groups) * len(strata) * len(families) * len(positions)
                )
    require(np.isfinite(weights).all() and (weights > 0).all(), "Invalid sampling weights")
    require(np.isclose(weights.sum(), 1), "Sampling mass differs from one")
    return weights / weights.sum()


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
    torch.set_num_threads(4)
    seed(42)
    start = time.perf_counter()
    encoded, numeric = Path(plan["encoded"]), Path(plan["numeric"])
    prior = read(encoded / "report.json")
    control = read(numeric / "report.json")
    rows = lines(encoded / "targets.jsonl")
    payload = torch.load(numeric / "features.pt", map_location="cpu", weights_only=True)
    initial = payload["initial"]
    features = [(x.to("cuda"), y.to("cuda"), c) for x, y, c in payload["features"]]
    require(len(features) == len(rows) == prior["states"], "Cached state count changed")
    families = set(prior["partitions"]["fit"]["families"])
    fit = [i for i, row in enumerate(rows) if row["family"] in families]
    held = [i for i, row in enumerate(rows) if row["family"] not in families]
    require(
        {rows[i]["family"] for i in held} == set(prior["partitions"]["held"]["families"]),
        "Registered source split differs",
    )
    probabilities = sampling_probabilities(rows, fit, plan["mixture"])
    head = Head(features[0][0].shape[-1], 128, 2).to("cuda").eval()
    head.load_state_dict(initial)
    measured_initial = measure(head, features, rows, held)
    require(measured_initial == control["initial"], "Cached initial predictions changed")
    updates = (len(fit) + plan["batchSize"] - 1) // plan["batchSize"]
    for key in ("epochs", "batchSize", "lr", "rankWeight", "seeds"):
        require(plan[key] == control["plan"][key], "Uniform control budget differs: " + key)
    report = {
        "scope": "Observed-only balanced sampling on cached own-state Train returns; frozen parent continuations, not full candidate-policy outcomes.",
        "testUsed": False,
        "devUsed": False,
        "plan": plan,
        "source": provenance({}),
        "initial": measured_initial,
        "initialHeadHash": weights_hash(initial),
        "fitStates": len(fit),
        "heldStates": len(held),
        "partitions": prior["partitions"],
        "updatesPerEpoch": updates,
        "results": {},
    }
    jsonl(
        root / "sampling.jsonl",
        [
            {
                "index": i,
                "stateKey": rows[i]["inputHash"],
                "family": rows[i]["family"],
                "benchmark": rows[i]["benchmark"],
                "response": response_group(rows[i]["input"]),
                "probability": float(p),
            }
            for i, p in zip(fit, probabilities, strict=True)
        ],
    )
    write(root / "registered.json", report)
    for random_seed in plan["seeds"]:
        head.load_state_dict(initial)
        random = np.random.default_rng(random_seed)
        optimizer = torch.optim.AdamW(
            head.parameters(), lr=plan["lr"], weight_decay=0.01, fused=True
        )
        best, selected = measured_initial["meanTargetLoss"], 0
        saved, saved_optimizer = copy.deepcopy(initial), None
        history, orders = [], []
        counts = np.zeros(len(rows), dtype=np.int64)
        for epoch in range(plan["epochs"]):
            losses = []
            for _ in range(updates):
                order = random.choice(fit, size=plan["batchSize"], replace=True, p=probabilities)
                orders.append(order.tolist())
                np.add.at(counts, order, 1)
                optimizer.zero_grad(set_to_none=True)
                loss = value_loss(head, [features[int(i)] for i in order], plan["rankWeight"])
                require(torch.isfinite(loss).item(), "Nonfinite value loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 5, error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach()))
            measured = measure(head, features, rows, held)
            if measured["meanTargetLoss"] < best - 1e-10:
                best, selected = measured["meanTargetLoss"], epoch + 1
                saved, saved_optimizer = (
                    copy.deepcopy(head.state_dict()),
                    copy.deepcopy(optimizer.state_dict()),
                )
            history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), **measured})
        final = copy.deepcopy(head.state_dict())
        head.load_state_dict(saved)
        key = str(random_seed)
        report["results"][key] = {
            "selectedEpoch": selected,
            "selected": measure(head, features, rows, held),
            "fit": measure(head, features, rows, fit),
            "epochs": history,
            "sampleCounts": counts.tolist(),
        }
        torch.save(
            {
                "weights": saved,
                "optimizer": saved_optimizer,
                "finalWeights": final,
                "finalOptimizer": optimizer.state_dict(),
                "seed": random_seed,
                "selectedEpoch": selected,
                "finalEpoch": plan["epochs"],
                "numericState": True,
                "actionPrecision": "float32",
                "parentWeightsHash": prior["plan"]["weightsHash"],
                "initialHeadHash": report["initialHeadHash"],
            },
            root / f"balanced-{random_seed}.pt",
        )
        write(root / f"batches-{random_seed}.json", orders)
        write(root / "partial.json", report)
        print({"seed": random_seed, "epoch": selected, "heldLoss": best}, flush=True)
    report["gates"] = {}
    for random_seed in plan["seeds"]:
        a = control["results"][f"continuations-{random_seed}"]["selected"]
        b = report["results"][str(random_seed)]["selected"]
        report["gates"][str(random_seed)] = {
            "improvesUniformControl": b["meanTargetLoss"] < a["meanTargetLoss"] - 1e-10,
            "neitherBenchmarkWorse": all(
                b["benchmarks"][k]["targetLoss"] <= a["benchmarks"][k]["targetLoss"] + 1e-10
                for k in b["benchmarks"]
            ),
            "noMoreFalseHandoffs": all(
                b["benchmarks"][k]["falseHandoff"] <= a["benchmarks"][k]["falseHandoff"] + 1e-10
                for k in b["benchmarks"]
            ),
        }
    report["eligibleForSequentialTrainCheck"] = all(
        all(g.values()) for g in report["gates"].values()
    )
    report["seconds"] = time.perf_counter() - start
    report["peakGpuBytes"] = torch.cuda.max_memory_allocated()
    report["rawHashes"] = {p.name: file_hash(p) for p in root.iterdir() if p.is_file()}
    write(root / "report.json", report)
    print({"seconds": report["seconds"], "gates": report["gates"]}, flush=True)


if __name__ == "__main__":
    main()
