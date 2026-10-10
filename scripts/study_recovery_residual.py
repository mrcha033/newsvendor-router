"""Test a zero-initialized numeric correction to the preserved value head on Train."""

import argparse
import copy
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from evaluate_responses import file_hash
from study_recovery_replay import measure
from torch import nn

from newsvendor.cli import provenance
from newsvendor.heads import Head
from newsvendor.io import lines, read, require, write
from newsvendor.structured_critic import value_loss
from newsvendor.structured_tool_eval import weights_hash


class ResidualHead(nn.Module):
    """Preserve the prior estimate and learn a correction from observable numeric state."""

    def __init__(self, initial, seed):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.base = Head(350, 128, 2)
            self.correction = Head(94, 128, 2)
        self.base.load_state_dict(initial)
        self.base.requires_grad_(False)
        nn.init.zeros_(self.correction.layers[-1].weight)
        nn.init.zeros_(self.correction.layers[-1].bias)

    def forward(self, features):
        require(features.shape[-1] == 350, "Encoded and numeric state required")
        return self.base(features) + self.correction(features[..., 256:])


def gates(candidate, reference):
    return {
        "lowerTargetLoss": candidate["meanTargetLoss"] < reference["meanTargetLoss"] - 1e-10,
        "neitherBenchmarkWorse": all(
            candidate["benchmarks"][k]["targetLoss"]
            <= reference["benchmarks"][k]["targetLoss"] + 1e-10
            for k in reference["benchmarks"]
        ),
        "noMoreFalseHandoffs": all(
            candidate["benchmarks"][k]["falseHandoff"]
            <= reference["benchmarks"][k]["falseHandoff"] + 1e-10
            for k in reference["benchmarks"]
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True)
    args = parser.parse_args()
    plan = read(args.registration)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == plan["gpu"], "Assigned L40S only")
    require(torch.cuda.device_count() == 1 and "L40S" in torch.cuda.get_device_name(0), "L40S only")
    require(provenance({})["sourceHash"] == plan["sourceHash"], "Package source differs")
    require(file_hash(__file__) == plan["scriptHash"], "Registered study source differs")
    for path, expected in plan["files"].items():
        require(file_hash(path) == expected, "Registered asset changed: " + path)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    root = Path(plan["output"])
    require(not root.exists(), "Preserve previous studies")
    root.mkdir()
    rows = lines(Path(plan["encoded"]) / "targets.jsonl")
    require(all(r["split"] == "train" for r in rows), "Train sources only")
    payload = torch.load(Path(plan["numeric"]) / "features.pt", weights_only=True)
    features = [(x.cuda(), y.cuda(), c) for x, y, c in payload["features"]]
    require(len(rows) == len(features) == 1204, "Train cohort changed")
    prior = read(Path(plan["control"]) / "report.json")
    fit_families = set(prior["partitions"]["fit"]["families"])
    held = [i for i, r in enumerate(rows) if r["family"] not in fit_families]
    fitting = [i for i, r in enumerate(rows) if r["family"] in fit_families]
    require(
        {rows[i]["family"] for i in held} == set(prior["partitions"]["held"]["families"]),
        "Source partitions changed",
    )
    initial = payload["initial"]
    initial_hash = weights_hash(initial)
    report = {
        "scope": "Train-only numeric residual comparison with the same actual returns and saved batch draws. Frozen parent value head; no encoder or demand updates.",
        "devUsed": False,
        "testUsed": False,
        "plan": plan,
        "source": provenance({}),
        "partitions": prior["partitions"],
        "baseHeadHash": initial_hash,
        "inner": {},
        "refit": {},
        "gates": {},
    }
    write(root / "registered.json", report)
    started = time.perf_counter()

    def train(stage, random_seed, batches, fixed_epoch=None):
        head = ResidualHead(initial, random_seed).cuda().eval()
        require(weights_hash(head.base.state_dict()) == initial_hash, "Base head changed")
        before = measure(head, features, rows, held)
        require(before == prior["initial"], "Zero residual changed initial predictions")
        trainable = [p for p in head.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=plan["lr"], weight_decay=0.01, fused=True)
        updates = 15 if stage == "inner" else 19
        require(len(batches) == plan["epochs"] * updates, "Matched update budget differs")
        allowed = set(fitting if stage == "inner" else range(len(rows)))
        require(
            all(set(batch) <= allowed and len(batch) == 64 for batch in batches), "Invalid batch"
        )
        name = f"{stage}-{random_seed}"
        saved, saved_optimizer = copy.deepcopy(head.state_dict()), None
        selected, best, history = 0, before["meanTargetLoss"], []
        write(root / f"{name}-batches.json", batches)
        torch.save({"weights": saved, "seed": random_seed}, root / f"{name}-initial.pt")
        for epoch in range(1, plan["epochs"] + 1):
            losses = []
            for batch in batches[(epoch - 1) * updates : epoch * updates]:
                optimizer.zero_grad(set_to_none=True)
                loss = value_loss(head, [features[i] for i in batch], plan["rankWeight"])
                require(torch.isfinite(loss).item(), "Nonfinite residual loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 5, error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach()))
            measured = measure(head, features, rows, held)
            accepted = (
                measured["meanTargetLoss"] < best - 1e-10
                if fixed_epoch is None
                else epoch == fixed_epoch
            )
            if accepted:
                selected, best = epoch, measured["meanTargetLoss"]
                saved, saved_optimizer = (
                    copy.deepcopy(head.state_dict()),
                    copy.deepcopy(optimizer.state_dict()),
                )
            history.append({"epoch": epoch, "loss": float(np.mean(losses)), **measured})
        require(fixed_epoch is None or selected == fixed_epoch, "Fixed refit epoch not preserved")
        require(weights_hash(head.base.state_dict()) == initial_hash, "Parent estimate was updated")
        torch.save(
            {
                "format": "numeric-residual-v1",
                "weights": saved,
                "optimizer": saved_optimizer,
                "finalWeights": head.state_dict(),
                "finalOptimizer": optimizer.state_dict(),
                "seed": random_seed,
                "selectedEpoch": selected,
                "finalEpoch": plan["epochs"],
                "baseHeadHash": initial_hash,
                "parentWeightsHash": plan["parentWeightsHash"],
                "numericState": True,
                "actionPrecision": "float32",
            },
            root / f"{name}.pt",
        )
        head.load_state_dict(saved)
        result = {
            "selectedEpoch": selected,
            "trainableParameters": sum(p.numel() for p in trainable),
            "totalHeadParameters": sum(p.numel() for p in head.parameters()),
            "epochs": history,
            "selected": measure(head, features, rows, held),
            "fit": measure(head, features, rows, fitting),
            "weightsHash": weights_hash(saved),
        }
        print(
            {
                "stage": stage,
                "seed": random_seed,
                "epoch": selected,
                "targetLoss": result["selected"]["meanTargetLoss"],
            },
            flush=True,
        )
        return result

    for random_seed in plan["seeds"]:
        result = train(
            "inner", random_seed, read(Path(plan["control"]) / f"batches-{random_seed}.json")
        )
        report["inner"][str(random_seed)] = result
        report["gates"][str(random_seed)] = gates(
            result["selected"], prior["results"][str(random_seed)]["selected"]
        )
        write(root / "partial.json", report)
    report["eligibleForRefit"] = all(all(g.values()) for g in report["gates"].values())
    if report["eligibleForRefit"]:
        for random_seed in plan["seeds"]:
            result = train(
                "refit",
                random_seed,
                read(Path(plan["fullControl"]) / f"balanced-{random_seed}-batches.json"),
                fixed_epoch=report["inner"][str(random_seed)]["selectedEpoch"],
            )
            report["refit"][str(random_seed)] = result
            inner = report["inner"][str(random_seed)]["selected"]
            current = result["selected"]
            report["gates"][str(random_seed)]["refitPreservesSelectedTrainBehavior"] = all(
                current["benchmarks"][k][metric] <= inner["benchmarks"][k][metric] + 1e-10
                for k in inner["benchmarks"]
                for metric in ("targetLoss", "falseHandoff")
            )
            write(root / "partial.json", report)
    report["eligibleForSequentialTrainCheck"] = bool(report["refit"]) and all(
        all(g.values()) for g in report["gates"].values()
    )
    report["seconds"] = time.perf_counter() - started
    report["peakGpuBytes"] = torch.cuda.max_memory_allocated()
    report["rawHashes"] = {p.name: file_hash(p) for p in root.iterdir() if p.is_file()}
    write(root / "report.json", report)
    print({"completed": True, "seconds": report["seconds"], "gates": report["gates"]}, flush=True)


if __name__ == "__main__":
    main()
