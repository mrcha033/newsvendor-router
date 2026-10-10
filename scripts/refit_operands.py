"""Refit the Train-selected operand condition at fixed epochs on all Train rows."""

import argparse
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from study_operands import losses, measure, score
from torch import nn

from newsvendor.bundle import file_hash
from newsvendor.cli import provenance
from newsvendor.io import lines, read, require, write
from newsvendor.structured_operands import OperandPairs, positive_pairs, proposals
from newsvendor.structured_tool_eval import weights_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True)
    args = parser.parse_args()
    plan = read(args.registration)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == plan["gpu"], "Assigned L40S only")
    require(torch.cuda.device_count() == 1 and "L40S" in torch.cuda.get_device_name(), "L40S only")
    require(provenance({})["sourceHash"] == plan["sourceHash"], "Package source changed")
    require(file_hash(__file__) == plan["scriptHash"], "Refit script changed")
    for path, expected in plan["files"].items():
        require(file_hash(path) == expected, "Registered input changed: " + path)
    study = read(Path(plan["study"]) / "report.json")
    require(study["eligibleForRefit"], "Train study gate failed")
    arm = study["selectedCondition"]
    require(arm == plan["condition"], "Selected condition changed")
    cache = Path(plan["cache"])
    cached = read(cache / "report.json")
    for name, expected in cached["rawHashes"].items():
        require(file_hash(cache / name) == expected, "Cache changed: " + name)
    root = Path(plan["output"])
    require(not root.exists(), "Preserve previous refits")
    root.mkdir()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    rows = lines(cache / "rows.jsonl")
    require(all(r["split"] == "train" for r in rows), "Train only")
    payload = torch.load(cache / "features.pt", map_location="cpu", weights_only=True)
    features = [{k: v.cuda() for k, v in f.items()} for f in payload["features"]]
    pairs, targets = [], []
    for row, feature in zip(rows, features, strict=True):
        data = proposals({"atoms": row["atoms"]}, feature["field"], feature["atoms"], feature)
        one = torch.zeros(len(row["atoms"]), dtype=torch.bool, device="cuda")
        two = torch.zeros_like(one)
        one[row["target"].get("operand1", [])] = True
        two[row["target"].get("operand2", [])] = True
        targets.append(
            {
                "pair": positive_pairs(data["pairs"][0], data["mask"][0], row["target"]),
                "first": one,
                "second": two,
            }
        )
        pairs.append(data)
    indices = list(range(len(rows)))
    counts = Counter(r["kind"] for r in rows)
    weights = torch.tensor([len(rows) / (2 * counts[r["kind"]]) for r in rows], device="cuda")
    report = {
        "plan": plan,
        "source": provenance({}),
        "testUsed": False,
        "devUsed": False,
        "newSourceHoldoutUsed": False,
        "condition": arm,
        "rows": len(rows),
        "refits": {},
    }
    write(root / "registered.json", report)
    for seed in plan["seeds"]:
        epochs = study["conditions"][f"{arm}-{seed}"]["selectedEpoch"]
        require(epochs == plan["epochs"][str(seed)] and epochs > 0, "Fixed epoch changed")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            if arm == "joint":
                head = OperandPairs().cuda()
            else:
                head = nn.ModuleDict(
                    {name: nn.Linear(256, 256, bias=False) for name in ("operand1", "operand2")}
                ).cuda()
                head.load_state_dict(payload["initial"])
        optimizer = torch.optim.AdamW(
            head.parameters(), lr=plan["lr"], weight_decay=0.01, fused=True
        )
        rng = np.random.default_rng(seed)
        orders = [rng.permutation(indices).tolist() for _ in range(epochs)]
        write(root / f"batches-{seed}.json", orders)
        initial_hash = weights_hash(head.state_dict())
        history = []
        for epoch, order in enumerate(orders, 1):
            raw_losses = []
            for offset in range(0, len(order), plan["batchSize"]):
                batch = order[offset : offset + plan["batchSize"]]
                optimizer.zero_grad(set_to_none=True)
                output, _ = score(head, arm, features, pairs, batch)
                loss = (losses(output, arm, targets, batch) * weights[batch]).mean()
                require(torch.isfinite(loss).item(), "Nonfinite operand loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 5, error_if_nonfinite=True)
                optimizer.step()
                raw_losses.append(float(loss.detach()))
            history.append({"epoch": epoch, "batchLosses": raw_losses})
            torch.save(
                {"weights": head.state_dict(), "epoch": epoch}, root / f"{seed}-epoch-{epoch}.pt"
            )
            write(
                root / "progress.json",
                {"seed": seed, "epoch": epoch, "seconds": time.perf_counter() - started},
            )
        torch.save(
            {
                "weights": head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "seed": seed,
                "epochs": epochs,
                "parentWeightsHash": plan["parentWeightsHash"],
            },
            root / f"refit-{seed}.pt",
        )
        measured = measure(
            head,
            arm,
            rows,
            features,
            pairs,
            targets,
            indices,
            plan["batchSize"],
            root / f"refit-{seed}-all-train.jsonl",
        )
        report["refits"][str(seed)] = {
            "epochs": epochs,
            "initialWeightsHash": initial_hash,
            "weightsHash": weights_hash(head.state_dict()),
            "diagnostics": measured,
            "history": history,
        }
        write(root / "partial.json", report)
        print({"seed": seed, "epochs": epochs, **measured}, flush=True)
    report.update(
        seconds=time.perf_counter() - started,
        peakGpuGiB=torch.cuda.max_memory_allocated() / 1024**3,
        rawHashes={p.name: file_hash(p) for p in root.iterdir() if p.suffix in (".pt", ".jsonl")},
    )
    write(root / "report.json", report)
    print({"complete": True, "seconds": report["seconds"]}, flush=True)


if __name__ == "__main__":
    main()
