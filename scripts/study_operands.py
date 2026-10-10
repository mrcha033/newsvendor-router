"""Compare joint operands with independent pointer fitting on fixed Train representations."""

import argparse
import copy
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from torch import nn

from newsvendor.bundle import file_hash
from newsvendor.cli import provenance
from newsvendor.io import jsonl, lines, read, require, write
from newsvendor.structured_inputs import MODES, OPS
from newsvendor.structured_model import compute
from newsvendor.structured_operands import OperandPairs, positive_pairs, proposals
from newsvendor.structured_tool_eval import weights_hash
from newsvendor.suite_score import metrics, number


def pack(items, indices):
    return {
        key: (
            torch.stack([items[i][key][0] for i in indices])
            if key in ("field", "operation")
            else nn.utils.rnn.pad_sequence([items[i][key][0] for i in indices], batch_first=True)
        )
        for key in items[0]
    }


def score(head, arm, features, pairs, indices):
    if arm == "joint":
        data = pack(pairs, indices)
        return head(data), data["pairs"]
    fields = torch.stack([features[i]["field"][0] for i in indices])
    atoms = nn.utils.rnn.pad_sequence([features[i]["atoms"] for i in indices], batch_first=True)
    lengths = torch.tensor([len(features[i]["atoms"]) for i in indices], device=fields.device)
    mask = torch.arange(atoms.shape[1], device=fields.device)[None] >= lengths[:, None]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = [
            (torch.bmm(head[name](fields)[:, None], atoms.transpose(1, 2))[:, 0] / math.sqrt(256))
            .float()
            .masked_fill(mask, -1e9)
            for name in ("operand1", "operand2")
        ]
    return out, None


def losses(output, arm, targets, indices):
    names = ["pair"] if arm == "joint" else ["first", "second"]
    outputs = [output] if arm == "joint" else output
    values = []
    for name, logits in zip(names, outputs, strict=True):
        positive = nn.utils.rnn.pad_sequence([targets[i][name] for i in indices], batch_first=True)
        active = positive.any(-1)
        values.append(
            -torch.logsumexp(logits.log_softmax(-1).masked_fill(~positive, -1e9), -1) * active
        )
    return sum(values)


def outcome(row, picked):
    parent = row["parent"]
    active = parent["mode"] == "compute" and parent.get("reason") not in (
        "missing",
        "conflict",
        "unsupported_value",
    )
    ids = picked[:1] if OPS[row["operation"]] == "copy" else picked
    value = parent["value"]
    if active:
        try:
            value = compute(OPS[row["operation"]], [row["atoms"][i]["value"] for i in ids])
        except ValueError:
            value = None
    result = {
        "value": value,
        "activeCalculation": active,
        "chosenOperands": ids if active else [],
        "sourceCorrect": None,
    }
    target = row["target"]
    if "operand1" in target:
        result["sourceCorrect"] = bool(
            active
            and row["operation"] == target["relation"]
            and picked[0] in target["operand1"]
            and ("operand2" not in target or picked[1] in target["operand2"])
        )
    if row["kind"] == "public":
        prediction = {**row["parentAnswer"], "answer": value}
        result["prediction"] = prediction
        result["metrics"] = metrics({"component": "tatqa"}, row["answerTarget"], prediction)
    else:
        expected = None
        supported = "operand1" in target or target.get("mode") in (
            MODES.index("missing"),
            MODES.index("conflict"),
        )
        if "operand1" in target:
            gold = [target["operand1"][0]] + (
                [target["operand2"][0]] if "operand2" in target else []
            )
            expected = compute(OPS[target["relation"]], [row["atoms"][i]["value"] for i in gold])
        actual = number(value)
        result["scalarCorrect"] = (
            None
            if not supported
            else (
                value is None
                if expected is None
                else actual is not None and abs(actual - expected) <= 1e-7 * max(1, abs(expected))
            )
        )
    return result


@torch.no_grad()
def measure(head, arm, rows, features, pairs, targets, indices, size, path):
    records = []
    counts = Counter()
    nll = Counter()
    for offset in range(0, len(indices), size):
        batch = indices[offset : offset + size]
        output, candidates = score(head, arm, features, pairs, batch)
        raw_loss = losses(output, arm, targets, batch).tolist()
        selected = (
            candidates[torch.arange(len(batch), device=output.device), output.argmax(-1)].tolist()
            if arm == "joint"
            else torch.stack([out.argmax(-1) for out in output], -1).tolist()
        )
        for index, chosen, value in zip(batch, selected, raw_loss, strict=True):
            row = rows[index]
            measured = outcome(row, chosen)
            kind = row["kind"]
            counts[kind + "Count"] += 1
            nll[kind] += value
            if kind == "public":
                counts["publicExact"] += measured["metrics"]["answerExact"]
                counts["publicRoundedExact"] += measured["metrics"]["answerRoundedExact"]
            else:
                if measured["scalarCorrect"] is not None:
                    counts["researchScalarCount"] += 1
                    counts["researchScalarCorrect"] += measured["scalarCorrect"]
                if measured["sourceCorrect"] is not None:
                    counts["researchSourceCount"] += 1
                    counts["researchSourceCorrect"] += measured["sourceCorrect"]
            records.append(
                {
                    "id": row["id"],
                    "family": row["family"],
                    "field": row["field"],
                    "inputHash": row["inputHash"],
                    "loss": value,
                    **measured,
                }
            )
    if path is not None:
        jsonl(path, records)
    return dict(counts) | {
        "balancedNll": sum(nll[k] / counts[k + "Count"] for k in ("public", "research")) / 2
    }


def preserved(value, parent):
    return all(value[k] >= parent[k] for k in ("researchScalarCorrect", "researchSourceCorrect"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True)
    args = parser.parse_args()
    plan = read(args.registration)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == plan["gpu"], "Assigned L40S only")
    require(torch.cuda.device_count() == 1 and "L40S" in torch.cuda.get_device_name(), "L40S only")
    require(provenance({})["sourceHash"] == plan["sourceHash"], "Package source changed")
    require(file_hash(__file__) == plan["scriptHash"], "Study script changed")
    cache = Path(plan["cache"])
    cached = read(cache / "report.json")
    require(file_hash(cache / "report.json") == plan["cacheReportHash"], "Registered cache changed")
    for name, expected in cached["rawHashes"].items():
        require(file_hash(cache / name) == expected, "Cached input changed: " + name)
    root = Path(plan["output"])
    require(not root.exists(), "Preserve previous fitting")
    root.mkdir()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    rows = lines(cache / "rows.jsonl")
    payload = torch.load(cache / "features.pt", map_location="cpu", weights_only=True)
    features = [{k: v.cuda() for k, v in f.items()} for f in payload["features"]]
    initial = payload["initial"]
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
    held_families = set()
    for n, kind in enumerate(("public", "research")):
        names = sorted({r["family"] for r in rows if r["kind"] == kind})
        shuffled = np.random.default_rng(plan["partitionSeed"] + n).permutation(names)
        held_families.update(shuffled[: math.ceil(len(shuffled) * 0.2)])
    fitting = [i for i, r in enumerate(rows) if r["family"] not in held_families]
    held = [i for i, r in enumerate(rows) if r["family"] in held_families]
    counts = Counter(rows[i]["kind"] for i in fitting)
    weights = torch.tensor([len(fitting) / (2 * counts[r["kind"]]) for r in rows], device="cuda")
    coverage = {
        kind: {
            "rows": sum(r["kind"] == kind for r in rows),
            "supervisedOperands": sum(
                r["kind"] == kind and "operand1" in r["target"] for r in rows
            ),
            "positiveCandidateRows": sum(
                r["kind"] == kind and bool(t["pair"].any())
                for r, t in zip(rows, targets, strict=True)
            ),
        }
        for kind in ("public", "research")
    }
    report = {
        "plan": plan,
        "source": provenance({}),
        "testUsed": False,
        "devUsed": False,
        "newSourceHoldoutUsed": False,
        "coverage": coverage,
        "partitions": {
            name: {"rows": len(group), "families": sorted({rows[i]["family"] for i in group})}
            for name, group in [("fit", fitting), ("held", held)]
        },
        "conditions": {},
    }
    write(root / "registered.json", report)

    def make(arm, seed):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            if arm == "joint":
                return OperandPairs().cuda()
            head = nn.ModuleDict(
                {name: nn.Linear(256, 256, bias=False) for name in ("operand1", "operand2")}
            ).cuda()
            head.load_state_dict(initial)
            return head

    parent = measure(
        make("joint", 42),
        "joint",
        rows,
        features,
        pairs,
        targets,
        held,
        plan["batchSize"],
        root / "parent-held.jsonl",
    )
    baseline = measure(
        make("pointers", 42),
        "pointers",
        rows,
        features,
        pairs,
        targets,
        held,
        plan["batchSize"],
        root / "initial-pointers-held.jsonl",
    )
    require(
        all(parent[k] == baseline[k] for k in parent if k != "balancedNll"),
        "Initial conditions predict different outcomes",
    )
    for a, b in zip(
        lines(root / "parent-held.jsonl"), lines(root / "initial-pointers-held.jsonl"), strict=True
    ):
        require(
            {k: v for k, v in a.items() if k != "loss"}
            == {k: v for k, v in b.items() if k != "loss"},
            "Initial per-field predictions differ",
        )
    report["initial"] = {"joint": parent, "pointers": baseline}
    for seed in plan["seeds"]:
        rng = np.random.default_rng(seed)
        orders = [rng.permutation(fitting).tolist() for _ in range(plan["epochs"])]
        write(root / f"batches-{seed}.json", orders)
        for arm in ("pointers", "joint"):
            name = f"{arm}-{seed}"
            head = make(arm, seed)
            optimizer = torch.optim.AdamW(
                head.parameters(), lr=plan["lr"], weight_decay=0.01, fused=True
            )
            first = report["initial"][arm]
            best = (-first["publicExact"], first["balancedNll"])
            selected, saved, saved_optimizer, history = (
                0,
                copy.deepcopy(head.state_dict()),
                None,
                [],
            )
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
                measured = measure(
                    head,
                    arm,
                    rows,
                    features,
                    pairs,
                    targets,
                    held,
                    plan["batchSize"],
                    root / f"{name}-epoch-{epoch}.jsonl",
                )
                key = (-measured["publicExact"], measured["balancedNll"])
                if preserved(measured, first) and key < best:
                    selected, best = epoch, key
                    saved, saved_optimizer = (
                        copy.deepcopy(head.state_dict()),
                        copy.deepcopy(optimizer.state_dict()),
                    )
                history.append({"epoch": epoch, "batchLosses": raw_losses, **measured})
                torch.save(
                    {"weights": head.state_dict(), "epoch": epoch},
                    root / f"{name}-epoch-{epoch}.pt",
                )
                write(
                    root / "progress.json",
                    {
                        "condition": name,
                        "epoch": epoch,
                        "selected": selected,
                        "seconds": time.perf_counter() - started,
                    },
                )
            torch.save(
                {
                    "weights": saved,
                    "optimizer": saved_optimizer,
                    "finalWeights": head.state_dict(),
                    "finalOptimizer": optimizer.state_dict(),
                    "selectedEpoch": selected,
                    "seed": seed,
                },
                root / f"{name}.pt",
            )
            head.load_state_dict(saved)
            measured = measure(
                head,
                arm,
                rows,
                features,
                pairs,
                targets,
                held,
                plan["batchSize"],
                root / f"{name}-selected.jsonl",
            )
            report["conditions"][name] = {
                "selectedEpoch": selected,
                "selected": measured,
                "history": history,
                "weightsHash": weights_hash(saved),
                "parameters": sum(p.numel() for p in head.parameters()),
            }
            write(root / "partial.json", report)
            print({"condition": name, "epoch": selected, **measured}, flush=True)
    eligible = []
    report["gates"] = {}
    for arm in ("pointers", "joint"):
        runs = [report["conditions"][f"{arm}-{s}"]["selected"] for s in plan["seeds"]]
        gate = all(
            preserved(v, report["initial"][arm])
            and v["publicExact"] > report["initial"][arm]["publicExact"]
            for v in runs
        )
        report["gates"][arm] = gate
        if gate:
            eligible.append(
                (sum(v["publicExact"] for v in runs) / len(runs), arm == "pointers", arm)
            )
    report["selectedCondition"] = max(eligible)[2] if eligible else None
    report["eligibleForRefit"] = bool(eligible)
    report.update(
        seconds=time.perf_counter() - started,
        peakGpuGiB=torch.cuda.max_memory_allocated() / 1024**3,
        rawHashes={p.name: file_hash(p) for p in root.iterdir() if p.suffix in (".pt", ".jsonl")},
    )
    write(root / "report.json", report)
    print(
        {"complete": True, "seconds": report["seconds"], "condition": report["selectedCondition"]},
        flush=True,
    )


if __name__ == "__main__":
    main()
