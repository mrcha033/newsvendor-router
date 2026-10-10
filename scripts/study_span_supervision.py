"""Paired Train-only fitting of span pointers with old and corrected loss targets."""

import argparse
import copy
import math
import os
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from torch import nn

from newsvendor import structured_train as training
from newsvendor.bundle import file_hash
from newsvendor.cli import provenance
from newsvendor.io import digest, jsonl, lines, read, require, write
from newsvendor.structured_inputs import best_span, prepare, span_value
from newsvendor.structured_labels import suite_targets
from newsvendor.structured_model import assemble
from newsvendor.structured_tool_eval import weights_hash
from newsvendor.suite_score import metrics


def scores(head, features, indices):
    tokens = nn.utils.rnn.pad_sequence([features[i][0] for i in indices], batch_first=True)
    fields = torch.stack([features[i][1] for i in indices])
    lengths = torch.tensor([len(features[i][0]) for i in indices], device=tokens.device)
    padding = torch.arange(tokens.shape[1], device=tokens.device)[None] >= lengths[:, None]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = {
            name: torch.bmm(layer(fields)[:, None], tokens.transpose(1, 2))[:, 0].float()
            / math.sqrt(256)
            for name, layer in head.items()
        }
    return {name: value.masked_fill(padding, -1e9) for name, value in output.items()}


def loss(output, records, indices, arm):
    start, end = (output[k].log_softmax(-1) for k in ("start", "end"))
    values = []
    for n, index in enumerate(indices):
        pairs = records[index][arm]
        if pairs:
            a, b = torch.tensor(pairs, device=start.device).T
            values.append(-torch.logsumexp(start[n, a] + end[n, b], 0))
    return torch.stack(values).sum() / len(indices) if values else start.sum() * 0


@torch.no_grad()
def measure(head, records, features, indices, batch_size, output=None):
    values, losses = [], []
    for offset in range(0, len(indices), batch_size):
        batch = indices[offset : offset + batch_size]
        out = scores(head, features, batch)
        losses.append(float(loss(out, records, batch, "corrected")) * len(batch))
        for n, i in enumerate(batch):
            row = records[i]
            length = len(features[i][0])
            a, b = best_span(row["view"], out["start"][n, :length], out["end"][n, :length])
            text, loc = span_value(row["view"], a, b)
            prediction = copy.deepcopy(row["parent"])
            if prediction["fields"][0]["mode"] == "span":
                prediction["fields"][0].update(value=text.strip(), evidence=[loc])
                prediction["answer"] = text.strip()
            values.append(
                {
                    "id": row["id"],
                    "family": row["family"],
                    "inputHash": row["inputHash"],
                    "prediction": prediction,
                    "metrics": metrics({"component": "tatqa"}, row["target"], prediction),
                    "pointerMatchesCorrectedTarget": [a, b] in row["corrected"],
                    "span": {"text": text, "location": loc},
                }
            )
    if output:
        jsonl(output, values)
    return {
        "cases": len(values),
        "answerExact": sum(r["metrics"]["answerExact"] for r in values),
        "pointerExact": sum(r["pointerMatchesCorrectedTarget"] for r in values),
        "correctedNll": sum(losses) / len(indices),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True)
    args = parser.parse_args()
    plan = read(args.registration)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == plan["gpu"], "Assigned L40S only")
    require(torch.cuda.device_count() == 1 and "L40S" in torch.cuda.get_device_name(), "L40S only")
    require(provenance({})["sourceHash"] == plan["sourceHash"], "Registered source changed")
    require(file_hash(__file__) == plan["scriptHash"], "Registered study changed")
    for path, expected in plan["files"].items():
        require(file_hash(path) == expected, "Registered asset changed: " + path)
    root = Path(plan["output"])
    require(not root.exists(), "Preserve previous study")
    root.mkdir()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    rows = [
        r
        for r in lines(Path(plan["dataset"]) / "inputs.jsonl")
        if r["split"] == "train" and r["component"] == "tatqa"
    ]
    ids = {r["id"] for r in rows}
    families = sorted({r["family"] for r in rows})
    order = np.random.default_rng(plan["partitionSeed"]).permutation(families)
    held_families = set(order[: math.ceil(len(order) * 0.2)])
    registration = {
        "plan": plan,
        "source": provenance({}),
        "testUsed": False,
        "devUsed": False,
        "newSourceHoldoutUsed": False,
        "inputRows": [
            {"id": r["id"], "family": r["family"], "inputHash": digest(r["input"])} for r in rows
        ],
        "heldFamilies": sorted(held_families),
        "scope": "Only two span pointers are fitted. Parent encoder/fusion/other heads are frozen. Internal Train families may have been seen by the parent: this is a head-selection split, not independent generalization evidence. Original scalar answer scoring is unchanged.",
    }
    write(root / "registered.json", registration)
    labels = {
        r["id"]: r["target"]
        for r in lines(Path(plan["dataset"]) / "labels.jsonl")
        if r["id"] in ids
    }
    rows = [r for r in rows if labels[r["id"]]["answerType"] == "span"]
    legacy_path = Path(plan["legacyLabels"])
    legacy = types.ModuleType("newsvendor._legacy_labels")
    legacy.__package__ = "newsvendor"
    exec(compile(legacy_path.read_bytes(), str(legacy_path), "exec"), legacy.__dict__)
    model, tokenizer, config, _ = training.load(plan["parent"], "cuda")
    model.requires_grad_(False).eval()
    require(weights_hash(model.state_dict()) == plan["parentWeightsHash"], "Wrong parent")
    initial = {
        k: v for k, v in model.pointers.state_dict().items() if k.startswith(("start.", "end."))
    }
    records, features = [], []
    with torch.no_grad():
        for offset in range(0, len(rows), plan["cacheBatch"]):
            batch = rows[offset : offset + plan["cacheBatch"]]
            views = [prepare(r, tokenizer, config["encoder"]) for r in batch]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                encoded = model.encode_batch(views)
                outputs = model.finish_batch(views, encoded)
            for row, view, states, out in zip(batch, views, encoded, outputs, strict=True):
                out = {k: v.float() for k, v in out.items()}
                prediction = assemble(view, out, use_value=False)
                # All annotation access occurs after ordinary public feature construction.
                old = legacy.suite_targets(view, "tatqa", labels[row["id"]])
                new = suite_targets(view, "tatqa", labels[row["id"]])
                a = sorted(set(map(tuple, old["fields"][0].get("spans", []))))
                b = sorted(set(map(tuple, new["fields"][0].get("spans", []))))
                require(set(b) <= set(a), "Corrected target introduced an answer")
                # Force segment construction even when the predicted mode is missing.
                best_span(view, out["start"][0], out["end"][0])
                records.append(
                    {
                        "id": row["id"],
                        "family": row["family"],
                        "inputHash": digest(row["input"]),
                        "target": labels[row["id"]],
                        "legacy": [list(p) for p in a],
                        "corrected": [list(p) for p in b],
                        "parent": prediction,
                        "view": {k: view[k] for k in ("sources", "locations", "segments")},
                    }
                )
                features.append(
                    (
                        states[0].detach().cpu(),
                        out["fieldState"][0].to(torch.bfloat16).detach().cpu(),
                    )
                )
            write(
                root / "progress.json",
                {"stage": "cache", "cases": len(records), "seconds": time.perf_counter() - started},
            )
    require(weights_hash(model.state_dict()) == plan["parentWeightsHash"], "Cache changed weights")
    initial = {k: v.detach().cpu().clone() for k, v in initial.items()}
    torch.save({"features": features, "initial": initial}, root / "features.pt")
    jsonl(root / "targets.jsonl", records)
    del model
    torch.cuda.empty_cache()
    features = [(a.cuda(), b.cuda()) for a, b in features]
    fitting = [i for i, r in enumerate(records) if r["family"] not in held_families]
    held = [i for i, r in enumerate(records) if r["family"] in held_families]
    report = registration | {
        "initialHash": weights_hash(initial),
        "fitCases": len(fitting),
        "heldCases": len(held),
        "conditions": {},
    }

    def make_head():
        head = nn.ModuleDict(
            {name: nn.Linear(256, 256, bias=False) for name in ("start", "end")}
        ).cuda()
        head.load_state_dict(initial, strict=True)
        return head

    report["initial"] = measure(
        make_head(), records, features, held, plan["batchSize"], root / "parent-held.jsonl"
    )
    for random_seed in plan["seeds"]:
        random = np.random.default_rng(random_seed)
        orders = [random.permutation(fitting).tolist() for _ in range(plan["epochs"])]
        write(root / f"batches-{random_seed}.json", orders)
        for arm in ("legacy", "corrected"):
            name = f"{arm}-{random_seed}"
            head = make_head()
            optimizer = torch.optim.AdamW(
                head.parameters(), lr=plan["lr"], weight_decay=0.01, fused=True
            )
            best = (-report["initial"]["answerExact"], report["initial"]["correctedNll"])
            selected, saved, saved_optimizer, history = (
                0,
                copy.deepcopy(head.state_dict()),
                None,
                [],
            )
            for epoch, order in enumerate(orders, 1):
                losses = []
                for offset in range(0, len(order), plan["batchSize"]):
                    batch = order[offset : offset + plan["batchSize"]]
                    optimizer.zero_grad(set_to_none=True)
                    value = loss(scores(head, features, batch), records, batch, arm)
                    require(torch.isfinite(value).item(), "Nonfinite pointer loss")
                    value.backward()
                    torch.nn.utils.clip_grad_norm_(head.parameters(), 5, error_if_nonfinite=True)
                    optimizer.step()
                    losses.append(float(value.detach()))
                measured = measure(head, records, features, held, plan["batchSize"])
                key = (-measured["answerExact"], measured["correctedNll"])
                if key < best:
                    selected, best = epoch, key
                    saved, saved_optimizer = (
                        copy.deepcopy(head.state_dict()),
                        copy.deepcopy(optimizer.state_dict()),
                    )
                history.append({"epoch": epoch, "fitLoss": float(np.mean(losses)), **measured})
                write(
                    root / "progress.json",
                    {
                        "stage": name,
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
                    "seed": random_seed,
                },
                root / f"{name}.pt",
            )
            head.load_state_dict(saved)
            selected_result = measure(
                head, records, features, held, plan["batchSize"], root / f"{name}-held.jsonl"
            )
            report["conditions"][name] = {
                "selectedEpoch": selected,
                "selected": selected_result,
                "history": history,
                "weightsHash": weights_hash(saved),
            }
            write(root / "partial.json", report)
            print({"condition": name, "epoch": selected, **selected_result}, flush=True)
    report.update(
        seconds=time.perf_counter() - started,
        rawHashes={p.name: file_hash(p) for p in root.iterdir() if p.suffix in (".pt", ".jsonl")},
    )
    write(root / "report.json", report)
    print({"complete": True, "seconds": report["seconds"]}, flush=True)


if __name__ == "__main__":
    main()
