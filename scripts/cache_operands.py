"""Cache frozen observed arithmetic representations for paired Train-only head studies."""

import argparse
import copy
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from newsvendor import corpus
from newsvendor import structured_train as training
from newsvendor.bundle import file_hash
from newsvendor.cli import provenance
from newsvendor.io import digest, jsonl, lines, read, require, write
from newsvendor.structured_inputs import prepare, research_input
from newsvendor.structured_labels import research_targets, suite_targets
from newsvendor.structured_model import Router, assemble, extract
from newsvendor.structured_rollout import FIELDS, TEXT
from newsvendor.structured_tool_eval import weights_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True)
    args = parser.parse_args()
    plan = read(args.registration)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == plan["gpu"], "Assigned L40S only")
    require(torch.cuda.device_count() == 1 and "L40S" in torch.cuda.get_device_name(), "L40S only")
    require(provenance({})["sourceHash"] == plan["sourceHash"], "Registered source changed")
    require(file_hash(__file__) == plan["cacheScriptHash"], "Cache script changed")
    for path, expected in plan["files"].items():
        require(file_hash(path) == expected, "Registered asset changed: " + path)
    root = Path(plan["cache"])
    require(not root.exists(), "Preserve earlier cache")
    root.mkdir()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    public = [
        r
        for r in lines(Path(plan["dataset"]) / "inputs.jsonl")
        if r["split"] == "train" and r["component"] == "tatqa"
    ]
    episodes = [e for e in corpus.generate(read(plan["researchConfig"])) if e["split"] == "train"]
    research = training.research_cases(episodes)
    registration = {
        "plan": plan,
        "source": provenance({}),
        "testUsed": False,
        "devUsed": False,
        "newSourceHoldoutUsed": False,
        "publicInputs": [
            {"id": r["id"], "family": r["family"], "inputHash": digest(r["input"])} for r in public
        ],
        "researchInputs": [
            {"id": r["id"], "family": r["family"], "inputHash": digest(r["input"])}
            for _, r in research
        ],
    }
    write(root / "registered.json", registration)
    ids = {r["id"] for r in public}
    labels = {
        r["id"]: r["target"]
        for r in lines(Path(plan["dataset"]) / "labels.jsonl")
        if r["id"] in ids
    }
    cases = [
        ("public", r) for r in public if labels[r["id"]]["answerType"] == "arithmetic"
    ] + research
    parent, tokenizer, config, _ = training.load(plan["parent"], "cuda")
    require(weights_hash(parent.state_dict()) == plan["parentWeightsHash"], "Wrong parent")
    encoder_config = copy.deepcopy(config["encoder"])
    encoder_config["jointOperands"] = True
    model = Router(parent.encoder, encoder_config).cuda()
    training.import_core(model, parent.state_dict())
    require(
        weights_hash(
            {k: v for k, v in model.state_dict().items() if not k.startswith("operand_pairs.")}
        )
        == plan["parentWeightsHash"],
        "Frozen parent changed",
    )
    initial = {
        k: v.detach().cpu().clone()
        for k, v in parent.pointers.state_dict().items()
        if k.startswith(("operand1.", "operand2."))
    }
    del parent
    model.requires_grad_(False).eval()
    features, records = [], []
    input_rows = []
    counts = {
        "cases": 0,
        "publicCases": 0,
        "researchStates": 0,
        "zeroCorrectionExactCases": 0,
        "rowsWithoutAtoms": 0,
    }
    with torch.no_grad():
        for offset in range(0, len(cases), plan["cacheBatch"]):
            batch = cases[offset : offset + plan["cacheBatch"]]
            views = []
            for mode, row in batch:
                observed = (
                    row["input"]
                    if mode == "public"
                    else research_input(row["input"], encoder_config)
                )
                options = (
                    {}
                    if mode == "public"
                    else {"fields": FIELDS, "actions": [{"id": "hold", "text": TEXT["hold"]}]}
                )
                views.append(prepare(observed, tokenizer, encoder_config, **options))
                input_rows.append(
                    {
                        "id": row["id"],
                        "family": row["family"],
                        "split": "train",
                        "kind": mode,
                        "input": observed,
                        "inputHash": digest(observed),
                    }
                )
            outputs = model(views)
            for (mode, row), view, out in zip(batch, views, outputs, strict=True):
                legacy = {
                    k: v
                    for k, v in out.items()
                    if k not in ("operandJoint", "operandPairs", "operandPairMask")
                }
                require(
                    extract(view, legacy) == extract(view, out),
                    "Zero pair correction changed extraction",
                )
                counts["cases"] += 1
                counts["zeroCorrectionExactCases"] += 1
                counts["publicCases" if mode == "public" else "researchStates"] += 1
                # Loss annotations are resolved only after ordinary observed feature encoding.
                targets = (
                    suite_targets(view, "tatqa", labels[row["id"]])
                    if mode == "public"
                    else research_targets(view, row["input"])
                )
                prediction = assemble(view, out, use_value=False)
                for i, field in enumerate(view["fields"]):
                    if mode == "research" and field["name"] == "F":
                        continue
                    if not view["atoms"]:
                        counts["rowsWithoutAtoms"] += 1
                        continue
                    records.append(
                        {
                            "id": row["id"],
                            "fieldIndex": i,
                            "field": field["name"],
                            "family": row["family"],
                            "kind": mode,
                            "split": "train",
                            "inputHash": view["inputHash"],
                            "target": targets["fields"][i],
                            "answerTarget": labels[row["id"]] if mode == "public" else None,
                            "parent": prediction["fields"][i],
                            "parentAnswer": {
                                k: prediction[k]
                                for k in ("action", "answer", "scale")
                                if k in prediction
                            }
                            if mode == "public"
                            else None,
                            "atoms": view["atoms"],
                            "operation": int(out["relation"][i].argmax()),
                        }
                    )
                    features.append(
                        {
                            "field": out["fieldState"][i : i + 1].to(torch.bfloat16).cpu(),
                            "atoms": out["atomState"].to(torch.bfloat16).cpu(),
                            "operand1": out["operand1"][i : i + 1].cpu(),
                            "operand2": out["operand2"][i : i + 1].cpu(),
                            "relation": out["relation"][i : i + 1].cpu(),
                        }
                    )
            write(
                root / "progress.json",
                {
                    "cases": counts["cases"],
                    "fieldRows": len(records),
                    "seconds": time.perf_counter() - started,
                },
            )
    require(
        weights_hash(
            {k: v for k, v in model.state_dict().items() if not k.startswith("operand_pairs.")}
        )
        == plan["parentWeightsHash"],
        "Cache changed parent",
    )
    torch.save({"features": features, "initial": initial}, root / "features.pt")
    jsonl(root / "rows.jsonl", records)
    jsonl(root / "inputs.jsonl", input_rows)
    report = registration | {
        "counts": counts,
        "fieldRows": len(records),
        "initialPointerHash": weights_hash(initial),
        "addedParameters": sum(p.numel() for p in model.operand_pairs.parameters()),
        "seconds": time.perf_counter() - started,
        "peakGpuGiB": torch.cuda.max_memory_allocated() / 1024**3,
        "rawHashes": {
            p.name: file_hash(p) for p in root.iterdir() if p.suffix in (".pt", ".jsonl")
        },
    }
    write(root / "report.json", report)
    print(
        {k: report[k] for k in ("counts", "fieldRows", "addedParameters", "seconds", "peakGpuGiB")},
        flush=True,
    )


if __name__ == "__main__":
    main()
