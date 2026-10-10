"""Reproduce fixed operand candidates from public inference weights and raw Dev records."""

import argparse
import copy
import gzip
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def main():
    import torch
    from evaluate_responses import CachedRouter, conditions, measurement

    from newsvendor import structured_retail, structured_train
    from newsvendor.bundle import file_hash, load_bundle
    from newsvendor.cli import provenance
    from newsvendor.io import lines, read, require, write
    from newsvendor.structured_model import Router
    from newsvendor.structured_rollout import rollout
    from newsvendor.structured_tool_eval import weights_hash
    from newsvendor.suite import public_input

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "refit", "dev", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    require(not Path(args.output).exists(), "Preserve previous reproduction")
    require(bool(os.environ.get("CUDA_VISIBLE_DEVICES")), "Choose the visible L40S")
    require(torch.cuda.device_count() == 1 and "L40S" in torch.cuda.get_device_name(), "L40S only")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    refit, dev = Path(args.refit), Path(args.dev)
    fitted, report = read(refit / "report.json"), read(dev / "report.json")
    require(provenance({})["sourceHash"] == report["source"]["sourceHash"], "Source differs")
    parent, tokenizer, config, manifest = load_bundle(args.bundle, "cuda")
    require(manifest["weightsHash"] == fitted["plan"]["parentWeightsHash"], "Wrong parent")
    parent.requires_grad_(False).eval()
    episodes = {}
    for kind in ("generated", "retail"):
        for part in ("inputs", "environment"):
            name = f"{kind}-{part}.jsonl"
            require(file_hash(dev / name) == report["rawHashes"][name], "Cohort changed")
        observed = lines(dev / f"{kind}-inputs.jsonl")
        environment = {r["id"]: r for r in lines(dev / f"{kind}-environment.jsonl")}
        observed = observed[:: max(1, len(observed) // 7)]
        require(all(r["split"] == "dev" for r in observed), "Original Dev reproduction only")
        episodes[kind] = [r | environment[r["id"]] for r in observed]
        if kind == "retail":
            for e in episodes[kind]:
                e.update(
                    benchmark=structured_retail.VERSION, cutoffIndex=len(e["input"]["observations"])
                )
    require(
        file_hash(dev / "public-inputs.jsonl") == report["rawHashes"]["public-inputs.jsonl"],
        "Public cohort changed",
    )
    public_rows = lines(dev / "public-inputs.jsonl")
    collection_path = dev / "public-collection.jsonl"
    require(
        file_hash(collection_path)
        == report["plan"]["files"][report["plan"]["publicDataset"] + "/collection.jsonl"],
        "Observed collection changed",
    )
    collection = lines(collection_path)
    require(
        len(public_rows) == 258 and all(r["split"] == "dev" for r in public_rows),
        "Public denominator changed",
    )
    started = time.perf_counter()
    result = {
        "scope": __doc__,
        "testUsed": False,
        "trainingPerformed": False,
        "sourceHash": report["source"]["sourceHash"],
        "selection": "Fixed stride over original research cohorts and every original public Dev case; no outcome selection.",
        "cases": {k: [r["id"] for r in v] for k, v in episodes.items()},
        "results": {},
    }
    for name in ["parent", "joint-42", "joint-43", "joint-44"]:
        options = copy.deepcopy(config)
        model = parent
        if name != "parent":
            seed = name.split("-")[1]
            path = refit / f"refit-{seed}.pt"
            require(file_hash(path) == fitted["rawHashes"][path.name], "Refit changed")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            options["encoder"]["jointOperands"] = True
            model = Router(parent.encoder, options["encoder"]).cuda()
            structured_train.import_core(model, parent.state_dict())
            model.operand_pairs.load_state_dict(payload["weights"], strict=True)
            model.requires_grad_(False).eval()
            require(
                weights_hash(model.state_dict()) == report["results"][name]["fullTensorHash"],
                "Candidate tensors differ",
            )
        router = CachedRouter(model, tokenizer, options["encoder"])
        router.reset()
        verified = {"generated": 0, "retail": 0, "public": 0}
        for kind, cases in episodes.items():
            raw_path = dev / f"{name}-{kind}.jsonl.gz"
            if raw_path.exists():
                require(
                    file_hash(raw_path) == report["rawHashes"][raw_path.name],
                    "Raw reference changed",
                )
                stream = gzip.open(raw_path, "rt")
            else:
                entry = read(dev / "raw-format.json")[raw_path.name]
                require(
                    entry["sourceGzipSha256"] == report["rawHashes"][raw_path.name],
                    "Raw provenance changed",
                )
                raw_path = dev / entry["path"]
                require(file_hash(raw_path) == entry["sha256"], "Restored raw bytes differ")
                stream = raw_path.open()
            ids = {e["id"] for e in cases}
            references = {}
            with stream:
                for row in map(json.loads, stream):
                    if row["id"] in ids:
                        references[row["id"], tuple(row.get("missingResponses", []))] = row
            for e in cases:
                for missing in conditions(0) if kind == "retail" else [()]:
                    actual = rollout(
                        e,
                        router,
                        **({"missing_responses": frozenset(missing)} if kind == "retail" else {}),
                    )
                    reference = references[e["id"], tuple(missing)]
                    require(
                        [r["action"] for r in actual["events"]]
                        == [r["action"] for r in reference["events"]],
                        "Actions differ",
                    )
                    require(
                        all(
                            actual[k] == reference[k]
                            for k in (
                                "q",
                                "result",
                                "total",
                                "terminalLoss",
                                "requestCost",
                                "falseHandoff",
                            )
                        ),
                        "Outcomes differ",
                    )
                    if kind == "retail":
                        require(
                            measurement(actual) == measurement(reference),
                            "Retail measurements differ",
                        )
                    verified[kind] += 1
        path = dev / f"{name}-public.jsonl"
        require(file_hash(path) == report["rawHashes"][path.name], "Public reference changed")
        expected = {r["id"]: r["prediction"] for r in lines(path)}
        for row in public_rows:
            actual, _ = structured_train.infer(
                model, tokenizer, public_input(row), options, collection
            )
            require(actual == expected[row["id"]], "Public prediction differs: " + row["id"])
            verified["public"] += 1
        result["results"][name] = verified
        print({"condition": name, "verified": verified}, flush=True)
    result.update(
        seconds=time.perf_counter() - started,
        verifiedRollouts=sum(v["generated"] + v["retail"] for v in result["results"].values()),
        verifiedPublicPredictions=sum(v["public"] for v in result["results"].values()),
    )
    write(args.output, result)
    print(
        {k: result[k] for k in ("seconds", "verifiedRollouts", "verifiedPublicPredictions")},
        flush=True,
    )


if __name__ == "__main__":
    main()
