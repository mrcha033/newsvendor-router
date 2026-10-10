"""Reproduce fixed Dev examples from the released base bundle and public numeric-residual records."""

import argparse
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
    from evaluate_responses import CachedRouter, conditions, file_hash, measurement
    from study_recovery_residual import ResidualHead

    from newsvendor import structured_retail
    from newsvendor.bundle import load_bundle
    from newsvendor.cli import provenance
    from newsvendor.heads import Head
    from newsvendor.io import lines, read, require, write
    from newsvendor.structured_rollout import rollout
    from newsvendor.structured_tool_eval import weights_hash

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--refit", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    require(not Path(args.output).exists(), "Preserve previous reproduction")
    require(bool(os.environ.get("CUDA_VISIBLE_DEVICES")), "Choose the visible L40S")
    require(torch.cuda.device_count() == 1 and "L40S" in torch.cuda.get_device_name(0), "L40S only")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    refit, dev = Path(args.refit), Path(args.dev)
    report, fitted = read(dev / "report.json"), read(refit / "report.json")
    require(provenance({})["sourceHash"] == report["source"]["sourceHash"], "Source differs")
    require(
        fitted["source"]["sourceHash"] == report["source"]["sourceHash"], "Training source differs"
    )
    model, tokenizer, config, manifest = load_bundle(args.bundle, "cuda")
    require(manifest["weightsHash"] == fitted["plan"]["parentWeightsHash"], "Wrong released parent")
    for name in ("value", "recovery"):
        old = model.heads[name]
        weights = dict(old.state_dict())
        weights["layers.0.weight"] = torch.cat(
            [weights["layers.0.weight"], weights["layers.0.weight"].new_zeros((128, 94))], dim=1
        )
        model.heads[name] = Head(350, 128, old.layers[-1].out_features).cuda().eval()
        model.heads[name].load_state_dict(weights)
    initial = {k: v.detach().cpu().clone() for k, v in model.heads["value"].state_dict().items()}
    require(weights_hash(initial) == fitted["baseHeadHash"], "Padded prior differs")
    model.config.update(numericState=True, actionPrecision="float32")
    config["encoder"].update(numericState=True, actionPrecision="float32")
    model.requires_grad_(False).eval()
    episodes = {}
    for benchmark in ("generated", "retail"):
        for part in ("inputs", "environment"):
            name = f"{benchmark}-{part}.jsonl"
            require(file_hash(dev / name) == report["rawHashes"][name], "Observed cohort changed")
        observed = lines(dev / f"{benchmark}-inputs.jsonl")
        environment = {r["id"]: r for r in lines(dev / f"{benchmark}-environment.jsonl")}
        # Fixed stride over the original order, independent of labels and outcomes.
        observed = observed[:: max(1, len(observed) // 7)]
        require(all(r["split"] == "dev" for r in observed), "Dev reproduction only")
        episodes[benchmark] = [r | environment[r["id"]] for r in observed]
        if benchmark == "retail":
            for e in episodes[benchmark]:
                e.update(
                    benchmark=structured_retail.VERSION, cutoffIndex=len(e["input"]["observations"])
                )
    started = time.perf_counter()
    result = {
        "scope": __doc__,
        "testUsed": False,
        "trainingPerformed": False,
        "sourceHash": report["source"]["sourceHash"],
        "selection": "Fixed stride of each original Dev cohort; no outcome selection.",
        "cases": {k: [e["id"] for e in v] for k, v in episodes.items()},
        "results": {},
    }
    state_cache = {}
    for name, expected in report["results"].items():
        mode, seed = name.split("-")
        stage = "inner" if mode == "selected" else "refit"
        path = refit / f"{stage}-{seed}.pt"
        require(
            file_hash(path) == expected["headHash"] == fitted["rawHashes"][path.name],
            "Head changed",
        )
        payload = torch.load(path, map_location="cpu", weights_only=True)
        model.heads["value"] = ResidualHead(initial, int(seed)).cuda().eval()
        model.heads["value"].load_state_dict(payload["weights"])
        model.heads["value"].requires_grad_(False)
        require(
            weights_hash(model.heads["value"].base.state_dict()) == fitted["baseHeadHash"],
            "Frozen prior changed",
        )
        require(
            weights_hash(model.heads["value"].state_dict()) == fitted[stage][seed]["weightsHash"],
            "Tensor identity differs",
        )
        router = CachedRouter(model, tokenizer, config["encoder"])
        router.reset()
        router.cache = state_cache
        compared = {"generated": 0, "retail": 0}
        for benchmark, cases in episodes.items():
            raw_path = dev / f"{name}-{benchmark}.jsonl.gz"
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
                for missing in conditions(0) if benchmark == "retail" else [()]:
                    options = (
                        {"missing_responses": frozenset(missing)} if benchmark == "retail" else {}
                    )
                    actual = rollout(e, router, **options)
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
                        "Measured outcomes differ",
                    )
                    if benchmark == "retail":
                        require(
                            measurement(actual) == measurement(reference),
                            "Retail measurements differ",
                        )
                    compared[benchmark] += 1
        result["results"][name] = compared
        print({"condition": name, "verified": compared}, flush=True)
    result["verifiedRollouts"] = sum(sum(r.values()) for r in result["results"].values())
    result["seconds"] = time.perf_counter() - started
    write(args.output, result)
    print(
        {
            "completed": True,
            "verifiedRollouts": result["verifiedRollouts"],
            "seconds": result["seconds"],
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
