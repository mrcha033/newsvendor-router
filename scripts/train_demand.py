# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["torch==2.14.1", "transformers==5.18.0", "numpy==2.5.3", "pydantic==2.12.5"]
# [tool.uv.sources]
# torch = { index = "cuda" }
# [[tool.uv.index]]
# name = "cuda"
# url = "https://download.pytorch.org/whl/cu130"
# explicit = true
# ///
"""Train a standalone GRU from the pinned snapshot, without a parent checkpoint."""

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/research-from-scratch.json")
    parser.add_argument(
        "--output", help="New directory; defaults to the configured demand checkpoint directory"
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    import torch

    from newsvendor import sequence
    from newsvendor.cli import provenance
    from newsvendor.io import digest, jsonl, lines, read, require, write
    from newsvendor.structured_tool_eval import weights_hash
    from newsvendor.structured_train import DEMAND_SCHEMA, dataset_hashes
    from newsvendor.suite import check
    from newsvendor.train import seed

    require(args.threads > 0, "--threads must be positive")
    torch.set_num_threads(args.threads)
    if args.device == "cuda":
        require(
            torch.cuda.is_available() and torch.cuda.device_count() == 1,
            "Expose the single assigned GPU with CUDA_VISIBLE_DEVICES",
        )
    config = read(args.config)
    checkpoint = Path(args.output) / "model.pt" if args.output else Path(config["demandCheckpoint"])
    directory = checkpoint.parent
    require(not directory.exists(), "Use a new demand output directory")
    check(config["dataset"])
    rows = [
        r
        for r in lines(Path(config["dataset"]) / "inputs.jsonl")
        if r["component"] == "retail" and r["split"] in ("train", "dev")
    ]
    ids = {r["id"] for r in rows}
    labels = {
        r["id"]: r["target"]
        for r in lines(Path(config["dataset"]) / "labels.jsonl")
        if r["id"] in ids
    }
    samples = sequence.samples(rows, labels, config["demand"]["minHistory"])
    train = [r for r in samples if r["split"] == "train"]
    dev = [r for r in samples if r["split"] == "dev"]
    require(train and dev, "Demand training requires Train and Dev observations")
    require(not ({r["family"] for r in train} & {r["family"] for r in dev}), "Source-group overlap")
    directory.mkdir(parents=True)
    seed(config["seed"])
    started = time.perf_counter()
    model = sequence.DemandEncoder().to(args.device)
    report = {
        "scope": "New training from the published snapshot; not restoration of historical weights",
        "observationModel": (
            "Joint daily observations under a zero-inflated total, conditionally nonempty active days and uniform allocation; assumes noninformative daily right censoring"
            if config["demand"].get("observation") == "daily_allocation"
            else "Complete totals use density; censored totals use an aggregate lower-bound survival score, which is not generally the likelihood of separately censored daily observations"
        ),
        "testUsed": False,
        "config": config,
        "device": args.device,
        "provenance": provenance(config),
        "scriptHash": digest(Path(__file__).read_bytes()),
        "initialWeightsHash": weights_hash(model.state_dict()),
        "samples": {
            split: {"forecasts": len(group), "families": sorted({r["family"] for r in group})}
            for split, group in (("train", train), ("dev", dev))
        },
    }
    write(directory / "registered.json", report)

    class Progress:
        def update(self, stage, **values):
            status = {"stage": stage, "seconds": time.perf_counter() - started, **values}
            write(directory / "progress.json", status)
            print(status, flush=True)

    report["training"], records = sequence.cross_fit(
        model, train, dev, config["demand"], config["seed"], Progress()
    )
    jsonl(directory / "crossfit.jsonl", records)
    report["crossfitHash"] = digest((directory / "crossfit.jsonl").read_bytes())
    report["finalWeightsHash"] = weights_hash(model.state_dict())
    report["seconds"] = time.perf_counter() - started
    temporary = checkpoint.with_suffix(".tmp")
    torch.save(
        {
            "schema": DEMAND_SCHEMA,
            "config": config,
            "report": report,
            "datasetHashes": dataset_hashes(config),
            "weights": {"demand." + k: v.detach().cpu() for k, v in model.state_dict().items()},
        },
        temporary,
    )
    temporary.replace(checkpoint)
    write(
        directory / "report.json",
        {
            **report,
            "checkpoint": str(checkpoint),
            "checkpointHash": digest(checkpoint.read_bytes()),
        },
    )
    print(
        {"checkpoint": str(checkpoint), "seconds": report["seconds"], "testUsed": False}, flush=True
    )


if __name__ == "__main__":
    main()
