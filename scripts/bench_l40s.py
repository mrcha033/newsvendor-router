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
"""Measure actual Train inputs and optimizer steps on L40S, without selecting on Test."""

import argparse
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--variant", choices=("base", "large"), default="base")
    parser.add_argument("--cases", type=int, default=64)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fresh", action="store_true", help="Benchmark the pinned initialization")
    args = parser.parse_args()
    devices = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,uuid", "--format=csv,noheader"], text=True
    )
    gpu = next(
        uuid.strip()
        for name, uuid in (line.split(",") for line in devices.splitlines())
        if name.strip() == "NVIDIA L40S"
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.chdir(ROOT)
    import numpy as np
    import torch

    from newsvendor import corpus, sequence, structured_data
    from newsvendor.cli import provenance
    from newsvendor.io import digest, read, require, write
    from newsvendor.structured_batches import case_batch
    from newsvendor.structured_model import Router, assemble, configure_optimizer, load_backbone
    from newsvendor.structured_train import (
        cases,
        dataset_hashes,
        language_backward,
        language_view,
        variant,
    )
    from newsvendor.train import seed

    config = variant(read(args.config), args.variant)
    accumulation = config["training"]["accumulation"]
    batch_size = case_batch(config)
    require(args.cases % accumulation == 0 and accumulation % batch_size == 0, "Invalid batch")
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    identity = {
        "config": config,
        "provenance": provenance(config),
        "datasetHashes": dataset_hashes(config),
        "scriptHash": digest(Path(__file__).read_bytes()),
        "gpuUuid": gpu,
        "checkpoint": args.checkpoint,
        "checkpointHash": digest(Path(args.checkpoint).read_bytes()),
        "fresh": args.fresh,
        "split": "train",
        "scope": "runtime measurement, not effectiveness evidence",
    }
    write(directory / "run.json", identity)
    with tarfile.open(directory / "source.tar.gz", "w:gz") as archive:
        for root in ("src", "configs", "scripts"):
            for file in sorted(Path(root).rglob("*")):
                if file.is_file() and "__pycache__" not in file.parts:
                    archive.add(file)
        for file in ("pyproject.toml", "uv.lock"):
            archive.add(file)
    rows, labels, collection = structured_data.load(config)
    train = cases(rows, corpus.generate(read(config["researchConfig"])), "train")
    parent = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    offset = parent["state"]["offset"]
    order = parent["state"]["order"][offset : offset + args.cases]
    random = np.random.default_rng()
    random.bit_generator.state = parent["state"]["random"]
    needed = config["training"].get("retrievalTraining") == "needed"
    rounds = [int(random.integers(config["maxRetrievals"] if needed else config["maxRetrievals"] + 1)) for _ in order]
    tokenizer, encoder = load_backbone(config["encoder"])
    seed(config["seed"])
    model = Router(encoder, config["encoder"]).cuda().train()
    optimizer = model.optimizer(config["training"])
    if args.variant == "base" and not args.fresh:
        model.load_state_dict(parent["weights"])
        optimizer.load_state_dict(parent["optimizer"])
        configure_optimizer(optimizer, config["training"])
    del parent
    stamp = time.perf_counter()
    prepared = []
    for index, round in zip(order, rounds, strict=True):
        view, target = language_view(train[index], tokenizer, config, labels, collection, 0 if needed else round)
        if needed:
            target |= {"caseIndex": int(index), "retrievalRound": round + 1}
        prepared.append((view, target))
    preparation = time.perf_counter() - stamp
    case_ids = [train[i][1]["id"] for i in order]
    write(
        directory / "inputs.json",
        {
            "caseIds": case_ids, "rounds": rounds, "preparationSeconds": preparation,
            "inputHashes": [view["inputHash"] for view, _ in prepared],
            "tokens": [int(view["batch"]["attention_mask"].sum()) for view, _ in prepared],
        },
    )
    records = []
    for repeat in range(args.passes):
        for start in range(0, len(prepared), accumulation):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            stamp = time.perf_counter()
            losses, encoded_tokens, retrievals = [], 0, 0
            for number in range(start, start + accumulation, batch_size):
                views, targets = zip(*prepared[number : number + batch_size], strict=True)
                for lo, hi in model.training_batches(views):
                    values, _, workload = language_backward(
                        model, list(zip(views[lo:hi], targets[lo:hi], strict=True)),
                        tokenizer, config, train, labels, collection,
                    )
                    losses.extend(values)
                    encoded_tokens += workload["tokens"]
                    retrievals += workload["retrievals"]
            torch._foreach_div_([p.grad for p in model.parameters() if p.grad is not None], accumulation)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            require(torch.isfinite(norm).item(), "Nonfinite gradients")
            if repeat == 0 and start == 0:
                gradient = model.encoder.embeddings.tok_embeddings.weight.grad
                encoder_gradient = float(gradient.abs().sum())
                require(encoder_gradient > 0, "Encoder did not receive gradients")
            optimizer.step()
            torch.cuda.synchronize()
            record = {
                "pass": repeat + 1, "start": start, "cases": accumulation,
                "seconds": time.perf_counter() - stamp,
                "loss": sum(losses) / len(losses), "gradientNorm": float(norm),
                "encodedTokens": encoded_tokens, "retrievalExamples": retrievals,
                "peakBytes": torch.cuda.max_memory_allocated(),
            }
            records.append(record)
            write(directory / "measurements.json", {"records": records})
            print(record, flush=True)
    warm = [r for r in records if r["pass"] > 1]
    result = {
        "casesPerSecond": sum(r["cases"] for r in warm) / sum(r["seconds"] for r in warm),
        "firstPassSeconds": sum(r["seconds"] for r in records if r["pass"] == 1),
        "peakBytes": torch.cuda.max_memory_allocated(), "records": records,
    }
    write(directory / "summary.json", result)
    model.eval()
    with torch.inference_mode():
        view = prepared[0][0]
        output = model(view)
        require(all(torch.isfinite(v).all() for v in output.values()), "Nonfinite inference")
        prediction = assemble(view, output)
    row = next(r for r in rows if r["split"] == "train" and r["component"] == "retail")
    values, _ = sequence.features(row["input"]["observations"])
    model.demand.train()
    model.zero_grad(set_to_none=True)
    output = model.demand([values, values], [1, 7])
    daily = sequence.losses(output, [{"y": 0.0, "censored": False}, {"y": 2.0, "censored": True}]).mean()
    daily.backward()
    demand_gradient = sum(float(p.grad.abs().sum()) for p in model.demand.parameters() if p.grad is not None)
    require(torch.isfinite(daily).item() and demand_gradient > 0, "Invalid demand gradients")
    write(directory / "numerical.json", {
        "encoderGradient": encoder_gradient, "demandGradient": demand_gradient,
        "horizons": [1, 7], "prediction": prediction,
        "scope": "numerical checks, not effectiveness evidence",
    })
    print({k: v for k, v in result.items() if k != "records"}, flush=True)


if __name__ == "__main__":
    main()
