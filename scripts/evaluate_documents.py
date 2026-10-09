"""Compare functional document heads on the same public Dev inputs and inference settings."""

import argparse
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, help="NAME=checkpoint.pt")
    parser.add_argument("--input-config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import numpy as np
    import torch

    from newsvendor.cli import provenance
    from newsvendor.io import digest, jsonl, lines, read, require, write
    from newsvendor.structured_model import Router, load_backbone
    from newsvendor.structured_train import dataset_hashes, infer
    from newsvendor.suite import check, public_input
    from newsvendor.suite_score import metrics

    config = read(args.input_config)
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1 and "L40S" in torch.cuda.get_device_name(0), "Use the assigned single L40S")
    torch.set_num_threads(8)
    paths = dict(item.split("=", 1) for item in args.checkpoint)
    require(len(paths) == len(args.checkpoint) and all(re.fullmatch(r"[a-z0-9_-]+", name) for name in paths), "Use distinct simple checkpoint names")
    directory = Path(args.output)
    require(not directory.exists(), "Use a new output directory")
    directory.mkdir(parents=True)
    check(config["dataset"])
    source = Path(config["dataset"])
    rows = [r for r in lines(source / "inputs.jsonl") if r["split"] == "dev" and r["component"] in ("cuad", "contractnli", "orsharc", "tatqa")]
    ids = {r["id"] for r in rows}
    labels = {r["id"]: r["target"] for r in lines(source / "labels.jsonl") if r["id"] in ids}
    collection = lines(source / "collection.jsonl")
    report = {"scope": "Public annotation Dev tasks, not joint SKU contracts or real organizational deployment",
              "testUsed": False, "inputConfig": config, "provenance": provenance(config),
              "scriptHash": digest(Path(__file__).read_bytes()), "observedInputHash": digest([public_input(r) for r in rows]),
              "collectionHash": digest(collection), "cases": len(rows), "models": {}}
    started = time.perf_counter()
    write(directory / "job.json", {"stage": "evaluate", "pid": os.getpid()})
    try:
        for name, path in paths.items():
            payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            require(dataset_hashes(payload["config"]) == payload["datasetHashes"], "Checkpoint data changed")
            settings = {**config["encoder"], **{k: payload["config"]["encoder"][k] for k in ("model", "revision")}}
            require(not settings.get("structuredTools") and not settings.get("dialogueController"), "Document comparison requires the core architecture")
            tokenizer, encoder = load_backbone(settings)
            model = Router(encoder, settings)
            core = {k: v for k, v in payload["weights"].items() if not k.startswith(("tools.", "control."))}
            model.load_state_dict(core, strict=True)
            current = {**config, "encoder": settings, "noValue": payload["config"]["noValue"]}
            del core, payload, encoder
            model = model.to("cuda").eval()
            torch.cuda.reset_peak_memory_stats()
            groups = defaultdict(lambda: defaultdict(list))
            records = []
            for index, row in enumerate(rows):
                observed = public_input(row)
                torch.cuda.synchronize()
                stamp = time.perf_counter()
                prediction, trace = infer(model, tokenizer, observed, current, collection)
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - stamp) * 1000
                measured = metrics(row, labels[row["id"]], prediction)
                records.append({"id": row["id"], "family": row["family"], "component": row["component"],
                                "inputHash": digest(observed), "prediction": prediction, "trace": trace,
                                "elapsedMs": elapsed, "metrics": measured})
                for key, value in measured.items():
                    if value is not None:
                        groups[row["component"]][key].append(float(value))
                if (index + 1) % 32 == 0:
                    write(directory / "progress.json", {"checkpoint": name, "processed": index + 1,
                          "total": len(rows), "seconds": time.perf_counter() - started})
            raw = directory / (name + ".jsonl")
            jsonl(raw, records)
            report["models"][name] = {
                "checkpoint": path, "checkpointHash": digest(Path(path).read_bytes()),
                "parameters": sum(p.numel() for p in model.parameters()), "encoder": settings,
                "metrics": {component: {key: {"mean": float(np.mean(v)), "count": len(v)} for key, v in values.items()} for component, values in groups.items()},
                "latencyMs": {"median": float(np.median([r["elapsedMs"] for r in records])),
                              "p95": float(np.quantile([r["elapsedMs"] for r in records], .95))},
                "peakGpuBytes": torch.cuda.max_memory_allocated(), "rawHash": digest(raw.read_bytes()),
            }
            write(directory / "partial.json", report)
            print({"checkpoint": name, "metrics": report["models"][name]["metrics"]}, flush=True)
            del model, tokenizer
            torch.cuda.empty_cache()
        report["seconds"] = time.perf_counter() - started
        write(directory / "report.json", report)
        write(directory / "job.json", {"stage": "complete", "pid": os.getpid(), "seconds": report["seconds"]})
    except BaseException as error:
        write(directory / "job.json", {"stage": "failed", "pid": os.getpid(), "error": str(error)})
        raise


if __name__ == "__main__":
    main()
