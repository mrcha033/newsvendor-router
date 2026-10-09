"""Measure a fixed core model on the original generated Dev benchmark."""

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
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/full.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    import torch

    from newsvendor import corpus, structured_train
    from newsvendor.cli import provenance
    from newsvendor.io import digest, jsonl, read, require, write
    from newsvendor.structured_metrics import generated_metrics
    from newsvendor.structured_rollout import ResearchRouter, rollout

    torch.set_num_threads(args.threads)
    if args.device == "cuda":
        require(os.environ.get("CUDA_VISIBLE_DEVICES") == "GPU-674d64b8-4bdf-7006-1791-5dc7f7245409", "Assigned L40S only")
    output = Path(args.output)
    require(not output.exists(), "Preserve prior evaluations")
    output.mkdir(parents=True)
    config = read(args.config)
    corpus_rows = corpus.generate(config)
    sources = corpus.audit(corpus_rows)
    episodes = [e for e in corpus_rows if e["split"] == "dev"]
    observed = [{k: e[k] for k in ("id", "family", "split", "input")} for e in episodes]
    jsonl(output / "inputs.jsonl", observed)
    report = {
        "scope": "Original generated Dev benchmark, separate from retail; not final Test or real organizational evidence",
        "testUsed": False, "device": args.device, "threads": args.threads,
        "checkpoint": args.checkpoint, "checkpointHash": digest(Path(args.checkpoint).read_bytes()),
        "provenance": provenance(config), "scriptHash": digest(Path(__file__).read_bytes()),
        "sources": sources, "inputHash": digest(observed), "results": {},
    }
    started = time.perf_counter()
    write(output / "job.json", {"stage": "loading", "pid": os.getpid()})
    try:
        model, tokenizer, settings, _ = structured_train.load(args.checkpoint, args.device)
        router = ResearchRouter(model, tokenizer, settings["encoder"], settings["noValue"])
        for policy in ("learned", "checklist"):
            records = []
            for index, episode in enumerate(episodes):
                records.append(rollout(episode, router, explore=policy == "checklist"))
                if (index + 1) % 5 == 0:
                    write(output / "progress.json", {"policy": policy, "processed": index + 1,
                          "total": len(episodes), "seconds": time.perf_counter() - started})
            path = output / f"{policy}.jsonl"
            jsonl(path, records)
            scenarios = {e["id"]: e["scenario"] for e in episodes}
            report["results"][policy] = {
                **generated_metrics(records), "rawHash": digest(path.read_bytes()),
                "scenarios": {s: generated_metrics([r for r in records if scenarios[r["id"]] == s]) for s in corpus.SCENARIOS},
            }
            write(output / "partial.json", report)
        report["seconds"] = time.perf_counter() - started
        write(output / "report.json", report)
        write(output / "job.json", {"stage": "complete", "pid": os.getpid(), "seconds": report["seconds"]})
        print(report["results"], flush=True)
    except BaseException as error:
        write(output / "job.json", {"stage": "failed", "pid": os.getpid(), "error": str(error)})
        raise


if __name__ == "__main__":
    main()
