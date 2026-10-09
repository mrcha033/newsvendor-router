"""Evaluate fixed research policies on Dev with predeclared missing-response rates."""

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
    parser.add_argument("--base", required=True)
    parser.add_argument("--no-value", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--noise", type=float, nargs="+", default=[.1, .2])
    args = parser.parse_args()

    import torch

    from newsvendor import structured_retail, structured_train
    from newsvendor.cli import provenance
    from newsvendor.io import digest, jsonl, lines, read, require, write
    from newsvendor.structured_rollout import ResearchRouter, rollout
    from newsvendor.suite import check

    require(len(set(args.noise)) == len(args.noise) and all(0 <= n <= 1 for n in args.noise), "Invalid response rates")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1 and "L40S" in torch.cuda.get_device_name(0), "Use the assigned single L40S")
    torch.set_num_threads(8)
    directory = Path(args.output)
    require(not directory.exists(), "Use a new evaluation directory")
    directory.mkdir(parents=True)
    base_config = read(Path(args.base).parent / "config.json")
    check(base_config["dataset"])
    rows = [r for r in lines(Path(base_config["dataset"]) / "inputs.jsonl") if r["component"] == "retail" and r["split"] in ("train", "dev")]
    episodes = structured_retail.cases(rows, base_config["dataSeed"], base_config["cutoffStride"])
    observed = [{k: e[k] for k in ("id", "family", "split", "input", "provenance")} for e in episodes]
    require(observed == lines(Path(args.base).parent / "inputs.jsonl"), "Evaluation inputs changed")
    ids = {r["id"] for r in rows}
    labels = {r["id"]: r["target"] for r in lines(Path(base_config["dataset"]) / "labels.jsonl") if r["id"] in ids}
    dev = [e for e in structured_retail.attach_targets(episodes, rows, labels) if e["split"] == "dev"]
    require(read(Path(args.base).parent / "manifest.json") == read(Path(args.no_value).parent / "manifest.json"), "Paired model datasets differ")
    started = time.perf_counter()
    report = {"scope": "Fixed checkpoints, controlled missing manager replies, Dev only", "testUsed": False,
              "noise": args.noise, "manifest": read(Path(args.base).parent / "manifest.json"),
              "provenance": provenance(base_config), "scriptHash": digest(Path(__file__).read_bytes()),
              "models": {}, "results": {}}
    write(directory / "job.json", {"stage": "evaluate", "pid": os.getpid()})
    try:
        for name, path in (("value", args.base), ("no_value", args.no_value)):
            model, tokenizer, config, _ = structured_train.load(path, "cuda")
            require(config["noValue"] == (name == "no_value"), "Wrong policy mode for checkpoint")
            report["models"][name] = {"path": path, "hash": digest(Path(path).read_bytes())}
            router = ResearchRouter(model, tokenizer, config["encoder"], config["noValue"])
            router.cache_states()
            for noise in args.noise:
                for policy in ([name, "checklist"] if name == "value" else [name]):
                    records = []
                    for index, episode in enumerate(dev):
                        records.append(rollout(episode, router, explore=policy == "checklist", noise=noise))
                        if (index + 1) % 64 == 0:
                            write(directory / "progress.json", {"policy": policy, "noise": noise,
                                  "processed": index + 1, "total": len(dev), "seconds": time.perf_counter()-started})
                    key = f"{policy}-{noise:g}"
                    raw = directory / (key + ".jsonl")
                    jsonl(raw, records)
                    report["results"][key] = {**structured_retail.summarize(records), "rawHash": digest(raw.read_bytes())}
                    write(directory / "partial.json", report)
                    print({"condition": key, "loss": report["results"][key]["meanTotalOnCompletePeriods"]}, flush=True)
            router.cache_states(False)
            del model, tokenizer, router
            torch.cuda.empty_cache()
        report["seconds"] = time.perf_counter()-started
        report["peakGpuBytes"] = torch.cuda.max_memory_allocated()
        write(directory / "report.json", report)
        write(directory / "job.json", {"stage": "complete", "pid": os.getpid(), "seconds": report["seconds"]})
    except BaseException as error:
        write(directory / "job.json", {"stage": "failed", "pid": os.getpid(), "error": str(error)})
        raise


if __name__ == "__main__":
    main()
