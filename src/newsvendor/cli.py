import argparse
import importlib.metadata
import os
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from . import corpus, data
from .encoder import embed
from .io import digest, read, require, write
from .policy import trajectory
from .report import save
from .train import fit, load, seed


def provenance(config):
    commit = (
        subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        or None
    )
    sources = {str(p): digest(p.read_bytes()) for p in sorted(Path("src/newsvendor").glob("*.py"))}
    return {
        "configHash": digest(config),
        "sourceHash": digest(sources),
        "sources": sources,
        "lockHash": digest(Path("uv.lock").read_bytes()),
        "python": platform.python_version(),
        "packages": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "numpy")},
        "platform": platform.platform(),
        "commit": commit,
        "timestamp": datetime.now(UTC).isoformat(),
    }


def smoke(config):
    episodes = corpus.generate(config)
    chosen = [
        next(e for e in episodes if e["family"] == f"{scenario}-{g:02}")
        for scenario in corpus.SCENARIOS
        for g in range(5)
    ]
    rows = [trajectory(e, "reference", construction="rules") for e in chosen]
    require(
        len(rows) == len({r["family"] for r in rows}) == 30,
        "Smoke must cover 30 distinct source bundles",
    )
    require(
        all(not r["falseHandoff"] for r in rows),
        "Reference rules falsely certified a parameter set",
    )
    report = {
        "episodes": len(episodes),
        "smokeEpisodes": 30,
        "split": corpus.audit(episodes),
        "falseHandoffs": 0,
        "provenance": provenance(config),
    }
    write("results/smoke.json", report)
    print({k: report[k] for k in ("episodes", "smokeEpisodes", "falseHandoffs")}, flush=True)


def evaluate(config, episodes, model, cache, directory, started):
    test = [e for e in episodes if e["split"] == "test"]
    rows, timing = [], []
    for e in test:
        for construction in ("rules", "learned"):
            for method in (
                "checklist",
                "ask_all",
                "uncertainty",
                "one_step",
                "reference",
                "learned",
            ):
                stamp = time.perf_counter()
                rows.append(trajectory(e, method, model, cache, construction=construction))
                timing.append((time.perf_counter() - stamp) * 1000)
        rows.append(trajectory(e, "oracle"))
        for ablation in ("evidence", "type", "impact", "update"):
            rows.append(trajectory(e, "learned", model, cache, ablation=ablation))
        for noise in (0.1, 0.2):
            rows.append(trajectory(e, "learned", model, cache, noise=noise))
    prov = {
        **provenance(config),
        "split": corpus.audit(episodes),
        "encoderHash": cache.id,
        "wallSeconds": time.perf_counter() - started,
        "inferenceMs": {
            "median": float(np.median(timing)),
            "p95": float(np.quantile(timing, 0.95)),
        },
    }
    summary = save(rows, directory, config, prov)
    print(
        {
            "directory": directory,
            "testEpisodes": len(test),
            "rollouts": len(rows),
            "learned": summary["learned/learned"],
        },
        flush=True,
    )


def experiment(config, checkpoint=None):
    started = time.perf_counter()
    seed(config["seed"])
    episodes = corpus.save(config)
    cache = embed(episodes, config["encoder"])
    if checkpoint:
        model, saved = load(checkpoint, cache)
        require(
            digest(saved) == digest(config),
            "Evaluation configuration differs from training checkpoint",
        )
    else:
        model = fit(config, episodes, cache, config["output"])
    evaluate(config, episodes, model, cache, config["output"], started)


def doctor():
    from .providers import envfile

    envfile()
    report = {
        "python": platform.python_version(),
        "torch": importlib.metadata.version("torch"),
        "encoderCached": Path(".cache/torch-embeddings.json").exists(),
        "pilotModel": Path("results/pilot/model.pt").exists(),
        "datasets": read("data/processed/check.json")
        if Path("data/processed/check.json").exists()
        else None,
        "providers": {
            p: {
                k.lower() + "Set": bool(os.environ.get(p + "_" + k))
                for k in ("URL", "MODEL", "KEY")
            }
            for p in ("JEV", "SGLANG", "AGENT")
        },
    }
    write("results/readiness.json", report)
    print(report, flush=True)


def main():
    parser = argparse.ArgumentParser(description="Controlled Newsvendor routing experiments")
    parser.add_argument(
        "command",
        choices=(
            "prepare",
            "fetch",
            "check",
            "smoke",
            "experiment",
            "evaluate",
            "sweep",
            "doctor",
            "external",
            "benchmark",
        ),
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--checkpoint")
    parser.add_argument("--provider", choices=("jev", "sglang", "agent"))
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-calls", type=int, default=0)
    parser.add_argument("--suite", choices=("public", "business"), default="public")
    args = parser.parse_args()
    config = read(
        args.config
        or (
            "configs/pilot.json"
            if args.command in ("experiment", "evaluate", "benchmark")
            else "configs/full.json"
        )
    )
    if args.seed is not None:
        config["seed"] = args.seed
    if args.command == "prepare":
        print(corpus.audit(corpus.save(config)))
    elif args.command == "fetch":
        print(data.fetch(read("configs/datasets.json")))
    elif args.command == "check":
        print(data.check())
    elif args.command == "smoke":
        smoke(config)
    elif args.command in ("experiment", "evaluate"):
        require(args.command != "evaluate" or args.checkpoint, "Evaluate requires --checkpoint")
        experiment(config, args.checkpoint)
    elif args.command == "sweep":
        schedule = read("configs/seeds.json")
        for value in schedule["seeds"]:
            current = read(args.config or schedule["config"])
            current.update(seed=value, output=f"results/seeds/{value}")
            experiment(current)
    elif args.command == "doctor":
        doctor()
    elif args.command == "external":
        from .providers import envfile, external

        envfile()
        require(args.provider, "External comparison requires --provider")
        print(external(args.provider, args.limit, args.suite, config))
    elif args.command == "benchmark":
        from .benchmark import run
        from .providers import envfile

        envfile()
        require(args.provider, "Benchmark requires --provider")
        run(config, args.provider, args.max_calls)


if __name__ == "__main__":
    main()
