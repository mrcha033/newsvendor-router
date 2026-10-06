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
    rows, timing, policy_timing = [], [], {}
    for e in test:
        for construction in ("rules", "learned"):
            for method in (
                "checklist",
                "ask_all",
                "uncertainty",
                "learned",
            ):
                stamp = time.perf_counter()
                rows.append(trajectory(e, method, model, cache, construction=construction))
                elapsed = (time.perf_counter() - stamp) * 1000
                rows[-1]["elapsedMs"] = elapsed
                timing.append(elapsed)
                policy_timing.setdefault(method + "/" + construction, []).append(elapsed)
        for ablation in ("evidence", "type", "impact", "update"):
            rows.append(trajectory(e, "learned", model, cache, ablation=ablation))
        for noise in (0.1, 0.2):
            for construction in ("rules", "learned"):
                for method in (
                    "checklist",
                    "ask_all",
                    "uncertainty",
                    "learned",
                ):
                    rows.append(
                        trajectory(e, method, model, cache, construction=construction, noise=noise)
                    )
    prov = {
        **provenance(config),
        "split": corpus.audit(episodes),
        "inputHash": digest([e["input"] for e in episodes]),
        "labelHash": digest([e["gold"] for e in episodes]),
        "encoderHash": cache.id,
        "wallSeconds": time.perf_counter() - started,
        "inferenceMs": {
            "median": float(np.median(timing)),
            "p95": float(np.quantile(timing, 0.95)),
        },
        "inferenceByPolicyMs": {
            name: {
                "n": len(times),
                "median": float(np.median(times)),
                "p95": float(np.quantile(times, 0.95)),
            }
            for name, times in policy_timing.items()
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
    episodes = corpus.save(config, directory=config["output"] + "/data")
    cache = embed(episodes, config["encoder"])
    if checkpoint:
        model, saved = load(checkpoint, cache)
        require(
            digest(saved) == digest(config),
            "Evaluation configuration differs from training checkpoint",
        )
    else:
        model = fit(config, episodes, cache, config["output"])
    from .diagnostics import inspect

    write(config["output"] + "/diagnostics.json", inspect(episodes, model, cache))
    evaluate(config, episodes, model, cache, config["output"], started)


def doctor():
    from .providers import envfile

    envfile()
    report = {
        "python": platform.python_version(),
        "torch": importlib.metadata.version("torch"),
        "encoderCached": Path(".cache/torch-embeddings.json").exists(),
        "pilotModel": Path(read("configs/pilot.json")["output"] + "/model.pt").exists(),
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
            "raw-benchmark",
            "diagnose",
            "audit-workload",
            "check-workload",
            "rescore",
            "prepare-suite",
            "check-suite",
            "score-suite",
            "retrieve-suite",
            "train-native",
            "prepare-orders",
            "check-orders",
            "pack-eval",
            "restore-eval",
        ),
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--checkpoint")
    parser.add_argument("--provider", choices=("jev", "sglang", "agent"))
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-calls", type=int, default=0)
    parser.add_argument("--suite", choices=("public", "business"), default="public")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--inputs", default="cases/pilot/inputs.jsonl")
    parser.add_argument("--annotations", default="cases/pilot/annotations.jsonl")
    parser.add_argument("--design", default="configs/workload-v2.json")
    parser.add_argument("--trajectories")
    parser.add_argument("--predictions")
    parser.add_argument("--query")
    args = parser.parse_args()
    config = read(
        args.config
        or (
            "configs/native.json"
            if args.command == "train-native"
            else "configs/orders.json"
            if args.command in ("prepare-orders", "check-orders")
            else "configs/complementary.json"
            if args.command.endswith("-suite")
            else "configs/pilot.json"
            if args.command in ("experiment", "evaluate", "benchmark")
            else "configs/full.json"
        )
    )
    if args.seed is not None:
        config["seed"] = args.seed
    if args.command == "train-native":
        from .native_model import run

        run(config)
    elif args.command in ("prepare-orders", "check-orders"):
        from . import orders

        print(
            orders.prepare(config) if args.command == "prepare-orders" else orders.check(config),
            flush=True,
        )
    elif args.command in ("pack-eval", "restore-eval"):
        from . import snapshot

        print(snapshot.pack() if args.command == "pack-eval" else snapshot.restore(), flush=True)
    elif args.command.endswith("-suite"):
        from . import suite
        from .io import lines

        if args.command == "prepare-suite":
            print(suite.prepare(config), flush=True)
        elif args.command == "check-suite":
            print(suite.check(config["output"]), flush=True)
        elif args.command == "retrieve-suite":
            require(args.query, "Retrieval requires --query")
            print(
                suite.retrieve(
                    lines(Path(config["output"]) / "collection.jsonl"), args.query, args.limit
                ),
                flush=True,
            )
        else:
            from .suite_score import score

            require(args.predictions, "Scoring requires --predictions JSONL")
            print(score(config["output"], lines(args.predictions), args.split), flush=True)
    elif args.command == "prepare":
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
    elif args.command == "rescore":
        from .evaluation import VERSION, rescore
        from .io import jsonl, lines
        from .report import label, summarize

        require(
            args.trajectories and args.config,
            "Rescore requires --trajectories and the original --config",
        )
        require(
            config.get("workload", "legacy") == "legacy",
            "Legacy rescore needs the original corpus configuration",
        )
        episodes = corpus.generate(config)
        metadata = Path(args.trajectories).parent / "metrics.json"
        require(
            metadata.exists(),
            "Rescore needs the original adjacent metrics.json for corpus provenance",
        )
        original = read(metadata)

        def corpus_config(cfg):
            return {
                "dataSeed": cfg.get("dataSeed", cfg["seed"]),
                **{k: cfg[k] for k in ("groups", "variants", "budget")},
                "workload": cfg.get("workload", "legacy"),
                "language": cfg.get("language", "equations"),
                "economics": cfg.get("economics", {}),
            }

        require(
            corpus_config(config) == corpus_config(original["config"]),
            "Rescore corpus differs from recorded original configuration",
        )
        original_hash = original.get("provenance", {}).get("inputHash")
        require(
            not original_hash or original_hash == digest([e["input"] for e in episodes]),
            "Rescore inputs differ from original input hash",
        )
        rows = rescore(lines(args.trajectories), episodes)
        jsonl(config["output"] + "/rescored.jsonl", rows)
        groups = {label(r) for r in rows}
        summary = {k: summarize([r for r in rows if label(r) == k]) for k in sorted(groups)}
        write(
            config["output"] + "/rescore.json",
            {
                "evaluationVersion": VERSION,
                "sourceFile": args.trajectories,
                "sourceHash": digest(Path(args.trajectories).read_bytes()),
                "inputHash": digest([e["input"] for e in episodes]),
                "originalCorpusVerifiedBy": "inputHash"
                if original_hash
                else "original configuration; historical inputHash unavailable",
                "originalMetadataHash": digest(metadata.read_bytes()),
                "provenance": provenance(config),
                "summary": summary,
                "scope": "Recorded coverage and economic losses retained; observed authorization rechecked.",
            },
        )
        print(
            {
                "directory": config["output"],
                "rows": len(rows),
                "summary": summary.get("learned/learned"),
            },
            flush=True,
        )
    elif args.command in ("audit-workload", "check-workload"):
        from .io import lines
        from .workload import check, controlled

        if args.command == "audit-workload":
            report = controlled(config)
        else:
            report = check(lines(args.inputs), lines(args.annotations), read(args.design))
        report["provenance"] = provenance(config)
        path = config["output"] + "/" + args.command + ".json"
        write(path, report)
        print({"report": path, "scope": report["scope"]}, flush=True)
    elif args.command == "diagnose":
        from .diagnostics import inspect

        require(args.checkpoint, "Diagnose requires --checkpoint")
        episodes = corpus.generate(config)
        cache = embed(episodes, config["encoder"])
        model, saved = load(args.checkpoint, cache)
        require(
            {k: v for k, v in config.items() if k != "output"}
            == {k: v for k, v in saved.items() if k != "output"},
            "Diagnosis configuration differs from training checkpoint",
        )
        report = inspect(episodes, model, cache, args.split)
        report["provenance"] = provenance(config)
        write(config["output"] + "/diagnostics.json", report)
        print(report, flush=True)
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

    elif args.command == "raw-benchmark":
        from .providers import envfile
        from .raw import run

        envfile()
        require(args.provider, "Raw comparison requires --provider")
        print(
            run(config, args.provider, args.max_calls, args.inputs, args.annotations, args.limit),
            flush=True,
        )


if __name__ == "__main__":
    main()
