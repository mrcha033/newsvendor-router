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
"""Train and measure the small Newsvendor model on linked, explicitly controlled inputs."""

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
    parser.add_argument("--config", default="configs/l40s-research.json")
    parser.add_argument(
        "--stage", choices=("check", "diagnose", "train", "policy"), default="check"
    )
    parser.add_argument("--output")
    args = parser.parse_args()

    import torch

    from newsvendor import corpus, structured_critic, structured_retail, structured_train
    from newsvendor.io import digest, jsonl, lines, read, require, write
    from newsvendor.structured_metrics import generated_metrics
    from newsvendor.structured_progress import Progress
    from newsvendor.structured_rollout import ResearchRouter, rollout
    from newsvendor.structured_tool_eval import weights_hash
    from newsvendor.suite import check
    from newsvendor.train import seed

    config = read(args.config)
    if args.output:
        config["output"] = args.output
    directory = Path(config["output"])
    require(
        not (directory / "run.json").exists(),
        "Use a new output directory; never overwrite a research run",
    )
    directory.mkdir(parents=True, exist_ok=True)
    check(config["dataset"])
    source_rows = [
        r for r in lines(Path(config["dataset"]) / "inputs.jsonl") if r["split"] in ("train", "dev")
    ]
    rows = [r for r in source_rows if r["component"] == "retail"]
    components = config.get("documentComponents", [])
    require(
        set(components) <= {"cuad", "contractnli", "orsharc", "tatqa"},
        "Unsupported core document component",
    )
    documents = [r for r in source_rows if r["component"] in components]
    collection = lines(Path(config["dataset"]) / "collection.jsonl") if documents else []
    episodes = structured_retail.cases(rows, config["dataSeed"], config["cutoffStride"])
    observed = [
        {k: e[k] for k in ("id", "family", "split", "input", "provenance")} for e in episodes
    ]
    jsonl(directory / "inputs.jsonl", observed)
    # Freeze the exact constructed inputs before opening outcome annotations.
    input_hash = digest((directory / "inputs.jsonl").read_bytes())
    if documents:
        jsonl(directory / "documents-inputs.jsonl", documents)
    ids = {r["id"] for r in rows + documents}
    labels = {
        r["id"]: r["target"]
        for r in lines(Path(config["dataset"]) / "labels.jsonl")
        if r["id"] in ids
    }
    episodes = structured_retail.attach_targets(episodes, rows, labels)
    jsonl(
        directory / "environment.jsonl",
        [
            {k: e[k] for k in ("id", "responses", "target", "financialAnnotations")}
            for e in episodes
        ],
    )
    require(
        digest((directory / "inputs.jsonl").read_bytes()) == input_hash,
        "Scoring labels changed observed inputs",
    )
    manifest = {
        "benchmark": structured_retail.VERSION,
        "scope": "Controlled financial conditions paired with observed retail histories; not actual organizational effectiveness",
        "inputFileHash": input_hash,
        "environmentFileHash": digest((directory / "environment.jsonl").read_bytes()),
        "sourceManifestHash": digest(read(Path(config["dataset"]) / "manifest.json")),
        "dataSeed": config["dataSeed"],
        "cutoffStride": config["cutoffStride"],
        "splits": {
            s: {
                "episodes": sum(e["split"] == s for e in episodes),
                "sourceFamilies": sorted({e["family"] for e in episodes if e["split"] == s}),
                "completeOutcomeEpisodes": sum(
                    e["split"] == s and all(e["target"]["complete"]) for e in episodes
                ),
            }
            for s in ("train", "dev")
        },
        "sourceOverlaps": 0,
        "testUsed": False,
    }
    original = []
    if config.get("originalResearch", False):
        source = corpus.generate(read(config["researchConfig"]))
        audited = corpus.audit(source)
        original = [e for e in source if e["split"] in ("train", "dev")]
        jsonl(
            directory / "generated-inputs.jsonl",
            [{k: e[k] for k in ("id", "family", "split", "input")} for e in original],
        )
        jsonl(
            directory / "generated-environment.jsonl",
            [{"id": e["id"], "gold": e["gold"]} for e in original],
        )
        manifest["generated"] = {
            "scope": "Original generated benchmark, with unchanged source split and loss definition; response probabilities are environment-only",
            "corpus": audited,
            "testUsed": False,
            "inputFileHash": digest((directory / "generated-inputs.jsonl").read_bytes()),
            "environmentFileHash": digest((directory / "generated-environment.jsonl").read_bytes()),
            "splits": {
                s: {
                    "episodes": sum(e["split"] == s for e in original),
                    "families": len({e["family"] for e in original if e["split"] == s}),
                }
                for s in ("train", "dev")
            },
        }
        require(
            config["policy"].get("selection") == "benchmark_normalized",
            "Mixed policy training requires explicit benchmark normalization",
        )
    require(
        not (
            set(manifest["splits"]["train"]["sourceFamilies"])
            & set(manifest["splits"]["dev"]["sourceFamilies"])
        ),
        "Source-group overlap",
    )
    require(
        not (
            {e["family"] for e in episodes + original if e["split"] == "train"}
            & {e["family"] for e in episodes + original if e["split"] == "dev"}
        ),
        "Combined source-group overlap",
    )
    if documents:
        document_labels = {r["id"]: labels[r["id"]] for r in documents}
        write(directory / "documents-labels.json", document_labels)
        manifest["documents"] = {
            "components": components,
            "inputsHash": digest((directory / "documents-inputs.jsonl").read_bytes()),
            "labelsHash": digest(document_labels),
            "collectionHash": digest(collection),
            "splits": {
                s: {
                    c: sum(r["split"] == s and r["component"] == c for r in documents)
                    for c in components
                }
                for s in ("train", "dev")
            },
        }
    write(directory / "manifest.json", manifest)
    if args.stage == "check":
        print(manifest, flush=True)
        return
    config["linkedManifest"] = str(directory / "manifest.json")
    require(
        torch.cuda.is_available()
        and torch.cuda.device_count() == 1
        and "L40S" in torch.cuda.get_device_name(0),
        "Run on the assigned, single visible L40S",
    )
    torch.set_num_threads(8)
    seed(config["seed"])
    started = time.perf_counter()
    model, tokenizer, parent = structured_train.initialize(config)
    model = model.to("cuda").eval()
    progress = Progress(config, structured_train.dataset_hashes(config), "cuda", None)
    write(directory / "parent.json", parent)
    write(directory / "job.json", {"stage": "initial_dev", "pid": os.getpid()})
    report = {
        "config": config,
        "parent": parent,
        "trainableParameters": sum(p.numel() for p in model.parameters()),
        "manifest": manifest,
    }
    dev = [e for e in episodes if e["split"] == "dev"]
    original_dev = [e for e in original if e["split"] == "dev"]

    def evaluate(name, cohort=dev, summarize=structured_retail.summarize):
        router = ResearchRouter(model, tokenizer, config["encoder"], config["noValue"])
        router.cache_states()
        measured = {}
        for policy in ("learned", "checklist"):
            records = []
            for index, episode in enumerate(cohort):
                records.append(rollout(episode, router, explore=policy == "checklist"))
                if (index + 1) % 32 == 0:
                    progress.update(name, policy=policy, processed=index + 1, total=len(cohort))
            path = directory / f"{name}-{policy}.jsonl"
            jsonl(path, records)
            measured[policy] = {**summarize(records), "rawHash": digest(path.read_bytes())}
        router.cache_states(False)
        write(directory / f"{name}.json", measured)
        return measured

    if config.get("evaluateInitial", True):
        report["initialDev"] = evaluate("initial-dev")
        if original_dev:
            report["initialGeneratedDev"] = evaluate(
                "initial-generated-dev", original_dev, generated_metrics
            )
    if args.stage == "train":
        training_cases, development_cases = [], []
        for mode, episode in structured_train.research_cases(episodes + original):
            destination = training_cases if episode["split"] == "train" else development_cases
            destination.append((mode, episode))
        for row in documents:
            destination = training_cases if row["split"] == "train" else development_cases
            destination.append(("public", row))
        training_cases = structured_train.balance_cases(training_cases, config["training"])
        write(
            directory / "language-cases.json",
            {
                "train": [
                    {
                        "mode": mode,
                        "id": row["id"],
                        "family": row["family"],
                        "inputHash": digest(row["input"]),
                    }
                    for mode, row in training_cases
                ],
                "dev": [
                    {
                        "mode": mode,
                        "id": row["id"],
                        "family": row["family"],
                        "inputHash": digest(row["input"]),
                    }
                    for mode, row in development_cases
                ],
            },
        )
        write(directory / "job.json", {"stage": "language", "pid": os.getpid()})
        report["language"] = structured_train.train_language(
            model,
            tokenizer,
            config,
            training_cases,
            development_cases,
            labels,
            collection,
            progress,
        )
        require(
            weights_hash(model.demand.state_dict()) == parent["demandWeightsHash"],
            "Language training changed the shared demand model",
        )
        report["demand"] = {
            "stage": "reuse_trained_GRU",
            "parent": parent,
            "unit": "globally-normalized-sales",
        }
        structured_train.save(directory / "common.pt", model, config, report)
        report["constructionDev"] = evaluate("construction-dev")
        if original_dev:
            report["constructionGeneratedDev"] = evaluate(
                "construction-generated-dev", original_dev, generated_metrics
            )
        if documents:
            from newsvendor.structured_policy import public_validation

            report["documentsDev"] = public_validation(
                model,
                tokenizer,
                config,
                development_cases,
                labels,
                collection,
                directory / "documents-dev.jsonl",
            )
    if args.stage in ("train", "policy"):
        if args.stage == "policy":
            report["language"] = {
                "stage": "reuse_trained_encoder_and_constructor",
                "parent": parent,
            }
            report["demand"] = {
                "stage": "reuse_trained_GRU",
                "parent": parent,
                "unit": "globally-normalized-sales",
            }
        write(
            directory / "job.json",
            {"stage": "recovery" if config["noValue"] else "value", "pid": os.getpid()},
        )
        exact = [e for e in episodes if all(e["target"]["complete"])] + original
        report["policyOutcomeCoverage"] = {
            s: sum(e["split"] == s for e in exact) for s in ("train", "dev")
        }
        report["policySelection"] = config["policy"].get("selection", "mean_total")
        report["policy"] = structured_critic.fit(
            model, tokenizer, config, exact, progress, [], {}, []
        )
        structured_train.save(directory / "model.pt", model, config, report)
        report["finalDev"] = evaluate("final-dev")
        if original_dev:
            report["finalGeneratedDev"] = evaluate(
                "final-generated-dev", original_dev, generated_metrics
            )
    report["seconds"] = time.perf_counter() - started
    report["peakGpuBytes"] = torch.cuda.max_memory_allocated()
    write(directory / "report.json", report)
    write(
        directory / "job.json",
        {"stage": "complete", "pid": os.getpid(), "seconds": report["seconds"], "testUsed": False},
    )
    print({"output": str(directory), "stage": "complete", "seconds": report["seconds"]}, flush=True)


if __name__ == "__main__":
    main()
