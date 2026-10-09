"""Evaluate frozen Newsvendor models on newly reserved source-disjoint retail histories."""

import argparse
import os
import sys
import tarfile
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from compare_observations import observed_intervals
from study_demand import evaluate, summarize

from newsvendor import sequence, structured_retail, structured_train
from newsvendor.cli import provenance
from newsvendor.io import digest, jsonl, lines, read, require, write
from newsvendor.structured_rollout import ResearchRouter, rollout
from newsvendor.structured_tool_eval import weights_hash
from newsvendor.suite import check


def paired_difference(left, right, metric, complete=False):
    old = {r["id"]: r for r in right}
    require(set(old) == {r["id"] for r in left}, "Unpaired rollout identities")
    groups, periods = defaultdict(list), set()
    for row in left:
        previous = old[row["id"]]
        require(
            row["family"] == previous["family"]
            and row["outcomeComplete"] == previous["outcomeComplete"],
            "Different paired source or outcome",
        )
        if complete and not row["outcomeComplete"]:
            continue
        if row[metric] is None or previous[metric] is None:
            continue
        groups[row["family"]].append(row[metric] - previous[metric])
        task = row["events"][0]["input"]["task"]
        periods.add((task["sku"], task["period"]))
    if not groups:
        return {
            "meanDifference": None,
            "sourceBootstrap95": None,
            "sourceFamilies": 0,
            "uniquePeriods": 0,
            "conditions": 0,
            "reason": "No comparable observed periods; exact loss is not identified",
        }
    sums = np.array([sum(groups[k]) for k in sorted(groups)])
    counts = np.array([len(groups[k]) for k in sorted(groups)])
    ids = np.random.default_rng(61).integers(len(sums), size=(5000, len(sums)))
    bootstrap = sums[ids].sum(-1) / counts[ids].sum(-1)
    return {
        "meanDifference": float(sums.sum() / counts.sum()),
        "sourceBootstrap95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        "sourceFamilies": len(groups),
        "uniquePeriods": len(periods),
        "conditions": int(counts.sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True)
    args = parser.parse_args()
    plan = read(args.registration)
    require(
        plan["trainingAllowed"] is False and plan["newSourceHoldout"],
        "Frozen source holdout required",
    )
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == plan["gpu"], "Assigned GPU only")
    require(
        torch.cuda.device_count() == 1 and "L40S" in torch.cuda.get_device_name(0),
        "Single L40S required",
    )
    require(provenance(plan)["sourceHash"] == plan["sourceHash"], "Inference source changed")
    for path, expected in plan["scripts"].items():
        require(digest(Path(path).read_bytes()) == expected, "Registered script changed")
    for entry in plan["models"].values():
        require(
            digest(Path(entry["path"]).read_bytes()) == entry["hash"], "Registered model changed"
        )
    dataset, root = Path(plan["dataset"]), Path(plan["output"])
    require(not root.exists(), "Preserve earlier evaluation")
    check(dataset)
    manifest = read(dataset / "manifest.json")
    selection = read(dataset / "selection.json")
    require(
        selection["registrationHash"] == digest(Path(args.registration).read_bytes()),
        "Data registered against a different plan",
    )
    ids = set(manifest["evaluationExpansion"]["ids"])
    require(len(ids) == plan["expectedSources"], "Holdout source count changed")
    rows = [r for r in lines(dataset / "inputs.jsonl") if r["id"] in ids]
    require(
        all(r["component"] == "retail" and r["split"] == "test" for r in rows),
        "Wrong evaluation split",
    )
    episodes = [
        e
        for e in structured_retail.cases(rows, plan["dataSeed"], 7)
        if e["cutoffIndex"] in plan["cutoffs"]
    ]
    require(
        len(episodes) == len(rows) * len(plan["cutoffs"]) * len(structured_retail.SCENARIOS),
        "Missing registered condition",
    )
    root.mkdir(parents=True)
    jsonl(
        root / "inputs.jsonl",
        [{k: e[k] for k in ("id", "family", "split", "input")} for e in episodes],
    )
    input_hash = digest((root / "inputs.jsonl").read_bytes())
    # Outcome values are passed only to scoring after the rollout input file is frozen.
    labels = {r["id"]: r["target"] for r in lines(dataset / "labels.jsonl") if r["id"] in ids}
    episodes = structured_retail.attach_targets(episodes, rows, labels)
    forecasts = sequence.samples(rows, labels, plan["minHistory"])
    require(
        {e["family"] for e in episodes} == set(manifest["evaluationExpansion"]["families"]),
        "Different evaluation sources",
    )
    report = {
        "scope": "Frozen models on previously unused project source groups; original public Train provenance retained. Generated financial documents and manager replies are controlled conditions, not organizational effectiveness.",
        "testUsed": True,
        "newSourceHoldout": True,
        "legacyTestUsed": False,
        "trainingPerformed": False,
        "registration": plan,
        "registrationHash": digest(Path(args.registration).read_bytes()),
        "provenance": provenance(plan),
        "inputHash": input_hash,
        "dataManifestHash": digest(manifest),
        "selectionHash": digest(selection),
        "episodes": len(episodes),
        "sourceFamilies": len(rows),
        "cutoffs": plan["cutoffs"],
        "results": {},
        "forecast": {},
        "models": {},
    }
    write(root / "registered.json", report)
    with tarfile.open(root / "source.tar.gz", "w:gz") as archive:
        for path in sorted(Path("src/newsvendor").glob("*.py")):
            archive.add(path, arcname=str(path))
        for path in [*map(Path, plan["scripts"]), Path(args.registration)]:
            archive.add(path, arcname=str(path))
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    write(root / "job.json", {"stage": "running", "pid": os.getpid()})
    fixed_language, fixed_common, fixed_actions = None, None, None
    all_records = {}
    try:
        for name, entry in plan["models"].items():
            model, tokenizer, config, _ = structured_train.load(entry["path"], "cuda")
            require(config["noValue"] == (name == "no_value"), "Wrong policy mode")
            weights = model.state_dict()
            language = weights_hash(
                {
                    k: v
                    for k, v in weights.items()
                    if not k.startswith(("demand.", "heads.value.", "heads.recovery."))
                }
            )
            require(
                fixed_language is None or fixed_language == language, "Changed document constructor"
            )
            fixed_language = language
            if name != "no_value":
                actions = weights_hash(
                    {
                        k: v
                        for k, v in weights.items()
                        if k.startswith(("heads.value.", "heads.recovery."))
                    }
                )
                require(
                    fixed_actions is None or fixed_actions == actions,
                    "Forecast comparison changed action heads",
                )
                fixed_actions = actions
            if name in ("selected", "no_value"):
                common = weights_hash(
                    {
                        k: v
                        for k, v in weights.items()
                        if not k.startswith(("heads.value.", "heads.recovery."))
                    }
                )
                require(fixed_common is None or fixed_common == common, "Changed paired GRU")
                fixed_common = common
            report["models"][name] = {
                **entry,
                "languageHash": language,
                "demandHash": weights_hash(model.demand.state_dict()),
            }
            router = ResearchRouter(model, tokenizer, config["encoder"], config["noValue"])
            router.cache_states()
            policies = ("selected", "checklist") if name == "selected" else (name,)
            for policy in policies:
                records = []
                for index, episode in enumerate(episodes):
                    records.append(rollout(episode, router, explore=policy == "checklist", noise=0))
                    if (index + 1) % 32 == 0:
                        progress = {
                            "model": name,
                            "policy": policy,
                            "processed": index + 1,
                            "total": len(episodes),
                            "seconds": time.perf_counter() - started,
                        }
                        write(root / "progress.json", progress)
                        print(progress, flush=True)
                path = root / f"{policy}.jsonl"
                jsonl(path, records)
                report["results"][policy] = {
                    **structured_retail.summarize(records),
                    "rawHash": digest(path.read_bytes()),
                }
                all_records[policy] = records
                write(root / "partial.json", report)
            router.cache_states(False)
            if name != "no_value":
                measured = evaluate(
                    model.demand,
                    forecasts,
                    plan["observationNodes"],
                    plan["allocationConcentration"],
                )
                path = root / f"{name}-forecasts.jsonl"
                jsonl(path, measured)
                report["forecast"][name] = {
                    "metrics": summarize(measured),
                    "rawHash": digest(path.read_bytes()),
                    "coverage": {
                        str(h): observed_intervals([r for r in measured if r["horizon"] == h])
                        for h in (1, 7)
                    },
                    "scope": "Secondary rolling-origin forecast metrics; primary economic evaluation uses registered non-overlapping weeks",
                }
            require(
                digest(Path(entry["path"]).read_bytes()) == entry["hash"],
                "Evaluation altered checkpoint",
            )
            del router, model, tokenizer, weights
            torch.cuda.empty_cache()
        report["comparisons"] = {
            "selectedMinusPreviousTotal": paired_difference(
                all_records["selected"], all_records["previous"], "total", complete=True
            ),
            "valueMinusChecklistTotal": paired_difference(
                all_records["selected"], all_records["checklist"], "total", complete=True
            ),
            "valueMinusNoValueTotal": paired_difference(
                all_records["selected"], all_records["no_value"], "total", complete=True
            ),
            "selectedMinusPreviousCensoredBound": paired_difference(
                all_records["selected"], all_records["previous"], "censoredOrderLossLowerBound"
            ),
        }
        require(
            digest((root / "inputs.jsonl").read_bytes()) == input_hash,
            "Input file modified by scoring",
        )
        report.update(
            seconds=time.perf_counter() - started,
            peakGpuBytes=torch.cuda.max_memory_allocated(),
            fixedLanguageHash=fixed_language,
            pairedConstructorAndDemandHash=fixed_common,
            fixedValueAndRecoveryHash=fixed_actions,
        )
        write(root / "report.json", report)
        write(
            root / "job.json",
            {"stage": "complete", "seconds": report["seconds"], "legacyTestUsed": False},
        )
        print(
            {
                "stage": "complete",
                "comparisons": report["comparisons"],
                "results": {
                    k: v["meanTotalOnCompletePeriods"] for k, v in report["results"].items()
                },
            },
            flush=True,
        )
    except Exception as error:
        write(
            root / "job.json",
            {"stage": "failed", "type": type(error).__name__, "message": str(error)},
        )
        raise


if __name__ == "__main__":
    main()
