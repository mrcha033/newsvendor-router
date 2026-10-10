"""Pair parameter decoders on observed documents and their actual ordering trajectories."""

import argparse
import copy
import gzip
import json
import math
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import torch
from evaluate_operands import observed_key, research_gates
from evaluate_responses import CachedRouter, conditions, expectations, measurement
from prepare_parameter_contrasts import alternate_scope

from newsvendor import structured_compare, structured_retail
from newsvendor.bundle import file_hash, load_bundle
from newsvendor.cli import provenance
from newsvendor.construction import atoms, parameter_record
from newsvendor.io import digest, jsonl, lines, read, require, write
from newsvendor.structured_inputs import research_input
from newsvendor.structured_metrics import generated_metrics
from newsvendor.structured_rollout import rollout
from newsvendor.structured_tool_eval import weights_hash

METHODS = {"parent": "independent", "scoped": "scoped", "first": "first"}
CONDITIONS = ("wrong_sku", "wrong_period", "older_version", "refund_order")


def documents(value, condition, seed):
    """Perturb observed documents only; never inspect reference values or future outcomes."""
    require(condition in CONDITIONS, "Unknown observed document condition")
    require(not value["history"] and not value.get("memory"), "Initial observations only")
    result = copy.deepcopy(value)
    rng = random.Random(digest([seed, research_input(value), condition]))
    added = []
    if condition == "refund_order":
        pattern = re.compile(
            r"(Refund per unit\s*=\s*-?\d+(?:\.\d+)?);\s*"
            r"(handling fee per unit\s*=\s*-?\d+(?:\.\d+)?)(.*)",
            re.I,
        )
        for doc in result["docs"]:
            match = pattern.fullmatch(doc["text"])
            if match:
                doc["text"] = match[2] + "; " + match[1] + match[3]
    else:
        key = {"wrong_sku": "sku", "wrong_period": "period"}.get(condition)
        foreign = alternate_scope(result["task"][key], key, rng) if key else None
        for doc in result["docs"]:
            numbers = [a for a in atoms(doc) if a["unit"].startswith("currency/")]
            if not numbers:
                continue
            extra = copy.deepcopy(doc)
            for atom in sorted(numbers, key=lambda a: a["span"][0], reverse=True):
                lo, hi = atom["span"]
                extra["text"] = (
                    extra["text"][:lo] + f"{atom['value'] * 1.13 + 0.7:.12g}" + extra["text"][hi:]
                )
            if key:
                extra[key] = foreign
                extra["version"] = doc["version"] + 1
            else:
                extra["version"] = 0
                doc["version"] = max(1, doc["version"])
            added.append(extra)
        result["docs"].extend(added)
    for i, doc in enumerate(result["docs"]):
        doc["id"] = digest([seed, value["task"]["sku"], condition, i])[:16]
    rng.shuffle(result["docs"])
    return result


def check_observed(original, changed):
    """Post-construction scoring audit; annotations never influence the transformation."""
    a, b = parameter_record(original), parameter_record(changed)
    require(a["state"] == b["state"] and a["types"] == b["types"], "Document states changed")
    require(a["values"].keys() == b["values"].keys(), "Observed field availability changed")
    require(
        all(
            math.isclose(v, b["values"][k], rel_tol=1e-7, abs_tol=1e-7)
            for k, v in a["values"].items()
        ),
        "Observed financial conditions changed",
    )


def metric(value, state):
    metrics = structured_retail.parameter_metrics(value, state)
    ref = parameter_record(value)
    metrics["wrongAccepted"] = sum(
        k not in ref["values"] or not math.isclose(v, ref["values"][k], rel_tol=1e-7, abs_tol=1e-7)
        for k, v in state["values"].items()
        if k in ("c", "p", "v", "b")
    )
    return metrics


def summarize(items):
    result = {"cases": len(items), "fields": 4 * len(items)}
    for key in (
        "parameterAccuracy",
        "stateAccuracy",
        "typeAccuracy",
        "rawParameterAccuracy",
        "rawStateAccuracy",
    ):
        result[key] = sum(r[key] for r in items) / len(items)
    for key in ("evidenceCorrect", "evidenceCount", "wrongAccepted"):
        result[key] = sum(r[key] for r in items)
    result["evidenceAccuracy"] = result["evidenceCorrect"] / max(1, result["evidenceCount"])
    return result


def measured(record):
    result = measurement(record)
    recovered = 0
    for index, event in enumerate(record["events"][:-1]):
        slot = event["action"]
        if slot not in ("c", "p", "v", "b") or slot in parameter_record(event["input"])["values"]:
            continue
        after = record["events"][index + 1]
        expected = parameter_record(after["input"])["values"].get(slot)
        actual = after["state"]["values"].get(slot)
        recovered += int(
            expected is not None
            and actual is not None
            and math.isclose(actual, expected, rel_tol=1e-7, abs_tol=1e-7)
        )
    result["metrics"]["neededRecovered"] = recovered
    return result


def diagnose(plan, root, routers, cache):
    report = {"partitions": {}}
    manifest = read(plan["contrasts"])
    require(
        not set(manifest["partitions"]["fit"]["families"])
        & set(manifest["partitions"]["held"]["families"]),
        "Contrast source overlap",
    )
    partitions = plan.get("diagnosticPartitions", ["fit", "held"])
    require("held" in partitions and set(partitions) <= {"fit", "held"}, "Train partitions only")
    for partition in partitions:
        rows = lines(manifest["partitions"][partition]["path"])
        require(all(r["split"] == "train" for r in rows), "Train diagnosis only")
        metrics = {name: defaultdict(list) for name in METHODS}
        streams = {
            name: gzip.open(root / f"{partition}-{name}.jsonl.gz", "wt", compresslevel=3)
            for name in METHODS
        }
        try:
            for index, row in enumerate(rows):
                cache.clear()
                for name, router in routers.items():
                    state = router.construct(row["input"])
                    scores = metric(row["input"], state)
                    metrics[name]["all"].append(scores)
                    metrics[name][row["contrast"]].append(scores)
                    record = {k: row[k] for k in ("id", "family", "contrast", "inputHash")}
                    record.update(state=state, metrics=scores)
                    streams[name].write(json.dumps(record, allow_nan=False) + "\n")
                if (index + 1) % 100 == 0:
                    write(
                        root / "progress.json",
                        {"partition": partition, "cases": index + 1, "total": len(rows)},
                    )
        finally:
            for stream in streams.values():
                stream.close()
        report["partitions"][partition] = {
            name: {k: summarize(v) for k, v in groups.items()} for name, groups in metrics.items()
        }
        write(root / "report.json", report)
    a = report["partitions"]["held"]["parent"]["all"]
    b = report["partitions"]["held"]["scoped"]["all"]
    report["eligibleForSequential"] = (
        b["parameterAccuracy"] > a["parameterAccuracy"]
        and b["evidenceAccuracy"] > a["evidenceAccuracy"]
        and b["wrongAccepted"] <= a["wrongAccepted"]
        and b["stateAccuracy"] >= a["stateAccuracy"]
    )
    return report


def episodes(plan, split):
    cohort = plan["cohorts"][split]
    values = {r["id"]: r for r in lines(cohort["inputs"])}
    result = []
    for environment in lines(cohort["environment"]):
        episode = copy.deepcopy(values[environment["id"]]) | environment
        require(episode["split"] == split, "Evaluation split changed")
        if "forecast" in episode["input"]["task"]:
            episode.update(
                benchmark=structured_retail.VERSION,
                cutoffIndex=len(episode["input"]["observations"]),
                kind="retail",
            )
        else:
            episode["kind"] = "generated"
        result.append(episode)
    return result


def sequential(plan, root, routers, cache, split):
    base = episodes(plan, split)
    report = {
        "conditions": {},
        "counts": {k: sum(e["kind"] == k for e in base) for k in ("retail", "generated")},
    }
    require(report["counts"] == plan["cohorts"][split]["counts"], "Cohort size changed")
    for condition in ("original", *CONDITIONS):
        current = copy.deepcopy(base)
        if condition != "original":
            if split == "train":
                unique = {}
                for episode in sorted(
                    current, key=lambda e: digest([plan["seed"], research_input(e["input"])])
                ):
                    unique.setdefault((episode["family"], episode.get("scenario", "")), episode)
                current = list(unique.values())
            for episode in current:
                before = episode["input"]
                episode["input"] = documents(before, condition, plan["seed"])
                check_observed(before, episode["input"])
        jsonl(
            root / f"{condition}-inputs.jsonl",
            [{k: e[k] for k in ("id", "family", "split", "kind", "input")} for e in current],
        )
        generated = {name: [] for name in METHODS}
        retail = {name: [] for name in METHODS}
        initial = {name: [] for name in METHODS}
        streams = {
            name: gzip.open(root / f"{condition}-{name}.jsonl.gz", "wt", compresslevel=3)
            for name in METHODS
        }
        try:
            for index, episode in enumerate(current):
                cache.clear()
                for name, router in routers.items():
                    router.reset()
                    initial[name].append(
                        metric(episode["input"], router.construct(episode["input"]))
                    )
                    masks = (
                        conditions(0)
                        if episode["kind"] == "retail" and condition == "original"
                        else [()]
                    )
                    for missing in masks:
                        options = (
                            {"missing_responses": frozenset(missing)}
                            if episode["kind"] == "retail"
                            else {}
                        )
                        record = rollout(episode, router, **options)
                        record.update(
                            documentCondition=condition,
                            missingResponses=list(missing),
                            kind=episode["kind"],
                        )
                        streams[name].write(
                            json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n"
                        )
                        if episode["kind"] == "retail":
                            retail[name].append(
                                measured(record) | {"missingResponses": list(missing)}
                            )
                        else:
                            generated[name].append(record)
                if (index + 1) % 20 == 0:
                    write(
                        root / "progress.json",
                        {"condition": condition, "cases": index + 1, "total": len(current)},
                    )
        finally:
            for stream in streams.values():
                stream.close()
        result = {}
        for name in METHODS:
            expectations_by_rate = (
                {str(rate): expectations(retail[name], rate) for rate in (0, 0.1, 0.2)}
                if condition == "original"
                else {"0": retail[name]}
            )
            result[name] = {
                "initial": summarize(initial[name]),
                "generated": generated_metrics(generated[name]),
                "generatedTerminalLoss": sum(r["terminalLoss"] for r in generated[name])
                / len(generated[name]),
                "retail": {
                    rate: structured_compare.means(values)
                    for rate, values in expectations_by_rate.items()
                },
            }
            jsonl(root / f"{condition}-{name}-retail-measurements.jsonl", retail[name])
        report["conditions"][condition] = result
        write(root / "report.json", report)
    a, b = report["conditions"]["original"]["parent"], report["conditions"]["original"]["scoped"]
    report["originalGates"] = research_gates(a, b)
    report["challengeGates"] = {}
    for condition in CONDITIONS:
        a, b = report["conditions"][condition]["parent"], report["conditions"][condition]["scoped"]
        report["challengeGates"][condition] = {
            "initialParameters": b["initial"]["parameterAccuracy"]
            >= a["initial"]["parameterAccuracy"],
            "initialEvidence": b["initial"]["evidenceAccuracy"] >= a["initial"]["evidenceAccuracy"],
            "initialWrongAccepted": b["initial"]["wrongAccepted"] <= a["initial"]["wrongAccepted"],
            "generatedTotal": b["generated"]["meanTotal"] <= a["generated"]["meanTotal"] + 1e-9,
            "generatedFalseHandoffs": b["generated"]["falseHandoffs"]
            <= a["generated"]["falseHandoffs"],
            "generatedUnnecessaryQuestions": b["generated"]["resolvedFieldRequests"]
            <= a["generated"]["resolvedFieldRequests"],
            "retailTotal": b["retail"]["0"]["totalOnCompletePeriods"]["mean"]
            <= a["retail"]["0"]["totalOnCompletePeriods"]["mean"] + 1e-9,
            "retailRecovery": b["retail"]["0"]["neededRecovered"]["mean"]
            >= a["retail"]["0"]["neededRecovered"]["mean"] - 1e-12,
            "retailFalseHandoffs": b["retail"]["0"]["falseHandoff"]["mean"]
            <= a["retail"]["0"]["falseHandoff"]["mean"],
            "retailUnnecessaryQuestions": b["retail"]["0"]["unnecessaryRequests"]["mean"]
            <= a["retail"]["0"]["unnecessaryRequests"]["mean"],
            "retailFinalParameters": b["retail"]["0"]["parameterAccuracy"]["mean"]
            >= a["retail"]["0"]["parameterAccuracy"]["mean"] - 1e-12,
            "generatedFinalParameters": b["generated"]["finalParameters"]["parameterAccuracy"]
            >= a["generated"]["finalParameters"]["parameterAccuracy"] - 1e-12,
        }
    report["eligible"] = all(report["originalGates"].values()) and all(
        all(g.values()) for g in report["challengeGates"].values()
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True)
    parser.add_argument("--stage", choices=("diagnose", "train", "dev"), required=True)
    args = parser.parse_args()
    plan = read(args.registration)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == plan["gpu"], "Assigned L40S only")
    require(torch.cuda.device_count() == 1 and "L40S" in torch.cuda.get_device_name(), "L40S only")
    require(provenance({})["sourceHash"] == plan["sourceHash"], "Registered source changed")
    require(file_hash(__file__) == plan["scriptHash"], "Registered evaluator changed")
    for path, expected in plan["files"].items():
        require(file_hash(path) == expected, "Registered file changed: " + path)
    root = Path(plan["output"]) / args.stage
    require(not root.exists(), "Preserve previous measurements")
    if args.stage == "train":
        require(
            read(root.parent / "diagnose/report.json")["eligibleForSequential"],
            "Train diagnosis gate failed",
        )
    if args.stage == "dev":
        require(read(root.parent / "train/report.json")["eligible"], "Sequential Train gate failed")
    root.mkdir(parents=True)
    write(root / "registered.json", plan | {"stage": args.stage})
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    model, tokenizer, config, manifest = load_bundle(plan["bundle"], "cuda")
    require(manifest["weightsHash"] == plan["weightsHash"], "Parent weights changed")
    model.eval().requires_grad_(False)
    forward = model.forward
    cache, checks = {}, []

    @torch.inference_mode()
    def cached(view):
        key = digest(observed_key(view))
        if key not in cache:
            cache[key] = forward(view)
        if len(checks) < 12:
            direct = forward(view)
            require(
                all(torch.equal(v, direct[k]) for k, v in cache[key].items()),
                "Cache changed model output",
            )
            checks.append(key)
        return cache[key]

    model.forward = cached
    routers = {
        name: CachedRouter(model, tokenizer, config["encoder"] | {"parameterDecoding": method})
        for name, method in METHODS.items()
    }
    report = (
        diagnose(plan, root, routers, cache)
        if args.stage == "diagnose"
        else sequential(plan, root, routers, cache, args.stage)
    )
    require(
        weights_hash(model.state_dict()) == plan["weightsHash"], "Weights changed during comparison"
    )
    report.update(
        source=provenance({}),
        stage=args.stage,
        testUsed=False,
        newSourceHoldoutUsed=False,
        trainingPerformed=False,
        weightsHash=plan["weightsHash"],
        directForwardChecks=len(checks),
        seconds=time.perf_counter() - started,
        peakGpuBytes=torch.cuda.max_memory_allocated(),
        rawHashes={p.name: file_hash(p) for p in root.iterdir() if p.suffix in (".gz", ".jsonl")},
    )
    write(root / "report.json", report)
    print(
        {
            k: v
            for k, v in report.items()
            if k not in ("source", "conditions", "partitions", "rawHashes")
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
