"""Evaluate fixed operand refits on their own sequential states and paired manager replies."""

import argparse
import copy
import gzip
import json
import os
import time
from pathlib import Path

import torch
from evaluate_responses import CachedRouter, conditions, expectations, measurement

from newsvendor import structured_compare, structured_retail, structured_train
from newsvendor.bundle import file_hash
from newsvendor.cli import provenance
from newsvendor.io import digest, jsonl, lines, read, require, write
from newsvendor.structured_metrics import generated_metrics
from newsvendor.structured_model import Router
from newsvendor.structured_operands import proposals
from newsvendor.structured_policy import public_validation, retained
from newsvendor.structured_rollout import rollout
from newsvendor.structured_tool_eval import weights_hash


def observed_key(value):
    if torch.is_tensor(value):
        return ["tensor", str(value.dtype), list(value.shape), value.tolist()]
    if isinstance(value, dict):
        pairs = [[observed_key(k), observed_key(v)] for k, v in value.items()]
        return ["dict", sorted(pairs, key=lambda pair: digest(pair[0]))]
    if isinstance(value, (list, tuple)):
        return [type(value).__name__, [observed_key(v) for v in value]]
    return value


def research_gates(parent, current):
    a, b = parent["generated"], current["generated"]
    result = {
        "generatedTotal": b["meanTotal"] <= a["meanTotal"] + 1e-9,
        "generatedFalseHandoffs": b["falseHandoffs"] <= a["falseHandoffs"],
        "generatedResolvedRequests": b["resolvedFieldRequests"] <= a["resolvedFieldRequests"],
        "generatedParameters": all(
            b["finalParameters"][k] >= a["finalParameters"][k] - 1e-12
            for k in ("parameterAccuracy", "stateAccuracy", "typeAccuracy")
        ),
        "generatedEvidence": b["finalEvidenceAccuracy"] >= a["finalEvidenceAccuracy"] - 1e-12,
    }
    for rate in ("0", "0.1", "0.2"):
        a, b = parent["retail"][rate], current["retail"][rate]
        for metric in ("totalOnCompletePeriods", "falseHandoff", "unnecessaryRequests"):
            result[metric + "-" + rate] = b[metric]["mean"] <= a[metric]["mean"] + 1e-9
        for metric in ("parameterAccuracy", "stateAccuracy", "typeAccuracy"):
            result[metric + "-" + rate] = b[metric]["mean"] >= a[metric]["mean"] - 1e-12
        result["evidence-" + rate] = (
            b["evidenceCorrect"]["mean"] / b["evidenceCount"]["mean"]
            >= a["evidenceCorrect"]["mean"] / a["evidenceCount"]["mean"] - 1e-12
        )
    for metric in ("necessaryRequests", "recoveredAfterRequest", "handoffRate"):
        result[metric + "-no-dropout"] = (
            current["retail"]["0"][metric]["mean"] >= parent["retail"]["0"][metric]["mean"] - 1e-12
        )
    result["missingAtStop-no-dropout"] = (
        current["retail"]["0"]["missingAtStop"]["mean"]
        <= parent["retail"]["0"]["missingAtStop"]["mean"] + 1e-12
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True)
    args = parser.parse_args()
    plan = read(args.registration)
    require(plan["split"] in ("train", "dev"), "Train or original Dev only")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == plan["gpu"], "Assigned L40S only")
    require(torch.cuda.device_count() == 1 and "L40S" in torch.cuda.get_device_name(), "L40S only")
    require(provenance({})["sourceHash"] == plan["sourceHash"], "Package source changed")
    require(file_hash(__file__) == plan["scriptHash"], "Evaluation script changed")
    for path, expected in plan["files"].items():
        require(file_hash(path) == expected, "Registered input changed: " + path)
    if plan["split"] == "dev":
        require(read(plan["trainReport"])["eligibleForDev"], "Sequential Train gate failed")
    refit = read(Path(plan["refit"]) / "report.json")
    require(refit["condition"] == "joint", "This comparison requires the selected joint arm")
    root = Path(plan["output"])
    require(not root.exists(), "Preserve earlier evaluations")
    root.mkdir()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    episodes = {"generated": [], "retail": []}
    for cohort in plan["cohorts"]:
        observed = {r["id"]: r for r in lines(cohort["inputs"])}
        environment = lines(cohort["environment"])
        for row in environment:
            episode = copy.deepcopy(observed[row["id"]]) | row
            require(episode["split"] == plan["split"], "Wrong evaluation split")
            if "forecast" in episode["input"]["task"]:
                episode.update(
                    benchmark=structured_retail.VERSION,
                    cutoffIndex=len(episode["input"]["observations"]),
                )
                episodes["retail"].append(episode)
            else:
                episodes["generated"].append(episode)
        jsonl(root / (cohort["name"] + "-inputs.jsonl"), [observed[r["id"]] for r in environment])
        jsonl(root / (cohort["name"] + "-environment.jsonl"), environment)
    require({k: len(v) for k, v in episodes.items()} == plan["counts"], "Cohort changed")
    parent, tokenizer, config, _ = structured_train.load(plan["parent"], "cuda")
    require(weights_hash(parent.state_dict()) == plan["parentWeightsHash"], "Wrong parent")
    config = copy.deepcopy(config)
    config["encoder"]["jointOperands"] = True
    model = Router(parent.encoder, config["encoder"]).cuda()
    structured_train.import_core(model, parent.state_dict())
    model.requires_grad_(False).eval()
    del parent
    original_forward = model.forward
    outputs, checks = {}, []

    @torch.inference_mode()
    def cached_forward(view):
        if isinstance(view, list):
            return [cached_forward(v) for v in view]
        key = digest(observed_key(view))
        if key not in outputs:
            outputs[key] = original_forward(view)
        result = dict(outputs[key])
        if "atomState" in result:
            data = proposals(view, result["fieldState"], result["atomState"], result)
            result["operandJoint"] = model.operand_pairs(data).float()
            result["operandPairs"] = data["pairs"].float()
            result["operandPairMask"] = data["mask"].float()
        if len(checks) < 12:
            direct = original_forward(view)
            require(
                all(torch.equal(v, direct[k]) for k, v in result.items()),
                "Cached model output differs",
            )
            checks.append(key)
        return result

    model.forward = cached_forward
    report = {
        "plan": plan,
        "source": provenance({}),
        "testUsed": False,
        "newSourceHoldoutUsed": False,
        "devUsed": plan["split"] == "dev",
        "trainingPerformed": False,
        "primary": "joint-42",
        "results": {},
        "gates": {},
    }
    write(root / "registered.json", report)
    public_cases, labels, collection = [], {}, []
    if plan["split"] == "dev":
        dataset = Path(plan["publicDataset"])
        rows = [
            r
            for r in lines(dataset / "inputs.jsonl")
            if r["split"] == "dev" and r["component"] in ("contractnli", "cuad", "orsharc", "tatqa")
        ]
        require(len(rows) == 258, "Full public Dev denominator changed")
        public_cases = [("public", r) for r in rows]
        ids = {r["id"] for r in rows}
        labels = {r["id"]: r["target"] for r in lines(dataset / "labels.jsonl") if r["id"] in ids}
        collection = lines(dataset / "collection.jsonl")
        jsonl(root / "public-inputs.jsonl", rows)
    for name in ["parent"] + [f"joint-{s}" for s in plan["seeds"]]:
        if name != "parent":
            seed = name.split("-")[1]
            payload = torch.load(
                Path(plan["refit"]) / f"refit-{seed}.pt", map_location="cpu", weights_only=True
            )
            require(payload["epochs"] == refit["refits"][seed]["epochs"], "Refit epoch changed")
            require(
                weights_hash(payload["weights"]) == refit["refits"][seed]["weightsHash"],
                "Refit weights changed",
            )
            model.operand_pairs.load_state_dict(payload["weights"])
        frozen_hash = weights_hash(
            {k: v for k, v in model.state_dict().items() if not k.startswith("operand_pairs.")}
        )
        require(frozen_hash == plan["parentWeightsHash"], "Frozen parent weights changed")
        router = CachedRouter(model, tokenizer, config["encoder"])
        router.reset()  # Constructor and decision caches belong to this candidate only.
        checks.clear()
        generated = []
        with gzip.open(root / f"{name}-generated.jsonl.gz", "wt", compresslevel=3) as stream:
            for index, episode in enumerate(episodes["generated"]):
                record = rollout(episode, router)
                stream.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
                generated.append(record)
                if (index + 1) % 30 == 0:
                    write(
                        root / "progress.json",
                        {
                            "condition": name,
                            "generated": index + 1,
                            "seconds": time.perf_counter() - started,
                        },
                    )
        measured = []
        with gzip.open(root / f"{name}-retail.jsonl.gz", "wt", compresslevel=3) as stream:
            for index, episode in enumerate(episodes["retail"]):
                for missing in conditions(0):
                    record = rollout(episode, router, missing_responses=frozenset(missing))
                    record["missingResponses"] = list(missing)
                    stream.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
                    measured.append(measurement(record) | {"missingResponses": list(missing)})
                if (index + 1) % 14 == 0:
                    write(
                        root / "progress.json",
                        {
                            "condition": name,
                            "retail": index + 1,
                            "seconds": time.perf_counter() - started,
                        },
                    )
        jsonl(root / f"{name}-retail-measurements.jsonl", measured)
        expected = {str(rate): expectations(measured, rate) for rate in (0, 0.1, 0.2)}
        write(root / f"{name}-retail-expectations.json", expected)
        current = {
            "generated": generated_metrics(generated),
            "generatedTerminalLoss": sum(r["terminalLoss"] for r in generated) / len(generated),
            "retail": {k: structured_compare.means(v) for k, v in expected.items()},
            "headTensorHash": weights_hash(model.operand_pairs.state_dict()),
            "fullTensorHash": weights_hash(model.state_dict()),
            "frozenWeightsHash": frozen_hash,
            "directForwardChecks": len(checks),
            "rollouts": len(generated) + len(measured),
        }
        if public_cases:
            current["public"] = public_validation(
                model,
                tokenizer,
                config,
                public_cases,
                labels,
                collection,
                root / f"{name}-public.jsonl",
            )
        report["results"][name] = current
        if name != "parent":
            prior = report["results"]["parent"]
            gates = research_gates(prior, current)
            if public_cases:
                current["publicRetentionFailures"] = retained(prior["public"], current["public"])
                gates["publicRetention"] = not current["publicRetentionFailures"]
                gates["publicArithmeticImproves"] = (
                    current["public"]["tatqa"]["answerExact"]
                    > prior["public"]["tatqa"]["answerExact"]
                )
            report["gates"][name] = gates
        write(root / "partial.json", report)
        print(
            {
                "condition": name,
                "generated": current["generated"]["meanTotal"],
                "retail": {
                    k: v["totalOnCompletePeriods"]["mean"] for k, v in current["retail"].items()
                },
                "public": current.get("public"),
                "seconds": time.perf_counter() - started,
            },
            flush=True,
        )
    report.update(
        **{
            "eligibleForDev" if plan["split"] == "train" else "eligibleForAdoption": all(
                all(g.values()) for g in report["gates"].values()
            )
        },
        seconds=time.perf_counter() - started,
        peakGpuGiB=torch.cuda.max_memory_allocated() / 1024**3,
        cachedViews=len(outputs),
        rawHashes={p.name: file_hash(p) for p in root.iterdir() if p.is_file()},
    )
    write(root / "report.json", report)
    print(
        {
            "complete": True,
            "seconds": report["seconds"],
            "failedGates": {
                k: [m for m, v in g.items() if not v] for k, g in report["gates"].items()
            },
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
