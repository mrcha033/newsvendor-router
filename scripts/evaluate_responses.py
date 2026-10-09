"""Compare frozen retail Dev policies under all 16 shared manager-response conditions."""

import argparse
import copy
import gzip
import hashlib
import itertools
import json
import math
import os
import sys
import tarfile
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from newsvendor import corpus, structured_compare, structured_retail, structured_train
from newsvendor.cli import provenance
from newsvendor.io import digest, jsonl, lines, read, require, write
from newsvendor.structured_rollout import ResearchRouter, rollout
from newsvendor.structured_tool_eval import weights_hash
from newsvendor.suite import check

ROOT = Path(__file__).resolve().parents[1]


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def conditions(noise):
    require(math.isfinite(noise) and 0 <= noise <= 1, "Invalid missing-response probability")
    fields = sorted(corpus.SLOTS)
    return {
        tuple(s for s, missing in zip(fields, bits, strict=True) if missing): noise ** sum(bits)
        * (1 - noise) ** (len(fields) - sum(bits))
        for bits in itertools.product((False, True), repeat=len(fields))
    }


def measurement(record):
    complete = record["outcomeComplete"]
    return {
        "key": record["id"],
        "family": record["family"],
        "outcomeComplete": complete,
        "metrics": {
            **{
                k: record[k]
                for k in (
                    "requestCost",
                    "interactions",
                    "falseHandoff",
                    "necessaryRequests",
                    "unnecessaryRequests",
                    "recoveredAfterRequest",
                    "unresolvedAfterRequest",
                )
            },
            "totalOnCompletePeriods": record["total"] if complete else None,
            "terminalOnCompletePeriods": record["terminalLoss"] if complete else None,
            "handoffRate": float(record["result"] == "handoff"),
            "missingAtStop": len(record["unresolvedAtStop"]),
            **{
                k: record["parameters"][k]
                for k in (
                    "parameterAccuracy",
                    "stateAccuracy",
                    "typeAccuracy",
                    "evidenceCorrect",
                    "evidenceCount",
                )
            },
        },
    }


def expectations(records, noise):
    probabilities = conditions(noise)
    groups = defaultdict(dict)
    for record in records:
        missing = tuple(record["missingResponses"])
        require(missing in probabilities, "Unknown response condition")
        require(missing not in groups[record["key"]], "Duplicate response condition")
        groups[record["key"]][missing] = record
    require(groups, "No response measurements")
    result = []
    for key, group in groups.items():
        require(group.keys() == probabilities.keys(), "Incomplete response conditions")
        first = group[()]
        require(
            all(
                r["family"] == first["family"]
                and r["outcomeComplete"] == first["outcomeComplete"]
                and r["metrics"].keys() == first["metrics"].keys()
                for r in group.values()
            ),
            "Response conditions changed the evaluation case",
        )
        metrics = {}
        for name in first["metrics"]:
            values = [r["metrics"][name] for r in group.values()]
            if all(v is None for v in values):
                metrics[name] = None
            else:
                require(
                    all(v is not None and math.isfinite(v) for v in values),
                    "Inconsistent measured outcome availability",
                )
                metrics[name] = math.fsum(
                    probabilities[missing] * r["metrics"][name] for missing, r in group.items()
                )
        result.append({"key": key, "family": first["family"], "metrics": metrics})
    return result


class CachedRouter(ResearchRouter):
    """Reuse deterministic decisions for identical observed states, never response conditions."""

    def reset(self):
        self.cache_states()
        self.decisions = {}

    def choose(self, value, state):
        key = digest([value, state])
        if key not in self.decisions:
            self.decisions[key] = copy.deepcopy(self.decision(value, state))
        self.last_decision = copy.deepcopy(self.decisions[key])
        return self.last_decision["action"]


def main():
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True)
    args = parser.parse_args()
    plan = read(args.registration)
    require(plan["split"] == "dev" and plan["trainingAllowed"] is False, "Dev evaluation only")
    require(plan["noiseLevels"] == [0, 0.1, 0.2], "Use the registered sensitivity levels")
    require(set(plan["models"]) == {"value", "no_value"}, "Both frozen policies required")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == plan["gpu"], "Assigned GPU only")
    require(
        torch.cuda.device_count() == 1 and "L40S" in torch.cuda.get_device_name(0),
        "Single L40S required",
    )
    require(provenance(plan)["sourceHash"] == plan["sourceHash"], "Registered source changed")
    require(file_hash(__file__) == plan["scriptHash"], "Registered evaluation script changed")
    for entry in [*plan["models"].values(), *plan["references"].values(), plan["inputs"]]:
        require(file_hash(entry["path"]) == entry["sha256"], "Registered artifact changed")
    output = Path(plan["output"])
    require(not output.exists(), "Preserve prior evaluations")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    source_audit = check(plan["dataset"])
    require(
        digest(read(Path(plan["dataset"]) / "manifest.json")) == plan["dataManifestHash"],
        "Registered data changed",
    )
    rows = [
        r
        for r in lines(Path(plan["dataset"]) / "inputs.jsonl")
        if r["split"] == "dev" and r["component"] == "retail"
    ]
    episodes = structured_retail.cases(rows, plan["dataSeed"], plan["cutoffStride"])
    observed = [{k: e[k] for k in ("id", "family", "split", "input")} for e in episodes]
    require(observed == lines(plan["inputs"]["path"]), "Original Dev inputs changed")
    jsonl(output / "inputs.jsonl", observed)
    input_hash = file_hash(output / "inputs.jsonl")
    ids = {r["id"] for r in rows}
    labels = {
        r["id"]: r["target"]
        for r in lines(Path(plan["dataset"]) / "labels.jsonl")
        if r["id"] in ids
    }
    episodes = structured_retail.attach_targets(episodes, rows, labels)
    require(
        observed == [{k: e[k] for k in observed[0]} for e in episodes],
        "Outcomes changed observed inputs",
    )
    jsonl(
        output / "environment.jsonl",
        [
            {k: e[k] for k in ("id", "responses", "target", "financialAnnotations")}
            for e in episodes
        ],
    )
    report = {
        "scope": "Frozen retail Dev policies; exact expectation over independent per-field response availability. Controlled documents and manager conditions, not organizational effectiveness.",
        "testUsed": False,
        "trainingPerformed": False,
        "modelSelectionPerformed": False,
        "registration": plan,
        "registrationHash": file_hash(args.registration),
        "source": provenance(plan),
        "sourceAudit": source_audit,
        "inputHash": input_hash,
        "environmentHash": file_hash(output / "environment.jsonl"),
        "episodes": len(episodes),
        "families": sorted({e["family"] for e in episodes}),
        "completeEpisodes": sum(all(e["target"]["complete"]) for e in episodes),
        "completePeriods": len(
            {(e["sourceId"], e["cutoffIndex"]) for e in episodes if all(e["target"]["complete"])}
        ),
        "conditions": [
            {
                "missingResponses": list(k),
                "probabilities": {str(n): conditions(n)[k] for n in plan["noiseLevels"]},
            }
            for k in conditions(0)
        ],
        "results": {},
    }
    write(output / "registered.json", report)
    with tarfile.open(output / "source.tar.gz", "w:gz") as archive:
        for p in [*sorted(Path("src").rglob("*.py")), Path(__file__), Path(args.registration)]:
            archive.add(p, arcname=str(p.relative_to(ROOT) if p.is_absolute() else p))
    started = time.perf_counter()
    common = None
    expected = {}
    for name, entry in plan["models"].items():
        model, tokenizer, config, _ = structured_train.load(entry["path"], "cuda")
        require(config["noValue"] == (name == "no_value"), "Wrong checkpoint policy mode")
        fixed = weights_hash(
            {
                k: v
                for k, v in model.state_dict().items()
                if not k.startswith(("heads.value.", "heads.recovery."))
            }
        )
        require(common is None or common == fixed, "Policy comparison changed the constructor/GRU")
        common = fixed
        router = CachedRouter(model, tokenizer, config["encoder"], config["noValue"])
        for policy in (name, "checklist") if name == "value" else (name,):
            reference = {r["id"]: r for r in lines(plan["references"][policy]["path"])}
            require(set(reference) == {e["id"] for e in episodes}, "Reference Dev cohort changed")
            measurements = []
            path = output / f"{policy}-rollouts.jsonl.gz"
            with gzip.open(path, "wt", encoding="utf-8", compresslevel=3) as stream:
                for index, episode in enumerate(episodes):
                    router.reset()
                    for missing in conditions(0):
                        record = rollout(
                            episode,
                            router,
                            explore=policy == "checklist",
                            missing_responses=frozenset(missing),
                        )
                        if not missing:
                            require(
                                measurement(record) == measurement(reference[episode["id"]]),
                                "No-dropout measurements changed",
                            )
                            require(
                                record["q"] == reference[episode["id"]]["q"],
                                "No-dropout order changed",
                            )
                        record["missingResponses"] = list(missing)
                        stream.write(
                            json.dumps(
                                record, ensure_ascii=False, allow_nan=False, separators=(",", ":")
                            )
                            + "\n"
                        )
                        measurements.append(
                            measurement(record) | {"missingResponses": list(missing)}
                        )
                    if (index + 1) % 16 == 0:
                        progress = {
                            "policy": policy,
                            "episodes": index + 1,
                            "total": len(episodes),
                            "seconds": time.perf_counter() - started,
                        }
                        write(output / "progress.json", progress)
                        print(progress, flush=True)
            expected[policy] = {str(n): expectations(measurements, n) for n in plan["noiseLevels"]}
            jsonl(output / f"{policy}-measurements.jsonl", measurements)
            write(output / f"{policy}-expectations.json", expected[policy])
            report["results"][policy] = {
                "fixedWeightsHash": fixed,
                "noDropoutMatchesHistoricalMeasurementsAndOrders": True,
                "rawHash": file_hash(path),
                "rawRollouts": len(measurements),
                "metrics": {
                    n: structured_compare.means(rows) for n, rows in expected[policy].items()
                },
            }
            write(output / "partial.json", report)
        del router, model, tokenizer
        torch.cuda.empty_cache()
    report["comparisons"] = {
        f"value-minus-{other}": {
            n: structured_compare.paired(expected["value"][n], expected[other][n])
            for n in expected["value"]
        }
        for other in ("no_value", "checklist")
    }
    for entry in plan["models"].values():
        require(file_hash(entry["path"]) == entry["sha256"], "Evaluation mutated checkpoint")
    require(file_hash(output / "inputs.jsonl") == input_hash, "Observed inputs changed")
    report["seconds"] = time.perf_counter() - started
    report["peakGpuBytes"] = torch.cuda.max_memory_allocated()
    report["timingScope"] = (
        "All condition rollouts with cached deterministic states and decisions; not an operational latency benchmark"
    )
    write(output / "report.json", report)
    print({"completed": True, "seconds": report["seconds"]}, flush=True)


if __name__ == "__main__":
    main()
