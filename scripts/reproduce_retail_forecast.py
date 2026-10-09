# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["torch==2.14.1", "transformers==5.18.0", "numpy==2.5.3", "pydantic==2.12.5"]
# [[tool.uv.index]]
# name = "pytorch-cuda"
# url = "https://download.pytorch.org/whl/cu130"
# explicit = true
# [tool.uv.sources]
# torch = { index = "pytorch-cuda" }
# ///
"""Reproduce the selected holdout GRU from published archives only, without ModernBERT."""

import argparse
import io
import json
import os
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from study_demand import evaluate, summarize

from newsvendor import sequence
from newsvendor.cli import provenance
from newsvendor.io import digest, jsonl, read, require, write
from newsvendor.structured_tool_eval import weights_hash
from newsvendor.structured_train import DEMAND_SCHEMA

ROOT = Path(__file__).resolve().parents[1]


def contents(record, names):
    path = Path(record["path"])
    require(digest(path.read_bytes()) == record["sha256"], "Published archive changed")
    result = {}
    with tarfile.open(path, "r:xz") as archive:
        for name in names:
            result[name] = archive.extractfile(name).read()
            require(
                digest(result[name]) == record["files"][name]["sha256"], "Archive member changed"
            )
    return result


def main():
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/reproduce-retail-forecast")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    output = Path(args.output)
    require(not output.exists(), "Preserve previous reproduction")
    if args.device == "cuda":
        require(
            bool(os.environ.get("CUDA_VISIBLE_DEVICES"))
            and torch.cuda.device_count() == 1
            and "L40S" in torch.cuda.get_device_name(0),
            "Expose the assigned single L40S through CUDA_VISIBLE_DEVICES",
        )
    published = read("docs/evidence/research-retail-holdout-results.json")
    selected = read("docs/evidence/research-selection-results.json")
    require(
        provenance({})["sourceHash"] == published["provenance"]["sourceHash"],
        "Use the recorded evaluation source",
    )

    # Enforce independence from ignored working datasets and all local model files.
    blocked_roots = [ROOT / "data", ROOT / "models"]

    def audit(event, values):
        if event != "open" or not isinstance(values[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(values[0])).resolve()
        require(
            not any(path.is_relative_to(root) for root in blocked_roots),
            "Working data/model access is forbidden",
        )
        require(
            not (path.is_relative_to(ROOT / "results") and path.suffix == ".pt"),
            "Local checkpoint access is forbidden",
        )

    sys.addaudithook(audit)
    model_name = "results/l40s-demand-selection-v2/42/model.pt"
    model_archive = next(r for r in selected["publishedRaw"] if model_name in r["files"])
    model_bytes = contents(model_archive, [model_name])[model_name]
    prefix = "results/l40s-retail-holdout-v1/"
    names = [
        prefix + suffix
        for suffix in ("retail-inputs.jsonl", "retail-labels.jsonl", "selected-forecasts.jsonl")
    ]
    data = contents(published["publishedRaw"], names)
    rows, annotations, reference = [
        [json.loads(line) for line in data[name].splitlines()] for name in names
    ]
    require(len(rows) == 27 and all(r["split"] == "test" for r in rows), "Wrong published cohort")
    labels = {r["id"]: r["target"] for r in annotations}
    payload = torch.load(io.BytesIO(model_bytes), map_location="cpu", weights_only=True)
    require(payload["schema"] == DEMAND_SCHEMA, "Demand checkpoint required")
    require(
        all(k.startswith("demand.") for k in payload["weights"]), "Unexpected non-demand weights"
    )
    model = sequence.DemandEncoder()
    model.load_state_dict({k.removeprefix("demand."): v for k, v in payload["weights"].items()})
    identity = weights_hash(model.state_dict())
    require(
        identity == published["models"]["selected"]["demandHash"],
        "Different published demand model",
    )
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    model.to(args.device).eval()
    plan = published["registration"]
    started = time.perf_counter()
    cases = sequence.samples(rows, labels, plan["minHistory"])
    records = evaluate(model, cases, plan["observationNodes"], plan["allocationConcentration"])
    # Keep raw predictions even when a platform comparison fails.
    jsonl(output / "predictions.jsonl", records)
    require(len(records) == len(reference), "Changed forecast origin count")
    f_errors, q_errors, mismatches, families = [], [], [], []
    for actual, expected in zip(records, reference, strict=True):
        require(
            all(
                actual[k] == expected[k]
                for k in ("id", "cutoff", "horizon", "historyHash", "y", "scale", "censored")
            ),
            "Changed observed input/target",
        )
        left, right = actual["prediction"], expected["prediction"]
        same_family = left["distribution"]["family"] == right["distribution"]["family"]
        families.append(same_family)
        for field, a, b, errors in (
            ("F", left["F"], right["F"], f_errors),
            (
                "orders",
                [r["q"] for r in left["orders"]],
                [r["q"] for r in right["orders"]],
                q_errors,
            ),
        ):
            a, b = np.asarray(a), np.asarray(b)
            error = float(np.abs(a - b).max()) if a.shape == b.shape else None
            if error is not None:
                errors.append(error)
            if not same_family or a.shape != b.shape or not np.allclose(a, b, rtol=1e-5, atol=1e-4):
                mismatches.append(
                    {
                        **{k: actual[k] for k in ("id", "cutoff", "horizon")},
                        "field": field,
                        "maxAbsoluteError": error,
                        "actualShape": list(a.shape),
                        "expectedShape": list(b.shape),
                        "sameFamily": same_family,
                    }
                )
    report = {
        "scope": "Published small GRU and new holdout inputs only; this reproduces forecasts, not the full ModernBERT manager-request evaluation",
        "device": args.device,
        "sourceHash": published["provenance"]["sourceHash"],
        "demandWeightsHash": identity,
        "weightsArchiveHash": model_archive["sha256"],
        "dataArchiveHash": published["publishedRaw"]["sha256"],
        "scriptHash": digest(Path(__file__).read_bytes()),
        "localDataAndCheckpointReadsForbidden": True,
        "modernBertLoaded": False,
        "origins": len(records),
        "sources": len(rows),
        "allFamiliesIdentical": all(families),
        "numericalTolerancePassed": not mismatches,
        "mismatches": mismatches,
        "maxAbsoluteFError": max(f_errors),
        "maxAbsoluteOrderError": max(q_errors),
        "numericalTolerance": {"rtol": 1e-5, "atol": 1e-4},
        "metrics": summarize(records),
        "seconds": time.perf_counter() - started,
        "rawHash": digest((output / "predictions.jsonl").read_bytes()),
        "environment": provenance({}),
    }
    write(output / "report.json", report)
    print(
        {k: v for k, v in report.items() if k not in ("metrics", "environment", "mismatches")},
        flush=True,
    )
    print({"mismatchCount": len(mismatches), "firstMismatches": mismatches[:3]}, flush=True)
    require(
        not mismatches,
        "CPU/GPU reproduction exceeded unchanged numerical tolerance; raw results preserved",
    )


if __name__ == "__main__":
    main()
