"""Recalculate the published four-condition comparison from raw rows on CPU."""

import argparse
import hashlib
import json
import math
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from newsvendor.factorial import factorial
from newsvendor.io import read, require, write


def compare(actual, expected):
    if isinstance(expected, dict):
        require(actual.keys() == expected.keys(), "Analysis keys differ")
        return max((compare(actual[k], v) for k, v in expected.items()), default=0)
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        require(
            math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9), "Analysis value differs"
        )
        return abs(actual - expected)
    require(actual == expected, "Analysis metadata differs")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", default="docs/evidence/research-ai-comparison-results.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    require(not Path(args.output).exists(), "Preserve earlier analyses")
    torch.set_num_threads(4)
    evidence = read(args.evidence)
    archive = evidence["archive"]
    require(
        hashlib.sha256(Path(archive["path"]).read_bytes()).hexdigest() == archive["sha256"],
        "Measurement archive changed",
    )
    results, maximum = {}, 0
    with tarfile.open(archive["path"]) as source:
        require(set(source.getnames()) == set(archive["files"]), "Archive members differ")
        for name, item in archive["files"].items():
            require(
                hashlib.sha256(source.extractfile(name).read()).hexdigest() == item["sha256"],
                "Archive member changed",
            )
        for split in ("train", "dev"):
            prefix = f"results/ai-comparison-v1/{split}"
            report = json.load(source.extractfile(f"{prefix}/report.json"))
            rows = [json.loads(line) for line in source.extractfile(f"{prefix}/outcomes.jsonl")]
            result = {}
            for stratum, conditions in report["analysis"].items():
                result[stratum] = {}
                for condition, metrics in conditions.items():
                    selected = [
                        r for r in rows if r["stratum"] == stratum and r["condition"] == condition
                    ]
                    result[stratum][condition] = {
                        metric: factorial(selected, metric) for metric in metrics
                    }
            maximum = max(maximum, compare(result, report["analysis"]))
            results[split] = result
    write(
        args.output,
        {
            "allMatch": True,
            "maxAbsoluteDifference": maximum,
            "analysis": results,
            "archiveHash": archive["sha256"],
            "scriptHash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scope": "CPU reanalysis of existing raw measurements; no new model or human evaluation",
        },
    )
    print(json.dumps({"allMatch": True, "splits": len(results), "maxAbsoluteDifference": maximum}))


if __name__ == "__main__":
    main()
