"""Analyze published paired Dev outcomes without model inference or hidden-label inputs."""

import argparse
import hashlib
import json
import platform
import sys
import tarfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from newsvendor.cli import provenance
from newsvendor.decision_effects import pair_outcomes, paired_regression, theory_checks
from newsvendor.io import jsonl, read, require, write


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def published_rows(plan):
    for path, expected in plan["files"].items():
        require(sha(path) == expected, "Published analysis input changed: " + path)
    evidence = read("docs/evidence/research-parameter-decoding-results.json")
    artifacts = evidence["artifacts"]
    prefix = "results/parameter-decoding-v2/dev/"
    rows, identities, members = [], {}, {}
    with tarfile.open(artifacts["records"]["path"], "r:xz") as archive:
        for condition in plan["conditions"]:
            name = prefix + condition + "-inputs.jsonl"
            data = archive.extractfile(name).read()
            require(
                hashlib.sha256(data).hexdigest() == artifacts["records"]["files"][name]["sha256"],
                "Observed input bytes changed",
            )
            observed = [json.loads(line) for line in data.splitlines()]
            require(all(r["split"] == "dev" for r in observed), "Only existing Dev is authorized")
            require(len({r["id"] for r in observed}) == len(observed), "Duplicate observed input")
            identities[condition] = {r["id"]: (r["family"], r["kind"]) for r in observed}
            members[name] = artifacts["records"]["files"][name]
            for method in plan["methods"]:
                name = prefix + f"{condition}-{method}-retail-measurements.jsonl"
                data = archive.extractfile(name).read()
                require(
                    hashlib.sha256(data).hexdigest()
                    == artifacts["records"]["files"][name]["sha256"],
                    "Retail measurements changed",
                )
                members[name] = artifacts["records"]["files"][name]
                for line in data.splitlines():
                    r = json.loads(line)
                    if r["missingResponses"]:
                        continue
                    require(
                        identities[condition][r["key"]] == (r["family"], "retail"),
                        "Retail source mismatch",
                    )
                    metrics = r["metrics"]
                    rows.append(
                        {
                            "id": r["key"],
                            "family": r["family"],
                            "kind": "retail",
                            "period": r["key"].rsplit(":", 1)[0],
                            "complete": r["outcomeComplete"],
                            "condition": condition,
                            "method": method,
                            "metrics": {
                                "total": metrics["totalOnCompletePeriods"],
                                "terminal": metrics["terminalOnCompletePeriods"],
                                **{
                                    k: metrics[k]
                                    for k in plan["metrics"]["retail"]
                                    if k not in ("total", "terminal")
                                },
                            },
                        }
                    )
    wanted = {
        prefix + f"{c}-{m}.jsonl": (c, m) for c in plan["conditions"] for m in plan["methods"]
    }
    seen = set()
    with tarfile.open(artifacts["devRaw"]["path"], "r|xz") as archive:
        for member in archive:
            if member.name not in wanted:
                continue
            require(member.name not in seen, "Duplicate raw archive member")
            seen.add(member.name)
            condition, method = wanted[member.name]
            checksum, size = hashlib.sha256(), 0
            for line in archive.extractfile(member):
                checksum.update(line)
                size += len(line)
                # Published compact JSON has this marker. Confirm the top-level
                # kind after parsing; skip large retail traces already summarized.
                if b'"kind":"generated"' not in line:
                    continue
                r = json.loads(line)
                if r["kind"] != "generated":
                    continue
                require(not r["missingResponses"], "Unexpected generated response mask")
                require(r["documentCondition"] == condition, "Document condition changed")
                require(
                    identities[condition][r["id"]] == (r["family"], "generated"),
                    "Generated source mismatch",
                )
                require(
                    abs(r["total"] - r["terminalLoss"] - r["requestCost"]) < 1e-8,
                    "Cost decomposition changed",
                )
                rows.append(
                    {
                        "id": r["id"],
                        "family": r["family"],
                        "kind": "generated",
                        "period": None,
                        "complete": True,
                        "condition": condition,
                        "method": method,
                        "metrics": {
                            "total": r["total"],
                            "terminal": r["terminalLoss"],
                            **{
                                k: r[k]
                                for k in plan["metrics"]["generated"]
                                if k not in ("total", "terminal")
                            },
                        },
                    }
                )
            actual = {"bytes": size, "sha256": checksum.hexdigest()}
            require(
                actual == artifacts["devRaw"]["files"][member.name], "Raw trajectory bytes changed"
            )
            members[member.name] = actual
    require(seen == set(wanted), "Missing raw trajectory member")
    for condition in plan["conditions"]:
        for method in plan["methods"]:
            for kind, count in plan["counts"].items():
                selected = [
                    r
                    for r in rows
                    if (r["condition"], r["method"], r["kind"]) == (condition, method, kind)
                ]
                require(len(selected) == count, "Incomplete cohort extraction")
    return sorted(rows, key=lambda r: (r["kind"], r["condition"], r["id"], r["method"])), members


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/research-economics-v1.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root, plan = Path(args.output), read(args.config)
    require(not root.exists(), "Preserve prior analysis output")
    require(
        plan["status"] == "exploratory-existing-dev" and plan["split"] == "dev",
        "Unsupported analysis scope",
    )
    require(
        plan["methods"] == ["parent", "scoped"], "Do not relabel model modification as AI adoption"
    )
    torch.set_num_threads(2)
    root.mkdir(parents=True)
    write(root / "registered.json", plan)
    rows, members = published_rows(plan)
    jsonl(root / "outcomes.jsonl", rows)
    analyses = {}
    for kind, metrics in plan["metrics"].items():
        analyses[kind] = {}
        for metric in metrics:
            pairs = pair_outcomes(rows, kind, metric)
            analyses[kind][metric] = paired_regression(pairs, plan["conditions"])
    # Compare question cost on the SAME complete cohort as total and terminal,
    # while keeping the primary all-retail question analysis separate.
    complete = [r for r in rows if r["kind"] == "retail" and r["complete"]]
    analyses["retail"]["requestCostComplete"] = paired_regression(
        pair_outcomes(complete, "retail", "requestCost"), plan["conditions"]
    )
    for kind in ("retail", "generated"):
        cost = "requestCostComplete" if kind == "retail" else "requestCost"
        for condition in plan["conditions"]:
            cells = [analyses[kind][m]["cells"][condition] for m in ("total", "terminal", cost)]
            require(
                abs(cells[0]["effect"] - cells[1]["effect"] - cells[2]["effect"]) < 1e-8,
                "Paired cost decomposition failed",
            )
    result = {
        "status": "exploratory-existing-dev",
        "newInference": False,
        "trainingPerformed": False,
        "testUsed": False,
        "humanParticipants": 0,
        "source": provenance(plan),
        "scriptHash": sha(__file__),
        "configHash": sha(args.config),
        "sourceMembers": members,
        "outcomeHash": sha(root / "outcomes.jsonl"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "analyses": analyses,
        "theoryChecks": theory_checks(),
        "limitations": [
            "All arms contain AI; effects identify the existing decoder modification only.",
            "Previously inspected Dev; no new confirmatory sample or human-behavior evidence.",
            "Saturated paired OLS equals paired condition means, not an additional discovery.",
            "Repeated conditions are not independent demand observations. Few source families.",
            "Source-deletion ranges are sensitivities, not confidence intervals or causal guarantees.",
            "Censored retail demand is excluded from exact loss only, not from question metrics.",
            "Generated and retail loss definitions remain separate; hold/invalid penalties are retained.",
        ],
    }
    write(root / "report.json", result)
    print(
        json.dumps(
            {
                "output": str(root),
                "outcomes": len(rows),
                "analyses": sum(map(len, analyses.values())),
                "newInference": False,
            }
        )
    )


if __name__ == "__main__":
    main()
