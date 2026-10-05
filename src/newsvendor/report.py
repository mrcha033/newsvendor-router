import csv
import random
import statistics
from pathlib import Path

import numpy as np

from .io import jsonl, require, write


def mean(values):
    values = list(values)
    return statistics.mean(values) if values else None


def label(row):
    return (
        row["method"]
        + "/"
        + row["construction"]
        + ("/no-" + row["ablation"] if row["ablation"] else "")
        + ("/noise-" + str(row["noise"]) if row["noise"] else "")
    )


def summarize(rows):
    return {
        "n": len(rows),
        "total": mean(r["total"] for r in rows),
        "regretAmongOrders": mean(r["regret"] for r in rows if r["regret"] is not None),
        "requests": mean(r["requests"] for r in rows),
        "holdRate": mean(r["result"] == "hold" for r in rows),
        "coverage": mean(r["coverage"] for r in rows),
        "falseHandoff": mean(r["falseHandoff"] for r in rows),
    }


def paired(rows, a, b, repetitions=2000, seed=42):
    aa = {r["id"]: r for r in rows if label(r) == a}
    bb = {r["id"]: r for r in rows if label(r) == b}
    require(aa.keys() == bb.keys(), "Paired comparison must have identical episode IDs")
    groups = {}
    for id, r in aa.items():
        groups.setdefault(r["family"], []).append(r["total"] - bb[id]["total"])
    differences = [mean(group) for group in groups.values()]
    rand = random.Random(seed)
    samples = [mean(rand.choices(differences, k=len(differences))) for _ in range(repetitions)]
    return {
        "a": a,
        "b": b,
        "families": len(groups),
        "difference": mean(differences),
        "ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
        "unit": "original source bundle",
        "metric": "scripted total loss; negative favors a",
    }


def save(rows, directory, config, provenance):
    groups = {}
    for row in rows:
        groups.setdefault(label(row), []).append(row)
    summary = {k: summarize(r) for k, r in groups.items()}
    comparisons = []
    for construction in sorted({r["construction"] for r in rows if r["method"] == "learned"}):
        for method in ("reference", "one_step"):
            comparisons.append(
                paired(
                    rows,
                    "learned/" + construction,
                    method + "/" + construction,
                    config["bootstrap"],
                    config["seed"],
                )
            )
    jsonl(directory + "/trajectories.jsonl", rows)
    write(
        directory + "/metrics.json",
        {
            "scope": "controlled synthetic execution and learning check",
            "config": config,
            "provenance": provenance,
            "summary": summary,
            "comparisons": comparisons,
            "byScenario": {
                k: {
                    s: summarize([e for e in r if e["scenario"] == s])
                    for s in sorted({e["scenario"] for e in r})
                }
                for k, r in groups.items()
            },
        },
    )
    with Path(directory + "/metrics.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=["policy", *next(iter(summary.values()))])
        writer.writeheader()
        writer.writerows({"policy": k, **v} for k, v in summary.items())
    table = [
        "| Policy / constructor | n | Total loss | Requests | Hold | False handoff |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, s in summary.items():
        table.append(
            f"| {key} | {s['n']} | {s['total']:.3f} | {s['requests']:.3f} | "
            f"{s['holdRate']:.1%} | {s['falseHandoff']:.1%} |"
        )
    Path(directory + "/report.md").write_text(
        "# Controlled synthetic experiment\n\n"
        "Generated English business documents test execution, leakage controls and learning. "
        "They do not establish effectiveness on real organizational work. "
        "Loss includes declared scripted request and hold costs.\n\n"
        + "\n".join(table)
        + "\n\n## Paired differences\n\n"
        + "\n\n".join(
            f"{c['a']} vs {c['b']}: {c['difference']:.3f}, 95% source-bundle bootstrap CI "
            f"[{c['ci95'][0]:.3f}, {c['ci95'][1]:.3f}], {c['families']} bundles."
            for c in comparisons
        )
        + "\n\nAblations and response-error stress tests here are inference diagnostics. "
        "Retrained ablation studies require separate runs.\n"
    )
    return summary
