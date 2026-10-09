"""Aligned source-family comparisons; heterogeneous tasks have separate metrics."""

from collections import defaultdict
from math import isclose
from pathlib import Path

import numpy as np

from .construction import reference
from .io import digest, lines, require


def means(rows):
    grouped, families = defaultdict(list), defaultdict(lambda: defaultdict(list))
    for row in rows:
        for name, value in row["metrics"].items():
            if value is not None:
                require(np.isfinite(value), "Nonfinite comparison measurement")
                grouped[name].append(float(value))
                families[name][row["family"]].append(float(value))
    return {
        name: {
            "mean": float(np.mean(values)),
            "cases": len(values),
            "familyMean": float(np.mean([np.mean(v) for v in families[name].values()])),
            "families": len(families[name]),
        }
        for name, values in grouped.items()
    }


def paired(first, second, seed=42):
    a, b = {r["key"]: r for r in first}, {r["key"]: r for r in second}
    require(len(a) == len(first) and len(b) == len(second), "Duplicate comparison key")
    require(a.keys() == b.keys(), "Comparison evaluation cases differ")
    differences = defaultdict(lambda: defaultdict(list))
    for key in a:
        require(a[key]["family"] == b[key]["family"], "Comparison family differs")
        for metric in a[key]["metrics"].keys() & b[key]["metrics"].keys():
            x, y = a[key]["metrics"][metric], b[key]["metrics"][metric]
            if x is not None and y is not None:
                differences[metric][a[key]["family"]].append(float(x) - float(y))
    result = {}
    for metric, groups in sorted(differences.items()):
        values = np.array([np.mean(v) for _, v in sorted(groups.items())])
        rand = np.random.default_rng(seed)
        bootstrap = values[rand.integers(len(values), size=(2000, len(values)))].mean(-1)
        result[metric] = {
            "familyMeanDifference": float(values.mean()),
            "ci95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
            "families": len(groups),
            "pairedCases": sum(map(len, groups.values())),
        }
    return result


def research_measurements(rows):
    def wrong(event):
        expected, actual = reference(event["input"]), event["state"]
        for key in ("state", "types", "values"):
            for slot in ("c", "p", "v", "b", "F"):
                a, b = expected[key].get(slot), actual[key].get(slot)
                equal = isclose(a, b, rel_tol=1e-6, abs_tol=1e-6) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else a == b
                if not equal:
                    return True
        return False

    result = []
    for row in rows:
        event = row["events"][0]
        observed, state = reference(event["input"]), event["state"]
        initial_error, final_error = wrong(event), wrong(row["events"][-1])
        result.append(
            {
                "key": row["id"],
                "family": row["family"],
                "metrics": {
                    k: row[k]
                    for k in (
                        "total",
                        "terminalLoss",
                        "requestCost",
                        "interactions",
                        "falseHandoff",
                    )
                }
                | {
                    "initialTypeAccuracy": float(
                        np.mean([state["types"].get(k) == v for k, v in observed["types"].items()])
                    ),
                    "initialStateAccuracy": float(
                        np.mean([state["state"].get(k) == v for k, v in observed["state"].items()])
                    ),
                    "handoffRate": float(row["result"] == "handoff"),
                    "initialConstructionError": float(initial_error),
                    "finalConstructionError": float(final_error),
                    "constructionRecoveryRate": float(not final_error) if initial_error else None,
                    "errorAccumulationRate": float(final_error) if not initial_error else None,
                },
            }
        )
    return result


def compare(reports):
    require(
        {r["config"]["variant"] for r in reports} == {"base", "large", "no_value"},
        "Three complete variants are required",
    )
    first = reports[0]
    split = first["evaluation"]["componentMetrics"]["split"]
    models, raw = {}, {}
    for report in reports:
        config = report["config"]
        name, directory = config["variant"], Path(config["output"])
        for key in ("trainIdsHash", "developmentIdsHash", "limitedCasesPerComponent"):
            require(report[key] == first[key], "Comparison training scope differs: " + key)
        for key in ("inputHash", "labelHash"):
            require(
                report["evaluation"]["componentMetrics"][key]
                == first["evaluation"]["componentMetrics"][key],
                "Held-out data differ",
            )
        for key in ("training", "demand", "policy", "seed"):
            require(config[key] == first["config"][key], "Comparison configuration differs: " + key)
        require(report["hardware"]["name"] == "NVIDIA L40S", "Comparison requires L40S")
        require(
            report["provenance"]["sourceHash"] == first["provenance"]["sourceHash"],
            "Comparison source versions differ",
        )
        require(
            report["evaluation"]["componentMetrics"]["split"] == split,
            "Comparison evaluation split differs",
        )
        components = lines(directory / f"scores-{split}" / "measurements.jsonl")
        periods = lines(directory / f"demand-{split}.jsonl")
        research = lines(directory / f"research-{split}.jsonl")
        raw[name] = {
            component: [r | {"key": r["id"]} for r in components if r["component"] == component]
            for component in {r["component"] for r in components}
        }
        raw[name]["demandRolling"] = [
            r | {"key": digest([r["id"], r["cutoff"], r["dates"]])} for r in periods
        ]
        raw[name]["researchControlled"] = research_measurements(research)
        models[name] = {
            "parameters": report["trainableParameters"],
            "continuation": report.get("continuation"),
            "sharedCommonTraining": report.get("sharedCommonTraining"),
            "trainingSeconds": report["trainingSeconds"],
            "trainingPeakGpuBytes": report["trainingPeakGpuBytes"],
            "inferenceMs": report["evaluation"]["inferenceMs"],
            "inferencePeakGpuBytes": report["evaluation"]["peakGpuBytes"],
            "measurements": {component: means(rows) for component, rows in raw[name].items()},
            "artifactDirectory": str(directory),
        }
    comparisons = {}
    for a, b in (("large", "base"), ("base", "no_value")):
        comparisons[a + "Minus" + b.title().replace("_", "")] = {
            component: paired(raw[a][component], raw[b][component]) for component in raw[a]
        }
    return {
        "models": models,
        "comparisons": comparisons,
        "seed": first["config"]["seed"],
        "split": split,
        "limitedCasesPerComponent": first["limitedCasesPerComponent"],
        "sourceHash": first["provenance"]["sourceHash"],
        "inputHash": first["evaluation"]["componentMetrics"]["inputHash"],
        "confidenceMethod": "2000 paired source-family bootstrap resamples; 95% percentile interval",
        "scope": "Public component data and generated controlled rollouts. No real organizational effectiveness claim.",
        "notes": [
            "Differences are first model minus second; loss metrics favor negative values.",
            "Censored order losses are lower bounds; uncensored loss and interval coverage use complete periods.",
            "The first seed is not a multi-seed result; source-family intervals measure held-out family variation.",
        ],
    }


def markdown(report):
    lines = [
        f"L40S 비교 · seed {report['seed']} · {report['split']}",
        "",
        f"Train 제한: {report['limitedCasesPerComponent'] if report['limitedCasesPerComponent'] else '없음'}",
        "",
        "| 모델 | 파라미터 | 추론 중앙값 ms | p95 ms | 추론 peak GiB | 통제 발주 손실 | 행동 수 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("base", "large", "no_value"):
        model = report["models"][name]
        research = model["measurements"]["researchControlled"]
        lines.append(
            f"| {name} | {model['parameters']:,} | {model['inferenceMs']['median']:.1f} | "
            f"{model['inferenceMs']['p95']:.1f} | {model['inferencePeakGpuBytes'] / 2**30:.2f} | "
            f"{research['total']['mean']:.3f} | {research['interactions']['mean']:.2f} |"
        )
    lines.extend(
        [
            "",
            "과제별 지표와 source family 단위 95% 신뢰구간은 같은 이름의 JSON에 저장합니다.",
            "통제 rollout은 생성된 환경의 결과이며 실제 조직 업무의 효과성 근거가 아닙니다.",
            "한 seed의 비교입니다. 품절 구간 발주 손실은 하한이며 calibration·정확한 손실은 비품절 구간에서 측정합니다.",
            "모델 선택에 Test를 사용하지 않았습니다.",
        ]
    )
    return "\n".join(lines) + "\n"
