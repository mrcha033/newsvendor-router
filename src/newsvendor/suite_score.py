"""Independent native scoring; heterogeneous tasks never share a fabricated loss."""

import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from .io import digest, jsonl, lines, read, require, write
from .suite import check

ACTIONS = {"answer", "ask", "speak", "call_tool", "abstain"}


def number(value):
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return None
    text = str(value).strip().replace(",", "").replace("−", "-").rstrip("%")
    if text.startswith("(") and text.endswith(")"):
        text = "-" + text[1:-1]
    try:
        result = float(text)
        return result if math.isfinite(result) else None
    except ValueError:
        return None


def normalized(value):
    numeric = number(value)
    if numeric is not None:
        return format(numeric, ".12g")
    return " ".join(re.findall(r"[\w]+(?:[.-][\w]+)*", str(value).casefold()))


def exact(a, b):
    if a is None or isinstance(a, (bool, dict)):
        return False
    if isinstance(a, list) or isinstance(b, list):
        a, b = a if isinstance(a, list) else [a], b if isinstance(b, list) else [b]
        return Counter(map(normalized, a)) == Counter(map(normalized, b))
    x, y = number(a), number(b)
    if y is not None:
        return x is not None and math.isclose(x, y, abs_tol=1e-4, rel_tol=1e-6)
    return normalized(a) == normalized(b)


def text_f1(a, b):
    if not isinstance(a, (str, int, float, list)):
        return 0.0
    if number(b) is not None:
        return float(exact(a, b))

    def tokens(x):
        return Counter(normalized(" ".join(map(str, x)) if isinstance(x, list) else x).split())

    x, y = tokens(a), tokens(b)
    if not x or not y:
        return float(x == y)
    common = sum((x & y).values())
    return 2 * common / (sum(x.values()) + sum(y.values()))


def intervals(spans):
    grouped = defaultdict(list)
    for span in spans:
        grouped[span["document"]].append((span["start"], span["end"]))
    merged = {}
    for document, values in grouped.items():
        result = []
        for start, end in sorted(values):
            if result and start <= result[-1][1]:
                result[-1] = (result[-1][0], max(end, result[-1][1]))
            else:
                result.append((start, end))
        merged[document] = result
    return merged


def evidence_f1(predicted, target, documents):
    if not isinstance(predicted, list):
        return 0.0
    docs = {d["id"]: len(d["text"]) for d in documents}
    for span in predicted:
        if not isinstance(span, dict) or set(span) != {"document", "start", "end"}:
            return 0.0
        if (
            span["document"] not in docs
            or type(span["start"]) is not int
            or type(span["end"]) is not int
            or not 0 <= span["start"] < span["end"] <= docs[span["document"]]
        ):
            return 0.0
    a, b = intervals(predicted), intervals(target)

    def size(grouped):
        return sum(end - start for values in grouped.values() for start, end in values)

    count_a, count_b = size(a), size(b)
    if count_a + count_b == 0:
        return 1.0
    overlap = sum(
        max(0, min(x[1], y[1]) - max(x[0], y[0]))
        for doc in a.keys() & b.keys()
        for x in a[doc]
        for y in b[doc]
    )
    return 2 * overlap / (count_a + count_b)


def metrics(row, target, prediction):
    component = row["component"]
    action = prediction.get("action")
    correct_action = float(action == target["action"])
    result = {"actionAccuracy": correct_action}
    answer = prediction.get("answer")
    if component in {"cuad", "contractnli"}:
        evidence = evidence_f1(
            prediction.get("evidence", []), target["spans"], row["input"]["documents"]
        )
        accuracy = correct_action
        if component == "contractnli":
            accuracy *= float(answer == target["answer"])
            result["stateAccuracy"] = accuracy
        else:
            result["answerabilityAccuracy"] = accuracy
        result.update(evidenceF1=evidence, groundedScore=accuracy * evidence)
    elif component == "orsharc":
        accuracy = correct_action
        if target["action"] == "answer":
            accuracy *= float(
                isinstance(answer, str) and answer.casefold() == target["answer"].casefold()
            )
        result["decisionAccuracy"] = accuracy
        if target["action"] == "ask":
            result["followupReferenceF1"] = correct_action * text_f1(answer, target["answer"])
        retrieved = prediction.get("retrieved", [])
        result["ruleRecallAt5"] = float(
            isinstance(retrieved, list) and target["retrieved"][0] in retrieved[:5]
        )
    elif component == "abcd":
        if target["action"] == "call_tool":
            args = prediction.get("arguments", [])
            result["toolExact"] = correct_action * float(prediction.get("tool") == target["tool"])
            if target["argumentsObservable"]:
                result["observableToolAndArgumentsExact"] = result["toolExact"] * float(
                    isinstance(args, list)
                    and [normalized(a) for a in args]
                    == [normalized(a) for a in target["arguments"]]
                )
        else:
            result["spokenReferenceF1"] = correct_action * text_f1(answer, target["answer"])
    elif component == "tatqa":
        scale = float(prediction.get("scale", "") == target["scale"])
        result.update(
            scaleAccuracy=scale,
            answerExact=correct_action * scale * float(exact(answer, target["answer"])),
            answerF1=correct_action * scale * text_f1(answer, target["answer"]),
        )
    elif component == "retail":
        predicted = [number(v) for v in answer] if isinstance(answer, list) else []
        valid = bool(
            correct_action
            and len(predicted) == len(target["answer"])
            and all(v is not None and v >= 0 for v in predicted)
        )
        result["forecastValid"] = float(valid)
        for name, complete in (("uncensoredSalesMAE", True), ("censoredObservedSalesMAE", False)):
            errors = (
                [
                    abs(a - b)
                    for a, b, observed in zip(
                        predicted, target["answer"], target["complete"], strict=True
                    )
                    if observed == complete
                ]
                if valid
                else []
            )
            result[name] = sum(errors) / len(errors) if errors else None
    return result


def score(directory, predictions, split="test", output="results/complementary"):
    require(split in {"dev", "test"}, "Score only held-out splits")
    check(directory)
    directory = Path(directory)
    rows = {r["id"]: r for r in lines(directory / "inputs.jsonl") if r["split"] == split}
    targets = {r["id"]: r["target"] for r in lines(directory / "labels.jsonl") if r["id"] in rows}
    provided = {}
    for item in predictions:
        require(set(item) == {"id", "prediction"}, "Prediction requires id and prediction only")
        require(item["id"] in rows, "Prediction outside requested split")
        require(item["id"] not in provided, "Duplicate prediction id")
        require(isinstance(item["prediction"], dict), "Prediction must be an object")
        provided[item["id"]] = item["prediction"]
    measurements = []
    for id, row in rows.items():
        prediction = provided.get(id, {})
        values = metrics(row, targets[id], prediction)
        valid = prediction.get("action") in ACTIONS
        if not valid:
            values = {key: None if value is None else 0.0 for key, value in values.items()}
        measurements.append(
            {
                "id": id,
                "component": row["component"],
                "family": row["family"],
                "provided": id in provided,
                "validAction": valid,
                "metrics": values,
            }
        )
    summary = {}
    for component in sorted({r["component"] for r in measurements}):
        selected = [r for r in measurements if r["component"] == component]
        names = sorted({k for r in selected for k in r["metrics"]})
        values = {}
        for name in names:
            observed = [r for r in selected if r["metrics"].get(name) is not None]
            families = defaultdict(list)
            for r in observed:
                families[r["family"]].append(r["metrics"][name])
            means = [sum(v) / len(v) for v in families.values()]
            values[name] = {
                "mean": sum(r["metrics"][name] for r in observed) / len(observed)
                if observed
                else None,
                "familyMean": sum(means) / len(means) if means else None,
                "cases": len(observed),
                "families": len(families),
            }
        comparable = all(r["provided"] and r["validAction"] for r in selected)
        if component == "retail":
            comparable &= all(r["metrics"]["forecastValid"] == 1 for r in selected)
        summary[component] = {
            "cases": len(selected),
            "provided": sum(r["provided"] for r in selected),
            "coverage": sum(r["provided"] for r in selected) / len(selected),
            "comparable": comparable,
            "metrics": values,
        }
    manifest = read(directory / "manifest.json")
    report = {
        "split": split,
        "inputHash": manifest["inputHash"],
        "labelHash": manifest["labelHash"],
        "predictionHash": digest(predictions),
        "scorerHash": digest(Path(__file__).read_bytes()),
        "components": summary,
        "scope": manifest["scope"],
        "pooledScore": None,
        "notes": [
            "Missing predictions remain in classification denominators.",
            "Forecast MAE covers valid forecasts only; incomplete coverage is not comparable.",
            "Reference wording scores are lexical diagnostics, not action-value labels.",
            "Censored observed-sales error is not latent-demand error.",
        ],
    }
    jsonl(Path(output) / "measurements.jsonl", measurements)
    jsonl(Path(output) / "predictions.jsonl", predictions)
    write(Path(output) / "metrics.json", report)
    return report
