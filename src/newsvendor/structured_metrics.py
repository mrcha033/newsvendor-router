"""Research outcomes and auxiliary tool metrics over every decision."""

from .suite_score import metrics


def generated_metrics(rows):
    from .construction import demand, parameter_record
    from .structured_retail import parameter_metrics

    def mean(values):
        return sum(values) / len(values) if values else None

    initial = [parameter_metrics(r["events"][0]["input"], r["events"][0]["state"]) for r in rows]
    final = [parameter_metrics(r["events"][-1]["input"], r["events"][-1]["state"]) for r in rows]
    requests = {"missingFieldRequests": 0, "resolvedFieldRequests": 0, "resolvedAfterRequest": 0}
    for row in rows:
        for index, event in enumerate(row["events"][:-1]):
            action = event["action"]
            if action not in ("c", "p", "v", "b", "demand"):
                continue
            observed = parameter_record(event["input"])
            demand(event["input"], observed)
            slot = "F" if action == "demand" else action
            missing = observed["state"][slot] != "verified"
            requests["missingFieldRequests" if missing else "resolvedFieldRequests"] += 1
            after = row["events"][index + 1]
            accepted = after["state"]["state"].get(slot) == "verified"
            if slot != "F":
                truth = parameter_record(after["input"])["values"].get(slot)
                actual = after["state"]["values"].get(slot)
                accepted = (
                    accepted
                    and truth is not None
                    and actual is not None
                    and abs(truth - actual) <= 1e-7 * max(1, abs(truth))
                )
            requests["resolvedAfterRequest"] += int(missing and accepted)
    return {
        "episodes": len(rows),
        "sourceFamilies": len({r["family"] for r in rows}),
        "meanTotal": mean([r["total"] for r in rows]),
        "meanRequestCost": mean([r["requestCost"] for r in rows]),
        "meanInteractions": mean([r["interactions"] for r in rows]),
        "falseHandoffs": sum(r["falseHandoff"] for r in rows),
        "holds": sum(r["result"] == "hold" for r in rows),
        "initialParameters": {
            k: mean([m[k] for m in initial])
            for k in ("parameterAccuracy", "stateAccuracy", "typeAccuracy")
        },
        "finalParameters": {
            k: mean([m[k] for m in final])
            for k in ("parameterAccuracy", "allParametersCorrect", "stateAccuracy", "typeAccuracy")
        },
        "finalEvidenceAccuracy": sum(m["evidenceCorrect"] for m in final)
        / max(1, sum(m["evidenceCount"] for m in final)),
        **requests,
        "requestMetricScope": "Missing observed fields; this does not mean each request was economically necessary under the legacy minimax task.",
    }


def tool_metrics(row, label, prediction):
    result = metrics(row, label, prediction)
    expected = label["action"] == "call_tool"
    called = prediction.get("action") == "call_tool"
    correct = float(expected and called and label["tool"] == prediction.get("tool"))
    result.update(expectedCall=float(expected), predictedCall=float(called), correctCall=correct)
    result["toolDecisionAccuracy"] = (
        correct if expected else float(prediction.get("action") == label["action"])
    )
    result["argumentsDecisionAccuracy"] = (
        result.get("observableToolAndArgumentsExact")
        if expected
        else result["toolDecisionAccuracy"]
    )
    return result


def performance_goal(public, economic_loss, goal):
    observed = {
        "toolExact": public["toolExact"],
        "observableToolAndArgumentsExact": public["observableToolAndArgumentsExact"],
        "callPrecision": public["correctCall"] / max(public["predictedCall"], 1e-12),
        "generatedTotalLoss": economic_loss,
    }
    failures = [
        name
        for name in ("toolExact", "observableToolAndArgumentsExact", "callPrecision")
        if observed[name] + 1e-12 < goal[name]
    ]
    if economic_loss >= goal["generatedTestTotalLossBelow"]:
        failures.append("generatedTotalLoss")
    return {"passed": not failures, "failures": failures, "observed": observed, "required": goal}
