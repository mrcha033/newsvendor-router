"""Tool metrics over every decision, including unnecessary calls."""

from .suite_score import metrics


def tool_metrics(row, label, prediction):
    result = metrics(row, label, prediction)
    expected = label["action"] == "call_tool"
    called = prediction.get("action") == "call_tool"
    correct = float(expected and called and label["tool"] == prediction.get("tool"))
    result.update(expectedCall=float(expected), predictedCall=float(called), correctCall=correct)
    result["toolDecisionAccuracy"] = correct if expected else float(prediction.get("action") == label["action"])
    result["argumentsDecisionAccuracy"] = result.get("observableToolAndArgumentsExact") if expected else result["toolDecisionAccuracy"]
    return result


def performance_goal(public, economic_loss, goal):
    observed = {"toolExact": public["toolExact"],
                "observableToolAndArgumentsExact": public["observableToolAndArgumentsExact"],
                "callPrecision": public["correctCall"] / max(public["predictedCall"], 1e-12),
                "generatedTotalLoss": economic_loss}
    failures = [name for name in ("toolExact", "observableToolAndArgumentsExact", "callPrecision")
                if observed[name] + 1e-12 < goal[name]]
    if economic_loss >= goal["generatedTestTotalLossBelow"]:
        failures.append("generatedTotalLoss")
    return {"passed": not failures, "failures": failures, "observed": observed, "required": goal}
