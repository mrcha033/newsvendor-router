"""Loss adapters apply only source annotations; labels never enter prepared model views."""

import ast
import math
import re

import torch
from torch.nn import functional as fn

from .construction import candidates, parameter_record, reference
from .corpus import KINDS, STATUSES
from .structured_inputs import DECISIONS, MODES, OPS, SCALES, same_source, span_indices
from .structured_model import compute
from .structured_procedures import visible
from .structured_tools import role_targets
from .suite import canonical
from .suite_score import rounded_numeric_exact


def occurrences(view, text):
    result = []
    if not str(text).strip():
        return result
    for source in view["sources"]:
        for match in re.finditer(re.escape(str(text).strip()), source["text"], re.I):
            loc = {k: source[k] for k in ("kind", "id", "row", "column") if k in source}
            loc |= {"start": match.start(), "end": match.end()}
            pair = span_indices(view, loc)
            if pair is not None:
                result.append(pair)
    return result


def suite_targets(view, component, target):
    result = {"fields": [{} for _ in view["fields"]]}
    fields = result["fields"]
    if "spans" in target:
        pairs = [
            span_indices(
                view,
                {"kind": "document", "id": s["document"], "start": s["start"], "end": s["end"]},
            )
            for s in target["spans"]
        ]
        pairs = [p for p in pairs if p is not None]
        fields[0]["evidence"] = int(bool(pairs))
        result["needsRetrieval"] = bool(target["spans"]) and not pairs
        if pairs:
            fields[0]["spans"] = pairs
        if target["action"] == "abstain":
            fields[0]["mode"] = MODES.index("missing")
    if component == "contractnli":
        fields[0]["decision"] = DECISIONS.index(target["answer"])
    if component == "orsharc" and target["action"] != "ask":
        fields[0]["decision"] = DECISIONS.index(target["answer"].title())
    action = target["action"]
    if component == "abcd":
        if target.get("pastTools") and view.get("controllerTurns"):
            turns = {turn["historyIndex"]: i for i, turn in enumerate(view["controllerTurns"]) if turn["role"] == "tool"}
            tools = {view["actions"][action]["id"].removeprefix("call_tool:"): i
                     for i, action in enumerate(view["historyActions"])}
            result["pastTools"] = [[turns[t["historyIndex"]], tools[t["tool"]]] for t in target["pastTools"]
                                   if t["historyIndex"] in turns and t["tool"] in tools]
        action = "call_tool:" + target["tool"] if action == "call_tool" else "respond"
        if target["action"] == "speak" and "?" in target.get("answer", ""):
            text = " " + canonical(target["answer"]) + " "
            asked = [
                i
                for i, field in enumerate(view["fields"])
                if (field.get("tool") or field.get("questionField"))
                and " " + canonical(field["name"].replace("_", " ")) + " " in text
            ]
            if asked:
                result["question"] = asked
                action = "confirm" if " confirm " in text else "ask"
        if target["action"] == "call_tool":
            indices = [i for i, f in enumerate(view["fields"]) if f.get("tool") == target["tool"]]
            positional = any("position" in view["fields"][i] for i in indices)
            if positional:
                for number, index in enumerate(indices):
                    fields[index]["use"] = int(number < len(target["arguments"]))
            if len(target["arguments"]) > len(indices):
                result["unsupportedArgumentCount"] = True
                indices = []
            for position, (index, value) in enumerate(zip(indices, target["arguments"], strict=False)):
                pairs = occurrences(view, value)
                choices = view["fields"][index].get("choices", [])
                role_ids = role_targets(view["fields"][index], value) if view.get("roles") else []
                annotated_roles = target.get("argumentRoles", [])
                if not role_ids and position < len(annotated_roles) and annotated_roles[position] and view.get("roles"):
                    # Source scenario slots are Train loss annotations only. Restrict
                    # them to this observed tool's declared role vocabulary. Shared
                    # enum values can denote a different role in the current action;
                    # preserve those existing marginal labels rather than narrowing.
                    role_ids = [i for i, role in enumerate(view["fields"][index].get("roles", []))
                                if role["name"] in annotated_roles[position]] or role_ids
                if role_ids:
                    fields[index]["role"] = [view["roles"].index(view["fields"][index]["roles"][i]["name"]) for i in role_ids]
                entity_ids = [i for i, e in enumerate(view.get("entities", [])) if canonical(str(e["value"])) == canonical(str(value))]
                if value in choices:
                    fields[index]["mode"] = MODES.index("choice")
                    fields[index]["choice"] = next(
                        i
                        for i, choice in enumerate(view["choices"])
                        if view["choiceFields"][i] == index and choice == str(value)
                    )
                elif entity_ids:
                    fields[index].update(mode=MODES.index("entity"), entity=entity_ids, evidence=1)
                    if pairs:
                        fields[index]["spans"] = pairs
                elif pairs:
                    fields[index]["mode"] = MODES.index("span")
                    fields[index]["spans"] = pairs
                    fields[index]["evidence"] = 1
                elif any(re.search(re.escape(str(value).strip()), s["text"], re.I) for s in view["sources"] if str(value).strip()):
                    # Train/Dev loss metadata only: observed support exists outside the current view.
                    result["needsRetrieval"] = True
                # A hidden/unobservable argument is not a missing-state annotation.
        if target.get("procedure") and view.get("procedures"):
            matches = [i for i, p in enumerate(view["procedures"]) if p["id"] == target["procedure"]]
            if matches:
                result["procedure"] = matches
                if view.get("workflowProgress"):
                    if target.get("workflowNodes"):
                        result["workflowNodes"] = target["workflowNodes"]
                elif "stage" in target:
                    result["stage"] = target["stage"]
                if view.get("controllerStart") is None and target["action"] == "call_tool" and not visible(view, view["procedures"][matches[0]]):
                    result["needsRetrieval"] = True
    if component == "tatqa":
        fields[0]["scale"] = SCALES.index(target["scale"])
        # Required multiple answers are not interchangeable single-span targets.
        # The scalar head can still learn scale/action from those annotations.
        if target["answerType"] == "span":
            answers = target["answer"] if isinstance(target["answer"], list) else [target["answer"]]
            pairs = [p for answer in answers for p in occurrences(view, answer)]
            if pairs:
                fields[0] |= {"spans": pairs, "mode": MODES.index("span"), "evidence": 1}
        elif target.get("derivation"):
            relation = derivation(view, target["derivation"], target["scale"], target["answer"])
            if relation is not None:
                fields[0] |= relation | {"mode": MODES.index("compute"), "evidence": 1}
    if action == "abstain":
        action = "hold"
    ids = [a["id"] for a in view["actions"]]
    if (
        result.get("needsRetrieval")
        and view["selectedChunks"] < view["indexedChunks"]
    ):
        # Existing span annotation supervises retrieval coverage, not a fabricated parameter state.
        action = "retrieve"
    if action in ids:
        result["recovery"] = ids.index(action)
        if component == "abcd" and view.get("controllerStart") is not None:
            result["control"] = ids.index(action)
    return result


def derivation(view, text, scale="", answer=None):
    """Map source formulas to the existing calculator; annotations are loss-only.

    TAT-QA sometimes omits percent conversion in its written derivation. Use the
    annotated scale and rounded answer to resolve that unit conversion, never to
    search for an unrelated expression. Unsupported/inconsistent trees stay masked.
    """
    try:
        node = ast.parse(text.replace(",", "").replace("$", "").strip(), mode="eval").body

        def constant(n):
            if isinstance(n, ast.Constant) and type(n.value) in (int, float):
                return float(n.value)
            if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
                return -constant(n.operand)
            raise ValueError("Not a numeric literal")

        percent = False
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            for factor, expression in ((node.left, node.right), (node.right, node.left)):
                if isinstance(factor, ast.Constant) and factor.value == 100 and isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Div):
                    node, percent = expression, True
                    break
        # Both written forms denote the same ordered pair of source values.
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub) and isinstance(node.left, ast.BinOp) and isinstance(node.left.op, ast.Div) and constant(node.right) == 1:
            vals = [constant(node.left.left), constant(node.left.right)]
            if scale != "percent":
                return None
            op = "change"
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div) and isinstance(node.left, ast.BinOp):
            left = node.left
            vals = [constant(left.left), constant(left.right)]
            denominator = constant(node.right)
            if isinstance(left.op, ast.Add) and denominator == 2 and not percent:
                op = "average"
            elif isinstance(left.op, ast.Sub) and denominator == vals[1] and (percent or scale == "percent"):
                op = "change"
            else:
                return None
        elif isinstance(node, ast.BinOp):
            op = {ast.Add: "add", ast.Sub: "subtract", ast.Mult: "multiply", ast.Div: "divide"}[
                type(node.op)
            ]
            vals = [constant(node.left), constant(node.right)]
            if percent:
                op = "percent"
            elif op == "divide" and scale == "percent":
                # Preserve rare raw-ratio annotations; do not multiply every
                # percent-labelled answer by 100 unconditionally.
                if answer is None or not rounded_numeric_exact(compute(op, vals), answer):
                    op = "percent"
        else:
            op, vals = "copy", [constant(node)]
        value = compute(op, vals)
        if answer is not None and not rounded_numeric_exact(value, answer):
            return None
        indices = [
            [i for i, a in enumerate(view["atoms"]) if math.isclose(a["value"], v, abs_tol=1e-7)]
            for v in vals
        ]
        if not all(indices):
            return None
        return {
            "relation": OPS.index(op),
            "operand1": indices[0],
            **({"operand2": indices[1]} if len(indices) > 1 else {}),
        }
    except (SyntaxError, KeyError, TypeError, ValueError, OverflowError):
        return None


def research_targets(view, value):
    linked = "forecast" in value["task"]
    ref = parameter_record(value) if linked else reference(value)
    result = {"fields": []}
    for field in view["fields"]:
        slot = field["name"]
        if linked and slot == "F":
            # The numeric demand path is supervised by observed future likelihood,
            # not a fabricated language-state label or a true distribution family.
            result["fields"].append({"type": KINDS.index("estimate")})
            continue
        target = {
            "type": KINDS.index(ref["types"][slot]),
            "state": STATUSES.index(ref["state"][slot]),
        }
        if slot != "F":
            target["evidence"] = int(slot in ref["links"])
            if ref["state"][slot] == "conflict":
                target["mode"] = MODES.index("conflict")
            elif slot not in ref["values"]:
                target["mode"] = MODES.index("missing")
            else:
                expr = next(
                    e for e in candidates(value, slot) if e["id"] == ref["expressions"][slot]
                )
                op = {"divide": "divide", "subtract": "subtract", "copy": "copy"}[expr["op"]]
                operands = []
                for atom in expr["args"]:
                    loc = {
                        "kind": "document",
                        "id": expr["doc"]["id"],
                        "start": atom["span"][0],
                        "end": atom["span"][1],
                    }
                    ids = [
                        i
                        for i, a in enumerate(view["atoms"])
                        if same_source(a["location"], loc)
                        and a["location"]["start"] == loc["start"]
                        and a["location"]["end"] == loc["end"]
                    ]
                    operands.append(ids)
                if all(operands):
                    target |= {
                        "relation": OPS.index(op),
                        "operand1": operands[0],
                        "mode": MODES.index("compute"),
                        **({"operand2": operands[1]} if len(operands) > 1 else {}),
                    }
        result["fields"].append(target)
    return result


def objective(output, targets, *, no_value=False):
    losses = {}
    if targets.get("pastTools"):
        pairs = torch.tensor(targets["pastTools"], device=output["pastTools"].device)
        losses["pastTools"] = [fn.cross_entropy(output["pastTools"][pairs[:, 0]], pairs[:, 1])]
    if "control" in targets:
        target = targets["control"]
        call = int(output["callMask"][target])
        device = output["controlRecovery"].device
        control_key = "baseControlRecovery" if "baseControlRecovery" in output else "controlRecovery"
        call_key = "baseCallGate" if "baseCallGate" in output else "callGate"
        for suffix, action_key, gate_key in (("", control_key, call_key),
                                              ("Context", "contextRecovery", "contextCallGate")):
            if action_key not in output:
                continue
            losses["callGate" + suffix] = [fn.cross_entropy(output[gate_key][None], torch.tensor([call], device=device))]
            scores = output[action_key].masked_fill(output["callMask"].bool() != bool(call), -1e9)
            losses["control" + suffix] = [fn.cross_entropy(scores[None], torch.tensor([target], device=device))]
        if "rerankJoint" in output:
            # Supervise the exact combined call/tool distribution regardless of
            # the inference ablation toggle. No gold state enters its features.
            losses["rerank"] = [fn.cross_entropy(output["rerankJoint"][None], torch.tensor([target], device=device))]
    for i, field in enumerate(targets.get("fields", [])):
        for name, target in field.items():
            if name == "spans":
                # Multiple genuine source occurrences are marginal alternatives, not negatives.
                start = output["start"][i].log_softmax(-1)
                end = output["end"][i].log_softmax(-1)
                unique = set(tuple(p) for p in target)
                loss = -torch.logsumexp(torch.stack([start[a] + end[b] for a, b in unique]), 0)
            elif name in ("operand1", "operand2", "role", "entity"):
                loss = -torch.logsumexp(output[name][i].log_softmax(-1)[target], 0)
            else:
                loss = fn.cross_entropy(
                    output[name][i : i + 1], torch.tensor([target], device=output[name].device)
                )
            losses.setdefault(name, []).append(loss)
    if "recovery" in targets:
        recovery = output.get("rerankRecovery", output["recovery"])
        losses["recovery"] = [
            fn.cross_entropy(
                recovery[None],
                torch.tensor([targets["recovery"]], device=recovery.device),
            )
        ]
        if "baseRecovery" in output:
            losses["baseRecovery"] = [fn.cross_entropy(output["baseRecovery"][None],
                torch.tensor([targets["recovery"]], device=output["recovery"].device))]
    if "procedure" in targets:
        losses["procedure"] = [-torch.logsumexp(output["procedure"].log_softmax(-1)[targets["procedure"]], 0)]
    if "workflowNodes" in targets:
        # Marginalize genuinely ambiguous node alternatives, conditional on the
        # annotated procedure. No forced target for out-of-catalog actions.
        conditional = output["stage"][targets["procedure"]].log_softmax(-1)
        losses["stage"] = [-torch.logsumexp(conditional[:, targets["workflowNodes"]], -1).mean()]
    elif "stage" in targets:
        losses["stage"] = [fn.cross_entropy(output["stage"][None],
            torch.tensor([targets["stage"]], device=output["stage"].device))]
    if "question" in targets:
        ids = targets["question"]
        ids = ids if isinstance(ids, list) else [ids]
        losses["question"] = [
            -torch.logsumexp(output["question"].flatten().log_softmax(-1)[ids], 0)
        ]
    if "values" in targets and not no_value:
        indices = targets["valueIndices"]
        truth = torch.tensor(targets["values"], device=output["value"].device)
        losses["value"] = [fn.smooth_l1_loss(output["value"][indices], truth)]
    means = {name: torch.stack(values).mean() for name, values in losses.items()}
    total = sum(means.values()) if means else output["state"].sum() * 0
    values = torch.stack([v.detach() for v in means.values()]).cpu().tolist() if means else []
    return total, dict(zip(means, values, strict=True))
