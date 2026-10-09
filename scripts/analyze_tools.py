"""Diagnose saved tool outputs and supervision without changing or rerunning models."""

import argparse
import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from newsvendor.io import digest, jsonl, lines, read, require, write
from newsvendor.structured_inputs import schemas
from newsvendor.suite import canonical
from newsvendor.suite_score import metrics, normalized


def summarize(records):
    counts, errors, tools, arities = Counter(), Counter(), defaultdict(Counter), defaultdict(Counter)
    fields = Counter()
    for row in records:
        counts["cases"] += 1
        if row["goldAction"] != "call_tool":
            counts["spokenCases"] += 1
            counts["falseCalls"] += row["action"] == "call_tool"
            continue
        counts["callCases"] += 1
        counts["callPredictionsOnCalls"] += row["action"] == "call_tool"
        counts["toolCorrect"] += row["toolCorrect"]
        errors[row["error"]] += 1
        tool, arity = tools[row["goldTool"]], arities[str(row["arity"])]
        for group in (tool, arity):
            group["calls"] += 1
            group["toolCorrect"] += row["toolCorrect"]
        if row["observable"]:
            counts["observableCalls"] += 1
            counts["observableToolCorrect"] += row["toolCorrect"]
            counts["toolAndArgumentsCorrect"] += row["exact"]
            counts["latentArgumentListsCorrect"] += row["latentExact"]
            for group in (tool, arity):
                group["observableCalls"] += 1
                group["exact"] += row["exact"]
                group["latentExact"] += row["latentExact"]
            fields["arguments"] += row["arity"]
            fields["correctLatentValues"] += sum(row["fieldCorrect"])
    counts["predictedCalls"] = counts["falseCalls"] + counts["callPredictionsOnCalls"]
    return {
        "counts": dict(counts), "errorsOnCalls": dict(errors),
        "perTool": dict(tools), "perArity": dict(arities), "fields": dict(fields),
        "macroToolAccuracy": sum(t["toolCorrect"] / t["calls"] for t in tools.values()) / len(tools),
        "callDetectionPrecision": counts["callPredictionsOnCalls"] / max(1, counts["predictedCalls"]),
        "callDetectionRecall": counts["callPredictionsOnCalls"] / counts["callCases"],
        "toolsWithoutAccountLookup": {
            "correct": counts["toolCorrect"] - tools["pull-up-account"]["toolCorrect"],
            "cases": counts["callCases"] - tools["pull-up-account"]["calls"],
        },
    }


def measurements(predictions, rows, labels, traces=None):
    result = []
    for item in predictions:
        if item["id"] not in rows:
            continue
        row, target, prediction = rows[item["id"]], labels[item["id"]], item["prediction"]
        measured = metrics(row, target, prediction)
        current = {
            "id": row["id"], "family": row["family"], "split": row["split"],
            "goldAction": target["action"], "goldTool": target.get("tool"),
            "action": prediction["action"], "tool": prediction.get("tool"),
            "nextAction": prediction.get("nextAction"),
            "question": prediction.get("question"), "reason": prediction.get("reason"),
            "historyLength": len(row["input"]["history"]),
            "lastRole": row["input"]["history"][-1]["role"],
        }
        if target["action"] == "call_tool":
            fs = [f for f in prediction["fields"] if f["field"].startswith(target["tool"] + ":")]
            latent = [f["value"] for f in fs if f["use"]]
            tool_correct = bool(measured["toolExact"])
            error = (
                "correct_tool" if tool_correct else
                "decoder_rejected_correct_tool" if prediction.get("tool") == target["tool"] else
                "wrong_tool" if prediction["action"] == "call_tool" else
                "wrong_tool_then_decoder_rejected" if prediction.get("tool") else
                "retrieval_exhausted" if prediction.get("reason") == "retrieval_budget_exhausted" else
                "no_call"
            )
            current.update(
                arity=len(target["arguments"]), observable=target["argumentsObservable"],
                toolCorrect=tool_correct, exact=bool(measured.get("observableToolAndArgumentsExact")),
                error=error, goldArguments=target["arguments"], arguments=prediction.get("arguments"),
                latentArguments=latent,
                latentExact=[normalized(v) for v in latent] == [normalized(v) for v in target["arguments"]],
                fieldCorrect=[normalized(f["value"]) == normalized(v) for f, v in zip(fs, target["arguments"], strict=False)],
            )
        if traces and row["id"] in traces:
            current["retrievalTrace"] = [
                {k: t[k] for k in ("round", "indexedChunks", "selectedChunks", "queryTruncated")}
                | {"action": t["prediction"]["action"]}
                for t in traces[row["id"]]
            ]
        result.append(current)
    return result


def supervision(snapshot, labels, extension, fields, names):
    stats, tools, rejects = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    by_tool = defaultdict(list)
    for field in fields:
        by_tool[field["tool"]].append(field)

    def count(row):
        target, split = labels[row["id"]], row["split"]
        current = stats[split]
        current["cases"] += 1
        current[target["action"]] += 1
        if target["action"] == "speak":
            answer = target.get("answer", "")
            if "?" in answer:
                current["spokenWithQuestionMark"] += 1
                text = " " + canonical(answer) + " "
                current["questionFieldLexicalMatch"] += any(
                    " " + canonical(name.replace("_", " ")) + " " in text for name in names
                )
            return
        tools[split][target["tool"]] += 1
        current["noArgumentCalls"] += not target["arguments"]
        current["observableCalls"] += target["argumentsObservable"]
        fs = by_tool[target["tool"]]
        require(len(target["arguments"]) <= len(fs), "Unsupported argument count")
        invalid = any(f["choices"] and v not in f["choices"] for f, v in zip(fs, target["arguments"], strict=False))
        unrepresentable = any(
            f["choices"] and normalized(v) not in {normalized(choice) for choice in f["choices"]}
            for f, v in zip(fs, target["arguments"], strict=False)
        )
        current["enumRejectsReferenceArguments"] += invalid
        current["enumRejectsObservableReferenceArguments"] += invalid and target["argumentsObservable"]
        current["enumCannotRepresentNormalizedReference"] += unrepresentable
        current["enumCannotRepresentObservableReference"] += unrepresentable and target["argumentsObservable"]
        if invalid:
            rejects[split][target["tool"]] += 1

    for row in snapshot.values():
        count(row)
    with (extension / "inputs.jsonl").open() as handle:
        for line in handle:
            count(json.loads(line))
    return {"counts": dict(stats), "tools": dict(tools), "enumRejectsByTool": dict(rejects)}


def policy_coverage(config, rows, labels, output):
    """Prepare public Dev views first; read source subflow labels only for scoring coverage."""
    import torch
    from transformers import AutoTokenizer

    from newsvendor.structured_inputs import prepare, source_tokens
    from newsvendor.structured_labels import suite_targets
    from newsvendor.suite import public_input

    torch.set_num_threads(1)
    encoder = config["encoder"]
    tokenizer = AutoTokenizer.from_pretrained(
        encoder["model"], revision=encoder["revision"], cache_dir=".cache/torch-models",
        local_files_only=True, trust_remote_code=False,
    )
    dev = [r for r in rows.values() if r["split"] == "dev"]
    views = {r["id"]: prepare(public_input(r), tokenizer, encoder) for r in dev}
    policy = dev[0]["input"]["documents"][0]["text"]
    guidelines = json.loads(policy)
    definition = read("configs/complementary.json")["sources"]["abcd"]
    assets = {}
    for item in definition["files"]:
        url = f"https://raw.githubusercontent.com/{definition['repo']}/{definition['revision']}/{item['path']}"
        path = Path("data/raw/complementary/abcd") / (digest(url)[:20] + ".raw")
        data = path.read_bytes()
        require(digest(data) == item["sha256"], "ABCD source changed")
        assets[Path(item["path"]).name] = data
    ontology = json.loads(assets["ontology.json"])
    flows = {canonical(flow): value for flow, value in guidelines.items()}
    subflows = {}
    for flow, names in ontology["intents"]["subflows"].items():
        policy_names = list(flows[canonical(flow.replace("_", " "))]["subflows"])
        require(len(names) == len(policy_names), "Subflow order mapping changed")
        subflows.update(zip(names, policy_names, strict=True))
    original = json.loads(gzip.decompress(assets["abcd_v1.1.json.gz"]))
    annotations = {
        f"abcd:{c['convo_id']}:{i}": turn["targets"]
        for records in original.values() for c in records for i, turn in enumerate(c["delexed"])
        if f"abcd:{c['convo_id']}:{i}" in views
    }
    ranges = {}
    for match in re.finditer(r'^      "([^"]+)": \{', policy, re.M):
        end = re.search(r"^      \}", policy[match.end():], re.M)
        require(end is not None and match[1] not in ranges, "Ambiguous workflow block")
        ranges[match[1]] = (match.start(), match.end() + end.end())
    offsets = source_tokens(tokenizer, policy)["offset_mapping"]
    records = []
    for row in dev:
        view, target = views[row["id"]], labels[row["id"]]
        title = subflows[annotations[row["id"]][0]]
        lo, hi = ranges[title]
        locations = [p for p in view["locations"] if p["kind"] == "document"]
        supported = [p for p in locations if lo <= p["start"] < hi]
        whole = sum(lo <= start < hi for start, _ in offsets)
        supervision = suite_targets(view, "abcd", target)
        history_tokens = sum(len(source_tokens(tokenizer, h["text"])["input_ids"]) for h in row["input"]["history"])
        records.append({
            "id": row["id"], "split": "dev", "goldAction": target["action"],
            "goldTool": target.get("tool"), "workflowForScoringOnly": title,
            "workflowTokens": whole, "visibleWorkflowTokens": len(supported),
            "workflowFraction": len(supported) / max(1, whole),
            "documentTokens": len(locations), "fullDocumentTokens": len(offsets),
            "historyTokens": history_tokens,
            "visibleHistoryTokens": sum(p["kind"] == "history" for p in view["locations"]),
            "queryTruncated": view["queryTruncated"],
            "needsRetrievalTarget": bool(supervision.get("needsRetrieval")),
            "supervisedAction": view["actions"][supervision["recovery"]]["id"] if "recovery" in supervision else None,
            "supervisedFieldCounts": dict(Counter(k for f in supervision["fields"] for k in f)),
            "inputHash": view["inputHash"],
        })
    jsonl(output / "dev-policy-coverage.jsonl", records)
    calls = [r for r in records if r["goldAction"] == "call_tool"]
    return {
        "scope": "Dev-only deterministic view audit. Source subflow labels score coverage after views are built; they never enter retrieval, model inputs or selection.",
        "mapping": "Public ontology subflow order aligned with each public guideline flow's ordered subflows.",
        "cases": len(records), "calls": len(calls),
        "fullPolicyTokens": len(offsets),
        "meanSelectedDocumentTokens": sum(r["documentTokens"] for r in records) / len(records),
        "callsWithNoWorkflowTokens": sum(r["visibleWorkflowTokens"] == 0 for r in calls),
        "callsWithUnderHalfWorkflow": sum(r["workflowFraction"] < .5 for r in calls),
        "callsSupervisingRetrieval": sum(r["supervisedAction"] == "retrieve" for r in calls),
        "allCasesWithTruncatedQueryPrefix": sum(r["queryTruncated"] for r in records),
        "allCasesWithCompleteHistory": sum(r["historyTokens"] == r["visibleHistoryTokens"] for r in records),
    }


def dev_scores(root, rows, labels):
    """Frozen first-pass action rankings on every public Dev input, never Test."""
    import gc
    import hashlib

    import torch

    from newsvendor.structured_inputs import prepare
    from newsvendor.structured_model import Router, load_backbone
    from newsvendor.suite import public_input

    torch.set_num_threads(1)
    torch.manual_seed(42)
    dev = [r for r in rows.values() if r["split"] == "dev"]
    output = root / "tool-analysis"
    report = {"scope": "Frozen Dev first-pass scores. Public views are built without labels; targets are read only after inference. No Test rerun or parameter updates.", "variants": {}}
    for variant in ("base", "large"):
        config = read(root / variant / "42/config.json")
        config["encoder"]["compileLayers"] = False
        tokenizer, encoder = load_backbone(config["encoder"])
        model = Router(encoder, config["encoder"])
        checkpoint = root / variant / "42/model.pt"
        payload = torch.load(checkpoint, weights_only=True, mmap=True, map_location="cpu")
        model.load_state_dict(payload["weights"], strict=True)
        del payload
        model.cuda().eval()
        records = []
        with torch.inference_mode():
            for row in dev:
                view = prepare(public_input(row), tokenizer, config["encoder"])
                logits = model(view)["recovery"]
                ids = [a["id"] for a in view["actions"]]
                ranking = sorted((i for i, a in enumerate(view["allowedActions"]) if a), key=lambda i: float(logits[i]), reverse=True)
                tool_ranking = [i for i in ranking if ids[i].startswith("call_tool:")]
                probabilities = logits.softmax(-1).tolist()
                # Labels enter only this measurement record, after inference.
                target = labels[row["id"]]
                gold = "call_tool:" + target["tool"] if target["action"] == "call_tool" else "respond"
                records.append({
                    "id": row["id"], "split": "dev", "inputHash": view["inputHash"],
                    "goldAction": target["action"], "goldTool": target.get("tool"),
                    "rawAction": ids[ranking[0]],
                    "rankAmongActions": ranking.index(ids.index(gold)) + 1,
                    "rankAmongTools": tool_ranking.index(ids.index(gold)) + 1 if target["action"] == "call_tool" else None,
                    "toolProbabilityMass": sum(probabilities[i] for i in tool_ranking),
                    "topActions": [{"action": ids[i], "probability": probabilities[i]} for i in ranking[:8]],
                })
                if len(records) % 32 == 0:
                    print({"variant": variant, "devScored": len(records)}, flush=True)
        calls = [r for r in records if r["goldAction"] == "call_tool"]
        with checkpoint.open("rb") as handle:
            checkpoint_hash = hashlib.file_digest(handle, "sha256").hexdigest()
        report["variants"][variant] = {
            "checkpoint": str(checkpoint), "checkpointHash": checkpoint_hash,
            "executionOverride": {"compileLayers": False}, "devCases": len(records), "callCases": len(calls),
            "rawTop1Correct": sum(r["rankAmongActions"] == 1 for r in calls),
            "top1AmongToolsCorrect": sum(r["rankAmongTools"] == 1 for r in calls),
            "top3AmongToolsCorrect": sum(r["rankAmongTools"] <= 3 for r in calls),
            "top5AmongToolsCorrect": sum(r["rankAmongTools"] <= 5 for r in calls),
            "nonCallButCorrectToolRankedFirst": sum(not r["rawAction"].startswith("call_tool:") and r["rankAmongTools"] == 1 for r in calls),
            "rawActionsOnCalls": dict(Counter(r["rawAction"] for r in calls)),
        }
        jsonl(output / f"{variant}-dev-scores.jsonl", records)
        write(output / "dev-scores.json", report)
        print({"variant": variant, "summary": report["variants"][variant]}, flush=True)
        del model, encoder
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/l40s-policy-v2"))
    parser.add_argument("--coverage", action="store_true", help="Audit deterministic Dev policy coverage; no model inference")
    parser.add_argument("--scores-only", action="store_true", help="Score frozen first-pass Dev action rankings on the visible GPU")
    args = parser.parse_args()
    root, output = args.root, args.root / "tool-analysis"
    config = read(root / "base/42/config.json")
    dataset, extension = Path(config["dataset"]), Path(config["expansion"])
    rows = {r["id"]: r for r in lines(dataset / "inputs.jsonl") if r["component"] == "abcd"}
    labels = {r["id"]: r["target"] for r in lines(dataset / "labels.jsonl")}
    if args.scores_only:
        dev_scores(root, rows, labels)
        return
    labels.update({r["id"]: r["target"] for r in lines(extension / "labels.jsonl")})
    definitions = read(config["encoder"]["toolSchema"])
    tools = next(iter(rows.values()))["input"]["tools"]
    fields = schemas(tools, definitions["fields"], positional=True)
    report = {
        "scope": "Post-hoc analysis of immutable predictions and Train supervision; no model inference, fitting, changed targets or checkpoint selection.",
        "sourceHash": read(root / "fix-verification.json")["sourceHash"],
        "scriptHash": digest(Path(__file__).read_bytes()),
        "datasetHashes": {"snapshot": digest(read(dataset / "manifest.json")), "expansion": digest(read(extension / "manifest.json"))},
        "supervision": supervision(rows, labels, extension, fields, definitions["names"]),
        "schema": {"tools": len(tools), "argumentQueries": len(fields), "questionQueries": len(definitions["names"]),
                   "toolDescriptions": tools, "enumUnions": {t["id"]: [f["choices"] for f in fields if f["tool"] == t["id"]] for t in tools}},
        "variants": {},
    }
    if args.coverage:
        report["policyCoverage"] = policy_coverage(config, rows, labels, output)
    for variant in ("base", "large", "no_value"):
        directory = root / variant / "42"
        training = read(directory / "training.json")
        selection = read(root / "fix-verification.json")["results"][variant]["selection"]
        final_dev = "initial" if selection["iteration"] is None else f"{selection['iteration']}-{selection['epoch']}"
        info = {
            "languageEpochs": training["language"],
            "selected": selection,
            "policyCandidates": [
                {"iteration": iteration["iteration"], "epoch": epoch["epoch"],
                 "economicDevLoss": epoch["devActualTotalLoss"], "abcd": epoch["publicDev"]["abcd"],
                 "accepted": epoch["accepted"], "retentionFailures": epoch["retentionFailures"]}
                for iteration in training["policy"] for epoch in iteration["epochs"]
            ],
        }
        traces = {r["id"]: r["trace"] for r in lines(directory / "traces-test.jsonl")}
        for stage, path in (
            ("test", directory / "predictions-test.jsonl"),
            ("initialDev", directory / "retention-initial.jsonl"),
            ("selectedDev", directory / f"retention-{final_dev}.jsonl"),
        ):
            records = measurements(lines(path), rows, labels, traces if stage == "test" else None)
            info[stage] = summarize(records)
            jsonl(output / f"{variant}-{stage}.jsonl", records)
        saved = read(directory / "scores-test/metrics.json")["components"]["abcd"]["metrics"]
        require(abs(info["test"]["counts"]["toolCorrect"] / info["test"]["counts"]["callCases"] - saved["toolExact"]["mean"]) < 1e-10, "Metric mismatch")
        report["variants"][variant] = info
    write(output / "report.json", report)
    print(json.dumps({"report": str(output / "report.json"), "supervision": report["supervision"],
                      "test": {v: r["test"] for v, r in report["variants"].items()}}, indent=2))


if __name__ == "__main__":
    main()
