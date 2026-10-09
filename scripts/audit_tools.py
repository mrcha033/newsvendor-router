"""Audit fixed tool predictions and Train/Dev stage changes without training or Test probes."""

import argparse
import gc
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results/l40s-efficient")
    parser.add_argument("--output", default="results/l40s-efficient/tool-audit")
    args = parser.parse_args()
    devices = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,uuid", "--format=csv,noheader"], text=True
    )
    gpu = next(uuid.strip() for name, uuid in (r.split(",") for r in devices.splitlines()) if name.strip() == "NVIDIA L40S")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.chdir(ROOT)
    import torch

    from newsvendor import structured_data
    from newsvendor.cli import provenance
    from newsvendor.corpus import STATUSES
    from newsvendor.io import digest, jsonl, lines, read, require, write
    from newsvendor.structured_inputs import prepare
    from newsvendor.structured_model import Router, assemble, extract, load_backbone
    from newsvendor.structured_train import infer
    from newsvendor.suite import public_input
    from newsvendor.suite_score import metrics, normalized
    from newsvendor.train import seed

    root, output = Path(args.root), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = read(root / "base/42/config.json")
    rows, labels, collection = structured_data.load(config)
    snapshot_ids = {r["id"] for r in lines(Path(config["dataset"]) / "inputs.jsonl")}
    native = [r for r in rows if r["component"] == "abcd" and r["id"] in snapshot_ids]
    selected = [r for r in native if r["split"] in ("train", "dev")]
    expansion = sorted(
        (r for r in rows if r["component"] == "abcd" and r["id"] not in snapshot_ids),
        key=lambda r: digest([42, r["id"]]),
    )[:128]
    selected += expansion
    native_ids = {r["id"] for r in native}

    def measurement(row, prediction, details=None):
        # Annotations are accessed only after model inference has returned.
        target = labels[row["id"]]
        record = {
            "id": row["id"], "family": row["family"], "split": row["split"],
            "group": "native" if row["id"] in native_ids else "expanded",
            "goldAction": target["action"], "goldTool": target.get("tool"),
            "goldArity": len(target.get("arguments", [])),
            "observable": target.get("argumentsObservable", False),
            "prediction": prediction,
            "metrics": metrics(row, target, prediction),
        }
        if details is not None:
            record["diagnostics"] = details
        if target["action"] == "call_tool":
            fields = [f for f in prediction["fields"] if f["field"].startswith(target["tool"] + ":")]
            record["argumentFields"] = [
                {"mode": f["mode"], "state": f["state"], "use": f["use"],
                 "hasValue": f["value"] is not None,
                 "valueCorrect": normalized(f["value"]) == normalized(value)}
                for f, value in zip(fields, target["arguments"], strict=False)
            ]
            if details is not None:
                record["rawToolCorrect"] = details["rawAction"] == "call_tool:" + target["tool"]
                record["stateBypassToolAndArgsExact"] = metrics(row, target, details["stateBypass"]).get("observableToolAndArgumentsExact")
                record["actionMaskToolAndArgsExact"] = metrics(row, target, details["actionMask"]).get("observableToolAndArgumentsExact")
                record["unblockedFieldsCorrect"] = [
                    normalized(f["value"]) == normalized(value)
                    for f, value in zip(
                        [f for f in details["unblockedFields"] if f["field"].startswith(target["tool"] + ":")],
                        target["arguments"], strict=False,
                    )
                ]
        return record

    def summary(records):
        calls = [r for r in records if r["goldAction"] == "call_tool"]
        observed = [r for r in calls if r["observable"]]
        field_rows = [f for r in observed for f in r.get("argumentFields", [])]
        result = {
            "cases": len(records), "callCases": len(calls), "observableCallCases": len(observed),
            "predictedActions": dict(Counter(r["prediction"]["action"] for r in records)),
            "actionsOnGoldCalls": dict(Counter(r["prediction"]["action"] for r in calls)),
            "toolCorrect": sum(r["metrics"].get("toolExact", 0) for r in calls),
            "toolCorrectObservable": sum(r["metrics"].get("toolExact", 0) for r in observed),
            "toolAndArgumentsCorrect": sum(r["metrics"].get("observableToolAndArgumentsExact", 0) for r in observed),
            "correctTools": dict(Counter(r["goldTool"] for r in calls if r["metrics"].get("toolExact"))),
            "fieldCases": len(field_rows),
            "fieldModes": dict(Counter(f["mode"] for f in field_rows)),
            "fieldStates": dict(Counter(f["state"] for f in field_rows)),
            "fieldValuesCorrect": sum(f["valueCorrect"] for f in field_rows),
            "fieldsWithoutValues": sum(not f["hasValue"] for f in field_rows),
            "correctByArity": dict(Counter(r["goldArity"] for r in observed if r["metrics"].get("observableToolAndArgumentsExact"))),
        }
        if records and "diagnostics" in records[0]:
            result.update(
                rawActionsOnGoldCalls=dict(Counter(r["diagnostics"]["rawAction"] for r in calls)),
                rawToolCorrect=sum(r["rawToolCorrect"] for r in calls),
                stateBypassExact=sum(r["stateBypassToolAndArgsExact"] or 0 for r in observed),
                actionMaskExact=sum(r["actionMaskToolAndArgsExact"] or 0 for r in observed),
                unblockedFieldValuesCorrect=sum(sum(r["unblockedFieldsCorrect"]) for r in observed),
            )
        return result

    report = {
        "provenance": provenance(config), "gpuUuid": gpu,
        "scriptHash": digest(Path(__file__).read_bytes()),
        "scope": "Fixed Test prediction error analysis; new checkpoint and decoder probes use Train/Dev only. No training, Test-conditioned inputs, or changed experiment outputs.",
        "inputHash": digest([public_input(r) for r in selected]),
        "caseIds": [r["id"] for r in selected],
        "selection": "All native ABCD Train/Dev plus 128 hash-selected expanded Train inputs; no label-based input selection.",
        "fixedTest": {}, "stages": {},
    }
    test_rows = {r["id"]: r for r in rows if r["component"] == "abcd" and r["split"] == "test"}
    for variant in ("base", "no_value", "large"):
        predictions = lines(root / variant / "42/predictions-test.jsonl")
        records = [measurement(test_rows[r["id"]], r["prediction"]) for r in predictions if r["id"] in test_rows]
        jsonl(output / (variant + "-fixed-test.jsonl"), records)
        report["fixedTest"][variant] = summary(records)
    write(output / "report.json", report)
    print({"fixedTest": report["fixedTest"]}, flush=True)

    seed(42)
    for variant in ("base", "large", "no_value"):
        current = read(root / variant / "42/config.json")
        tokenizer, encoder = load_backbone(current["encoder"])
        model = Router(encoder, current["encoder"]).cuda().eval()
        stages = ["language", "policy"] if variant != "no_value" else ["policy"]
        for stage in stages:
            checkpoint = root / variant / "42" / ("language-done.pt" if stage == "language" else "model.pt")
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
            model.load_state_dict(payload["weights"], strict=True)
            del payload
            records = []
            stamp = time.perf_counter()
            for number, row in enumerate(selected):
                value = public_input(row)
                with torch.inference_mode():
                    view = prepare(value, tokenizer, current["encoder"], collection=collection)
                    logits = model(view)
                    raw_action = view["actions"][int(logits["recovery"].argmax())]["id"]
                    unblocked = {**logits, "state": torch.full_like(logits["state"], -1000)}
                    unblocked["state"][:, STATUSES.index("verified")] = 0
                    bypass = assemble(view, unblocked, no_value=current["noValue"], use_value=False)
                    masked = {**logits, "recovery": logits["recovery"].clone()}
                    for i, action in enumerate(view["actions"]):
                        if action["id"] in ("answer", "hold"):
                            masked["recovery"][i] = -1000
                    action_mask = assemble(view, masked, no_value=current["noValue"], use_value=False)
                    prediction, trace = infer(model, tokenizer, value, current, collection)
                    details = {
                        "rawAction": raw_action,
                        "modeScores": logits["mode"].argmax(-1).tolist(),
                        "stateScores": logits["state"].argmax(-1).tolist(),
                        "stateBypass": bypass, "actionMask": action_mask,
                        "unblockedFields": extract(view, unblocked),
                        "inputHash": view["inputHash"], "rounds": len(trace),
                        "indexedChunks": view["indexedChunks"], "selectedChunks": view["selectedChunks"],
                    }
                records.append(measurement(row, prediction, details))
                if (number + 1) % 100 == 0:
                    print({"variant": variant, "stage": stage, "processed": number + 1, "total": len(selected)}, flush=True)
            key = variant + "-" + stage
            jsonl(output / (key + ".jsonl"), records)
            results = {
                group + ":" + split: summary([r for r in records if r["group"] == group and r["split"] == split])
                for group, split in (("native", "train"), ("native", "dev"), ("expanded", "train"))
            }
            report["stages"][key] = {
                "checkpoint": str(checkpoint), "checkpointHash": digest(checkpoint.read_bytes()),
                "seconds": time.perf_counter() - stamp, "summaries": results,
            }
            write(output / "report.json", report)
            print({"stage": key, "summaries": results}, flush=True)
        del model, encoder
        gc.collect()
        torch.cuda.empty_cache()
    require(all(r["split"] in ("train", "dev") for r in selected), "Unexpected diagnostic split")
    print({"complete": True, "report": str(output / "report.json")}, flush=True)


if __name__ == "__main__":
    main()
