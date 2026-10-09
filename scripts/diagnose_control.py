"""Replay Dev conversations with isolated controller scoring paths."""

import argparse
import copy
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    modes = ("baseline", "rerank_off", "controller_off", "context_only",
             "prior_gate", "prior_rank", "context_gate", "context_rank", "dialogue_state_off")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--checkpoint", help="Preserved optimizer checkpoint to diagnose instead of final model.pt")
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--output")
    parser.add_argument("--conditions", nargs="+", choices=modes, default=list(modes[:-1]))
    parser.add_argument("--reference", help="Saved GPU Dev predictions to compare against the baseline diagnostic")
    args = parser.parse_args()
    os.chdir(ROOT)
    import torch

    from newsvendor.cli import provenance
    from newsvendor.io import digest, jsonl, lines, read, require, write
    from newsvendor.structured_compare import means
    from newsvendor.structured_model import Router, load_backbone
    from newsvendor.structured_tool_eval import conversation_replay, weights_hash
    from newsvendor.structured_train import load

    if args.device != "cpu":
        require(torch.cuda.device_count() == 1 and torch.cuda.get_device_name() == "NVIDIA L40S",
                "Exactly one visible L40S required")
    torch.set_num_threads(4)
    directory = Path(args.run)
    destination = Path(args.output) if args.output else directory.parent.parent / "control-dev"
    require(not destination.exists(), "Preserve previous diagnostic measurements")
    destination.mkdir(parents=True)
    checkpoint = Path(args.checkpoint) if args.checkpoint else directory / "model.pt"
    runtime_overrides, training, identity, checkpoint_state = {}, None, None, None
    if args.checkpoint:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        config = read(directory / "config.json")
        identity = read(directory / "run.json")["identity"]
        require(payload["identity"] == identity and digest(config) == identity["configHash"],
                "Optimizer checkpoint and run identity differ")
        require(provenance(config)["sourceHash"] == identity["sourceHash"],
                "Optimizer diagnostic requires the run's exact source")
        checkpoint_state = {k: payload["state"][k] for k in ("epoch", "offset")}
        if args.device == "cpu":
            runtime_overrides = {"compileLayers": False, "encoderBatch": 1, "precision": "fp32"}
        config = copy.deepcopy(config)
        config["encoder"].update(runtime_overrides)
        tokenizer, encoder = load_backbone(config["encoder"])
        model = Router(encoder, config["encoder"]).to(args.device).eval()
        model.load_state_dict(payload["weights"], strict=True)
        del payload
    else:
        require(args.device != "cpu", "CPU diagnostics require an explicit preserved checkpoint")
        model, tokenizer, config, training = load(checkpoint, args.device)
    modes = tuple(args.conditions)
    require(len(set(modes)) == len(modes) and "baseline" in modes, "Declare unique conditions including baseline")
    require("dialogue_state_off" not in modes or config["encoder"].get("dialogueState"),
            "This checkpoint has no dialogue-state module")
    rows = [r for r in lines(Path(config["dataset"]) / "inputs.jsonl")
            if r["split"] == "dev" and r["component"] == "abcd"]
    ids = {r["id"] for r in rows}
    labels = {r["id"]: r["target"] for r in lines(Path(config["dataset"]) / "labels.jsonl")
              if r["id"] in ids}
    collection = lines(Path(config["dataset"]) / "collection.jsonl")
    require(rows and ids == labels.keys(), "Incomplete Dev inputs or labels")
    original = model.control.forward
    residual = model.control.residual
    before = weights_hash(model.state_dict())
    reference = lines(args.reference) if args.reference else None
    if reference is not None:
        require(len(reference) == len(rows) and {r["id"] for r in reference} == ids,
                "Reference does not contain exactly the fixed Dev cases")
    report = {
        "scope": "Dev-only counterfactual scoring paths with full own-memory conversation replay; no new training or Test selection",
        "declaredConditions": modes,
        "checkpoint": str(checkpoint), "checkpointHash": digest(checkpoint.read_bytes()), "weightsHash": before,
        "checkpointIdentity": identity, "checkpointState": checkpoint_state,
        "device": args.device, "runtimeOverrides": runtime_overrides,
        "reference": args.reference, "referenceHash": digest(Path(args.reference).read_bytes()) if args.reference else None,
        "interventionScope": "All conditions replay their own conversation memory. Controller score interventions occur before optional final contextual reranking. CPU FP32 scores are diagnostic and are not substituted for original GPU goal measurements.",
        "sourceHash": provenance(config)["sourceHash"],
        "scriptHash": digest(Path(__file__).read_bytes()), "inputIdsHash": digest(sorted(ids)),
        "conditions": {},
    }
    write(destination / "plan.json", report)
    for name in modes:
        current = copy.deepcopy(config)
        current["encoder"]["rerank"] = name != "rerank_off"
        current["encoder"]["controllerEnabled"] = name != "controller_off"
        if name == "dialogue_state_off":
            current["encoder"]["dialogueState"] = False
        model.config = current["encoder"]
        model.control.residual = False if name == "context_only" else residual

        def forward(view, queries, output, mode=name):
            original(view, queries, output)
            if mode == "prior_gate":
                scores, mask = output["recovery"].float(), output["callMask"]
                output["callGate"] = torch.stack([scores[~mask].logsumexp(0), scores[mask].logsumexp(0)])
            elif mode == "prior_rank":
                output["controlRecovery"] = output["recovery"]
            elif mode == "context_gate":
                output["callGate"] = output["contextCallGate"]
            elif mode == "context_rank":
                output["controlRecovery"] = output["contextRecovery"]

        model.control.forward = forward
        print({"condition": name, "stage": "replaying", "device": args.device}, flush=True)
        measurements = conversation_replay(model, tokenizer, current, rows, labels, collection, "dev")
        path = destination / f"{name}.jsonl"
        jsonl(path, measurements)
        public = means(measurements)
        observed = {k: v["mean"] for k, v in public.items()}
        if name == "baseline" and training is not None:
            baseline = training["policy"][-1]["publicDevBaseline"]["abcd"]
            for key in ("toolExact", "observableToolAndArgumentsExact", "predictedCall", "correctCall"):
                require(abs(observed[key] - baseline[key]) < 1e-12, "Baseline replay differs")
        result = {"public": public,
                  "callPrecision": observed["correctCall"] / max(observed["predictedCall"], 1e-12),
                  "raw": str(path), "rawHash": digest(path.read_bytes())}
        if name == "baseline" and reference is not None:
            reference_by_id = {r["id"]: r["prediction"] for r in reference}
            fields = ("action", "tool", "arguments")
            differences = [{"id": row["key"],
                            "reference": {k: reference_by_id[row["key"]].get(k) for k in fields},
                            "diagnostic": {k: row["prediction"].get(k) for k in fields}}
                           for row in measurements if any(row["prediction"].get(k) != reference_by_id[row["key"]].get(k) for k in fields)]
            result["referenceAgreement"] = {"cases": len(rows), "matchingDecisions": len(rows) - len(differences),
                                            "differences": differences}
        report["conditions"][name] = result
        write(destination / "report.json", report)
        print({"condition": name, **{k: observed[k] for k in ("toolExact", "observableToolAndArgumentsExact")},
               "callPrecision": result["callPrecision"]}, flush=True)
    model.control.forward = original
    model.control.residual = residual
    require(weights_hash(model.state_dict()) == before, "Diagnostic changed model weights")
    report["status"] = "complete"
    write(destination / "report.json", report)


if __name__ == "__main__":
    main()
