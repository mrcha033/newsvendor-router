"""Held-out paired tool ablations and noisy own-state economic rollouts."""

import copy
import hashlib
import tarfile
import time
from pathlib import Path

import torch

from . import corpus, structured_data
from .cli import provenance
from .io import digest, jsonl, read, require, write
from .structured_compare import means, paired, research_measurements
from .structured_metrics import tool_metrics
from .structured_model import Router, load_backbone
from .structured_rollout import ResearchRouter, rollout
from .structured_train import infer, synchronize
from .suite import public_input


def weights_hash(weights):
    result = hashlib.sha256()
    for name, value in sorted(weights.items()):
        result.update((name + str(value.dtype) + str(tuple(value.shape))).encode())
        result.update(
            value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        )
    return result.hexdigest()


@torch.inference_mode()
def conversation_replay(model, tokenizer, config, rows, labels, collection, split="test"):
    """Copy own predictions across logged prefixes; responses remain the observed dataset."""
    records, states, histories = [], {}, {}
    selected = sorted(
        (r for r in rows if r["component"] == "abcd" and r["split"] == split),
        key=lambda r: (r["id"].rsplit(":", 1)[0], len(r["input"]["history"]), r["id"]),
    )
    for row in selected:
        value = public_input(row)
        family, history = row["family"], value["history"]
        conversation = row["id"].rsplit(":", 1)[0]
        old = histories.get(conversation, [])
        state = states.get(conversation) if history[: len(old)] == old else None
        if not config["encoder"].get("structuredTools", False):
            state = None
        synchronize(model)
        start = time.perf_counter()
        prediction, trace = infer(model, tokenizer, value, config, collection, state=state)
        synchronize(model)
        records.append(
            {
                "key": row["id"],
                "family": family,
                "prediction": prediction,
                "metrics": tool_metrics(row, labels[row["id"]], prediction),
                "trace": trace,
                "elapsedMs": (time.perf_counter() - start) * 1000,
                "priorOwnMemory": len((state or {}).get("memory", [])),
            }
        )
        states[conversation] = {
            "memory": prediction.get("memory", []),
            "procedureIds": prediction.get("procedureIds", []),
        }
        histories[conversation] = history
    return records


@torch.inference_mode()
def run(config, directory):
    """Run only after all Dev selection. Each toggle pair loads the identical checkpoint."""
    directory = Path(directory)
    evaluation_provenance = provenance(config)
    source_archive = (
        directory / f"evaluation-source-{evaluation_provenance['sourceHash'][:12]}.tar.gz"
    )
    if not source_archive.exists():
        with tarfile.open(source_archive, "w:gz") as archive:
            for root in ("src", "configs", "scripts"):
                for file in sorted(Path(root).rglob("*")):
                    if file.is_file() and "__pycache__" not in file.parts:
                        archive.add(file)
            for file in ("pyproject.toml", "uv.lock"):
                archive.add(file)
    rows, labels, collection = structured_data.load(config)
    episodes = corpus.generate(read(config["researchConfig"]))
    tokenizer, encoder = load_backbone(config["encoder"])
    model = Router(encoder, config["encoder"]).to("cuda:0").eval()
    conditions = [
        (
            "decoder_fixed",
            config.get("comparisonBaseline", str(Path(config["warmStart"]).parent / "model.pt")),
            False,
            False,
        ),
        ("common_rerank_off", "common.pt", False, False),
        ("common_rerank_on", "common.pt", True, False),
        ("policy_rerank_off", "policy-done.pt", False, False),
        ("policy_rerank_on", "policy-done.pt", True, False),
        ("policy_recovery_only", "policy-done.pt", True, True),
    ]
    if config["encoder"].get("dialogueController"):
        conditions.append(("policy_controller_off", "policy-done.pt", False, False))
    if config["encoder"].get("dialogueState"):
        conditions.append(("policy_dialogue_state_off", "policy-done.pt", True, False))
    report, raw, economic = {}, {}, {}
    for name, checkpoint, rerank, no_value in conditions:
        path = Path(checkpoint) if name == "decoder_fixed" else directory / checkpoint
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        parameter_hash = weights_hash(payload["weights"])
        current = copy.deepcopy(payload["config"] if name == "decoder_fixed" else config)
        active = (
            Router(encoder, {**current["encoder"], "compileLayers": False}).to("cuda:0").eval()
            if name == "decoder_fixed"
            else model
        )
        active.load_state_dict(payload["weights"], strict=True)
        if name != "decoder_fixed":
            require(
                payload["identity"] == read(directory / "run.json")["identity"],
                "Comparison checkpoint changed",
            )
        current["output"] = str(directory / "comparison" / name)
        current["encoder"]["rerank"] = rerank
        current["encoder"]["controllerEnabled"] = name != "policy_controller_off"
        if name == "policy_dialogue_state_off":
            current["encoder"]["dialogueState"] = False
        current["noValue"] = no_value
        active.config = current["encoder"]
        target = Path(current["output"])
        target.mkdir(parents=True, exist_ok=True)
        # Public values do not use economic heads: recovery-only reuses the same public run.
        if not no_value:
            measurements = conversation_replay(active, tokenizer, current, rows, labels, collection)
            jsonl(target / "logged-conversations.jsonl", measurements)
            raw[name] = measurements
            public = means(measurements)
        else:
            public = report["policy_rerank_on"]["public"]
        router = ResearchRouter(active, tokenizer, current["encoder"], no_value)
        research, research_paths = {}, {}
        # Economic inputs contain no tool procedures, so the rerank toggle has no effect there.
        key = (
            parameter_hash,
            no_value,
            digest(
                {
                    k: v
                    for k, v in current["encoder"].items()
                    if k not in ("rerank", "controllerEnabled", "dialogueState")
                }
            ),
        )
        if key in economic:
            research, research_paths = economic[key]
        else:
            for noise in (0.0, 0.1, 0.2):
                records = [
                    rollout(e, router, noise=noise) for e in episodes if e["split"] == "test"
                ]
                raw_path = target / f"research-noise-{noise}.jsonl"
                jsonl(raw_path, records)
                research[str(noise)] = means(research_measurements(records))
                research_paths[str(noise)] = str(raw_path)
            economic[key] = research, research_paths
        report[name] = {
            "checkpoint": str(path),
            "checkpointHash": digest(path.read_bytes()),
            "weightsHash": parameter_hash,
            "rerank": rerank,
            "controllerEnabled": bool(
                current["encoder"].get("dialogueController")
                and current["encoder"]["controllerEnabled"]
            ),
            "dialogueState": bool(current["encoder"].get("dialogueState")),
            "noValue": no_value,
            "public": public,
            "research": research,
            "researchMeasurements": research_paths,
            "publicMeasurements": str(
                directory
                / "comparison"
                / ("policy_rerank_on" if no_value else name)
                / "logged-conversations.jsonl"
            ),
        }
        write(directory / "tool-comparison.json", {"status": "running", "conditions": report})
        print({"comparison": name, "public": {k: v["mean"] for k, v in public.items()}}, flush=True)
    contrasts = {}
    for first, second in [
        ("common_rerank_on", "common_rerank_off"),
        ("policy_rerank_on", "policy_rerank_off"),
        ("policy_rerank_on", "common_rerank_on"),
        ("common_rerank_on", "decoder_fixed"),
    ]:
        contrasts[first + "_minus_" + second] = paired(raw[first], raw[second], config["seed"])
    report = {
        "status": "complete",
        "conditions": report,
        "pairedSourceFamilyDifferences": contrasts,
        "publicScope": "Fixed Test logged conversation prefixes with own predicted field memory; not interactive ABCD environment",
        "researchScope": "Generated economic simulator with own predicted state, actual response failures and measured terminal loss; not real-work efficacy",
        "lossDefinition": "Regret/hold plus requests; invalid handoff terminal loss floored at hold cost; raw economic loss and penalty stored separately",
        "trainingProvenance": read(directory / "run.json")["provenance"],
        "evaluationProvenance": evaluation_provenance,
        "evaluationSourceArchive": str(source_archive),
        "evaluationSourceArchiveHash": digest(source_archive.read_bytes()),
        "testIdsHash": digest([r["id"] for r in rows if r["split"] == "test"]),
    }
    if config["encoder"].get("dialogueController"):
        report["pairedSourceFamilyDifferences"]["policy_controller_on_minus_off"] = paired(
            raw["policy_rerank_off"], raw["policy_controller_off"], config["seed"]
        )
    if config["encoder"].get("dialogueState"):
        report["pairedSourceFamilyDifferences"]["policy_dialogue_state_on_minus_off"] = paired(
            raw["policy_rerank_on"], raw["policy_dialogue_state_off"], config["seed"]
        )
    if config.get("performanceGoal"):
        from .structured_metrics import performance_goal

        selected = report["conditions"][
            "policy_rerank_on" if config["encoder"].get("rerank") else "policy_rerank_off"
        ]
        report["goalTest"] = performance_goal(
            {k: v["mean"] for k, v in selected["public"].items()},
            selected["research"]["0.0"]["total"]["mean"],
            config["performanceGoal"],
        )
    write(directory / "tool-comparison.json", report)
    return report
