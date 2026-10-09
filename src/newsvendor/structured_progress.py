"""Durable training progress and exact optimizer/RNG resumption."""

import random
import tarfile
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from .cli import provenance
from .io import digest, read, require, write


def objective_config(config):
    value = {**config, "encoder": dict(config["encoder"]), "training": dict(config["training"])}
    value.pop("output", None)
    runtime = (
        "encoderBatch", "gradientCheckpointing", "bucketLengths", "checkpointTokenLimit",
        "attention", "compileLayers", "compileLimit", "padMultiple", "splitLongBatches",
        "paddingRatio",
    )
    value["backbones"] = {name: dict(backbone) for name, backbone in config["backbones"].items()}
    for encoder in [value["encoder"], *value["backbones"].values()]:
        for key in runtime:
            encoder.pop(key, None)
    for key in (
        "caseBatch", "prefetchWorkers", "progressEvery", "checkpointEvery", "fusedOptimizer",
    ):
        value["training"].pop(key, None)
    return value


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    write(temporary, value)
    temporary.replace(path)


def rng_state():
    name, values, position, gaussian, cached = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [name, values.tolist(), position, gaussian, cached],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    name, values, position, gaussian, cached = state["numpy"]
    np.random.set_state((name, np.array(values, dtype=np.uint32), position, gaussian, cached))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


class Progress:
    def __init__(self, config, dataset_hashes, device, limit, resume=False):
        self.directory = Path(config["output"])
        self.directory.mkdir(parents=True, exist_ok=True)
        self.started = time.perf_counter()
        self.previous_seconds = 0
        self.identity = {
            "configHash": digest(config),
            "datasetHashes": dataset_hashes,
            "sourceHash": provenance(config)["sourceHash"],
            "limit": limit,
        }
        self.config = config
        path = self.directory / "run.json"
        if path.exists():
            require(resume, "Run already exists; use --resume or a new output directory")
            saved = read(path)
            require(saved["identity"] == self.identity, "Resume code/config/data differ")
            self.provenance = saved["provenance"]
            self.hardware = saved["hardware"]
            progress = self.directory / "progress.json"
            if progress.exists():
                self.previous_seconds = read(progress)["elapsedSeconds"]
        else:
            self.provenance = provenance(config)
            self.hardware = {"device": str(device), "cuda": torch.version.cuda}
            if str(device).startswith("cuda"):
                properties = torch.cuda.get_device_properties(device)
                self.hardware.update(
                    name=properties.name,
                    totalBytes=properties.total_memory,
                    capability=list(torch.cuda.get_device_capability(device)),
                    uuid=str(properties.uuid),
                )
            atomic_json(
                path,
                {
                    "identity": self.identity,
                    "provenance": self.provenance,
                    "hardware": self.hardware,
                },
            )
            with tarfile.open(self.directory / "source.tar.gz", "w:gz") as archive:
                for root in ("src", "configs", "scripts"):
                    for file in sorted(Path(root).rglob("*")):
                        if file.is_file() and "__pycache__" not in file.parts:
                            archive.add(file)
                for file in ("pyproject.toml", "uv.lock"):
                    archive.add(file)
            atomic_json(self.directory / "config.json", config)

    def import_language(self, path):
        path = Path(path)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        old_config = read(path.parent / "config.json")
        old_run = read(path.parent / "run.json")
        require(payload["identity"] == old_run["identity"], "Parent checkpoint/run differ")
        require(digest(old_config) == payload["identity"]["configHash"], "Parent config changed")
        require(
            payload["identity"]["datasetHashes"] == self.identity["datasetHashes"]
            and payload["identity"]["limit"] == self.identity["limit"],
            "Parent data/scope differ",
        )
        old_objective, new_objective = objective_config(old_config), objective_config(self.config)
        controller_upgrade = bool(old_objective["encoder"].get("dialogueController")
                                  and not old_objective["encoder"].get("controllerResidual")
                                  and new_objective["encoder"].get("controllerResidual"))
        if controller_upgrade:
            # A declared score-composition upgrade keeps parameter/optimizer shapes,
            # observed inputs, targets, splits and all other training settings intact.
            old_objective["encoder"]["controllerResidual"] = True
        auxiliary_upgrade = bool(old_objective["encoder"].get("dialogueController")
                                 and old_objective["encoder"].get("controllerResidual")
                                 and not old_objective["encoder"].get("controllerAuxiliary")
                                 and new_objective["encoder"].get("controllerAuxiliary"))
        if auxiliary_upgrade:
            # Add direct supervision of the same contextual scores with the same
            # action labels. Parameters, optimizer moments and inputs are unchanged.
            old_objective["encoder"]["controllerAuxiliary"] = True
        context_upgrade = bool(old_objective["encoder"].get("dialogueController")
                               and not old_objective["encoder"].get("procedureContext")
                               and new_objective["encoder"].get("procedureContext"))
        if context_upgrade:
            # Versioned input correction: complete public action conditions, neutral
            # list wording, and no unsupported count-to-workflow transition feature.
            # Weight/optimizer shapes, labels and all other settings remain fixed.
            old_objective["encoder"].update(
                procedureContext=True, controllerLength=3072, controllerPolicyTokens=1536,
            )
        rerank_upgrade = bool(old_objective["encoder"].get("structuredTools")
                              and not old_objective["encoder"].get("rerankSupervision")
                              and new_objective["encoder"].get("rerankSupervision"))
        if rerank_upgrade:
            # Preserve the two recovery losses, but train base and reranked scores
            # separately even when inference currently selects the base branch.
            old_objective["encoder"]["rerankSupervision"] = True
            if not old_objective["encoder"].get("rerank", True) and new_objective["encoder"].get("rerank", True):
                old_objective["encoder"]["rerank"] = True
        workflow_upgrade = bool(old_objective["encoder"].get("procedureContext")
                                and old_objective["encoder"].get("rerankSupervision")
                                and not old_objective["encoder"].get("workflowProgress")
                                and new_objective["encoder"].get("workflowProgress"))
        if workflow_upgrade:
            # Explicit target correction: conditional next public tool nodes from
            # official Train traces replace past-call counts. Parameter shapes,
            # optimizer, observed inputs and every other setting remain intact.
            old_objective["encoder"]["workflowProgress"] = True
        role_upgrade = bool(old_objective["encoder"].get("structuredTools")
                            and not old_objective["training"].get("sourceRoleSupervision")
                            and new_objective["training"].get("sourceRoleSupervision"))
        if role_upgrade:
            # Add named source-slot labels only to official Train tool arguments.
            # No new parameters, inference candidates, inputs or data splits.
            old_objective["training"]["sourceRoleSupervision"] = True
        dialogue_upgrade = bool(old_objective["encoder"].get("dialogueState")
                                and old_objective["encoder"].get("workflowProgress")
                                and not old_objective["encoder"].get("dialogueWorkflow")
                                and new_objective["encoder"].get("dialogueWorkflow"))
        if dialogue_upgrade:
            # Route the existing learned state into procedure/progress/reranking.
            # Keep parameters, optimizer moments, observations and targets intact.
            old_objective["encoder"]["dialogueWorkflow"] = True
        require(old_objective == new_objective, "Continuation changes learning objective or inputs")
        require(
            payload["state"]["offset"] % self.config["training"]["accumulation"] == 0,
            "Parent has unapplied gradients",
        )
        destination = self.directory / "language-progress.pt"
        require(not destination.exists(), "Continuation already exists")
        measurement = {
            "checkpoint": str(path),
            "checkpointHash": digest(path.read_bytes()),
            "identity": payload["identity"],
            "config": old_config,
            "provenance": old_run["provenance"],
            "processed": payload["state"]["offset"],
            "epoch": payload["state"]["epoch"] + 1,
            "previousSeconds": read(path.parent / "progress.json")["elapsedSeconds"],
            "controllerResidualUpgrade": controller_upgrade,
            "controllerAuxiliaryUpgrade": auxiliary_upgrade,
            "procedureContextUpgrade": context_upgrade,
            "rerankSupervisionUpgrade": rerank_upgrade,
            "workflowProgressUpgrade": workflow_upgrade,
            "sourceRoleSupervisionUpgrade": role_upgrade,
            "dialogueWorkflowUpgrade": dialogue_upgrade,
        }
        previous = path.parent / "continuation.json"
        if previous.exists():
            measurement["parent"] = read(previous)
            measurement["previousSeconds"] += measurement["parent"]["previousSeconds"]
        payload["identity"] = self.identity
        if (controller_upgrade or auxiliary_upgrade or context_upgrade or rerank_upgrade or workflow_upgrade or role_upgrade or dialogue_upgrade or old_run["identity"]["sourceHash"] != self.identity["sourceHash"]) and payload["state"].get("saved") is not None:
            payload["state"]["validateBest"] = True
        torch.save(payload, destination)
        atomic_json(self.directory / "continuation.json", measurement)
        self.update("imported", processed=measurement["processed"], parent=str(path))

    def import_common(self, path, model):
        """Explicit warm start of completed language/demand stages with parent provenance."""
        path = Path(path)
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        parent = read(path.parent / "run.json")
        shared = payload["state"]["config"]
        require(payload["identity"] == parent["identity"], "Parent checkpoint/run differ")
        require(digest(shared) == parent["identity"]["configHash"], "Parent config changed")
        require(
            payload["identity"]["datasetHashes"] == self.identity["datasetHashes"]
            and payload["identity"]["limit"] == self.identity["limit"],
            "Common data/scope differ",
        )
        expected = "base" if self.config["variant"] == "no_value" else self.config["variant"]
        require(shared["variant"] == expected, "Common backbone/variant differ")
        for key in ("encoder", "training", "demand", "seed", "maxRetrievals"):
            require(shared[key] == self.config[key], "Common training setting differs: " + key)
        require("policy" not in payload["state"], "Common checkpoint contains policy training")
        model.load_state_dict(payload["weights"], strict=True)
        restore_rng(payload["rng"])
        lineage = {
            "checkpoint": str(path), "checkpointHash": digest(path.read_bytes()),
            "identity": payload["identity"], "config": shared,
            "provenance": parent["provenance"],
            "sourceArchiveHash": digest((path.parent / "source.tar.gz").read_bytes()),
            "scope": "Explicit language/demand warm start; policy code and objectives may change.",
        }
        state = payload["state"] | {
            "config": self.config, "reusedCommon": str(path),
            "commonHash": lineage["checkpointHash"], "commonLineage": lineage,
        }
        self.checkpoint("common", model, state)
        atomic_json(self.directory / "common-parent.json", lineage)
        return state

    def import_tools(self, path, model):
        """Add new tool heads to a verified parent; reuse only its completed demand stage."""
        path = Path(path)
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        parent = read(path.parent / "run.json")
        shared = payload["state"]["config"]
        require(payload["identity"] == parent["identity"], "Parent checkpoint/run differ")
        require(digest(shared) == parent["identity"]["configHash"], "Parent config changed")
        for key in ("snapshot", "expansion", "researchConfig"):
            require(payload["identity"]["datasetHashes"].get(key) == self.identity["datasetHashes"].get(key),
                    "Tool warm-start data differ: " + key)
        for key in ("model", "revision"):
            require(shared["encoder"][key] == self.config["encoder"][key], "Warm-start backbone differs")
        require(shared["demand"] == self.config["demand"], "Warm-start demand settings differ")
        require("policy" not in payload["state"], "Warm start must precede policy learning")
        mismatch = model.load_state_dict(payload["weights"], strict=False)
        expected = {k for k in model.state_dict() if k.startswith(("tools.", "control.")) and k not in payload["weights"]}
        require(set(mismatch.missing_keys) == expected and not mismatch.unexpected_keys,
                "Unexpected warm-start parameter mismatch")
        lineage = {"checkpoint": str(path), "checkpointHash": digest(path.read_bytes()),
                   "identity": payload["identity"], "config": shared,
                   "sourceArchiveHash": digest((path.parent / "source.tar.gz").read_bytes()),
                   "scope": "Tool-head continuation with full encoder fine-tuning; unchanged demand weights reused",
                   "initializedParameters": sorted(expected)}
        atomic_json(self.directory / "tools-parent.json", lineage)
        return {"demand": payload["state"]["demand"], "lineage": lineage}

    def update(self, stage, **values):
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "stage": stage,
            "elapsedSeconds": self.previous_seconds + time.perf_counter() - self.started,
            **values,
        }
        atomic_json(self.directory / "progress.json", record)
        print(record, flush=True)

    def checkpoint(self, name, model, state, optimizer=None):
        payload = {
            "identity": self.identity,
            "weights": model.state_dict(),
            "state": state,
            "rng": rng_state(),
            "optimizer": optimizer.state_dict() if optimizer else None,
        }
        path = self.directory / (name + ".pt")
        temporary = path.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    def restore(self, name, model, optimizer=None):
        path = self.directory / (name + ".pt")
        if not path.exists():
            return None
        payload = torch.load(path, map_location="cpu", weights_only=True)
        require(payload["identity"] == self.identity, "Checkpoint identity changed")
        model.load_state_dict(payload["weights"])
        if optimizer and payload["optimizer"] is not None:
            optimizer.load_state_dict(payload["optimizer"])
        restore_rng(payload["rng"])
        return payload["state"]
