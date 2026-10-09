"""Structured controller for the existing retail simulator; Qwen is only user/judge."""

import json
import time
from pathlib import Path

import torch

from .io import digest, jsonl, lines, read, write
from .orders import WRITES, argument_model, check
from .structured_train import infer, synchronize
from .suite import payload


class Actor:
    def __init__(self, model, tokenizer, config):
        self.model, self.tokenizer, self.config = model, tokenizer, config
        self.reset()

    def reset(self):
        self.state, self.history = None, []

    def decide(self, public, history, confirmed):
        if history[:len(self.history)] != self.history:
            self.reset()
        tools = [
            {
                "id": t["id"],
                "description": t["description"],
                "parameters": t["parameters"],
                "requiresConfirmation": t["id"] in WRITES,
            }
            for t in public["tools"]
        ]
        value = payload(
            "Resolve the customer request using the observed conversation, policy "
            "and tools. Ask for missing fields and confirm proposed changes.",
            documents=[{"id": "policy", "title": "Retail policy", "text": public["policy"]}],
            history=[{"role": h["role"], "text": h["content"]} for h in history],
            tools=tools,
        )
        synchronize(self.model)
        start = time.perf_counter()
        result, trace = infer(self.model, self.tokenizer, value, self.config, adapt=False,
                              state=self.state)
        self.state = {"memory": result.get("memory", []),
                      "procedureIds": result.get("procedureIds", [])}
        self.history = [dict(h) for h in history]
        action = result["action"]
        response = result
        if action in ("call_tool", "confirm") and result.get("tool"):
            try:
                args = (
                    argument_model(result["tool"]).model_validate(result["arguments"]).model_dump()
                )
                proposal = {"tool": result["tool"], "arguments": args}
                if result["tool"] in WRITES:
                    action = "call_tool" if confirmed == digest(proposal) else "confirm"
                response = {"action": action, **proposal}
                if action == "confirm":
                    response["text"] = "Please confirm this proposed change."
            except (ValueError, KeyError, TypeError) as error:
                response = {
                    "action": "speak",
                    "text": "Please provide valid observed arguments for " + result["tool"] + ".",
                    "reason": str(error),
                }
        elif action in ("ask", "confirm"):
            response = {
                "action": "speak",
                "text": result["question"].get(
                    "text", "Please confirm " + result["question"]["field"] + "."
                ),
                "question": result["question"],
            }
        elif action in ("answer", "respond"):
            observed = result.get("answer") or result["fields"][0]["value"]
            response = {
                "action": "speak",
                "text": "Observed information: " + str(observed)
                if observed is not None
                else "Please provide the next required details.",
            }
        else:
            response = {
                "action": "finish",
                "text": "I cannot complete this request with the currently supported information.",
            }
        synchronize(self.model)
        record = {
            "raw": json.dumps(response, ensure_ascii=False),
            "inputTokens": len(self.tokenizer.encode(json.dumps(value), add_special_tokens=False)),
            "outputTokens": 0,
            "latencyMs": (time.perf_counter() - start) * 1000,
            "cacheHit": False,
            "scope": "Structured controller; raw-input token count; no generated assistant tokens",
            "modelTrace": trace,
            "structuredOutput": result,
        }
        return response, record


def run(model, tokenizer, config, generator, orders_config, limit=None):
    from .cli import provenance
    from .order_benchmark import episode

    check(orders_config)
    directory = Path(orders_config["dataset"])
    rows = [r for r in lines(directory / "inputs.jsonl") if r["split"] == "test"]
    rows = rows[:limit] if limit else rows
    labels = {r["id"]: r for r in lines(directory / "labels.jsonl")}
    db = read(directory / "db.json")
    actor, records = Actor(model, tokenizer, config), []
    output = Path(config["output"]) / "orders"
    if next(model.parameters()).is_cuda:
        torch.cuda.reset_peak_memory_stats()
    for row in rows:
        actor.reset()
        records.append(
            episode(row, labels[row["id"]], db, "structured", generator, actor, None, orders_config)
        )
        jsonl(output / "episodes.jsonl", records)
    report = {
        "cases": len(records),
        "taskSuccesses": sum(r["measurement"]["taskSuccess"] is True for r in records),
        "violations": sum(r["measurement"]["policyViolations"] for r in records),
        "assistantGenerations": 0,
        "userAndJudgeModel": generator.config,
        "datasetHash": digest(read(directory / "manifest.json")),
        "provenance": provenance(config),
        "peakGpuBytes": torch.cuda.max_memory_allocated()
        if next(model.parameters()).is_cuda
        else None,
        "scope": "ABCD transfer to simulated retail, without retail fine-tuning; no real-work efficacy claim",
    }
    write(output / "metrics.json", report)
    return report
