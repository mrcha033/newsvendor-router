"""Build fixed laboratory tasks and cache actual frozen-model advice on observed paths."""

import argparse
import copy
import hashlib
import itertools
import json
import random
import sys
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch

from newsvendor.bundle import load_bundle
from newsvendor.cli import provenance
from newsvendor.corpus import answer_doc, document, outcome
from newsvendor.io import digest, require, write
from newsvendor.optimizer import optimal
from newsvendor.structured_rollout import ResearchRouter
from newsvendor.structured_tool_eval import weights_hash

CONDITIONS = [
    "sufficient",
    "missing_v",
    "sufficient",
    "missing_c",
    "missing_v",
    "conflict_v",
    "wrong_sku",
    "wrong_period",
    "older_version",
    "refund_order",
]


def build_tasks(seed=842719):
    """Independent controlled cases. Future demand never enters document/model construction."""
    observed, environments = [], {}
    for index, condition in enumerate(CONDITIONS):
        rng = random.Random(f"human-newsvendor-v1:{seed}:{index}")
        end = date(2026, 8, 1) + timedelta(days=7 * index)
        sku = f"LAB-{index + 1:02}"
        daily = [[float(d), 0.2] for d in range(1 + index % 3, 6 + index % 3)]
        counts = {0.0: 1.0}
        for _ in range(7):
            after = Counter()
            for total, probability in counts.items():
                for d, p in daily:
                    after[total + d] += probability * p
            counts = after
        F = sorted([d, p] for d, p in counts.items())
        c = float(rng.randint(5, 9))
        p = c + (4.0 if index % 2 == 0 else 12.0)
        v, b = float(rng.randint(1, int(c) - 2)), float(2 + index % 4)
        task = {
            "sku": sku,
            "period": f"{end.isoformat()}/{(end + timedelta(days=6)).isoformat()}",
            "forecast": {"start": end.isoformat(), "days": 7, "unit": "globally-normalized-sales"},
            "quantityUnit": "globally-normalized-sales",
            "bounds": [0, 70],
            "hold": 400.0,
            "costs": {"c": 4.0, "p": 4.0, "v": 4.0},
            "deadline": 2,
            "decision": "expected_loss",
            "tolerance": None,
        }
        history = []
        for day in range(35):
            quantity = rng.choice(daily)[0]
            stocked_out = index in (5, 8) and day % 6 == 0
            history.append(
                {
                    "date": (end - timedelta(days=35 - day)).isoformat(),
                    "sales": min(quantity, 2) if stocked_out else quantity,
                    "stockoutHours": 3 if stocked_out else 0,
                    "discount": 1.0,
                    "holidayFlag": 0,
                    "activityFlag": 0,
                    "precipitation": 0.0,
                    "temperature": 20.0,
                    "humidity": 50.0,
                    "wind": 1.0,
                }
            )
        value = {
            "task": task,
            "docs": [],
            "observations": history,
            "history": [],
            "remaining": 2,
            "historySource": {"id": sku, "sku": sku, "unit": "globally-normalized-sales"},
        }
        value["docs"] = [
            document(
                value,
                "quote",
                "Purchase quotation",
                f"Pack price = {10 * c:g}; units per pack = 10.",
                "buyer",
            ),
            answer_doc(value, "p", p, "price"),
            answer_doc(value, "v", v, "return"),
            answer_doc(value, "b", b, "manager"),
        ]
        for doc in value["docs"]:
            doc["version"] = 1
        if condition.startswith("missing_"):
            remove = {"c": "quote", "v": "return"}[condition[-1]]
            value["docs"] = [d for d in value["docs"] if d["id"] != remove]
        elif condition == "conflict_v":
            alternative = answer_doc(value, "v", v + 1, "return-alternative")
            alternative["version"] = 1
            value["docs"].append(alternative)
        elif condition in ("wrong_sku", "wrong_period", "older_version"):
            extra = document(
                value,
                "irrelevant",
                "Purchase quotation",
                f"Pack price = {10 * (c + 4):g}; units per pack = 10.",
                "buyer",
                2,
            )
            if condition == "wrong_sku":
                extra["sku"] = "ANOTHER-PRODUCT"
            elif condition == "wrong_period":
                extra["period"] = "2025-01-01/2025-01-07"
            else:
                extra["version"] = 0
            value["docs"].insert(0, extra)
        elif condition == "refund_order":
            value["docs"][2]["text"] = (
                f"Handling fee per unit = 1; refund per unit = {v + 1:g}. Unlimited returns."
            )
        task_id = f"practice-{index + 1}" if index < 2 else f"task-{index - 1}"
        observed.append(
            {"id": task_id, "practice": index < 2, "family": f"human-lab-{index}", "input": value}
        )
        theta = {"c": c, "p": p, "v": v, "b": b, "F": F}
        target = optimal(theta, task["bounds"])
        # Experimental responses are stored separately and revealed only after requests.
        environments[task_id] = {
            "theta": theta,
            "responses": {"c": c, "p": p, "v": v},
            "condition": condition,
            "optimal": target,
            "demand": sum(rng.choice(daily)[0] for _ in range(7)),
        }
    return observed, environments


def observed_path(value, environment, path):
    current = copy.deepcopy(value)
    for action in path:
        current = outcome(current, action, environment["responses"][action])
    return current


def summary(value, state):
    forecast = state.get("forecast")
    F = forecast["F"] if forecast else []

    def quantile(level):
        cumulative = 0.0
        for d, p in F:
            cumulative += p
            if cumulative >= level:
                return d
        return F[-1][0] if F else None

    return {
        "parameters": [
            {
                "name": s,
                "value": state["values"].get(s),
                "state": state["state"].get(s),
                "source": state["links"].get(s),
            }
            for s in ("c", "p", "v", "b")
        ],
        "forecast": {"mean": sum(d * p for d, p in F), "low": quantile(0.1), "high": quantile(0.9)}
        if F
        else None,
        "q": state["q"] if state["valid"] else None,
        "missing": state.get("missing", []),
        "valid": state["valid"],
        "inputHash": digest(value),
        "stateHash": digest(state),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    root = Path(args.output)
    require(not root.exists(), "Preserve fixed study material")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    model, tokenizer, config, manifest = load_bundle(args.bundle, args.device)
    model.eval().requires_grad_(False)
    require(
        manifest["weightsHash"]
        == "e01f07f8ab6325411975733303a277c95d5c914170fd833d18c452ab4bd105dd",
        "Use the selected frozen model",
    )
    router = ResearchRouter(model, tokenizer, config["encoder"] | {"parameterDecoding": "scoped"})
    tasks, environments = build_tasks()
    root.mkdir(parents=True)
    write(root / "tasks.json", tasks)
    write(root / "environments.json", environments)
    advice, states = {}, {}
    for task in tasks:
        advice[task["id"]], states[task["id"]] = {}, {}
        router.cache_states()
        for size in range(3):
            for path in itertools.permutations(("c", "p", "v"), size):
                current = observed_path(task["input"], environments[task["id"]], path)
                state = router.construct(current)
                key = ",".join(path)
                advice[task["id"]][key] = summary(current, state)
                states[task["id"]][key] = state
        print(json.dumps({"task": task["id"], "states": len(advice[task["id"]])}), flush=True)
    require(
        weights_hash(model.state_dict()) == manifest["weightsHash"], "Preparation changed weights"
    )
    write(root / "advice.json", advice)
    write(root / "model-states.json", states)
    write(
        root / "manifest.json",
        {
            "version": 1,
            "scope": "Prepared laboratory materials; zero human observations",
            "humanParticipants": 0,
            "tasks": len(tasks),
            "practiceTasks": 2,
            "mainTasks": 8,
            "adviceStates": sum(map(len, advice.values())),
            "seed": 842719,
            "source": provenance({}),
            "weightsHash": manifest["weightsHash"],
            "bundleManifest": manifest,
            "scriptHash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "files": {
                p.name: {
                    "bytes": p.stat().st_size,
                    "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                }
                for p in root.iterdir()
                if p.is_file()
            },
            "selection": "Fixed seed and conditions before model advice; no case selection by AI accuracy or treatment effect",
            "sourceLabels": "Generated controlled laboratory materials, separate from protected Test and holdout",
        },
    )


if __name__ == "__main__":
    main()
