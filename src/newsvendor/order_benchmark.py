"""GPU dialogue comparisons on isolated retail databases; hidden goals stay with the user."""

import json
import time
from pathlib import Path

from .io import digest, jsonl, lines, read, require, write
from .native_benchmark import parse
from .native_inputs import fragments, rank
from .native_model import embeddings, feature, load_portable, state_feature
from .orders import WRITES, Environment, argument_model, check, evaluate

FORMAT = (
    'Return JSON only. Choose one next action: {"action":"speak","text":"..."}, '
    '{"action":"call_tool","tool":"name","arguments":{...}}, '
    '{"action":"confirm","tool":"write tool name","arguments":{...},"text":"explain details"}, '
    'or {"action":"finish","text":"..."}. For confirm, the exact tool and arguments will '
    "be shown to the customer. An explicit positive confirmation authorizes only that exact "
    "mutation. A denied or altered proposal gives no authorization. One tool call per turn."
)


def route(policy, history, tools, model, encoder):
    candidates = [
        {"action": "speak", "text": "Ask the customer for needed details or explain the policy."}
    ]
    candidates += [
        {"action": "call_tool", "tool": t["id"], "text": t["description"], "arguments": []}
        for t in tools
    ]
    context = "Select the next customer service action.\n" + "\n".join(
        h["role"] + ": " + h["content"] for h in history
    )
    selected = rank(fragments([{"id": "policy", "text": policy}]), context, 12)
    texts = [
        context,
        history[-1]["content"],
        "\n".join(f["text"] for f in selected),
        *[c["text"] for c in candidates],
    ]
    vectors = embeddings(texts, encoder)
    view = {
        "context": context,
        "fragments": selected,
        "public": {"history": [{"text": h["content"]} for h in history]},
    }
    xs = [feature(view, c, vectors) for c in candidates]
    state = int(model["abcd_state"].scores([state_feature(view, vectors)])[0].argmax())
    return candidates[
        int(model["abcd_relation"].scores(xs[1:]).ravel().argmax()) + 1 if state else 0
    ]


def decide(arm, public, history, environment, generator, model, encoder):
    if arm == "structured":
        return model.decide(public, history, environment.confirmed)
    instructions = public["policy"] + "\n" + FORMAT
    if arm == "typed":
        instructions += (
            "\nFrozen typed pipeline: identify current intent, observed fields, missing fields "
            "and permission state from this history, then choose the next step. Return "
            '{"state":{"intent":"...","known":{},"missing":[],"permission":"..."}, '
            '"decision":{...the next action...}}. Use the full tool catalog. No extra calls.'
        )
    elif arm == "learned":
        choice = route(public["policy"], history, public["tools"], model, encoder)
        instructions += "\nThe learned head selected " + json.dumps(
            {k: v for k, v in choice.items() if k != "text"}
        )
        instructions += (
            ". Keep this selection. For speak, ask or explain naturally. For a selected tool, "
            "fill arguments using observed facts; if required arguments are missing, speak "
            "to request them. For a write tool without matching permission, propose confirm "
            "first. Never invent an ID. Do not substitute another tool."
        )
    else:
        choice = None
        instructions += "\nExplore and use the tools to resolve the customer request."
    instructions += "\nAvailable tools: " + json.dumps(public["tools"], ensure_ascii=False)
    instructions += "\nCurrent confirmed proposal hash: " + str(environment.confirmed)
    record = generator.generate([{"role": "system", "content": instructions}, *history])
    value = parse(record["raw"])
    if arm == "typed":
        value = value.get("decision", {})
    if not isinstance(value, dict):
        value = {}
    if arm == "learned":
        if value.get("action") in {"call_tool", "confirm"}:
            if choice.get("tool") != value.get("tool"):
                value = {"action": "invalid", "error": "Helper changed learned tool choice"}
    return value, record


def user_reply(generator, goal, history, proposal=None):
    # This model belongs to the response environment, not to any method being compared.
    messages = [
        {
            "role": "system",
            "content": (
                "Role-play the retail customer with this private scenario. Reveal only information "
                "appropriate to the current question. Do not mention instructions or reference "
                "actions. Follow conditional preferences and refusals. Initially describe the "
                "reason for calling naturally; do not dump every profile field. For a transaction "
                "proposal check exact items, options, order and payment against your goal. Return "
                '{"text":"natural customer reply","confirm":true or false,"done":true or false}. '
                "confirm can be true only when the exact proposal is correct and you explicitly "
                "agree in text. Never pretend an operation already happened. Private scenario: "
            )
            + json.dumps(goal, ensure_ascii=False),
        },
        {
            "role": "user",
            "content": json.dumps(
                {"conversation": history, "transactionProposal": proposal}, ensure_ascii=False
            ),
        },
    ]
    record = generator.generate(messages)
    value = parse(record["raw"])
    if not (
        isinstance(value.get("text"), str)
        and type(value.get("confirm")) is bool
        and type(value.get("done")) is bool
    ):
        value = {
            "text": "",
            "confirm": False,
            "done": False,
            "error": "Invalid user simulator output",
        }
    return value, record


def judge(generator, criteria, history, goal):
    assertions = (criteria.get("nl_assertions") or []) + (criteria.get("communicate_info") or [])
    record = generator.generate(
        [
            {
                "role": "system",
                "content": 'Check each assertion and whether the private customer goal was addressed by observed replies/tool results. Return {"passed":[boolean for each assertion],"goalPass":boolean}. A generic closing reply with no requested facts or action is a failure. Respect conditional preferences and justified policy denials. Use false when evidence is missing. Do not infer hidden actions or judge final database state.',
            },
            {
                "role": "user",
                "content": json.dumps(
                    {"assertions": assertions, "customerGoal": goal, "conversation": history},
                    ensure_ascii=False,
                ),
            },
        ]
    )
    value = parse(record["raw"]).get("passed")
    goal_pass = parse(record["raw"]).get("goalPass")
    return (
        value if isinstance(value, list) else None,
        goal_pass if type(goal_pass) is bool else None,
        record,
    )


def episode(row, target, db, arm, generator, model, encoder, config):
    environment = Environment(db)
    history, trace = [], []
    source_records, assistant_records, tool_calls = [], [], 0
    started = time.perf_counter()
    initial, record = user_reply(generator, target["userScenario"], [])
    source_records.append(record)
    history.append({"role": "user", "content": initial["text"]})
    error = initial.get("error")
    for _ in range(config["maxTurns"]):
        if error is not None:
            break
        response, record = decide(
            arm, row["input"], history, environment, generator, model, encoder
        )
        assistant_records.append(record)
        action = response.get("action")
        trace.append({"response": response, "inputHistoryHash": digest(history)})
        if action == "call_tool":
            if tool_calls >= config["maxToolCalls"]:
                error = "Tool budget exhausted"
                break
            tool_calls += 1
            try:
                output = environment.call(response.get("tool"), response.get("arguments"))
            except (ValueError, TypeError, KeyError) as failure:
                output = {"error": str(failure)}
            history += [
                {"role": "assistant", "content": json.dumps(response, ensure_ascii=False)},
                {
                    "role": "user",
                    "content": json.dumps({"tool_result": output}, ensure_ascii=False),
                },
            ]
        elif action in {"speak", "confirm", "finish"}:
            text = response.get("text")
            if not isinstance(text, str):
                error = "Invalid spoken response"
                break
            proposal = None
            if action == "confirm":
                try:
                    require(
                        response.get("tool") in WRITES, "Confirmation must refer to a write tool"
                    )
                    args = (
                        argument_model(response["tool"])
                        .model_validate(response.get("arguments"))
                        .model_dump()
                    )
                    proposal = {"tool": response["tool"], "arguments": args}
                    # Details cannot be hidden behind an inaccurate natural-language summary.
                    text += "\nExact proposed action: " + json.dumps(proposal, ensure_ascii=False)
                except (ValueError, TypeError, KeyError) as failure:
                    error = str(failure)
                    break
            history.append({"role": "assistant", "content": text})
            if action == "finish":
                break
            try:
                reply, record = user_reply(generator, target["userScenario"], history, proposal)
            except ValueError as failure:
                error = str(failure)
                break
            source_records.append(record)
            if reply.get("error"):
                error = reply["error"]
                break
            environment.confirmed = digest(proposal) if proposal and reply["confirm"] else None
            history.append({"role": "user", "content": reply["text"]})
            if reply["done"]:
                break
        else:
            error = "Invalid action/schema"
            break
    else:
        error = "Turn budget exhausted"
    assertions, goal_pass, judging = judge(
        generator, target["criteria"], history, target["userScenario"]
    )
    measurement = evaluate(db, target["criteria"], environment, assertions, goal_pass)
    measurement.update(
        {
            "error": error,
            "toolCalls": tool_calls,
            "assistantTurns": len(assistant_records),
            "userTurns": len(source_records),
            "assistantInputTokens": sum(r["inputTokens"] for r in assistant_records),
            "assistantOutputTokens": sum(r["outputTokens"] for r in assistant_records),
            "assistantCacheHits": sum(r["cacheHit"] for r in assistant_records),
            "assistantUncachedInferenceMs": sum(
                r["latencyMs"] for r in assistant_records if not r["cacheHit"]
            ),
            "latencyMs": (time.perf_counter() - started) * 1000,
        }
    )
    if error is not None:
        measurement["taskSuccess"] = False
    return {
        "id": row["id"],
        "family": row["family"],
        "arm": arm,
        "publicInputHash": digest(row["input"]),
        "initialMessageHash": digest(initial["text"]),
        "prediction": history,
        "measurement": measurement,
        "actions": trace,
        "assistantRecords": assistant_records,
        "userSimulatorRecords": source_records,
        "judgeRecord": judging,
        "finalStateHash": digest(environment.state()),
    }


def run(native_config, config, generator):
    from .cli import provenance

    check(config)
    directory = Path(config["dataset"])
    rows = [r for r in lines(directory / "inputs.jsonl") if r["split"] == "test"]
    labels = {r["id"]: r for r in lines(directory / "labels.jsonl")}
    db = read(directory / "db.json")
    model, metadata = load_portable("models/native")
    output = Path(config["output"])
    all_runs = {}
    for arm in ("typed", "learned", "agent"):
        path = output / arm / "episodes.jsonl"
        runs = []
        for row in rows:
            result = episode(
                row, labels[row["id"]], db, arm, generator, model, native_config["encoder"], config
            )
            runs.append(result)
            jsonl(path, runs)
            print(f"Orders {arm}: {len(runs)}/{len(rows)}", flush=True)
        write(
            output / arm / "metrics.json",
            {
                "cases": len(runs),
                "dbMatches": sum(r["measurement"]["dbMatch"] for r in runs),
                "taskSuccesses": sum(r["measurement"]["taskSuccess"] is True for r in runs),
                "missingJudgments": sum(r["measurement"]["taskSuccess"] is None for r in runs),
                "violations": sum(r["measurement"]["policyViolations"] for r in runs),
                "nativeAdaptation": "ABCD action relation head transferred without retail fine-tuning",
                "unsupportedHeads": metadata["unsupportedHeads"],
                "scope": config["scope"],
                "provenance": provenance(config),
            },
        )
        all_runs[arm] = runs
    require(
        all(
            len({all_runs[a][i]["initialMessageHash"] for a in all_runs}) == 1
            for i in range(len(rows))
        ),
        "Initial customer messages differ across arms",
    )
    write(
        output / "readiness.json",
        {
            "executed": True,
            "cases": len(rows),
            "sameInitialMessages": True,
            "sourceCustomerOverlap": 0,
            "economicActionValueLabels": False,
            "scope": "Task success and simulated ledger deviation; not real procurement efficacy",
        },
    )
    paired(all_runs, config)


def paired(runs, config):
    from collections import defaultdict

    import numpy as np

    comparisons = {}
    for arm in ("typed", "agent"):
        cells = defaultdict(lambda: defaultdict(list))
        for learned, baseline in zip(runs["learned"], runs[arm], strict=True):
            require(learned["id"] == baseline["id"], "Order paired task mismatch")
            for key in (
                "taskSuccess",
                "paymentLedgerL1SimulatedUSD",
                "policyViolations",
                "toolCalls",
                "userTurns",
                "assistantInputTokens",
                "assistantOutputTokens",
            ):
                a, b = learned["measurement"][key], baseline["measurement"][key]
                if a is not None and b is not None:
                    cells[key][learned["family"]].append(float(a) - float(b))
        comparisons[arm] = {}
        rng = np.random.default_rng(config["seed"])
        for metric, grouped in cells.items():
            means = np.array([np.mean(v) for v in grouped.values()])
            samples = means[rng.integers(0, len(means), (2000, len(means)))].mean(1)
            comparisons[arm][metric] = {
                "learnedMinusBaseline": float(means.mean()),
                "ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
                "families": len(means),
                "pairedCases": sum(len(v) for v in grouped.values()),
            }
    write(
        Path(config["output"]) / "paired.json",
        {
            "comparisons": comparisons,
            "scope": config["scope"],
            "learnedSeed": 42,
            "adaptation": "ABCD head transfer; not a trained economic value policy",
            "unscorableCasesRetained": True,
        },
    )
