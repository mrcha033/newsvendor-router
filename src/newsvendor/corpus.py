import copy
import random

from .io import digest, jsonl, require, write

SCENARIOS = (
    "sufficient",
    "missing_contract",
    "missing_demand",
    "unset_preference",
    "conflicting",
    "unavailable",
)
SLOTS = {
    "c": "Per-unit purchase cost",
    "p": "Per-unit selling price",
    "v": "Net return refund after handling fees",
    "b": "Additional shortage cost selected by the manager",
}
KINDS = ("fact", "estimate", "preference", "assumption")
STATUSES = ("verified", "candidate", "unconfirmed", "conflict", "unavailable")


def document(input, id, title, text, role="system", version=1, complete=True):
    return {
        "id": id,
        "title": title,
        "text": text,
        "sku": input["task"]["sku"],
        "period": input["task"]["period"],
        "version": version,
        "role": role,
        "complete": complete,
    }


def answer_doc(input, slot, value, id="response"):
    if slot == "v":
        return document(
            input,
            id,
            "Current return contract",
            f"Refund per unit = {value + 1:g}; handling fee per unit = 1. Unlimited returns.",
            "supplier",
            3,
        )
    if slot == "b":
        return document(
            input,
            id,
            "Manager decision",
            f"Selected additional shortage cost per unit = {value:g}.",
            "manager",
            3,
        )
    raise ValueError(f"Unknown response slot {slot}")


def possible(input, action):
    return input["task"]["allowed"]["F" if action == "demand" else action]


def outcome(input, action, value):
    next = copy.deepcopy(input)
    next["history"].append({"action": action, "answer": "no_response" if value is None else value})
    next["remaining"] = max(0, next["remaining"] - 1)
    if value is None:
        return next
    if value == "partial":
        texts = {
            "v": "Return terms received; the refund and handling fee are incomplete.",
            "b": "Manager received the question but has not selected a cost policy.",
            "demand": "Analyst returned a partial history with unresolved stockouts.",
        }
        titles = {
            "v": "Current return contract",
            "b": "Manager decision",
            "demand": "Partial demand response",
        }
        next["docs"].append(
            document(next, f"partial-{action}", titles[action], texts[action], complete=False)
        )
    elif action == "demand":
        observations = []
        for d, p in value:
            observations.extend(
                [
                    {"date": f"history-{i}", "demand": d, "complete": True, "stockout": False}
                    for i in range(round(p * 10))
                ]
            )
        next["observations"] = observations
    else:
        title = "return" if action == "v" else "Manager decision"
        next["docs"] = [d for d in next["docs"] if title not in d["title"]]
        next["docs"].append(answer_doc(next, action, value, f"response-{action}"))
    return next


def generate(config):
    rand = random.Random(config["seed"])
    groups, variants = config["groups"], config["variants"]
    require(groups >= 10 and groups % 10 == 0, "Use a multiple of 10 source groups per scenario")
    episodes = []
    for scenario in SCENARIOS:
        ids = list(range(groups))
        rand.shuffle(ids)
        splits = {
            g: "train"
            if i < groups * 0.6
            else "dev"
            if i < groups * 0.7
            else "cal"
            if i < groups * 0.8
            else "test"
            for i, g in enumerate(ids)
        }
        for g in range(groups):
            scale, high = 1 + (g % 5) * 0.25, 80 + 10 * (g % 5)
            c, p = 6 * scale, 10 * scale
            v, b = rand.choice((0, 4 * scale)), rand.choice((0, 8 * scale))
            F, alt = [[0, 0.6], [high, 0.4]], [[0, 0.4], [high, 0.6]]
            truth = alt if scenario == "missing_demand" and g % 2 else F
            for j in range(variants):
                input = {
                    "task": {
                        "sku": f"{scenario}-{g}",
                        "period": "next-day",
                        "allowed": {"v": [0, 4 * scale], "b": [0, 8 * scale], "F": [F, alt]},
                        "bounds": [0, high],
                        "decision": "minimax",
                        "tolerance": None,
                        "costs": {"v": 10 * scale, "b": 20 * scale, "demand": 15 * scale},
                        "hold": 400 * scale,
                        "rho": {"v": 1, "b": 1, "demand": 1},
                        "partial": {"v": 0, "b": 0, "demand": 0},
                        "deadline": 3,
                    },
                    "docs": [],
                    "observations": [],
                    "history": [],
                    "remaining": config["budget"],
                }
                pack = 5 + g % 6
                fields = [f"Pack size = {pack}", f"Pack price = {c * pack:g}"]
                if j % 2:
                    fields.reverse()
                separator = ("; ", "\n", " | ", ". ", "\n- ")[(g + j) % 5]
                input["docs"].extend(
                    [
                        document(
                            input, "quote", "Purchase quotation", separator.join(fields), "buyer"
                        ),
                        document(
                            input,
                            "price",
                            "Sales price list",
                            f"Selling price per unit = {p:g}.",
                            "sales",
                        ),
                        document(
                            input,
                            "policy",
                            "Allowed cost policies",
                            f"Additional shortage cost options per unit = 0 or {8 * scale:g}. "
                            "No policy has been selected.",
                        ),
                    ]
                )
                if scenario in ("sufficient", "missing_demand", "unset_preference"):
                    input["docs"].append(
                        answer_doc(
                            input, "v", 0 if scenario == "unset_preference" else v, "contract"
                        )
                    )
                if (
                    scenario in ("sufficient", "missing_demand")
                    or scenario == "missing_contract"
                    and g % 2
                ):
                    input["docs"].append(answer_doc(input, "b", b, "manager"))
                if scenario == "conflicting":
                    for k, value in enumerate((0, 4 * scale)):
                        d = answer_doc(input, "v", value, f"contract-{k}")
                        d["version"] = 1
                        input["docs"].append(d)
                    if g % 2:
                        input["docs"].append(answer_doc(input, "b", b, "manager"))
                for k in range(10):
                    complete = not (scenario == "missing_demand" and k % 3 == 0)
                    input["observations"].append(
                        {
                            "date": f"history-{k}",
                            "demand": 0 if k < round(truth[0][1] * 10) else high,
                            "complete": complete,
                            "stockout": not complete,
                        }
                    )
                if scenario == "unavailable":
                    input["task"]["rho"].update(b=0 if g % 2 == 0 else 0.6, v=0.8)
                    input["task"]["partial"].update(b=0.15, v=0.15)
                    input["task"]["deadline"] = 1 if g % 3 == 0 else 3
                if splits[g] == "test" and g % 2 == 0:
                    stale = document(
                        input,
                        "stale",
                        "Sales price list",
                        f"Selling price per unit = {p * 1.5:g}.",
                        "sales",
                    )
                    stale["period"] = "last-year"
                    input["docs"].append(stale)
                rand.shuffle(input["docs"])
                family = f"{scenario}-{g:02}"
                episodes.append(
                    {
                        "id": f"{family}-{j}",
                        "family": family,
                        "scenario": scenario,
                        "split": splits[g],
                        "input": input,
                        "gold": {
                            "theta": {
                                "c": c,
                                "p": p,
                                "v": 0 if scenario == "unset_preference" else v,
                                "b": b,
                                "F": truth,
                            },
                            "reviewed": False,
                        },
                    }
                )
    return episodes


def audit(episodes):
    sources, counts = {}, {}
    for e in episodes:
        require(
            e["family"] not in sources or sources[e["family"]] == e["split"],
            f"Source bundle crosses splits: {e['family']}",
        )
        sources[e["family"]] = e["split"]
        counts[e["split"]] = counts.get(e["split"], 0) + 1
    return {
        "counts": counts,
        "families": len(sources),
        "hash": digest([[e["id"], e["family"], e["split"]] for e in episodes]),
    }


def save(config, directory="data/synthetic"):
    episodes = generate(config)
    jsonl(
        f"{directory}/inputs.jsonl", [{k: v for k, v in e.items() if k != "gold"} for e in episodes]
    )
    jsonl(f"{directory}/labels.jsonl", [{"id": e["id"], "gold": e["gold"]} for e in episodes])
    write(
        f"{directory}/manifest.json",
        {
            "generator": "controlled-English-documents-v2",
            "config": config,
            "split": audit(episodes),
            "inputHash": digest([e["input"] for e in episodes]),
            "labelHash": digest([e["gold"] for e in episodes]),
            "humanReviewed": 0,
        },
    )
    return episodes
