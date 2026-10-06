import copy
import math
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
        if input["task"].get("demandResponse") == "forecast":
            next["docs"].append(
                document(
                    next,
                    "response-demand",
                    "Demand forecast",
                    "Estimated demand; this is not reconstructed lost sales.\nUnits | Probability\n"
                    + "\n".join(f"{d:g} | {p:g}" for d, p in value),
                    "analyst",
                )
            )
            return next
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


def diverse(seed, scenario, group):
    """Independent source-level economics; variants share one bundle and one theta."""
    rand = random.Random(f"{seed}:{scenario}:{group}:diverse-v1")
    c = round(rand.uniform(4, 25), 4)
    p = round(c + rand.uniform(3, 22), 4)
    v = rand.choice((0, round(c * rand.uniform(0.15, 0.8), 4)))
    b = rand.choice((0, round(rand.uniform(1, 30), 4)))
    high = rand.randint(80, 200)
    support = sorted(rand.sample(range(1, high), 5))
    counts = [1] * 5
    for _ in range(5):
        counts[rand.randrange(5)] += 1
    other = counts.copy()
    donor = next(i for i in range(5) if other[i] > 1)
    other[donor] -= 1
    other[(donor + 1) % 5] += 1
    F = [[d, n / 10] for d, n in zip(support, counts, strict=True)]
    alt = [[d, n / 10] for d, n in zip(support, other, strict=True)]
    return c, p, v, b, high, F, alt, rand


def prose(input, pack, style):
    """Render observed numeric facts as prose, retaining the legacy controlled source schema."""
    for doc in input["docs"]:
        from .construction import atoms

        values = {a["label"].lower(): a["value"] for a in atoms(doc)}
        if doc["title"] == "Purchase quotation":
            price = values["pack price"]
            doc["text"] = (
                f"Each carton contains {pack} units and is priced at USD {price:.12g} per carton."
                if style % 2
                else f"A box of {pack} items costs USD {price:.12g}. Freight is included."
            )
        elif doc["title"] == "Sales price list":
            price = values["selling price per unit"]
            doc["text"] = f"The retail price is USD {price:g} per unit."
        elif doc["title"] == "Current return contract":
            refund = values["refund per unit"]
            fee = values["handling fee per unit"]
            doc["text"] = (
                f"Unsold units earn a USD {refund:g} refund each, less a USD {fee:g} handling fee per unit. Unlimited returns."
            )
        elif doc["title"] == "Manager decision":
            chosen = values["selected additional shortage cost per unit"]
            doc["text"] = (
                f"Selected policy: I choose an additional shortage cost of USD {chosen:g} per unmet unit."
            )


def generate(config):
    data_seed = config.get("dataSeed", config["seed"])
    rand = random.Random(data_seed)
    groups, variants = config["groups"], config["variants"]
    economics = config.get("economics", {})
    mode = config.get("workload", "legacy")
    require(mode in ("legacy", "diverse"), "Unknown controlled workload")
    require(
        config.get("language", "equations") in ("equations", "prose"), "Unknown document language"
    )
    for key in ("requestMultipliers", "holdMultipliers"):
        choices = economics.get(key, [1])
        require(
            choices and all(math.isfinite(v) and v > 0 for v in choices), "Invalid cost multipliers"
        )
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
            if mode == "diverse":
                c, p, v, b, high, F, alt, business = diverse(data_seed, scenario, g)
            truth = alt if scenario == "missing_demand" and g % 2 else F
            economic_rand = random.Random(f"{data_seed}:{scenario}:{g}:economics")
            multipliers = {
                a: economic_rand.choice(economics.get("requestMultipliers", [1]))
                for a in ("v", "b", "demand")
            }
            hold_multiplier = economic_rand.choice(economics.get("holdMultipliers", [1]))
            business_task = {}
            if mode == "diverse":
                business_task = {
                    "costs": {
                        a: round(business.uniform(2, 120) * multipliers[a], 4)
                        for a in ("v", "b", "demand")
                    },
                    "hold": round(business.uniform(100, 1000) * hold_multiplier, 4),
                    "rho": {a: round(business.uniform(0.25, 1), 2) for a in ("v", "b", "demand")},
                    "partial": {
                        a: round(business.uniform(0, 0.25), 2) for a in ("v", "b", "demand")
                    },
                    "deadline": business.choice((1, 2, 3)),
                    "demandResponse": "forecast",
                }
            for j in range(variants):
                input = {
                    "task": {
                        "sku": f"{scenario}-{g}"
                        if mode == "legacy"
                        else "SKU-" + digest(f"{data_seed}:{scenario}:{g}")[:10],
                        "period": "next-day",
                        "allowed": {
                            "v": sorted({0, v, round(c * 0.8, 4)})
                            if mode == "diverse"
                            else [0, 4 * scale],
                            "b": sorted({0, b, 30}) if mode == "diverse" else [0, 8 * scale],
                            "F": [F, alt],
                        },
                        "bounds": [0, high],
                        "decision": "minimax",
                        "tolerance": None,
                        "costs": {
                            a: cost * scale * multipliers[a]
                            for a, cost in (("v", 10), ("b", 20), ("demand", 15))
                        },
                        "hold": 400 * scale * hold_multiplier,
                        "rho": {"v": 1, "b": 1, "demand": 1},
                        "partial": {"v": 0, "b": 0, "demand": 0},
                        "deadline": 3,
                    },
                    "docs": [],
                    "observations": [],
                    "history": [],
                    "remaining": config["budget"],
                }
                if mode == "diverse":
                    input["task"].update(copy.deepcopy(business_task))
                pack = 5 + g % 6
                fields = [f"Pack size = {pack}", f"Pack price = {c * pack:.12g}"]
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
                            "Additional shortage cost options per unit = "
                            + " or ".join(f"{cost:g}" for cost in input["task"]["allowed"]["b"])
                            + ". "
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
                    for k, value in enumerate((0, max(input["task"]["allowed"]["v"]))):
                        d = answer_doc(input, "v", value, f"contract-{k}")
                        d["version"] = 1
                        input["docs"].append(d)
                    if g % 2:
                        input["docs"].append(answer_doc(input, "b", b, "manager"))
                sampled = [d for d, prob in truth for _ in range(round(prob * 10))]
                for k in range(10):
                    complete = not (scenario == "missing_demand" and k % 3 == 0)
                    input["observations"].append(
                        {
                            "date": f"history-{k}",
                            "demand": sampled[k]
                            if complete or mode == "legacy"
                            else sampled[k] // 2,
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
                if config.get("language") == "prose":
                    prose(input, pack, j)
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
            "generator": "controlled-English-documents-v4",
            "config": config,
            "split": audit(episodes),
            "inputHash": digest([e["input"] for e in episodes]),
            "labelHash": digest([e["gold"] for e in episodes]),
            "humanReviewed": 0,
        },
    )
    return episodes
