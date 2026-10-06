from collections import Counter, defaultdict

from .construction import construct, reference
from .corpus import SLOTS, audit
from .policy import choose, planner, policy_heads, states


def inspect(episodes, model, cache, split="dev"):
    """Diagnose public branches, without reading hidden truth or weighting them as real traffic."""
    audit(episodes)
    selected = [e for e in episodes if e["split"] == split]
    counts, routing = defaultdict(Counter), {}
    for episode in selected:
        for input in states(episode["input"], reference):
            ref, pred = reference(input), construct(input, model, cache)
            answer = input["history"][-1]["answer"] if input["history"] else None
            kind = (
                "initial"
                if not input["history"]
                else answer
                if answer in ("partial", "no_response")
                else "complete"
            )
            count = counts[kind]
            count["states"] += 1
            for slot in SLOTS:
                count["slots"] += 1
                count["statusCorrect"] += pred["state"][slot] == ref["state"][slot]
                if slot in ref["values"]:
                    count["supportedSlots"] += 1
                    count["supportedMissed"] += pred["values"].get(slot) != ref["values"][slot]
                if ref["state"][slot] == "conflict":
                    count["conflicts"] += 1
                    count["conflictsLost"] += pred["state"][slot] != "conflict"
    for name, builder in (
        ("learned", lambda input: construct(input, model, cache)),
        ("rules", reference),
    ):
        regrets, memo = [], {}
        policy = policy_heads(model, name)
        for episode in selected:
            for input in states(episode["input"], builder):
                planned = planner(input, builder, memo, scoring="constructed")
                action = choose(
                    input, planned["state"], "learned", policy, cache, constructor=builder
                )
                regrets.append(
                    (planned["values"][action] - planned["value"]) / input["task"]["hold"]
                )
        routing[name] = {
            "states": len(regrets),
            "meanNormalizedDecisionRegret": sum(regrets) / max(1, len(regrets)),
            "optimalChoiceRate": sum(r <= 1e-8 for r in regrets) / max(1, len(regrets)),
        }
    return {
        "split": split,
        "families": sorted({e["family"] for e in selected}),
        "scope": "Uniform reachable public branches; rule-derived diagnostics, not traffic frequencies",
        "construction": dict(counts),
        "policy": routing,
    }
