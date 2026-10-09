"""Check numeric visibility in the same Train states as the token-context audit."""

import copy
from collections import Counter, defaultdict
from pathlib import Path

from newsvendor.io import digest, lines, require, write
from newsvendor.structured_value import STATE_FEATURES, state_features

source = Path("results/l40s-research-replay-v1/base/42/value-targets-0.jsonl")
counts = defaultdict(Counter)
for row in lines(source):
    require(row["split"] == "train", "Expected Train state")
    group = "retail" if "forecast" in row["input"]["task"] else "generated"
    before = state_features(row["input"], row["state"], row["actions"])
    for field in ("q", "gamma", "types", "values"):
        state = copy.deepcopy(row["state"])
        if field in ("q", "gamma"):
            if state[field] is None:
                continue
            state[field] += 1
        elif field == "types":
            state["types"]["b"] = "fact" if state["types"].get("b") != "fact" else "preference"
        else:
            if not isinstance(state["values"].get("p"), (int, float)):
                continue
            state["values"]["p"] *= 1.01
        after = state_features(row["input"], state, row["actions"])
        counts[group][field + "_tested"] += 1
        counts[group][field + "_same"] += int(before == after)
        require(before != after, "Own-state perturbation remains invisible")
write(
    "results/research-checks/numeric-state-visibility-verified.json",
    {
        "scope": "Train-only visibility diagnostic, not a counterfactual forecast or effectiveness test",
        "testUsed": False,
        "devUsed": False,
        "source": str(source),
        "sourceHash": digest(source.read_bytes()),
        "scriptHash": digest(Path(__file__).read_bytes()),
        "features": STATE_FEATURES,
        "counts": dict(counts),
    },
)
print(dict(counts))
