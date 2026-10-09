"""Measure Train numeric-feature visibility after the configured BF16 linear-input cast."""

import copy
from collections import Counter, defaultdict
from pathlib import Path

import torch

from newsvendor.io import digest, lines, require, write
from newsvendor.structured_value import state_features

source = Path("results/l40s-research-replay-v1/base/42/value-targets-0.jsonl")
counts = defaultdict(Counter)
for row in lines(source):
    require(row["split"] == "train", "Expected Train state")
    group = "retail" if "forecast" in row["input"]["task"] else "generated"
    before = torch.tensor(
        state_features(row["input"], row["state"], row["actions"]), dtype=torch.bfloat16
    )
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
        after = torch.tensor(
            state_features(row["input"], state, row["actions"]), dtype=torch.bfloat16
        )
        counts[group][field + "_tested"] += 1
        counts[group][field + "_same"] += int(torch.equal(before, after))
write(
    "results/research-checks/numeric-state-precision-verified.json",
    {
        "scope": "Train-only feature quantization diagnostic; not an end-to-end GPU or effectiveness test",
        "testUsed": False,
        "devUsed": False,
        "source": str(source),
        "sourceHash": digest(source.read_bytes()),
        "scriptHash": digest(Path(__file__).read_bytes()),
        "dtype": "bfloat16",
        "counts": dict(counts),
    },
)
print(dict(counts))
