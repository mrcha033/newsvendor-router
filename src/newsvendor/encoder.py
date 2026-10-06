from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from .corpus import SLOTS, outcome, possible
from .io import digest, read, require, write

EXTRA = 32
POLICY_EXTRA = 64
DEMAND = "Demand distribution estimated from complete uncensored historical observations"
CONSTRAINT = (
    "Declared single-period model assumptions, unlimited returns and bounded order quantities"
)
ACTIONS = {
    "v": "Retrieve the applicable return contract and handling fees",
    "b": "Ask the manager to select an additional shortage cost policy",
    "demand": "Ask the analyst for complete uncensored demand history",
    "handoff": "Pass validated parameters to the Newsvendor optimizer",
    "hold": "Hold the order because no valid decision can be executed",
}


@dataclass
class Embeddings:
    config: dict
    values: dict

    @property
    def id(self):
        return digest(self.config)

    def vector(self, text):
        require(text in self.values, f"Missing pretrained embedding: {text[:80]}")
        return np.asarray(self.values[text], dtype=np.float32)

    def pool(self, input):
        return np.mean([self.vector(d["title"] + ": " + d["text"]) for d in input["docs"]], axis=0)


def texts(input):
    result = [d["title"] + ": " + d["text"] for d in input["docs"]]
    for action in ("v", "b", "demand"):
        for value in [*possible(input, action), "partial"]:
            result.extend(
                d["title"] + ": " + d["text"] for d in outcome(input, action, value)["docs"]
            )
    return result


def embed(episodes, config, path=".cache/torch-embeddings.json"):
    saved = read(path) if Path(path).exists() else {}
    values = saved.get("values", {}) if saved.get("id") == digest(config) else {}
    alltexts = sorted(
        {
            *SLOTS.values(),
            DEMAND,
            CONSTRAINT,
            *ACTIONS.values(),
            *(text for e in episodes for text in texts(e["input"])),
        }
    )
    missing = [text for text in alltexts if text not in values]
    if missing:
        tokenizer = AutoTokenizer.from_pretrained(
            config["model"],
            revision=config["revision"],
            cache_dir=".cache/torch-models",
            token=False,
            trust_remote_code=False,
        )
        model = AutoModel.from_pretrained(
            config["model"],
            revision=config["revision"],
            cache_dir=".cache/torch-models",
            token=False,
            trust_remote_code=False,
            use_safetensors=True,
        )
        model.eval().requires_grad_(False)
        with torch.inference_mode():
            for start in range(0, len(missing), 32):
                batch = missing[start : start + 32]
                encoded = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=config["maxLength"],
                    return_tensors="pt",
                )
                # Detect truncation separately; these short templates must retain every evidence span.
                lengths = tokenizer(batch, truncation=False, return_length=True)["length"]
                require(
                    max(lengths) <= config["maxLength"], "Encoder would truncate a source document"
                )
                hidden = model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                pooled = torch.nn.functional.normalize(pooled, dim=1)
                require(pooled.shape[1] == config["dim"], "Encoder dimension mismatch")
                values.update(zip(batch, pooled.cpu().tolist(), strict=True))
                print(f"Encoded {min(start + 32, len(missing))}/{len(missing)} texts", flush=True)
        del model
    write(path, {"id": digest(config), "config": config, "values": values})
    return Embeddings(config, values)
