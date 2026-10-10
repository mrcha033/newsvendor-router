"""Learn a value correction only after an observed, unresolved request failure."""

import torch
from torch import nn

from .corpus import SLOTS
from .heads import Head
from .io import require
from .structured_value import STATE_FEATURES


def failed_fields(value, state):
    """Latest observed replies only; never predict whether a future reply will arrive."""
    latest = {
        "F" if event["action"] == "demand" else event["action"]: event.get("answer")
        for event in value["history"]
        if event["action"] in (*SLOTS, "demand")
    }
    return tuple(
        slot
        for slot, answer in latest.items()
        if state["state"].get(slot) != "verified"
        and (answer is None or isinstance(answer, str) and answer in ("no_response", "partial"))
    )


class RecoveryHead(nn.Module):
    """Preserve the prior outside observed recovery; learn the action inside it."""

    def __init__(self, base):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.numeric = len(STATE_FEATURES)
        require(base.layers[0].in_features == 256 + self.numeric, "Numeric prior required")
        require(base.layers[-1].out_features == 2, "Value head required")
        with torch.random.fork_rng(devices=[]):
            self.correction = Head(self.numeric, 128, 2)
        nn.init.zeros_(self.correction.layers[-1].weight)
        nn.init.zeros_(self.correction.layers[-1].bias)

    def forward(self, features):
        require(features.shape[-1] == 257 + self.numeric, "Observed recovery flag required")
        state, active = features[..., :-1], features[..., -1:].bool()
        prior = self.base(state)
        correction = self.correction(state[..., -self.numeric :])
        return torch.where(active, prior + correction, prior)
