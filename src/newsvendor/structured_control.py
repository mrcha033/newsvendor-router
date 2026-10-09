"""Contextual action candidates and a separately supervised call decision."""

import math

import torch
from torch import nn

from .heads import Head
from .io import require
from .native_inputs import rank
from .structured_inputs import query_tokens


def encode(value, actions, procedures, tokenizer, config, selected_ids=(), *, turns=False):
    """Encode observed dialogue and every allowed schema candidate, never target labels."""
    role_text = "\n".join(f"{h['role']}: {h['text']}" for h in value["history"])
    tools = {t["id"]: t for t in value["tools"]}
    candidates = []
    for action in actions:
        tool = tools.get(action["id"].removeprefix("call_tool:"), {})
        text = action["id"] + ": " + action["text"]
        slots = tool.get("argumentSlots", list(tool.get("parameters", {}).get("properties", {})))
        if slots:
            text += ". Argument roles: " + ", ".join(slots)
        candidates.append(query_tokens(tokenizer, text)[:config.get("controllerActionTokens", 48)])
    ranked = rank(procedures, role_text, len(procedures)) if procedures else []
    policy = sorted(ranked, key=lambda p: p["id"] not in selected_ids)[:config.get("controllerProcedures", 3)]
    policy_key = "context" if config.get("procedureContext") else "text"
    policy_ids = query_tokens(tokenizer, "\n".join(p[policy_key] for p in policy))
    policy_limit = config.get("controllerPolicyTokens", 384)
    if config.get("procedureContext"):
        # Fail visibly rather than silently removing later alternatives/conditions.
        require(len(policy_ids) <= policy_limit, "Public action conditions exceed policy budget")
    policy_ids = policy_ids[:policy_limit]
    length = config.get("controllerLength", 2048)
    available = length - len(policy_ids) - sum(len(c) + 1 for c in candidates) - 4
    require(available >= 128, "Controller candidates leave too little dialogue context")
    tool_count = sum(h["role"] == "tool" for h in value["history"])
    header = f"Observed tool results: {tool_count}.\n"
    text = header + role_text
    tokenized = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True) if turns else None
    history = tokenized["input_ids"] if turns else query_tokens(tokenizer, text)
    retained = list(range(len(history)))
    truncated = len(history) > available
    if truncated:
        # Keep the initial request and the newest observations when the dialogue is long.
        first = min(128, available // 4)
        retained = retained[:first] + retained[-(available - first):]
        history = [history[i] for i in retained]
    ids = [tokenizer.cls_token_id, *history, tokenizer.sep_token_id, *policy_ids, tokenizer.sep_token_id]
    spans = []
    for candidate in candidates:
        start = len(ids)
        ids += [*candidate, tokenizer.sep_token_id]
        spans.append((start, start + len(candidate)))
    require(len(ids) <= length, "Controller sequence exceeded its declared budget")
    if not turns:
        return ids, spans, truncated
    offsets = tokenized["offset_mapping"]
    layout, cursor = [], len(header)
    for index, turn in enumerate(value["history"]):
        end = cursor + len(turn["role"] + ": " + turn["text"])
        original = {i for i, (lo, hi) in enumerate(offsets) if lo < end and hi > cursor}
        positions = [i + 1 for i, source in enumerate(retained) if source in original]
        if positions:
            layout.append({"historyIndex": index, "role": turn["role"],
                           "start": min(positions), "end": max(positions) + 1,
                           "partial": len(positions) != len(original)})
        cursor = end + 1
    return ids, spans, truncated, layout


class History(nn.Module):
    """Observed turns and self-predicted past tool types; labels are loss-only."""

    def __init__(self):
        super().__init__()
        self.role = nn.Embedding(5, 16)
        self.tool = nn.Linear(256, 256, bias=False)
        self.value = nn.Linear(256, 64, bias=False)
        self.sequence = nn.GRU(336, 128, batch_first=True)
        self.project = nn.Linear(128, 256, bias=False)
        nn.init.zeros_(self.project.weight)

    def forward(self, view, queries, candidates, output):
        layout = view["controllerTurns"]
        states = queries[view["historyStart"]:view["historyStart"] + len(layout)]
        tools = candidates[view["historyActions"]]
        scores = self.tool(states) @ tools.T / math.sqrt(256)
        output["pastTools"] = scores
        role_ids = [{"user": 0, "assistant": 1, "tool": 2, "system": 3}.get(t["role"], 4) for t in layout]
        roles = torch.tensor(role_ids, device=states.device)
        expected = self.value(scores.softmax(-1) @ tools)
        expected = expected * (roles == 2)[:, None]
        features = torch.cat([states, self.role(roles).to(states.dtype), expected], -1)
        _, hidden = self.sequence(features[None])
        return self.project(hidden[-1, 0])


class Controller(nn.Module):
    def __init__(self, residual=False, auxiliary=False, history=False):
        super().__init__()
        require(not auxiliary or residual, "Auxiliary controller loss requires residual composition")
        self.residual = residual
        self.auxiliary = auxiliary
        self.action = Head(256, 128, 1)
        self.call = Head(256, 128, 2)
        if history:
            self.history = History()

    def forward(self, view, queries, output):
        start = view["controllerStart"]
        count = len(view["actions"])
        states = queries[start:start + count]
        summary = queries[start + count]
        if "dialogueState" in output or (hasattr(self, "history") and view.get("controllerTurns")):
            correction = output["dialogueState"] if "dialogueState" in output else self.history(view, queries, states, output)
            states, summary = states + correction, summary + correction
        output["controlRecovery"] = self.action(states).flatten()
        output["callGate"] = self.call(summary)
        output["callMask"] = torch.tensor([a["id"].startswith("call_tool:") for a in view["actions"]], device=states.device)
        if self.auxiliary:
            output["contextRecovery"] = output["controlRecovery"]
            output["contextCallGate"] = output["callGate"]
        if self.residual:
            # Retain learned decisions; contextual scores learn corrections to them.
            prior = output["recovery"].float()
            mask = output["callMask"]
            output["controlRecovery"] = prior + output["controlRecovery"]
            output["callGate"] = output["callGate"].float() + torch.stack([
                prior[~mask].logsumexp(0), prior[mask].logsumexp(0),
            ])


def selection(output, allowed):
    scores = output["controlRecovery"].detach()
    permitted = torch.zeros_like(output["callMask"], dtype=torch.bool)
    permitted[allowed] = True
    group = (output["callMask"] == output["callGate"].argmax().bool()) & permitted
    selected = torch.where(group.any(), group, permitted)
    return int(scores.masked_fill(~selected, -torch.inf).argmax())
