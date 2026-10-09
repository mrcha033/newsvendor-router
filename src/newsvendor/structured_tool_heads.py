"""Shared role/entity pointers, procedure retrieval, progress and candidate scoring."""

import math

import torch
from torch import nn

from .heads import Head
from .io import require
from .structured_inputs import MODES
from .structured_procedures import STAGES


class ToolHeads(nn.Module):
    def __init__(self):
        super().__init__()
        self.role = nn.Linear(256, 256, bias=False)
        self.entity = nn.Linear(256, 256, bias=False)
        self.procedure = nn.Linear(256, 256, bias=False)
        self.stage = Head(256, 128, STAGES)
        self.rerank = nn.Sequential(nn.Linear(262, 128), nn.GELU(), nn.Linear(128, 1))
        nn.init.zeros_(self.rerank[-1].weight)
        nn.init.zeros_(self.rerank[-1].bias)

    def forward(self, view, fields, actions, tokens, queries, output):
        device = fields.device
        contextual = view.get("contextRerank") and view.get("controllerStart") is not None
        if contextual:
            output["rerankFeatures"] = actions.new_zeros((len(actions), 6))
        if view["roles"]:
            values = queries[view["roleStart"] : view["procedureStart"]]
            logits = self.role(fields) @ values.T / math.sqrt(256)
            mask = torch.tensor(
                [
                    [name in {r["name"] for r in f.get("roles", [])} for name in view["roles"]]
                    for f in view["fields"]
                ],
                device=device,
            )
            output["role"] = logits.masked_fill(~mask, -1e9)
        if view["entities"]:
            values = torch.stack([tokens[e["tokens"]].mean(0) for e in view["entities"]])
            output["entity"] = self.entity(fields) @ values.T / math.sqrt(256)
        else:
            # Never let an unimplemented/empty entity path silently fall back to a span.
            output["mode"] = output["mode"].clone()
            output["mode"][:, MODES.index("entity")] = -1e9
        if not view["procedures"]:
            return
        context = fields[0] + actions.mean(0)
        if "dialogueState" in output:
            context = context + output["dialogueState"]
        values = queries[view["procedureStart"] : view["controllerStart"]]
        output["procedure"] = self.procedure(context) @ values.T / math.sqrt(256)
        if view.get("workflowProgress"):
            require(
                all(len(p["steps"]) < STAGES for p in view["procedures"]),
                "Public tool nodes exceed progress head capacity",
            )
            # Each procedure has its own node distribution. Its observed dialogue
            # and policy representations condition the existing shared head.
            stage_context = context[None] + values
            if view.get("controllerStart") is not None:
                stage_context = stage_context + queries[view["controllerStart"] + len(actions)]
            logits = self.stage(stage_context / 2)
            mask = torch.tensor(
                [
                    [s < len(p["steps"]) or s == STAGES - 1 for s in range(STAGES)]
                    for p in view["procedures"]
                ],
                device=device,
            )
            output["stage"] = logits.masked_fill(~mask, -1e9)
        else:
            output["stage"] = self.stage(context)
        probabilities = output["procedure"].float().softmax(-1)
        stages = output["stage"].float().softmax(-1)
        names = [a["id"].removeprefix("call_tool:") for a in view["actions"]]
        # Construct on the host and transfer once, avoiding thousands of scalar CUDA writes.
        membership = torch.tensor(
            [[float(n in p["steps"]) for n in names] for p in view["procedures"]], device=device
        )
        member = probabilities @ membership
        transition = torch.zeros_like(member)
        if view.get("workflowProgress") or not view.get("procedureContext"):
            # New targets identify public nodes independently of previous call count.
            # Keep the old count-indexed mapping only for legacy reproduction.
            transitions = torch.tensor(
                [
                    [
                        [float(s < len(p["steps"]) and n == p["steps"][s]) for n in names]
                        for s in range(STAGES)
                    ]
                    for p in view["procedures"]
                ],
                device=device,
            )
            transition = (
                torch.einsum("p,ps,psa->a", probabilities, stages, transitions)
                if view.get("workflowProgress")
                else torch.einsum("p,s,psa->a", probabilities, stages, transitions)
            )
        progress = (
            (probabilities * stages.max(-1).values).sum()
            if view.get("workflowProgress")
            else (stages * torch.arange(STAGES, device=device)).sum() / STAGES
        )
        modes, use = output["mode"].float().softmax(-1), output["use"].float().softmax(-1)[:, 1]
        features = []
        for a, action in enumerate(view["actions"]):
            indices = [
                i
                for i, f in enumerate(view["fields"])
                if action["id"] == "call_tool:" + f.get("tool", "")
            ]
            missing = modes[indices, MODES.index("missing")].mean() if indices else member[a] * 0
            conflict = modes[indices, MODES.index("conflict")].mean() if indices else member[a] * 0
            used = use[indices].mean() if indices else member[a] * 0
            features.append(
                torch.stack(
                    [
                        member[a],
                        transition[a],
                        missing,
                        conflict,
                        used,
                        progress,
                    ]
                )
            )
        features = torch.stack(features).to(actions.dtype)
        if contextual:
            # The final candidate scorer runs after the controller has read the
            # conversation; these are entirely self-predicted state features.
            output["rerankFeatures"] = features
            return
        output["rerankDelta"] = self.rerank(torch.cat([actions, features], -1)).flatten()
        output["baseRecovery"] = output["recovery"]
        if view.get("rerankSupervision"):
            # The inference ablation must not disable learning the comparison head.
            output["rerankRecovery"] = output["baseRecovery"] + output["rerankDelta"]
        if view["rerank"]:
            output["recovery"] = output["recovery"] + output["rerankDelta"]

    def rerank_context(self, view, actions, queries, output):
        """Rerank complete candidate probabilities, including the call decision."""
        start = view["controllerStart"]
        context = queries[start : start + len(actions)]
        if "dialogueState" in output:
            context = context + output["dialogueState"]
        delta = (
            self.rerank(torch.cat([actions + context, output["rerankFeatures"]], -1))
            .flatten()
            .float()
        )
        scores, gate, calls = (
            output["controlRecovery"].float(),
            output["callGate"].float(),
            output["callMask"],
        )
        allowed = torch.tensor(view["allowedActions"], device=scores.device, dtype=torch.bool)
        scores = scores.masked_fill(~allowed, -1e9)
        normalizer = torch.where(calls, scores[calls].logsumexp(0), scores[~calls].logsumexp(0))
        prior = scores - normalizer + gate.log_softmax(-1)[calls.long()]
        joint = (prior + delta).masked_fill(~allowed, -1e9)
        output.update(
            baseControlRecovery=output["controlRecovery"],
            baseCallGate=output["callGate"],
            rerankDelta=delta,
            rerankPrior=prior,
            rerankJoint=joint,
        )
        if view["rerank"]:
            output["controlRecovery"] = joint
            output["callGate"] = torch.stack(
                [joint[~calls].logsumexp(0), joint[calls].logsumexp(0)]
            )
