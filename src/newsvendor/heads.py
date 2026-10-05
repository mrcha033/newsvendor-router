import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Head(nn.Module):
    def __init__(self, dim, hidden, output):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, output))

    def forward(self, x):
        return self.layers(x)

    @torch.inference_mode()
    def scores(self, xs):
        return self(torch.as_tensor(np.asarray(xs), dtype=torch.float32)).cpu().numpy()

    def probabilities(self, x, temperature=1):
        logits = self.scores([x])[0] / temperature
        exp = np.exp(logits - max(logits))
        return exp / exp.sum()


def train(head, rows, epochs, seed, mode="class", lr=0.003, valid=None):
    """Batched autograd training; candidate sets are padded and masked, never flattened as labels."""
    size = len(rows)
    rows = rows + (valid or [])
    rand = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    if mode == "choice":
        width = max(len(r["xs"]) for r in rows)
        dim = len(rows[0]["xs"][0])
        x = torch.zeros(len(rows), width, dim)
        mask = torch.zeros(len(rows), width, dtype=torch.bool)
        values = torch.zeros(len(rows), width)
        for i, row in enumerate(rows):
            n = len(row["xs"])
            x[i, :n] = torch.as_tensor(np.asarray(row["xs"]))
            mask[i, :n] = True
            values[i, :n] = torch.tensor(row["values"]) / row["scale"]
        y = torch.tensor([r["y"] for r in rows], dtype=torch.long)
        actual = torch.tensor([r["value"] / r["scale"] for r in rows]).unsqueeze(-1)
    else:
        x = torch.as_tensor(np.asarray([r["x"] for r in rows]), dtype=torch.float32)
        y = torch.tensor(
            [r["y"] for r in rows],
            dtype=torch.long if mode in ("class", "repair") else torch.float32,
        )
        if mode == "repair":
            mask = torch.tensor([r["mask"] for r in rows], dtype=torch.bool)
    weights = None
    if mode in ("class", "repair"):
        counts = torch.bincount(y[:size], minlength=head.layers[-1].out_features).clamp(min=1)
        weights = size / (len(counts) * counts.float())

    def loss(indices):
        output = head(x[indices])
        if mode == "choice":
            logits = output.squeeze(-1).masked_fill(~mask[indices], -1e9)
            errors = F.huber_loss(
                values[indices], actual[indices].expand_as(values[indices]), reduction="none"
            )
            expected = (logits.softmax(-1) * errors).sum(-1).mean()
            return F.cross_entropy(logits, y[indices]) + 0.1 * expected
        if mode == "value":
            return F.mse_loss(output.squeeze(-1), y[indices])
        if mode == "repair":
            return F.cross_entropy(
                output.masked_fill(~mask[indices], -1e9), y[indices], weight=weights
            )
        targets = F.one_hot(y[indices], output.shape[-1]).float()
        return F.cross_entropy(output, y[indices], weight=weights) + 0.1 * (
            (output.softmax(-1) - targets).square().sum(-1).mean()
        )

    allrows = torch.arange(size)
    held = torch.arange(size, len(rows))
    with torch.no_grad():
        initial = loss(allrows).item()
        best = loss(held).item() if len(held) else math.inf
    saved = {k: v.detach().clone() for k, v in head.state_dict().items()}
    selected = 0
    for epoch in range(epochs):
        permutation = torch.randperm(size, generator=rand)
        for indices in permutation.split(128):
            optimizer.zero_grad(set_to_none=True)
            objective = loss(indices)
            if not torch.isfinite(objective):
                raise ValueError("Non-finite training objective")
            objective.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 5)
            optimizer.step()
        if len(held):
            with torch.no_grad():
                current = loss(held).item()
            if current < best:
                best, selected = current, epoch + 1
                saved = {k: v.detach().clone() for k, v in head.state_dict().items()}
    if len(held):
        head.load_state_dict(saved)
    head.eval()
    with torch.no_grad():
        final = loss(allrows).item()
    return {
        "initial": initial,
        "final": final,
        "epochs": epochs,
        "rows": size,
        "devRows": len(held),
        "selectedEpoch": selected if len(held) else epochs,
        "devLoss": best if len(held) else None,
    }


def calibrate(head, rows):
    logits = head.scores([r["x"] for r in rows])
    y = np.asarray([r["y"] for r in rows])
    best = {"temperature": 1.0, "nll": math.inf}
    for t in np.geomspace(0.25, 8, 60):
        scaled = logits / t
        exp = np.exp(scaled - scaled.max(1, keepdims=True))
        probs = exp / exp.sum(1, keepdims=True)
        nll = float(-np.log(probs[np.arange(len(y)), y].clip(1e-12)).mean())
        if nll < best["nll"]:
            best = {"temperature": float(t), "nll": nll}
    return best


def metrics(head, rows, temperature=1):
    logits = head.scores([r["x"] for r in rows]) / temperature
    exp = np.exp(logits - logits.max(1, keepdims=True))
    p = exp / exp.sum(1, keepdims=True)
    y = np.asarray([r["y"] for r in rows])
    correct = p.argmax(1) == y
    confidence = p.max(1)
    ece = 0.0
    for lo, hi in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:], strict=True):
        selected = (confidence >= lo) & (confidence <= hi if hi == 1 else confidence < hi)
        if selected.any():
            ece += selected.mean() * abs(correct[selected].mean() - confidence[selected].mean())
    return {
        "n": len(y),
        "accuracy": float(correct.mean()),
        "nll": float(-np.log(p[np.arange(len(y)), y].clip(1e-12)).mean()),
        "brier": float(((p - np.eye(p.shape[1])[y]) ** 2).sum(1).mean()),
        "ece": float(ece),
    }
