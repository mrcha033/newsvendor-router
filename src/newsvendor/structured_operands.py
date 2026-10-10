"""Joint ordered operand selection using observed tokens and source positions."""

import torch
from torch import nn
from torch.nn import functional as fn

from .heads import Head
from .io import require
from .structured_inputs import OPS


def coordinates(view, device):
    origins, rows = {}, []
    for atom in view["atoms"]:
        loc = atom["location"]
        key = loc["kind"], loc["id"]
        origin = origins.setdefault(key, len(origins))
        rows.append(
            [
                origin,
                loc["kind"] == "cell",
                loc.get("row", -1),
                loc.get("column", -1),
                loc["start"],
                loc["end"],
            ]
        )
    return torch.tensor(rows, device=device, dtype=torch.float32)


def layout(first, second):
    origin = first[..., 0] == second[..., 0]
    table = origin & first[..., 1].bool() & second[..., 1].bool()
    row = first[..., 2] == second[..., 2]
    column = first[..., 3] == second[..., 3]
    same_text = origin & (~first[..., 1].bool() | (row & column))
    same_span = same_text & (first[..., 4:] == second[..., 4:]).all(-1)
    return torch.stack(
        [
            origin.float(),
            same_span.float(),
            table.float(),
            (table & row).float(),
            (table & column).float(),
            ((first[..., 2] - second[..., 2]) / 32).clamp(-1, 1) * table,
            ((first[..., 3] - second[..., 3]) / 32).clamp(-1, 1) * table,
            ((first[..., 4] - second[..., 4]) / 4096).clamp(-1, 1) * same_text,
        ],
        -1,
    )


def proposals(view, fields, atoms, output, limit=5):
    """Build candidates from predictions alone; gold operands are never inserted."""
    require(limit > 0 and len(atoms) == len(view["atoms"]) > 0, "Invalid operand candidates")
    count = min(limit, len(atoms))
    left = output["operand1"].float().log_softmax(-1)
    right = output["operand2"].float().log_softmax(-1)
    # Stable ties preserve the original argmax at candidate zero.
    a = left.argsort(dim=-1, descending=True, stable=True)[:, :count]
    b = right.argsort(dim=-1, descending=True, stable=True)[:, :count]
    a = a[:, :, None].expand(-1, -1, count).flatten(1)
    b = b[:, None, :].expand(-1, count, -1).flatten(1)
    op = output["relation"].argmax(-1)
    unary = op == OPS.index("copy")
    b = torch.where(unary[:, None], a, b)
    mask = (~unary[:, None]) | (
        torch.arange(count * count, device=fields.device)[None] % count == 0
    )
    first, second = atoms[a], atoms[b]
    second = second.masked_fill(unary[:, None, None], 0)
    positions = coordinates(view, fields.device)
    context = layout(positions[a], positions[b]).masked_fill(unary[:, None, None], 0)
    marginal = torch.stack([left.gather(1, a), right.gather(1, b)], -1)
    marginal[..., 1] = marginal[..., 1].masked_fill(unary[:, None], 0)
    return {
        "field": fields,
        "first": first,
        "second": second,
        "operation": fn.one_hot(op, len(OPS)).float(),
        "layout": context,
        "marginals": marginal,
        "prior": marginal.sum(-1),
        "mask": mask,
        "pairs": torch.stack([a, b], -1),
    }


class OperandPairs(nn.Module):
    """A zero-initialized correction to the independent operand scores."""

    def __init__(self):
        super().__init__()
        self.field = nn.Linear(256, 64)
        self.atom = nn.Linear(256, 64)
        self.score = Head(64 * 5 + len(OPS) + 8 + 2, 64, 1)
        nn.init.zeros_(self.score.layers[-1].weight)
        nn.init.zeros_(self.score.layers[-1].bias)

    def forward(self, data):
        with torch.autocast(data["field"].device.type, enabled=False):
            query = self.field(data["field"].float())[:, None].expand(
                -1, data["first"].shape[1], -1
            )
            first, second = self.atom(data["first"].float()), self.atom(data["second"].float())
            operation = data["operation"][:, None].expand(-1, first.shape[1], -1)
            inputs = torch.cat(
                [
                    query,
                    first,
                    second,
                    first * second,
                    first - second,
                    operation,
                    data["layout"],
                    data["marginals"],
                ],
                -1,
            )
            correction = self.score(inputs).squeeze(-1)
            return (data["prior"] + correction).masked_fill(~data["mask"], -1e9)


def positive_pairs(pairs, mask, target):
    """Loss-only alternatives; missing gold candidates do not alter the pool."""
    if "operand1" not in target:
        return torch.zeros_like(mask, dtype=torch.bool)
    selected = torch.isin(
        pairs[:, 0].long(), pairs.new_tensor(target["operand1"], dtype=torch.long)
    )
    if "operand2" in target:
        selected &= torch.isin(
            pairs[:, 1].long(), pairs.new_tensor(target["operand2"], dtype=torch.long)
        )
    return selected & mask.bool()
