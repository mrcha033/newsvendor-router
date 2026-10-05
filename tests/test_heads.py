import numpy as np
import torch
from torch.nn import functional as F

from newsvendor.heads import Head, train


def test_expected_candidate_huber_gradient_matches_proposal():
    logits = torch.tensor([0.4, -0.7, 0.2], dtype=torch.float64, requires_grad=True)
    errors = F.huber_loss(
        torch.tensor([0.0, 2.0, 8.0], dtype=torch.float64),
        torch.full((3,), 3.0, dtype=torch.float64),
        reduction="none",
    )
    p = logits.softmax(-1)
    objective = (p * errors).sum()
    objective.backward()
    assert torch.allclose(logits.grad, p.detach() * (errors - objective.detach()), atol=1e-12)
    for i in range(3):
        plus, minus = logits.detach().clone(), logits.detach().clone()
        plus[i] += 1e-6
        minus[i] -= 1e-6
        numerical = ((plus.softmax(-1) - minus.softmax(-1)) * errors).sum() / 2e-6
        assert torch.allclose(logits.grad[i], numerical, atol=1e-8)


def test_actual_gradient_updates_reduce_class_and_value_objectives():
    torch.set_num_threads(1)
    torch.manual_seed(5)
    rows = [
        {"x": np.array([x, 1], dtype=np.float32), "y": int(x > 0)} for x in (-2.0, -1.0, 1.0, 2.0)
    ]
    result = train(Head(2, 8, 2), rows, 80, 5, lr=0.03)
    assert result["final"] < result["initial"] * 0.1
    values = [{"x": r["x"], "y": float(r["x"][0]) / 2} for r in rows]
    result = train(Head(2, 8, 1), values, 100, 5, "value", lr=0.02)
    assert result["final"] < result["initial"] * 0.1
