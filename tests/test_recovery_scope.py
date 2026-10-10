import copy

import torch

from newsvendor.heads import Head
from newsvendor.structured_recovery import RecoveryHead, failed_fields


def test_recovery_requires_latest_failed_reply_and_unresolved_own_state():
    state = {"state": {"b": "unconfirmed", "c": "verified", "F": "candidate"}}
    value = {"history": [], "hiddenAvailability": {"b": False}}
    assert failed_fields(value, state) == ()
    value["history"] = [{"action": "retrieve", "answer": None}]
    assert failed_fields(value, state) == ()
    value["history"] += [{"action": "b", "answer": "no_response"}]
    assert failed_fields(value, state) == ("b",)
    value["history"] += [{"action": "c", "answer": "partial"}]
    assert failed_fields(value, state) == ("b",)  # Existing verified evidence survives.
    value["history"] += [{"action": "b", "answer": 3.0}]
    assert failed_fields(value, state) == ()  # An older failure cannot reactivate it.
    value["history"] += [{"action": "demand", "answer": "partial"}]
    assert failed_fields(value, state) == ("F",)
    state["state"]["F"] = "verified"
    assert failed_fields(value, state) == ()
    value.update(hiddenAvailability={"b": True}, target={"action": "hold"})
    assert failed_fields(value, state) == ()


def test_scoped_training_changes_recovery_scores_but_preserves_normal_prior(tmp_path):
    torch.manual_seed(42)
    base = Head(350, 128, 2)
    saved = copy.deepcopy(base.state_dict())
    head = RecoveryHead(base)
    features = torch.randn(8, 351)
    features[:, -1] = torch.tensor([0, 1] * 4)
    initial = head(features).detach()
    assert torch.equal(initial, base(features[:, :-1]))
    optimizer = torch.optim.AdamW(head.correction.parameters(), lr=0.01)
    target = initial + 1
    for _ in range(10):
        optimizer.zero_grad(set_to_none=True)
        (head(features) - target).square().mean().backward()
        assert all(p.grad is None for p in head.base.parameters())
        optimizer.step()
    after = head(features)
    assert torch.equal(after[::2], initial[::2])
    assert not torch.equal(after[1::2], initial[1::2])
    assert all(torch.equal(v, saved[k]) for k, v in head.base.state_dict().items())
    optimizer.zero_grad(set_to_none=True)
    head(features[::2]).sum().backward()
    assert all(p.grad.count_nonzero() == 0 for p in head.correction.parameters())
    path = tmp_path / "recovery.pt"
    torch.save(head.state_dict(), path)
    restored = RecoveryHead(Head(350, 128, 2))
    restored.load_state_dict(torch.load(path, weights_only=True))
    assert torch.equal(after, restored(features))
