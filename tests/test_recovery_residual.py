"""The correction can learn under a saturated prior without changing that prior."""

import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from study_recovery_residual import ResidualHead

from newsvendor.heads import Head
from newsvendor.structured_critic import value_loss
from newsvendor.structured_value import constrain


def test_residual_learns_own_state_without_changing_saturated_prior():
    torch.manual_seed(2)
    prior = Head(350, 128, 2)
    with torch.no_grad():
        prior.layers[0].weight[:, :256].fill_(2)
        prior.layers[0].weight[:, 256:].zero_()
    saved = copy.deepcopy(prior.state_dict())
    head = ResidualHead(saved, 42)
    x = torch.ones(2, 350)
    x[1, 256:] = 0
    assert torch.equal(head(x), prior(x))
    assert torch.equal(prior.layers[0](x).tanh(), torch.ones(2, 128))
    costs = {"fixed": [[1, 0], [0, 0.03]], "learned": [[0, 0], [1, 1]]}
    target = torch.tensor([[1, 0], [1, 0.03]])
    optimizer = torch.optim.AdamW([p for p in head.parameters() if p.requires_grad], lr=0.01)
    before = value_loss(head, [(x, target, costs)]).item()
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        value_loss(head, [(x, target, costs)]).backward()
        assert all(p.grad is None for p in head.base.parameters())
        optimizer.step()
    assert value_loss(head, [(x, target, costs)]).item() < before
    assert all(torch.equal(v, saved[k]) for k, v in head.base.state_dict().items())
    assert constrain(torch.nn.functional.softplus(head(x)), costs)[0].tolist() == [1, 0]


def test_saved_residual_restores_predictions_and_trainable_parameters(tmp_path):
    head = ResidualHead(Head(350, 128, 2).state_dict(), 42)
    with torch.no_grad():
        head.correction.layers[-1].bias.add_(0.2)
    features = torch.randn(5, 350)
    path = tmp_path / "head.pt"
    torch.save(head.state_dict(), path)
    restored = ResidualHead(head.base.state_dict(), 3)
    restored.load_state_dict(torch.load(path, weights_only=True))
    assert torch.equal(head(features), restored(features))
    assert sum(p.numel() for p in restored.parameters() if p.requires_grad) == 12418
