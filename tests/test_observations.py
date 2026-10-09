import math
from datetime import date, timedelta

import pytest
import torch

from newsvendor import demand, observations, sequence


@pytest.mark.parametrize("family", demand.FAMILIES)
def test_daily_allocation_matches_existing_one_day_likelihood(family):
    sales = torch.tensor(
        [[0.0], [0.0], [0.1], [0.1], [2.0], [2.0], [20.0], [20.0]], dtype=torch.float64
    )
    censored = torch.tensor([[False], [True]] * 4)
    raw = torch.tensor([[-0.7, 0.2, -0.3]] * len(sales), dtype=torch.float64, requires_grad=True)
    expected = demand.nll(family, raw, sales[:, 0], censored[:, 0])
    measured = observations.allocation_nll(family, raw, sales, censored)
    torch.testing.assert_close(measured, expected, rtol=1e-9, atol=1e-9)
    gradient = torch.autograd.grad(measured.sum(), raw, retain_graph=True)[0]
    reference = torch.autograd.grad(expected.sum(), raw)[0]
    torch.testing.assert_close(gradient, reference, rtol=1e-8, atol=1e-8)
    assert torch.isfinite(gradient).all()


@pytest.mark.parametrize("family", demand.FAMILIES)
def test_allocation_observes_zero_days_and_ignores_padding(family):
    raw = torch.tensor([[-0.5, 0.3, -0.2]], dtype=torch.float64, requires_grad=True)
    sales = torch.tensor([[0.2, 0, 0.4]], dtype=torch.float64)
    censored = torch.zeros_like(sales, dtype=torch.bool)
    value = observations.allocation_nll(family, raw, sales, censored)
    log_zero = torch.nn.functional.logsigmoid(raw[:, 0])
    log_active = (-log_zero.expm1()).log()
    pattern = (
        torch.nn.functional.logsigmoid(-raw[:, 0])
        + 2 * log_active
        + log_zero
        - (-(3 * log_zero).expm1()).log()
    )
    density, _ = demand.positive_logs(family, *demand.parameters(family, raw), sales.sum(-1))
    expected = -(pattern + density - sales.sum(-1).log())
    torch.testing.assert_close(value, expected, rtol=1e-10, atol=1e-10)
    padded = torch.cat([sales, torch.zeros((1, 4))], dim=1)
    present = torch.tensor([[True] * 3 + [False] * 4])
    actual = observations.allocation_nll(family, raw, padded, ~present, present)
    torch.testing.assert_close(actual, value)
    zeros = observations.allocation_nll(family, raw, torch.zeros_like(sales), censored)
    torch.testing.assert_close(zeros, -torch.nn.functional.logsigmoid(raw[:, 0]))


@pytest.mark.parametrize("family", demand.FAMILIES)
def test_censored_zeros_supply_no_evidence_and_partial_zeros_remain_observed(family):
    raw = torch.tensor([[-0.5, 0.3, -0.2]], dtype=torch.float64, requires_grad=True)
    empty = torch.zeros((1, 7), dtype=torch.float64)
    loss = observations.allocation_nll(family, raw, empty, torch.ones_like(empty, dtype=torch.bool))
    assert loss.item() == 0
    gradient = torch.autograd.grad(loss.sum(), raw)[0]
    assert gradient.eq(0).all()
    sales = torch.tensor([[0.0, 0.6]], dtype=torch.float64)
    actual = observations.allocation_nll(family, raw, sales, torch.tensor([[False, True]]))
    log_zero = torch.nn.functional.logsigmoid(raw[:, 0])
    log_active = (-log_zero.expm1()).log()
    pattern = (
        torch.nn.functional.logsigmoid(-raw[:, 0])
        + log_zero
        + log_active
        - (-(2 * log_zero).expm1()).log()
    )
    _, tail = demand.positive_logs(family, *demand.parameters(family, raw), sales.sum(-1))
    torch.testing.assert_close(actual, -(pattern + tail), rtol=1e-9, atol=1e-9)


def test_log_probability_normal_quantiles_and_gradients():
    values = torch.tensor(
        [-10000.0, -100.0, -10.0, -1.0, -0.01, -1e-10], dtype=torch.float64, requires_grad=True
    )
    quantiles = observations.normal_isf(values)
    torch.testing.assert_close(torch.special.log_ndtr(-quantiles), values, rtol=1e-12, atol=1e-11)
    assert torch.autograd.gradcheck(observations.normal_isf, (values[:-1],), eps=1e-6)


def test_allocation_integral_matches_independent_exponential_censoring():
    # Seven iid exponentials have a Gamma(7, scale) total and uniform simplex
    # shares. Integrating missing days must recover the exact product likelihood.
    torch.set_num_threads(1)
    theta = 0.8
    lower = torch.tensor([0.3, 1.0, 3.0, 5.0, 9.0] * 7, dtype=torch.float64)
    exact = torch.arange(7).repeat_interleave(5).double()
    missing = (7 - exact)[:, None]
    log_fraction, weights = observations.quadrature(128)
    log_fraction = torch.as_tensor(log_fraction)
    log_weights = torch.as_tensor(weights).log()
    shape = torch.tensor(7.0, dtype=torch.float64)
    tail = torch.special.gammaincc(shape, lower / theta)
    desired = tail[:, None] * log_fraction.exp()
    lo = (lower / theta)[:, None].expand_as(desired).clone()
    hi = torch.full_like(lo, 100.0)
    for _ in range(60):
        middle = (lo + hi) / 2
        below = torch.special.gammaincc(shape, middle) > desired
        lo = torch.where(below, middle, lo)
        hi = torch.where(below, hi, middle)
    log_nodes = (((lo + hi) / 2) * theta).log()
    integrated = observations.log_integral(
        log_nodes, tail.log(), log_weights, lower, exact, missing
    )[:, 0]
    expected = -lower / theta - exact * math.log(theta)
    torch.testing.assert_close(integrated, expected, rtol=3e-4, atol=3e-4)


@pytest.mark.parametrize("family", demand.FAMILIES)
def test_mixed_daily_censoring_has_finite_correct_parameter_gradients(family):
    raw = torch.tensor(
        [[-0.8, 0.3, -0.2], [0.2, -0.1, 0.7]], dtype=torch.float64, requires_grad=True
    )
    sales = torch.tensor(
        [[0.1, 0, 0.2, 0.5, 0, 0.1, 0.1], [0, 0.5, 0, 0, 0, 0, 0]], dtype=torch.float64
    )
    mask = torch.tensor(
        [
            [False, True, False, True, False, False, True],
            [False, True, True, True, True, True, True],
        ]
    )

    def evaluate(value):
        return observations.allocation_nll(family, value, sales, mask)

    assert torch.autograd.gradcheck(evaluate, (raw,), eps=1e-6, atol=2e-5, rtol=1e-4)
    high = observations.allocation_nll(family, raw, sales * 100, torch.ones_like(mask))
    low = observations.allocation_nll(family, raw, sales, torch.ones_like(mask))
    assert torch.all(high >= low)


def observation_cases():
    start = date(2024, 1, 1)
    rows, labels = [], {}
    for index, split in enumerate(("train", "train", "dev")):
        identity = f"series-{index}"
        past = [
            {
                "date": (start + timedelta(days=i)).isoformat(),
                "sales": float((i + index) % 4),
                "stockoutHours": int(i % 3 == 0),
            }
            for i in range(30)
        ]
        rows.append(
            {
                "id": identity,
                "family": identity,
                "split": split,
                "component": "retail",
                "input": {"observations": past},
            }
        )
        labels[identity] = {
            "answer": [float(i % 3) for i in range(7)],
            "complete": [i % 2 == 0 for i in range(7)],
            "dates": [(start + timedelta(days=30 + i)).isoformat() for i in range(7)],
        }
    return rows, labels


def test_daily_observation_targets_never_change_forecast_features():
    rows, labels = observation_cases()
    original = sequence.samples(rows, labels)
    modified = {
        key: {**value, "answer": [100.0] * 7, "complete": [False] * 7}
        for key, value in labels.items()
    }
    changed = sequence.samples(rows, modified)
    assert len(original) == len(changed)
    for before, after in zip(original, changed, strict=True):
        assert before["historyHash"] == after["historyHash"]
        torch.testing.assert_close(before["sequence"], after["sequence"])
        assert before["scale"] == after["scale"]
    assert any(a["dailySales"] != b["dailySales"] for a, b in zip(original, changed, strict=True))


def test_daily_head_reuse_and_source_crossfit_train_the_same_gru():
    torch.manual_seed(42)
    torch.set_num_threads(1)
    rows, labels = observation_cases()
    samples = sequence.samples(rows, labels)
    train = [r for r in samples if r["split"] == "train"]
    dev = [r for r in samples if r["split"] == "dev"]
    model = sequence.DemandEncoder(hidden=8)
    states = [r["sequence"] for r in train]
    out = model(states, [r["horizon"] for r in train], daily=True)
    daily = model(states, [1] * len(train))
    for family in demand.FAMILIES:
        torch.testing.assert_close(out["dailyZero"][family], daily["raw"][family][:, 0])
    config = {
        "observation": "daily_allocation",
        "observationNodes": 64,
        "epochs": 1,
        "selectorEpochs": 1,
        "folds": 2,
        "lr": 0.001,
        "batchSize": 8,
    }
    initial = model.gru.weight_ih_l0.detach().clone()
    report, records = sequence.cross_fit(model, train, dev, config, 42)
    assert not torch.equal(model.gru.weight_ih_l0, initial)
    assert report["observation"] == "daily_allocation" and report["selectedEpoch"] == 1
    assert len(records) == len(train) and all("dailyCensored" in r for r in records)
    for fold in report["folds"]:
        assert not set(fold["fitFamilies"]) & set(fold["heldFamilies"])
    assert torch.isfinite(sequence.measure(model, dev, config=config)).all()
