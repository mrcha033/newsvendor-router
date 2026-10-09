"""Train response sampling preserves historical evaluation and observed-only state."""

import copy
import random
from collections import defaultdict

import numpy as np
import pytest

from newsvendor import corpus, policy, structured_forecast, structured_retail
from newsvendor.construction import reference
from newsvendor.io import digest, read
from newsvendor.structured_rollout import behavior, collect, rollout


def episode():
    row = next(
        e
        for e in corpus.generate(read("configs/full.json"))
        if e["split"] == "train" and e["scenario"] == "missing_contract"
    )
    row["input"]["task"].update(rho={"v": 0.55}, partial={"v": 0.25}, deadline=3)
    return row


class Router:
    config = {}

    def construct(self, value):
        # This test double checks the collector, not neural model effectiveness.
        assert not any("seed" in k.lower() for k in value)
        return reference(value)

    def allowed(self, value, state):
        return structured_forecast.actions(value, state)

    def choose(self, value, state):
        return behavior(value, state, self)


def test_response_sampling_keeps_legacy_draws_and_common_action_outcomes():
    row = episode()
    current = row["input"]
    action = "v"
    chance = int(digest(row["id"] + ":" + action)[:8], 16) / 2**32
    quality = int(digest(row["id"] + ":quality:" + action)[:8], 16) / 2**32
    expected = (
        None
        if chance >= current["task"]["rho"][action]
        else "partial"
        if quality < current["task"]["partial"][action]
        else row["gold"]["theta"][action]
    )
    assert policy.response(row, current, action, 0) == expected
    answers = [policy.response(row, current, action, 0, seed=i) for i in range(64)]
    assert set(answers) == {None, "partial", row["gold"]["theta"][action]}
    reordered = copy.deepcopy(current)
    reordered["history"].append({"action": "b", "answer": 0})
    assert answers == [policy.response(row, reordered, action, 0, seed=i) for i in range(64)]
    retail = {"split": "train", "family": "source", "cutoffIndex": 28, "responses": {"v": 2}}
    old = random.Random(digest(["source", 28, "v", 0])).random()
    assert structured_retail.response(retail, current, "v", 0.5) == (None if old < 0.5 else 2)
    answers = [structured_retail.response(retail, current, "v", 0.5, seed=i) for i in range(64)]
    assert set(answers) == {None, 2}
    assert answers == [
        structured_retail.response(retail, reordered, "v", 0.5, seed=i) for i in range(64)
    ]


def test_collector_averages_measured_returns_and_keeps_seeds_out_of_observed_inputs():
    row = episode()
    original = digest(row)
    rows, raw = collect([row], Router(), response_samples=8, response_seed=42)
    groups = defaultdict(list)
    for record in raw:
        groups[record["stateHash"], record["forcedAction"]].append(record)
    assert digest(row) == original
    assert any(r["requestCost"] > 0 for r in raw)
    for target in rows:
        assert len(set(target["responseSeeds"])) == 8
        assert target["observedResponseSeed"] not in target["responseSeeds"]
        for action, value in zip(target["actions"], target["values"], strict=True):
            measured = groups[digest(target["input"]), action]
            assert [r["responseSeed"] for r in measured] == target["responseSeeds"]
            np.testing.assert_allclose(
                value,
                np.mean([[r["terminalLoss"], r["requestCost"]] for r in measured], axis=0)
                / target["scale"],
            )
    again, other = collect([row], Router(), response_samples=8, response_seed=42)
    assert digest(rows) == digest(again)
    assert [r["total"] for r in raw] == [r["total"] for r in other]
    changed, _ = collect([row], Router(), response_samples=8, response_seed=43)
    assert rows[0]["responseSeeds"] != changed[0]["responseSeeds"]
    legacy, measured = collect([row], Router())
    assert all("responseSeeds" not in r for r in legacy)
    assert all("responseSeed" not in r for r in measured)
    assert len(measured) == sum(len(r["actions"]) for r in legacy)


@pytest.mark.parametrize("split", ["dev", "test"])
def test_response_resampling_rejects_held_out_partitions(split):
    row = dict(episode(), split=split)
    with pytest.raises(ValueError, match="Train only"):
        rollout(row, Router(), response_seed=42)
    with pytest.raises(ValueError, match="Train only"):
        policy.response(row, row["input"], "v", 0, seed=42)
    retail = {"split": split, "family": "source", "cutoffIndex": 28, "responses": {"v": 2}}
    with pytest.raises(ValueError, match="Train only"):
        structured_retail.response(retail, row["input"], "v", seed=42)
    with pytest.raises(ValueError, match="Train"):
        collect([row], Router(), split=split, response_seed=42)


@pytest.mark.parametrize("count", [0, -1, 1.5, True])
def test_response_sample_count_is_explicit(count):
    with pytest.raises(ValueError, match="positive integer"):
        collect([episode()], Router(), response_samples=count, response_seed=42)


def test_multiple_samples_cannot_silently_reuse_the_legacy_draw():
    with pytest.raises(ValueError, match="needs a seed"):
        collect([episode()], Router(), response_samples=2)
    with pytest.raises(ValueError, match="Recovery"):
        collect([episode()], Router(), with_values=False, response_samples=2, response_seed=42)
