import copy

import pytest

from newsvendor.construction import atoms, candidates, reference
from newsvendor.corpus import audit, generate, outcome
from newsvendor.policy import actions, planner

CONFIG = {"seed": 42, "groups": 20, "variants": 5, "budget": 2}


def test_source_splits_and_variants_preserve_truth():
    episodes = generate(CONFIG)
    assert episodes == generate(CONFIG)
    assert audit(episodes)["counts"] == {"train": 360, "dev": 60, "cal": 60, "test": 120}
    assert len({e["input"]["task"]["sku"] for e in episodes}) == 120
    for family in {e["family"] for e in episodes}:
        variants = [e for e in episodes if e["family"] == family]
        assert all(e["gold"] == variants[0]["gold"] for e in variants)
    poisoned = copy.deepcopy(episodes)
    poisoned[1]["split"] = "train" if poisoned[0]["split"] != "train" else "test"
    with pytest.raises(ValueError, match="crosses"):
        audit(poisoned)


def test_gold_poison_cannot_change_public_predictions():
    episode = next(e for e in generate(CONFIG) if e["scenario"] == "missing_contract")
    before = reference(episode["input"])
    action = planner(episode["input"])["action"]
    episode["gold"]["theta"]["v"] = episode["gold"]["theta"]["b"] = 1e9
    assert reference(episode["input"]) == before
    assert planner(episode["input"])["action"] == action


def test_provenance_conflicts_and_partial_response():
    episodes = generate(CONFIG)
    stale = next(e for e in episodes if any(d["id"] == "stale" for d in e["input"]["docs"]))
    assert "stale" not in reference(stale["input"])["links"].values()
    for doc in stale["input"]["docs"]:
        for atom in atoms(doc):
            lo, hi = atom["span"]
            assert float(doc["text"][lo:hi]) == atom["value"]
    assert all(c["unit"] == "currency/unit" for c in candidates(stale["input"], "c"))
    conflict = next(e for e in episodes if e["scenario"] == "conflicting")
    assert reference(conflict["input"])["state"]["v"] == "conflict"
    assert "handoff" not in actions(conflict["input"], reference(conflict["input"]))
    partial = outcome(conflict["input"], "v", "partial")
    assert reference(partial)["state"]["v"] == "conflict"
    absent = next(
        e
        for e in episodes
        if e["scenario"] == "unavailable" and e["input"]["task"]["rho"]["b"] == 0
    )
    assert "b" not in actions(absent["input"], reference(absent["input"]))


def test_return_update_removes_unnecessary_manager_query():
    episode = next(
        e
        for e in generate(CONFIG)
        if e["scenario"] == "missing_contract" and e["id"].endswith("-0")
    )
    value = max(episode["input"]["task"]["allowed"]["v"])
    updated = outcome(episode["input"], "v", value)
    assert reference(updated)["gamma"] == pytest.approx(0)
    assert planner(updated)["action"] == "handoff"


def test_censored_history_and_partial_receipt_cannot_certify_demand():
    episode = next(e for e in generate(CONFIG) if e["scenario"] == "missing_demand")
    assert not reference(episode["input"])["valid"]
    assert not reference(outcome(episode["input"], "demand", "partial"))["valid"]
    assert reference(outcome(episode["input"], "demand", episode["gold"]["theta"]["F"]))["valid"]
