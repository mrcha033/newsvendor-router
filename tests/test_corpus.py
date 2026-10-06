import copy

import pytest

from newsvendor.construction import atoms, candidates, reference
from newsvendor.corpus import audit, generate, outcome
from newsvendor.evaluation import covers
from newsvendor.io import digest, read
from newsvendor.optimizer import optimal
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


def test_weight_seeds_do_not_change_a_pinned_corpus():
    config = {**CONFIG, "dataSeed": 42}
    assert generate(config) == generate({**config, "seed": 43})


def test_cost_variation_preserves_truth_and_source_families():
    original = generate(CONFIG)
    varied = generate(
        {
            **CONFIG,
            "economics": {"requestMultipliers": [0.25, 1, 4, 10], "holdMultipliers": [0.5, 1, 2]},
        }
    )
    assert audit(original) == audit(varied)
    assert [e["gold"] for e in original] == [e["gold"] for e in varied]
    for family in {e["family"] for e in varied}:
        tasks = [e["input"]["task"] for e in varied if e["family"] == family]
        assert all(t["costs"] == tasks[0]["costs"] and t["hold"] == tasks[0]["hold"] for t in tasks)
    assert any(
        a["input"]["task"]["costs"] != b["input"]["task"]["costs"]
        for a, b in zip(original, varied, strict=True)
    )


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


def test_revised_workload_has_independent_economics_and_no_theta_overlap():
    config = read("configs/full.json")
    episodes = generate(config)
    assert audit(episodes)["counts"] == {"train": 360, "dev": 60, "cal": 60, "test": 120}
    assert len({digest(e["gold"]["theta"]) for e in episodes}) == 120
    train = {digest(e["gold"]["theta"]) for e in episodes if e["split"] == "train"}
    assert all(digest(e["gold"]["theta"]) not in train for e in episodes if e["split"] == "test")
    assert len({e["gold"]["theta"]["c"] / e["gold"]["theta"]["p"] for e in episodes}) == 120
    assert all(len(e["gold"]["theta"]["F"]) == 5 for e in episodes)
    assert all(
        optimal(e["gold"]["theta"], e["input"]["task"]["bounds"])["q"]
        not in e["input"]["task"]["bounds"]
        for e in episodes
    )
    assert all(e["scenario"] not in e["input"]["task"]["sku"] for e in episodes)
    for family in {e["family"] for e in episodes}:
        variants = [e for e in episodes if e["family"] == family]
        assert all(
            e["gold"] == variants[0]["gold"] and e["input"]["task"] == variants[0]["input"]["task"]
            for e in variants
        )
    assert len({digest(e["input"]["task"]["costs"]) for e in episodes}) == 120


def test_diverse_prose_and_forecast_preserve_numerical_evidence():
    episodes = generate(read("configs/pilot.json"))
    for e in episodes:
        if e["scenario"] == "sufficient":
            state = reference(e["input"])
            assert state["valid"] and covers(e["gold"]["theta"], state["omega"])
    episode = next(e for e in episodes if e["scenario"] == "missing_demand")
    input = episode["input"]
    assert not reference(input)["valid"]
    assert any(o["stockout"] for o in input["observations"])
    updated = outcome(input, "demand", episode["gold"]["theta"]["F"])
    assert updated["observations"] == input["observations"]
    assert "not reconstructed lost sales" in updated["docs"][-1]["text"]
    assert reference(updated)["types"]["F"] == "estimate"
    assert reference(updated)["valid"]
    assert covers(episode["gold"]["theta"], reference(updated)["omega"])
