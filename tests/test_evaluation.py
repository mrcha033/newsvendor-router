import copy

import pytest

from newsvendor.construction import reference
from newsvendor.corpus import generate, outcome
from newsvendor.evaluation import authorization, rescore, score
from newsvendor.report import save, summarize

CONFIG = {"seed": 42, "groups": 20, "variants": 5, "budget": 2}


def test_truth_coverage_does_not_authorize_unresolved_conflict():
    episodes = [
        e for e in generate(CONFIG) if e["split"] == "test" and e["scenario"] == "conflicting"
    ]
    assert len(episodes) == 20
    for episode in episodes:
        state = reference(episode["input"])
        state["omega"].append(episode["gold"]["theta"])
        state["valid"] = True  # Simulate the old decoder's erroneous certificate.
        evaluated = score(episode, episode["input"], state, "handoff")
        assert evaluated["coverage"]
        assert not evaluated["coverageErrorHandoff"]
        assert evaluated["authorizationErrorHandoff"] and evaluated["falseHandoff"]
        assert "unresolved-conflict" in evaluated["handoffReasons"]


def test_valid_clarification_removes_conflict_violation():
    episode = next(e for e in generate(CONFIG) if e["scenario"] == "conflicting")
    updated = outcome(episode["input"], "v", episode["gold"]["theta"]["v"])
    evaluated = score(episode, updated, reference(updated), "handoff")
    assert evaluated["authorized"] and not evaluated["falseHandoff"]


def test_censored_sales_and_wrong_quantity_are_separate_violations():
    episode = next(e for e in generate(CONFIG) if e["scenario"] == "missing_demand")
    result = authorization(episode["input"], -1)
    assert set(result["handoffReasons"]) >= {"censored-demand", "quantity"}


def test_retroactive_scoring_preserves_raw_loss_and_coverage():
    episode = next(e for e in generate(CONFIG) if e["scenario"] == "conflicting")
    row = {
        "id": episode["id"],
        "family": episode["family"],
        "method": "learned",
        "result": "handoff",
        "q": 90,
        "total": 0,
        "coverage": True,
        "falseHandoff": False,
        "events": [{"action": "handoff"}],
    }
    original = copy.deepcopy(row)
    result = rescore([row], [episode])[0]
    assert row == original
    assert result["total"] == 0 and result["coverage"]
    assert not result["legacyFalseHandoff"] and result["falseHandoff"]
    row["events"] = [{"action": "v"}, {"action": "handoff"}]
    with pytest.raises(ValueError, match="Recorded response"):
        rescore([row], [episode])


def test_unscored_authorization_cannot_be_reported_as_zero_errors(tmp_path):
    row = {
        "total": 0,
        "regret": 0,
        "requests": 0,
        "result": "handoff",
        "coverage": True,
        "falseHandoff": False,
    }
    assert summarize([row])["authorizationErrorHandoff"] is None
    assert summarize([row])["unresolvedConflictHandoff"] is None
    with pytest.raises(ValueError, match="require rescore"):
        save([row], str(tmp_path), {}, {})
