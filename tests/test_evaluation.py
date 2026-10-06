from newsvendor.construction import reference
from newsvendor.corpus import generate, outcome
from newsvendor.evaluation import score

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
    updated = outcome(episode["input"], "v", episode["gold"]["theta"]["v"])
    resolved = score(episode, updated, reference(updated), "handoff")
    assert resolved["authorized"] and not resolved["falseHandoff"]
