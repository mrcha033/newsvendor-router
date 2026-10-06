import copy

import numpy as np

from newsvendor.construction import construct, reference, training_rows
from newsvendor.corpus import SLOTS, generate, outcome
from newsvendor.encoder import ACTIONS, CONSTRAINT, DEMAND, EXTRA, POLICY_EXTRA, Embeddings, texts
from newsvendor.io import digest
from newsvendor.policy import actions, features, planner, response, trajectory

CONFIG = {"seed": 42, "groups": 10, "variants": 1, "budget": 2}


def cache_for(episode):
    vocabulary = {*texts(episode["input"]), *SLOTS.values(), *ACTIONS.values(), CONSTRAINT, DEMAND}
    return Embeddings(
        {"dim": 4, "revision": "routing-fixture"},
        {
            text: [
                float(
                    next(
                        (
                            i + 1
                            for i, title in enumerate(
                                (
                                    "Purchase quotation",
                                    "Sales price list",
                                    "Current return contract",
                                    "Manager decision",
                                )
                            )
                            if text.startswith(title)
                        ),
                        0,
                    )
                ),
                0.0,
                0.5,
                -0.5,
            ]
            for text in vocabulary
        },
    )


class Fixed:
    def __init__(self, probabilities):
        self.p = np.array(probabilities)

    def probabilities(self, *_):
        return self.p

    def scores(self, xs):
        return np.zeros((len(xs), 1))


class Evidence:
    def __init__(self, reject_return=False):
        self.reject_return = reject_return

    def probabilities(self, x, *_):
        slot = np.argmax(x[-EXTRA:][25:29]) + 1
        yes = x[0] == slot and not (slot == 3 and self.reject_return)
        return np.array([1 - int(yes), int(yes)])


def model_for(status, reject_return=False):
    return {
        "temperatures": {k: {"temperature": 1} for k in ("type", "state", "evidence")},
        "type": Fixed([0, 0, 1, 0]),
        "state": Fixed(status),
        "evidence": Evidence(reject_return),
        "relation": Fixed([1]),
    }


def test_low_evidence_cannot_erase_contract_conflicts():
    episode = next(e for e in generate(CONFIG) if e["scenario"] == "conflicting")
    pred = construct(episode["input"], model_for([1, 0, 0, 0, 0], True), cache_for(episode))
    assert pred["values"]["c"] == reference(episode["input"])["values"]["c"]
    assert pred["state"]["v"] == "conflict" and "conflict-v" in pred["errors"]
    assert not pred["valid"] and "handoff" not in actions(episode["input"], pred)


def test_constructed_planner_does_not_score_against_reference_semantics(monkeypatch):
    episode = next(e for e in generate(CONFIG) if e["scenario"] == "sufficient")
    input = copy.deepcopy(episode["input"])
    input["remaining"] = 0
    state = reference(input)
    state["q"] = input["task"]["bounds"][1] / 2

    def forbidden(_):
        raise AssertionError("Hidden helper constructor was called")

    monkeypatch.setattr("newsvendor.policy.reference", forbidden)
    result = planner(input, lambda _: state, scoring="constructed")
    from newsvendor.optimizer import regret

    assert result["values"]["handoff"] == max(
        regret(state["q"], theta, input["task"]["bounds"]) for theta in state["omega"]
    )


def test_planner_memo_does_not_mix_information_regimes():
    episode = next(e for e in generate(CONFIG) if e["scenario"] == "sufficient")
    memo = {}
    first = planner(episode["input"], memo=memo)
    second = planner(episode["input"], memo=memo, scoring="constructed")
    assert first["scoring"] == "reference" and second["scoring"] == "constructed"


def test_verified_response_is_consistent_with_accepted_expression():
    episode = next(e for e in generate(CONFIG) if e["scenario"] == "missing_contract")
    input = outcome(episode["input"], "v", max(episode["input"]["task"]["allowed"]["v"]))
    pred = construct(input, model_for([0, 0, 0, 1, 0]), cache_for(episode))
    assert pred["values"]["v"] == reference(input)["values"]["v"]
    assert pred["state"]["v"] == "verified" and pred["valid"]
    assert pred["probabilities"]["v"]["state"][3] == 1


def test_construction_trains_on_public_responses_without_gold():
    episode = next(e for e in generate(CONFIG) if e["scenario"] == "unavailable")
    cache = cache_for(episode)
    rows = training_rows([episode], cache)
    for answer in (*episode["input"]["task"]["allowed"]["v"], None, "partial"):
        key = digest(outcome(episode["input"], "v", answer))
        selected = [r for r in rows["evidence"] if r["stateHash"] == key]
        assert selected and all(r["family"] == episode["family"] for r in selected)
        if isinstance(answer, (int, float)):
            assert any(r["y"] == 1 and r["x"][0] == 3 for r in selected)
    poison = copy.deepcopy(episode)
    poison["gold"]["theta"]["v"] = 1e8
    other = training_rows([poison], cache)
    for name in rows:
        assert [r["y"] for r in rows[name]] == [r["y"] for r in other[name]]


class ActionScores:
    def __init__(self, preferred):
        self.preferred = list(ACTIONS).index(preferred)

    def scores(self, xs):
        return np.array([[-x[-POLICY_EXTRA:][18 + self.preferred]] for x in xs])


def test_other_request_costs_are_visible_to_the_value_model():
    episode = next(e for e in generate(CONFIG) if e["scenario"] == "missing_contract")
    input = episode["input"]
    other = copy.deepcopy(input)
    other["task"]["costs"]["b"] = 120
    cache = cache_for(episode)
    assert planner(input)["values"]["v"] != planner(other)["values"]["v"]
    assert np.array_equal(
        features(input, reference(input), "v", cache, context=False),
        features(other, reference(other), "v", cache, context=False),
    )
    assert not np.array_equal(
        features(input, reference(input), "v", cache),
        features(other, reference(other), "v", cache),
    )


def test_policy_context_handles_rejected_construction_without_values():
    episode = next(e for e in generate(CONFIG) if e["scenario"] == "sufficient")
    model = model_for([1, 0, 0, 0, 0])
    model["evidence"] = Fixed([1, 0])
    cache = cache_for(episode)
    state = construct(episode["input"], model, cache)
    assert not state["omega"] and not state["values"]
    assert np.isfinite(features(episode["input"], state, "hold", cache)).all()


def test_each_constructor_uses_its_own_policy_head():
    episode = next(e for e in generate(CONFIG) if e["scenario"] == "sufficient")
    model = {
        "value": ActionScores("hold"),
        "rule_value": ActionScores("handoff"),
        "repair": Fixed([1]),
        "rule_repair": Fixed([1]),
    }
    cache = cache_for(episode)
    assert trajectory(episode, "learned", model, cache, construction="rules")["result"] == "handoff"
    assert trajectory(episode, "learned", model, cache, constructor=reference)["result"] == "hold"


def test_error_probability_is_independent_of_partial_response_probability():
    episode = next(e for e in generate(CONFIG) if e["scenario"] == "unavailable")
    input = copy.deepcopy(episode["input"])
    input["task"]["rho"]["v"] = 1
    input["task"]["partial"]["v"] = 0.15
    for i in range(1000):
        episode["id"] = str(i)
        quality = int(digest(str(i) + ":quality:v")[:8], 16) / 2**32
        error = int(digest(str(i) + ":noise:v")[:8], 16) / 2**32
        if quality > 0.15 and error < 0.1:
            assert response(episode, input, "v", 0.1) != episode["gold"]["theta"]["v"]
            break
    else:
        raise AssertionError("Missing full response with a deterministic answer error")
