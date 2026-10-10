"""Paired manager availability is an evaluation condition, never an observed feature."""

import copy
import math
import runpy
from datetime import date, timedelta
from types import SimpleNamespace

import pytest
import torch

from newsvendor import corpus, sequence, structured_forecast, structured_retail
from newsvendor.construction import parameter_record
from newsvendor.io import digest
from newsvendor.structured_rollout import rollout

EVAL = runpy.run_path("scripts/evaluate_responses.py")


def episode():
    row = {
        "id": "retail-fixture",
        "family": "source",
        "split": "dev",
        "component": "retail",
        "input": {
            "observations": [
                {
                    "date": (date(2024, 1, 1) + timedelta(days=i)).isoformat(),
                    "sales": 2,
                    "stockoutHours": 0,
                }
                for i in range(28)
            ]
        },
    }
    labels = {
        row["id"]: {
            "dates": [(date(2024, 1, 29) + timedelta(days=i)).isoformat() for i in range(7)],
            "answer": [2] * 7,
            "complete": [True] * 7,
        }
    }
    return next(
        e
        for e in structured_retail.attach_targets(structured_retail.cases([row]), [row], labels)
        if e["scenario"] == "missing_v_b"
    )


class Router:
    """Grounded fixture for environment correctness, not neural effectiveness."""

    config = {}

    def __init__(self, value, order):
        torch.manual_seed(42)
        model = SimpleNamespace(demand=sequence.DemandEncoder(hidden=8))
        self.forecast = structured_forecast.predict(model, value)
        self.order = order

    def construct(self, value):
        assert not {"missingResponses", "missing_responses", "target", "responses"} & value.keys()
        record = parameter_record(value)
        record["fields"] = []
        return structured_forecast.finish(value, record, self.forecast)

    def allowed(self, value, state):
        return structured_forecast.actions(value, state)

    def choose(self, value, state):
        allowed = self.allowed(value, state)
        return next(
            (a for a in self.order if a in allowed), "handoff" if "handoff" in allowed else "hold"
        )


@pytest.mark.parametrize("split", ["train", "dev"])
def test_question_order_cannot_change_answers_and_exact_expectation_matches_formula(split):
    row = dict(episode(), split=split)
    before = digest(row)
    measurements = []
    routes = [Router(row["input"], order) for order in (("v", "b"), ("b", "v"))]
    for missing in EVAL["conditions"](0.2):
        records = [rollout(row, r, missing_responses=frozenset(missing)) for r in routes]
        expected = {a: None if a in missing else row["responses"][a] for a in ("v", "b")}
        assert all(
            {e["action"]: e["observedResponse"] for e in r["events"][:-1]} == expected
            for r in records
        )
        assert [e["action"] for e in records[0]["events"]][:2] == ["v", "b"]
        assert [e["action"] for e in records[1]["events"]][:2] == ["b", "v"]
        assert records[0]["total"] == pytest.approx(records[1]["total"])
        measurements.append(EVAL["measurement"](records[0]) | {"missingResponses": list(missing)})
    assert digest(row) == before
    costs = sum(row["input"]["task"]["costs"][a] for a in ("v", "b"))
    clean = measurements[0]["metrics"]["terminalOnCompletePeriods"]
    for noise in (0, 0.1, 0.2, 1):
        probs = EVAL["conditions"](noise)
        assert math.fsum(probs.values()) == pytest.approx(1)
        for field in corpus.SLOTS:
            assert math.fsum(p for mask, p in probs.items() if field in mask) == pytest.approx(
                noise
            )
        measured = EVAL["expectations"](measurements, noise)[0]["metrics"]
        expected = (
            (1 - noise) ** 2 * clean + (1 - (1 - noise) ** 2) * row["input"]["task"]["hold"] + costs
        )
        assert measured["totalOnCompletePeriods"] == pytest.approx(expected)
    altered = copy.deepcopy(row)
    altered["target"]["answer"] = [1e6] * 7
    left, right = [
        rollout(e, routes[0], missing_responses=frozenset({"v"})) for e in (row, altered)
    ]
    assert left["events"] == right["events"]


@pytest.mark.parametrize("split", ["cal", "test"])
def test_fixed_response_evaluation_rejects_protected_splits_before_inference(split):
    row = dict(episode(), split=split)
    with pytest.raises(ValueError, match="retail Train/Dev only"):
        rollout(row, None, missing_responses=frozenset())


def test_response_expectation_rejects_unpaired_or_invalid_conditions():
    row = episode()
    with pytest.raises(ValueError, match="financial request channels"):
        rollout(row, None, missing_responses={"F"})
    with pytest.raises(ValueError, match="financial request channels"):
        rollout(row, None, missing_responses="v")
    with pytest.raises(ValueError, match="fixed and random"):
        rollout(row, None, missing_responses=set(), noise=0.1)
    del row["input"]["task"]["forecast"]
    with pytest.raises(ValueError, match="retail Train/Dev only"):
        rollout(row, None, missing_responses=set())
    rows = [
        {
            "key": "one",
            "family": "source",
            "outcomeComplete": False,
            "missingResponses": list(mask),
            "metrics": {"loss": None, "cost": 1},
        }
        for mask in EVAL["conditions"](0)
    ]
    assert EVAL["expectations"](rows, 0.2)[0]["metrics"]["loss"] is None
    with pytest.raises(ValueError, match="Incomplete"):
        EVAL["expectations"](rows[:-1], 0.2)
    with pytest.raises(ValueError, match="Duplicate"):
        EVAL["expectations"](rows + rows[:1], 0.2)
    rows[0]["metrics"]["loss"] = 1
    with pytest.raises(ValueError, match="outcome availability"):
        EVAL["expectations"](rows, 0.2)


def test_decision_cache_uses_observed_state_and_returns_independent_records():
    router = object.__new__(EVAL["CachedRouter"])
    calls = []

    def decide(value, state):
        calls.append((copy.deepcopy(value), copy.deepcopy(state)))
        return {"action": "v" if state["missing"] else "handoff", "values": [value["price"]]}

    router.decision = decide
    router.reset()
    value, state = {"price": 1}, {"missing": True}
    assert router.choose(value, state) == "v"
    router.last_decision["values"][0] = 100
    assert router.choose(value, state) == "v" and router.last_decision["values"] == [1]
    assert len(calls) == 1
    value["price"] = 2
    router.choose(value, state)
    state["missing"] = False
    assert router.choose(value, state) == "handoff"
    assert len(calls) == 3
