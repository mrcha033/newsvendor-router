import copy
import math
from datetime import date, timedelta

import pytest
import torch

from newsvendor.conventional import (
    CommonPolicy,
    Conventional,
    rule_parameters,
    statistical_forecast,
)
from newsvendor.corpus import answer_doc, document, outcome
from newsvendor.factorial import factorial
from newsvendor.io import read


def value():
    end = date(2026, 1, 29)
    result = {
        "task": {
            "sku": "A",
            "period": "2026-01-29/2026-02-04",
            "quantityUnit": "globally-normalized-sales",
            "forecast": {"start": "2026-01-29", "days": 7, "unit": "globally-normalized-sales"},
            "bounds": [0, 100],
            "hold": 100,
            "costs": {"c": 1, "p": 1, "v": 2, "b": 3},
            "deadline": 3,
        },
        "historySource": {"id": "history-A", "sku": "A", "unit": "globally-normalized-sales"},
        "remaining": 3,
        "history": [],
        "observations": [
            {
                "date": (end - timedelta(days=28 - i)).isoformat(),
                "sales": 1 + i % 4 / 10,
                "stockoutHours": 0,
            }
            for i in range(28)
        ],
    }
    result["docs"] = [
        document(
            result, "c", "Purchase quotation", "Pack price = 60; units per pack = 10.", "buyer"
        ),
        answer_doc(result, "p", 10),
        answer_doc(result, "v", 1, "return"),
        answer_doc(result, "b", 2, "manager"),
    ]
    return result


def test_rule_constructor_reads_sources_not_reference(monkeypatch):
    import newsvendor.construction as construction

    def blocked(*args, **kwargs):
        raise AssertionError("Reference cannot enter construction")

    monkeypatch.setattr(construction, "parameter_record", blocked)
    monkeypatch.setattr(construction, "reference", blocked)
    v = value()
    v["docs"].append(
        document(v, "wrong", "Purchase quotation", "Purchase cost per unit = 99.", version=9)
        | {"sku": "B"}
    )
    v["docs"].append(
        document(v, "old", "Purchase quotation", "Purchase cost per unit = 8.", version=0)
    )
    result = rule_parameters(v)
    assert result["values"] == {"c": 6, "p": 10, "v": 1, "b": 2}
    assert result["links"]["c"] == "c"
    assert result["fields"][0]["expression"]["op"] == "divide"


def test_rule_conflict_and_missing_response_preserve_uncertainty():
    v = value()
    v["docs"].append(answer_doc(v, "v", 2, "other"))
    assert rule_parameters(v)["state"]["v"] == "conflict"
    assert rule_parameters(outcome(v, "v", None))["state"]["v"] == "conflict"
    assert rule_parameters(outcome(v, "v", 3))["values"]["v"] == 3


def test_unselected_preference_is_not_a_fact():
    v = value()
    v["docs"][-1]["text"] = "Recommended additional shortage cost per unit = 2."
    assert "b" not in rule_parameters(v)["values"]


def test_forecast_censoring_changes_estimate_and_has_no_future_access():
    torch.set_num_threads(2)
    settings = read("configs/ai-comparison-v1.json")["baseline"]
    v = value()
    forecast = statistical_forecast(v, settings)
    lower = copy.deepcopy(v)
    for row in lower["observations"]:
        row["stockoutHours"] = 1
    censored = statistical_forecast(lower, settings)
    assert forecast["fit"]["censoredBlocks"] == 0
    assert censored["fit"]["censoredBlocks"] == 4
    assert sum(d * p for d, p in censored["F"]) > sum(d * p for d, p in forecast["F"])
    assert math.isclose(sum(p for _, p in forecast["F"]), 1)
    with pytest.raises(ValueError, match="precede"):
        v["observations"][-1]["date"] = "2026-01-29"
        statistical_forecast(v, settings)


def test_same_question_rule_and_h0_prevents_requests():
    c = Conventional(read("configs/ai-comparison-v1.json")["baseline"])
    v = value()
    state = c.construct(v)
    assert state["valid"] and CommonPolicy(c, False).choose(v, state) == "handoff"
    v["docs"] = [d for d in v["docs"] if d["id"] != "return"]
    state = c.construct(v)
    assert CommonPolicy(c, True).choose(v, state) == "v"
    assert CommonPolicy(c, False).choose(v, state) == "hold"
    assert CommonPolicy(c, False).allowed(v, state) == ["hold"]
    replied = outcome(v, "v", 1)
    after = c.construct(replied)
    assert after["valid"] and after["values"]["v"] == 1


def factorial_rows():
    return [
        {
            "id": str(i),
            "family": str(i % 2),
            "period": str(i),
            "complete": True,
            "inputHash": str(i),
            "A": a,
            "H": h,
            "metrics": {"loss": 100 * i + 2 * a - 3 * h - 5 * a * h},
        }
        for i in range(4)
        for a in (0, 1)
        for h in (0, 1)
    ]


def test_factorial_case_effects_and_interaction():
    result = factorial(factorial_rows(), "loss")
    assert result["fixedEffectCoefficients"] == pytest.approx({"A": 2, "H": -3, "A*H": -5})
    assert result["effects"]["AI_with_questions"]["effect"] == -3
    assert result["effects"]["interaction"]["lower"] == 4


@pytest.mark.parametrize(
    "change", ["missing", "duplicate", "different_input", "one_missing_outcome"]
)
def test_factorial_rejects_unbalanced_or_changed_treatments(change):
    rows = factorial_rows()
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows.append(rows[-1])
    elif change == "different_input":
        rows[-1]["inputHash"] = "changed"
    else:
        rows[-1]["metrics"]["loss"] = None
    with pytest.raises(ValueError):
        factorial(rows, "loss")
