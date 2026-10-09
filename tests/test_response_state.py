"""Observed manager replies update state without replacing the model's raw prediction."""

import copy

import pytest

from newsvendor import corpus, structured_forecast
from newsvendor.construction import matches, parameter_record
from newsvendor.io import digest, read


@pytest.fixture
def value():
    return copy.deepcopy(
        next(
            e["input"]
            for e in corpus.generate(read("configs/full.json"))
            if e["split"] == "train" and e["scenario"] == "sufficient"
        )
    )


def missing(value, slot):
    value["docs"] = [d for d in value["docs"] if not matches(d, slot)]
    record = parameter_record(value)
    record["fields"] = [
        {
            "name": name,
            "field": name,
            "state": record["state"][name],
            "type": record["types"][name],
            "value": record["values"].get(name),
            "mode": "missing" if name == slot else "compute",
        }
        for name in corpus.SLOTS
    ]
    return record


@pytest.mark.parametrize("slot", corpus.SLOTS)
@pytest.mark.parametrize("answer", [None, "no_response"])
def test_missing_reply_updates_only_its_field_and_preserves_raw_predictions(value, slot, answer):
    record = missing(value, slot)
    raw = copy.deepcopy(record["fields"])
    value["history"].append({"action": slot, "answer": answer})
    original = digest(value)
    numeric = {k: copy.deepcopy(record[k]) for k in ("values", "links", "types", "expressions")}
    structured_forecast.copy_responses(value, record)
    assert record["state"][slot] == "unavailable"
    assert all(record[k] == v for k, v in numeric.items())
    assert all(record["state"][s] == "verified" for s in corpus.SLOTS if s != slot)
    assert record["responseStates"][slot]["responseHash"] == digest(value["history"][0])
    structured_forecast.resolved_fields(record, None)
    assert record["rawFields"] == raw
    field = next(f for f in record["fields"] if f["name"] == slot)
    assert field["value"] is None and field["state"] == "unavailable"
    assert field["reason"] == "no_response" and field["responseState"]["historyIndex"] == 0
    assert digest(value) == original
    changed = copy.deepcopy(value)
    changed["task"].update(rho={slot: 1}, partial={slot: 1}, allowed={slot: [99999]})
    changed["gold"] = {"theta": {slot: 99999}}
    other = parameter_record(value)
    other["state"][slot] = "unconfirmed"
    structured_forecast.copy_responses(changed, other)
    assert other["state"][slot] == record["state"][slot]
    assert other["responseStates"] == record["responseStates"]


@pytest.mark.parametrize("scope", [None, "sku", "period"])
def test_partial_status_needs_observed_material_for_the_current_task(value, scope):
    record = missing(value, "v")
    value = corpus.outcome(value, "v", "partial")
    doc = next(d for d in value["docs"] if d["id"] == "partial-v")
    if scope:
        doc[scope] = "other"
    structured_forecast.copy_responses(value, record)
    assert record["state"]["v"] == ("unconfirmed" if scope else "candidate")
    assert ("v" in record["responseStates"]) == (scope is None)
    if scope is None:
        assert record["responseStates"]["v"]["partialSources"] == {doc["id"]: digest(doc)}
        # A later unanswered request does not erase the material already received.
        value = corpus.outcome(value, "v", None)
        structured_forecast.copy_responses(value, record)
        assert record["state"]["v"] == "candidate"


def test_failed_reply_never_invalidates_complete_or_conflicting_evidence(value):
    record = parameter_record(value)
    before = copy.deepcopy(record)
    value = corpus.outcome(value, "v", None)
    structured_forecast.copy_responses(value, record)
    assert all(record[k] == v for k, v in before.items()) and not record["responseStates"]
    # Even when the model misses complete evidence, a failed query cannot certify absence.
    record["values"].pop("v")
    record["state"]["v"] = "unconfirmed"
    structured_forecast.copy_responses(value, record)
    assert record["state"]["v"] == "unconfirmed" and "v" not in record["responseStates"]
    other = corpus.answer_doc(value, "v", before["values"]["v"] + 5, "conflicting")
    other["version"] = max(d["version"] for d in value["docs"] if matches(d, "v"))
    value["docs"].append(other)
    record = parameter_record(value)
    assert record["state"]["v"] == "conflict"
    structured_forecast.copy_responses(value, record)
    assert record["state"]["v"] == "conflict" and "v" not in record["responseStates"]


def test_later_grounded_answer_restores_the_value_and_clears_failed_status(value):
    record = missing(value, "b")
    value = corpus.outcome(value, "b", None)
    structured_forecast.copy_responses(value, record)
    assert record["state"]["b"] == "unavailable"
    value = corpus.outcome(value, "b", 0)
    structured_forecast.copy_responses(value, record)
    assert record["state"]["b"] == "verified" and record["values"]["b"] == 0
    assert "b" not in record["responseStates"]
    assert record["copiedResponses"]["b"]["historyIndex"] == 1


def test_unavailable_needs_a_past_reply_for_that_field_not_a_hidden_response_rate(value):
    record = missing(value, "b")
    missing(value, "v")
    record = parameter_record(value)
    record["state"].update(b="unavailable", v="unavailable")
    value["task"]["rho"] = {"b": 0, "v": 0}
    value["history"].append({"action": "v", "answer": "no_response"})
    structured_forecast.copy_responses(value, record)
    assert record["state"]["v"] == "unavailable"
    assert record["state"]["b"] == "unconfirmed"
    assert record["rejectedStates"] == {"b": "no-observed-failed-reply"}
    assert set(record["responseStates"]) == {"v"}
