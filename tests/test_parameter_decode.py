import copy

import pytest
import torch

from newsvendor.construction import atoms
from newsvendor.structured_decode import decode_parameters, parameter_candidates
from newsvendor.structured_forecast import resolved_fields
from newsvendor.structured_inputs import OPS


def document(id, text, **kwargs):
    return {
        "id": id,
        "title": "Purchase quotation",
        "text": text,
        "sku": "A",
        "period": "2026-10",
        "version": 1,
        "role": "buyer",
        "complete": True,
    } | kwargs


def case(docs):
    value = {"task": {"sku": "A", "period": "2026-10"}, "docs": docs}
    candidates = [
        {
            "value": a["value"],
            "location": {
                "kind": "document",
                "id": d["id"],
                "start": a["span"][0],
                "end": a["span"][1],
            },
        }
        for d in docs
        for a in atoms(d)
    ]
    view = {"atoms": candidates}
    output = {
        "relation": torch.zeros(1, len(OPS)),
        "operand1": torch.zeros(1, len(candidates)),
        "operand2": torch.zeros(1, len(candidates)),
    }
    field = {
        "field": "c",
        "name": "c",
        "mode": "compute",
        "state": "verified",
        "type": "fact",
        "value": candidates[0]["value"],
        "evidence": [candidates[0]["location"]],
        "expression": {"op": "copy", "operands": [candidates[0]["location"]]},
    }
    return value, view, output, [field]


@pytest.mark.parametrize("metadata", [{"sku": "B"}, {"period": "2026-09"}, {"version": 0}])
def test_rejected_scope_uses_observed_current_source(metadata):
    value, view, output, fields = case(
        [
            document("wrong", "Purchase cost per unit = 99.", **metadata),
            document("current", "Pack price = 120; units per pack = 10."),
        ]
    )
    before = copy.deepcopy((value, view, fields))
    output["operand1"][0, 0] = 100  # Invalid evidence remains the neural argmax.
    decoded, log = decode_parameters(value, view, output, fields)
    assert decoded[0]["value"] == 12
    assert decoded[0]["expression"]["op"] == "divide"
    assert {x["id"] for x in decoded[0]["evidence"]} == {"current"}
    assert log[0]["changed"] and log[0]["candidates"] == 1
    assert (value, view, fields) == before


def test_scores_choose_ordered_supported_expression_and_ablation_is_separate():
    value, view, output, fields = case(
        [
            document("wrong", "Purchase cost per unit = 99.", sku="B"),
            document(
                "current", "Pack price = 120; units per pack = 10; purchase cost per unit = 12."
            ),
        ]
    )
    output["relation"][0, OPS.index("divide")] = 30
    output["operand1"][0, 1] = 30
    output["operand2"][0, 2] = 30
    scored, _ = decode_parameters(value, view, output, fields)
    first, _ = decode_parameters(value, view, output, fields, "first")
    assert scored[0]["value"] == first[0]["value"] == 12
    assert scored[0]["expression"]["op"] == "divide"
    assert first[0]["expression"]["op"] == "copy"


def test_valid_selection_with_arbitrary_title_is_preserved():
    value, view, output, fields = case(
        [document("current", "Purchase cost per unit = 12.", title="A supplier's terms")]
    )
    decoded, log = decode_parameters(value, view, output, fields)
    assert decoded == fields and log == []


@pytest.mark.parametrize("blocked", ["conflict", "unread", "partial", "absent"])
def test_no_repair_without_available_unambiguous_evidence(blocked):
    docs = [document("wrong", "Purchase cost per unit = 99.", sku="B")]
    if blocked != "absent":
        docs.append(
            document("current", "Purchase cost per unit = 12.", complete=blocked != "partial")
        )
    if blocked == "conflict":
        docs.append(document("conflict", "Purchase cost per unit = 13."))
    value, view, output, fields = case(docs)
    if blocked == "unread":
        view["atoms"] = view["atoms"][:1]
    decoded, log = decode_parameters(value, view, output, fields)
    if blocked == "conflict":
        assert decoded[0]["state"] == "conflict" and decoded[0]["value"] is None
        assert {loc["id"] for loc in decoded[0]["evidence"]} == {"current", "conflict"}
        assert log[0]["kind"] == "conflict" and log[0]["candidates"] == 2
    else:
        assert decoded == fields and log == [{"field": "c", "candidates": 0, "changed": False}]


@pytest.mark.parametrize("status", ["unavailable", "unconfirmed", "verified"])
def test_unselected_current_conflict_survives_missing_manager_reply(monkeypatch, status):
    import newsvendor.construction as construction
    from newsvendor.structured_forecast import copy_responses

    def forbidden(*args, **kwargs):
        raise AssertionError("Observed conflict checks must not call a scoring reference")

    monkeypatch.setattr(construction, "parameter_record", forbidden)
    value, view, output, fields = case(
        [
            document("one", "Purchase cost per unit = 12."),
            document("two", "Purchase cost per unit = 13."),
        ]
    )
    value["history"] = [{"action": "c", "answer": "no_response"}]
    fields[0].update(state=status, mode="missing", value=None, expression=None)
    decoded, log = decode_parameters(value, view, output, fields)
    assert decoded[0]["state"] == "conflict" and decoded[0]["value"] is None
    assert log[0]["kind"] == "conflict"
    record = {
        "values": {},
        "links": {},
        "expressions": {},
        "types": {"c": "fact"},
        "state": {"c": "conflict"},
        "errors": ["conflict-c"],
    }
    copy_responses(value, record)
    assert record["state"]["c"] == "conflict" and not record["values"]


def test_observed_conflict_blocks_order_until_manager_resolves_it():
    from newsvendor.construction import finish
    from newsvendor.corpus import outcome
    from newsvendor.structured_forecast import actions, copy_responses

    value, view, output, fields = case(
        [
            document(
                "one",
                "Refund per unit = 3; handling fee per unit = 1.",
                title="Current return contract",
            ),
            document(
                "two",
                "Refund per unit = 4; handling fee per unit = 1.",
                title="Current return contract",
            ),
        ]
    )
    value.update(history=[], remaining=2)
    value["task"].update(
        bounds=[0, 20],
        allowed={"v": [2, 3], "b": [0]},
        decision="minimax",
        tolerance=None,
        deadline=3,
        costs={"v": 1},
    )
    fields[0].update(
        name="v", field="v", mode="missing", state="unavailable", value=None, expression=None
    )
    decoded, _ = decode_parameters(value, view, output, fields)
    record = {
        "values": {"c": 5, "p": 10, "b": 0},
        "links": {},
        "expressions": {},
        "state": {
            "c": "verified",
            "p": "verified",
            "v": decoded[0]["state"],
            "b": "verified",
            "F": "verified",
        },
        "types": {"c": "fact", "p": "fact", "v": "fact", "b": "preference", "F": "estimate"},
        "errors": ["conflict-v"],
    }
    forecast = [[[5, 1.0]]]
    before = finish(value, record, forecast)
    assert "handoff" not in actions(value, before) and "v" in actions(value, before)
    value = outcome(value, "v", 2)
    copy_responses(value, record)
    after = finish(value, record, forecast)
    assert record["values"]["v"] == 2 and record["state"]["v"] == "verified"
    assert "handoff" in actions(value, after)


@pytest.mark.parametrize("unavailable", ["unread", "old"])
def test_conflict_requires_current_encoded_sources(unavailable):
    value, view, output, fields = case(
        [
            document("one", "Purchase cost per unit = 12."),
            document(
                "two", "Purchase cost per unit = 13.", version=0 if unavailable == "old" else 1
            ),
        ]
    )
    fields[0].update(mode="missing", state="unconfirmed", value=None, expression=None)
    if unavailable == "unread":
        view["atoms"] = view["atoms"][:1]
    decoded, log = decode_parameters(value, view, output, fields)
    assert decoded == fields and not log


def test_accepted_new_source_preserved_over_older_conflicting_titles():
    value, view, output, fields = case(
        [
            document("new", "Purchase cost per unit = 12.", title="Supplier terms", version=2),
            document("old-one", "Purchase cost per unit = 13."),
            document("old-two", "Purchase cost per unit = 14."),
        ]
    )
    decoded, log = decode_parameters(value, view, output, fields)
    assert decoded == fields and not log


@pytest.mark.parametrize(
    "change",
    [
        {"mode": "missing"},
        {"state": "conflict"},
        {"state": "candidate"},
        {"name": "b", "type": "fact"},
    ],
)
def test_missing_conflict_and_manager_preference_are_not_forced(change):
    value, view, output, fields = case(
        [
            document("wrong", "Purchase cost per unit = 99.", sku="B"),
            document("current", "Purchase cost per unit = 12."),
        ]
    )
    fields[0].update(change)
    decoded, log = decode_parameters(value, view, output, fields)
    assert decoded == fields and log == []


def test_no_hidden_reference_and_raw_predictions_are_preserved(monkeypatch):
    import newsvendor.construction as construction

    def forbidden(*args, **kwargs):
        raise AssertionError("Decoder must not call a scoring reference")

    monkeypatch.setattr(construction, "parameter_record", forbidden)
    value, view, output, fields = case(
        [
            document("wrong", "Purchase cost per unit = 99.", sku="B"),
            document("current", "Purchase cost per unit = 12."),
        ]
    )
    value["truth"] = {"c": 999999}
    decoded, _ = decode_parameters(value, view, output, fields)
    assert len(parameter_candidates(value, view, "c")) == 1
    record = {
        "fields": decoded,
        "unconstrainedFields": fields,
        "values": {"c": 12},
        "links": {"c": "current"},
        "types": {"c": "fact"},
        "state": {"c": "verified"},
        "errors": [],
    }
    resolved_fields(record, None)
    assert record["rawFields"][0]["value"] == 99
    assert record["fields"][0]["value"] == 12


@pytest.mark.parametrize(
    "condition", ["wrong_sku", "wrong_period", "older_version", "refund_order"]
)
def test_economic_document_conditions_use_observations_only(monkeypatch, condition):
    import importlib
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    evaluator = importlib.import_module("evaluate_parameters")
    value = {
        "task": {
            "sku": "SKU-0000000000",
            "period": "next-day",
            "bounds": [0, 100],
            "hold": 100,
            "costs": {k: 1 for k in ("c", "p", "v", "b")},
            "tolerance": 0,
            "deadline": 3,
            "decision": "expected_loss",
        },
        "docs": [
            document(
                "cost", "Purchase cost per unit = 12.", sku="SKU-0000000000", period="next-day"
            ),
            document(
                "refund",
                "Refund per unit = 5; handling fee per unit = 1. Unlimited returns.",
                sku="SKU-0000000000",
                period="next-day",
                title="Return contract",
                role="supplier",
            ),
        ],
        "observations": [],
        "history": [],
        "remaining": 3,
    }
    changed = evaluator.documents(value, condition, 7315)
    evaluator.check_observed(value, changed)

    def forbidden(*args, **kwargs):
        raise AssertionError("Input transformation must not use scoring annotations")

    monkeypatch.setattr(evaluator, "parameter_record", forbidden)
    poisoned = copy.deepcopy(value)
    poisoned["truth"] = {"c": 100000, "v": 90000, "futureDemand": [1000]}
    result = evaluator.documents(poisoned, condition, 7315)
    assert result["docs"] == changed["docs"] and result["task"] == changed["task"]
