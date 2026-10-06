"""Protocol fixtures only: these tests are not measurements of an actual language model."""

import copy

import pytest

from newsvendor import raw
from newsvendor.io import lines, read
from newsvendor.providers import Provider
from newsvendor.workload import observe, public_input


def fixture(id):
    row = next(r for r in lines("cases/pilot/inputs.jsonl") if r["id"] == id)
    annotation = next(r for r in lines("cases/pilot/annotations.jsonl") if r["id"] == id)
    return row, annotation


def answers(input, values):
    """Explicit manually chosen protocol answers, without pretending model inference."""
    menu = raw.catalog(input)
    result = {
        id: {"choice": next(iter(q["criteria"])), "probabilities": None}
        for id, q in raw.questions(input, menu).items()
    }
    result["scope"]["choice"] = "core"
    for doc in input["docs"]:
        result["e_" + doc["id"]]["choice"] = "yes"
    for slot, (value, source) in values.items():
        options = menu["distributions"] if slot == "F" else menu["numbers"]
        selected = next(
            k
            for k, v in options.items()
            if v["value"] == value and {e["docId"] for e in v["evidence"]} == set(source)
        )
        result["g_" + slot]["choice"] = selected
        result["s_" + slot]["choice"] = "verified"
        result["t_" + slot]["choice"] = (
            "preference" if slot == "b" else "estimate" if slot == "F" else "fact"
        )
    return menu, result


def complete(input):
    return answers(
        input,
        {
            "c": (6, ["d01"]),
            "p": (10, ["d02"]),
            "v": (4, ["d05"]),
            "b": (0, ["d03"]),
            "F": ([[10, 0.2], [30, 0.5], [60, 0.3]], ["d04"]),
        },
    )


def test_original_prose_to_own_record_and_optimizer_without_semantic_metadata():
    row, annotation = fixture("e7c4")
    input = public_input(row)
    menu, reply = complete(input)
    record = raw.decode(input, menu, reply)
    assert record["valid"] and record["q"] == 30
    assert record["values"]["v"] == 4
    assert raw.evaluate(annotation, input, record, "handoff")["authorized"]
    for doc in input["docs"]:
        assert set(doc) == {"id", "title", "text", "source"}
    # IDs embedded in product prose cannot become a pack count or a monetary amount.
    assert all(v["value"] != 314 for v in menu["numbers"].values())


def test_documented_low_impact_bounds_do_not_force_a_human_question():
    row, annotation = fixture("97e3")
    input = public_input(row)
    menu, reply = answers(
        input,
        {
            "c": (10, ["d01"]),
            "p": (14, ["d02"]),
            "v": ([0, 0.1], ["d05"]),
            "b": (0, ["d03"]),
            "F": ([[10, 0.2], [30, 0.5], [60, 0.3]], ["d04"]),
        },
    )
    record = raw.decode(input, menu, reply)
    assert record["valid"] and record["q"] == 30 and record["gamma"] == 0
    assert raw.checklist(input, record) == "handoff"
    assert raw.evaluate(annotation, input, record, "handoff")["authorized"]


def test_factual_penalty_and_chosen_preference_have_separate_source_evidence():
    row, annotation = fixture("f4a8")
    input = public_input(row)
    menu, reply = answers(
        input,
        {
            "c": (5, ["d01"]),
            "p": (9, ["d02"]),
            "v": (0, ["d05"]),
            "b": (2, ["d03", "d08"]),
            "F": ([[8, 0.25], [16, 0.25], [24, 0.25], [32, 0.25]], ["d04"]),
        },
    )
    record = raw.decode(input, menu, reply)
    assert record["valid"] and record["q"] == 24
    assert {e["docId"] for e in record["links"]["b"]} == {"d03", "d08"}
    assert raw.evaluate(annotation, input, record, "handoff")["authorized"]


def test_unauthorized_recommendation_cannot_be_promoted_by_provider_confidence():
    row, _ = fixture("e7c4")
    input = public_input(row)
    input["docs"][3]["source"]["author"] = "j22"
    menu, reply = complete(input)
    record = raw.decode(input, menu, reply)
    assert not record["valid"] and "unselected-preference" in record["errors"]
    assert "handoff" not in raw.permitted(input, record)


@pytest.mark.parametrize(
    "id,reason",
    [
        ("6fa2", "unresolved-v-conflict"),
        ("c9e1", "unresolved-b-unconfirmed"),
        ("d2c8", "unresolved-F-unconfirmed"),
        ("0e8f", "unsupported-scope"),
        ("5f1c", "unsupported-scope"),
    ],
)
def test_independent_adjudication_rejects_falsely_claimed_valid_state(id, reason):
    row, annotation = fixture(id)
    result = raw.evaluate(
        annotation, public_input(row), {"valid": True, "values": {}, "q": 30}, "handoff"
    )
    assert result["falseHandoff"] and reason in result["handoffReasons"]


def test_observed_clarification_resolves_conflict_but_late_reply_does_not():
    row, annotation = fixture("6fa2")
    event = annotation["environment"][0]
    input = observe(public_input(row), event["tool"], event["response"])
    assert raw.evaluate(annotation, input, {"q": 30, "values": {"v": 7}}, "handoff")["authorized"]
    row, annotation = fixture("4ab6")
    event = annotation["environment"][0]
    input = observe(public_input(row), event["tool"], event["response"])
    result = raw.evaluate(annotation, input, {"q": 30, "values": {}}, "handoff")
    assert set(result["handoffReasons"]) >= {"cutoff", "unresolved-v-unconfirmed"}


def test_raw_runtime_shares_only_unlabeled_candidates_and_records_actual_calls(
    monkeypatch, tmp_path
):
    row, annotation = fixture("e7c4")
    actual = []

    def protocol_fixture(provider, payload, questions):
        actual.append(copy.deepcopy(payload))
        if "action" in questions:
            reply = {"action": {"choice": "handoff", "probabilities": None}}
        else:
            _, reply = complete(payload["rawInput"])
        return reply, {
            "model": "protocol-fixture-only",
            "usage": None,
            "questionCount": len(questions),
        }

    monkeypatch.setattr(raw, "request_many", protocol_fixture)
    monkeypatch.setattr(
        raw, "settings", lambda _: Provider("agent", "http://localhost", "protocol-fixture-only")
    )
    import newsvendor.cli

    monkeypatch.setattr(newsvendor.cli, "provenance", lambda _: {"testFixture": True})
    monkeypatch.chdir(tmp_path)
    # Resolve files before changing cwd: public input and private annotations stay separate.
    from newsvendor.io import jsonl

    jsonl("inputs.jsonl", [row])
    jsonl("annotations.jsonl", [annotation])
    result = raw.run({"output": "results"}, "agent", 2, "inputs.jsonl", "annotations.jsonl", 1)
    assert result["newCalls"] == 2
    for summary in result["summary"].values():
        assert summary["handoffs"] == 1 and summary["falseHandoffs"] == 0
    assert all(p["rawInput"] == row["input"] for p in actual)
    assert all("reference" not in p and "environment" not in p for p in actual)
    # A replay consumes zero calls; a miss cannot silently exceed the authorized budget.
    again = raw.run({"output": "results"}, "agent", 0, "inputs.jsonl", "annotations.jsonl", 1)
    assert again["newCalls"] == 0 and len(actual) == 2
    assert read("results/raw/agent/metrics.json")["scope"].startswith("Constructed raw protocol")
    client = raw.Client(Provider("agent", "http://localhost", "other-fixture"), 0)
    with pytest.raises(ValueError, match="cache miss"):
        client.construct(row["input"])
