"""Protocol fixtures only: these tests are not measurements of an actual language model."""

import copy

import pytest

from newsvendor import raw
from newsvendor.io import lines, read
from newsvendor.providers import Provider
from newsvendor.workload import comparison_contract, observe, public_input


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


def test_observed_clarification_resolves_conflict_but_late_reply_does_not():
    row, annotation = fixture("6fa2")
    forged = raw.evaluate(
        annotation, public_input(row), {"valid": True, "q": 30, "values": {}}, "handoff"
    )
    assert forged["falseHandoff"] and "unresolved-v-conflict" in forged["handoffReasons"]
    event = annotation["environment"][0]
    input = observe(public_input(row), event["tool"], event["response"])
    assert raw.evaluate(annotation, input, {"q": 30, "values": {"v": 7}}, "handoff")["authorized"]
    row, annotation = fixture("4ab6")
    event = annotation["environment"][0]
    input = observe(public_input(row), event["tool"], event["response"])
    result = raw.evaluate(annotation, input, {"q": 30, "values": {}}, "handoff")
    assert set(result["handoffReasons"]) >= {"cutoff", "unresolved-v-unconfirmed"}
    for id, reason in (("d2c8", "unresolved-F-unconfirmed"), ("0e8f", "unsupported-scope")):
        row, annotation = fixture(id)
        result = raw.evaluate(
            annotation, public_input(row), {"valid": True, "q": 30, "values": {}}, "handoff"
        )
        assert result["falseHandoff"] and reason in result["handoffReasons"]


def test_raw_runtime_shares_only_unlabeled_candidates_and_records_actual_calls(
    monkeypatch, tmp_path
):
    design = read("configs/workload-v2.json")
    assert comparison_contract(design)["primaryArms"] == 3
    for policy in ("finite-rollout-on-estimated-belief", "one-step-value-of-information"):
        planned = copy.deepcopy(design)
        planned["comparisons"][0]["policy"] = policy
        with pytest.raises(ValueError, match="planners are not primary"):
            comparison_contract(planned)
    row, annotation = fixture("e7c4")
    menu, reply = complete(public_input(row))
    record = raw.decode(public_input(row), menu, reply)
    assert record["valid"] and record["q"] == 30
    assert all(v["value"] != 314 for v in menu["numbers"].values())
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
