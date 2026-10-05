import copy

import pytest

from newsvendor.construction import candidates, reference
from newsvendor.corpus import KINDS, SLOTS, STATUSES, generate
from newsvendor.io import digest
from newsvendor.providers import Provider
from newsvendor.typed import TypedBuilder, decode, questions


def response_for(input):
    ref = reference(input)
    answers = {}
    for id, q in questions(input).items():
        prefix, slot, *tail = id.split("_")
        if prefix == "e":
            choice = "yes" if input["docs"][int(tail[0])]["id"] == ref["links"].get(slot) else "no"
        elif prefix == "g":
            cs = candidates(input, slot)
            choice = next(
                (f"g{i}" for i, c in enumerate(cs) if c["id"] == ref["expressions"].get(slot)),
                "none",
            )
        elif prefix == "t":
            choice = ref["types"].get(slot, "assumption")
        else:
            choice = ref["state"].get(slot, "verified")
        answers[id] = {
            "choice": choice,
            "probabilities": {k: float(k == choice) for k in q["criteria"]},
            "confidence": None,
        }
    return answers


def test_typed_construction_uses_common_candidates_and_guards():
    config = {"seed": 42, "groups": 10, "variants": 1, "budget": 2}
    episodes = generate(config)
    for episode in episodes:
        input = episode["input"]
        decoded = decode(input, response_for(input))
        ref = reference(input)
        assert decoded["values"] == ref["values"]
        assert decoded["q"] == pytest.approx(ref["q"]) and decoded["valid"] == ref["valid"]
    input = next(e["input"] for e in episodes if e["scenario"] == "conflicting")
    hostile = response_for(input)
    hostile["s_v"]["choice"] = "verified"
    assert not decode(input, hostile)["valid"]


def test_provider_budget_and_cache_identity(tmp_path):
    input = generate({"seed": 42, "groups": 10, "variants": 1, "budget": 2})[0]["input"]
    builder = TypedBuilder(
        Provider("sglang", "http://127.0.0.1:1/v1/systemone", "contract-model"), 0, str(tmp_path)
    )
    with pytest.raises(ValueError, match="budget exhausted"):
        builder(input)
    assert builder.calls == 0
    for slot in (*SLOTS, "F", "C"):
        assert set(questions(input)[f"t_{slot}"]["criteria"]) == set(KINDS)
        assert set(questions(input)[f"s_{slot}"]["criteria"]) == set(STATUSES)
    answers = response_for(input)
    key = digest({"input": input, "questions": questions(input), "identity": builder.identity})
    from newsvendor.io import write

    write(
        builder.directory / (key + ".json"),
        {
            "requestHash": key,
            "answers": answers,
            "answerHash": digest(answers),
            "trace": {"usage": None},
        },
    )
    assert builder(input)["values"] == reference(input)["values"]
    poison = copy.deepcopy(answers)
    poison["t_c"]["choice"] = "preference"
    write(
        builder.directory / (key + ".json"),
        {"requestHash": key, "answers": poison, "answerHash": digest(answers), "trace": {}},
    )
    another = TypedBuilder(builder.provider, 0, str(tmp_path))
    with pytest.raises(ValueError, match="checksum"):
        another(input)
