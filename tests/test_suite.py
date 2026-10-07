import copy
import gzip
import io
import json
import zipfile

import pytest

from newsvendor.io import digest, jsonl, write
from newsvendor.suite import SCHEMA, abcd, check, cuad, partition, payload, public_input
from newsvendor.suite_score import score


def test_public_inputs_exclude_annotations_and_future_dialogue():
    stream = io.BytesIO()
    contract = {
        "title": "SUPPLY agreement",
        "paragraphs": [
            {
                "context": "Buy ten units.",
                "qas": [
                    {
                        "id": "supply__Minimum Commitment",
                        "question": "Minimum commitment?",
                        "answers": [{"text": "Buy ten", "answer_start": 0}],
                    }
                ],
            }
        ],
    }
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("CUADv1.json", json.dumps({"data": [contract]}))
    row = cuad(
        {"data.zip": stream.getvalue()},
        {"titleTerms": ["supply"], "categories": ["Minimum Commitment"]},
    )[0][0]
    original_input = public_input(row)
    row["target"] = {"action": "abstain", "spans": []}
    assert public_input(row) == original_input
    from newsvendor.native_benchmark import public_view
    from newsvendor.native_inputs import prepare

    config = {"fragments": 12, "candidates": 48}
    view = prepare(row, [], config)
    altered = copy.deepcopy(row)
    altered.update(
        id="PRIVATE_CASE_ID",
        family="PRIVATE_FAMILY",
        component="retail",
        split="test",
        target={"answer": "PRIVATE_LABEL"},
    )
    assert prepare(altered, [], config) == view
    assert view["indexedCharacters"] == view["sourceCharacters"]
    assert "PRIVATE_" not in json.dumps(public_view(view))
    row["input"]["tools"] = [{"id": "query", "future": "private response"}]
    with pytest.raises(ValueError):
        public_input(row)

    original = [
        ["customer", "I ordered a red jacket and would like to return it."],
        ["agent", "May I have your name?"],
        ["customer", "Ada Smith"],
        ["action", "Account opened."],
        ["customer", "UNOBSERVED_FUTURE"],
    ]
    delexed = [
        {
            "speaker": role,
            "text": text,
            "turn_count": i + 1 if i < 3 else i + 4,
            "targets": ["private_intent", None, "pull-up-account", ["ada smith"], -1],
        }
        for i, (role, text) in enumerate(original)
    ]
    raw = {"train": [{"convo_id": 1, "original": original, "delexed": delexed}]}
    contents = {
        "abcd_v1.1.json.gz": gzip.compress(json.dumps(raw).encode()),
        "guidelines.json": json.dumps({"Returns": "Policy A", "Shipping": "Policy B"}).encode(),
        "ontology.json": json.dumps(
            {"actions": {"interaction": {"pull-up-account": ["name"]}}}
        ).encode(),
    }
    rows, _ = abcd(contents, {"groups": 1, "seed": 53})
    action = next(r for r in rows if r["target"]["action"] == "call_tool")
    visible = json.dumps(public_input(action))
    assert "UNOBSERVED_FUTURE" not in visible and "private_intent" not in visible
    assert "Policy A" in visible and "Policy B" in visible
    assert len(action["input"]["history"]) == 3


def test_connected_sources_and_templates_stay_in_one_stable_split():
    rows = [
        {
            "id": str(i),
            "component": "retail",
            "input": payload(str(i)),
            "keys": [f"store:{i}", f"product:{i}"],
            "groupText": "",
        }
        for i in range(12)
    ]
    rows[1]["keys"] = ["store:0", "product:1"]
    rows[2]["keys"] = ["store:2", "product:1"]
    rows[3]["groupText"] = "Minimum purchase is 10 boxes per month."
    rows[4]["groupText"] = "Minimum purchase is 20 boxes per month."
    other = copy.deepcopy(rows)
    for row in other:
        row["input"]["request"] += " altered wording"
    config = {
        "seed": 53,
        "splits": {"train": 0.6, "dev": 0.2, "test": 0.2},
        "templateSimilarity": 0.85,
    }
    partition(rows, config)
    partition(other, config)
    assert len({r["family"] for r in rows[:3]}) == 1
    assert rows[3]["family"] == rows[4]["family"]
    assert [(r["family"], r["split"]) for r in rows] == [(r["family"], r["split"]) for r in other]
    for family in {r["family"] for r in rows}:
        assert len({r["split"] for r in rows if r["family"] == family}) == 1


def pack(directory):
    inputs, labels = [], []
    span = {"document": "contract", "start": 0, "end": 4}
    targets = {
        "cuad": {"action": "answer", "spans": [span]},
        "contractnli": {"action": "answer", "answer": "entailment", "spans": [span]},
        "orsharc": {"action": "answer", "answer": "yes", "retrieved": ["rule:0"]},
        "abcd": {"action": "speak", "answer": "Hello", "prefixLength": 0},
        "tatqa": {"action": "answer", "answer": 12, "scale": "million"},
        "retail": {
            "action": "answer",
            "answer": [2, 3],
            "dates": ["2024-01-02", "2024-01-03"],
            "complete": [True, False],
            "cutoff": "2024-01-01",
            "latentDemandLabels": False,
        },
    }
    for component, target in targets.items():
        for split in ("train", "dev", "test"):
            id = component + ":" + split
            docs = (
                []
                if component == "orsharc"
                else [{"id": "contract", "title": "Contract", "text": "Rule"}]
            )
            observations = [{"date": "2024-01-01", "sales": 1}] if component == "retail" else []
            inputs.append(
                {
                    "id": id,
                    "component": component,
                    "split": split,
                    "family": id,
                    "input": payload(
                        "Raw request " + id, documents=docs, observations=observations
                    ),
                }
            )
            labels.append({"id": id, "target": copy.deepcopy(target), "leakageKeys": [id]})
    collection = [{"id": "rule:0", "title": "Rule", "text": "Raw rule"}]
    jsonl(directory / "inputs.jsonl", inputs)
    jsonl(directory / "labels.jsonl", labels)
    jsonl(directory / "collection.jsonl", collection)
    manifest = {
        "schema": SCHEMA,
        "language": "en",
        "scope": "test fixture",
        "inputHash": digest(inputs),
        "labelHash": digest(labels),
        "collectionHash": digest(collection),
    }
    write(directory / "manifest.json", manifest)
    return inputs, labels, manifest


def test_scoring_rejects_omissions_bad_units_and_future_leakage(tmp_path):
    inputs, labels, manifest = pack(tmp_path)
    predictions = [
        {"id": "tatqa:test", "prediction": {"action": "answer", "answer": -12, "scale": "million"}},
        {
            "id": "contractnli:test",
            "prediction": {
                "action": "answer",
                "answer": "contradiction",
                "evidence": [{"document": "wrong", "start": 0, "end": 4}],
            },
        },
        {"id": "retail:test", "prediction": {"action": "answer", "answer": [2]}},
    ]
    report = score(tmp_path, predictions, output=tmp_path / "scores")["components"]
    assert report["tatqa"]["metrics"]["answerExact"]["mean"] == 0
    assert report["contractnli"]["metrics"]["groundedScore"]["mean"] == 0
    assert report["cuad"]["metrics"]["groundedScore"]["mean"] == 0
    assert report["retail"]["metrics"]["uncensoredCRPS"]["mean"] is None
    assert not report["retail"]["comparable"]
    for prediction in (
        {"action": "answer", "answer": 12},
        {"action": "answer", "answer": 12, "scale": "million"},
    ):
        result = score(
            tmp_path, [{"id": "tatqa:test", "prediction": prediction}], output=tmp_path / "scores"
        )
        assert result["components"]["tatqa"]["metrics"]["answerExact"]["mean"] == (
            "scale" in prediction
        )
        assert result["pooledScore"] is None
    with pytest.raises(ValueError, match="outside requested split"):
        score(tmp_path, [{"id": "tatqa:train", "prediction": {}}], output=tmp_path / "scores")
    inputs[0]["family"] = inputs[1]["family"]
    manifest["inputHash"] = digest(inputs)
    jsonl(tmp_path / "inputs.jsonl", inputs)
    write(tmp_path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="Source/template leakage"):
        check(tmp_path)
    inputs, labels, manifest = pack(tmp_path)
    labels[-1]["target"]["dates"][0] = "2024-01-01"
    manifest["labelHash"] = digest(labels)
    jsonl(tmp_path / "labels.jsonl", labels)
    write(tmp_path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="Future retail observation"):
        check(tmp_path)


def test_demand_likelihood_distribution_and_decision():
    import math
    from datetime import date, timedelta

    import numpy as np
    import torch

    from newsvendor import demand
    from newsvendor.heads import Head

    for family in demand.FAMILIES:
        logits = torch.zeros(3, 3, requires_grad=True)
        losses = demand.nll(
            family, logits, torch.tensor([1.0, 0.0, 0.0]), torch.tensor([True, False, True])
        )
        if family == "lognormal":
            # At the lognormal median: positive mass .5 times survival .5.
            assert losses.tolist() == pytest.approx([-math.log(0.25), math.log(2), 0])
        losses.sum().backward()
        assert torch.isfinite(logits.grad).all()

    heads = {name: Head(28, 32, 3) for name in demand.HEADS}
    for head in heads.values():
        for parameter in head.parameters():
            torch.nn.init.zeros_(parameter)
    heads["demand_family"].layers[-1].bias.data[1] = 2.0
    observations = [
        {
            "date": (date(2024, 1, 1) + timedelta(days=i)).isoformat(),
            "sales": 1.0,
            "stockoutHours": 0,
        }
        for i in range(60)
    ]
    value = payload(
        "Predict the next 7-day total demand distribution after 2024-02-29.",
        observations=observations,
    )
    output = demand.predict(value, heads)
    assert output["distribution"]["family"] == "lognormal"
    F = output["F"]
    assert len(F) == 65 and sum(p for _, p in F) == pytest.approx(1)
    sigma = math.log(2) + 0.05
    assert sum(d * p for d, p in F) == pytest.approx(0.5 * 7 * math.exp(sigma * sigma / 2))
    for index, family in enumerate(demand.FAMILIES):
        heads["demand_family"].layers[-1].bias.data.zero_()
        heads["demand_family"].layers[-1].bias.data[index] = 2.0
        prediction = demand.predict(value, heads)
        assert prediction["distribution"]["family"] == family
        assert set(prediction["distribution"]["parameters"]) == {
            "zeroProbability",
            *demand.PARAMETERS[family],
        }
        if family == "weibull":
            shape = math.log(2) + 0.2
            mean = 0.5 * 7 * sigma * math.gamma(1 + 1 / shape)
            assert sum(d * p for d, p in prediction["F"]) == pytest.approx(mean)
    assert demand.order([[0, 0.6], [100, 0.4]])["q"] == 0
    assert demand.order([[0, 0.6], [100, 0.4]], underage=3)["q"] == 100
    assert output["orders"][0]["q"] < output["orders"][-1]["q"]

    target = {
        "answer": [1.0] * 7,
        "complete": [True] * 7,
        "dates": [(date(2024, 3, 1) + timedelta(days=i)).isoformat() for i in range(7)],
    }
    exact = demand.metrics(value, target, output)
    assert exact["distributionValid"] == 1 and exact["uncensoredCRPS"] >= 0
    assert exact["observationNLL"] == pytest.approx(
        math.log(7 * sigma * math.sqrt(2 * math.pi) / 0.5)
    )
    altered = copy.deepcopy(output)
    altered["orders"][-1].update(q=0, cost=9 * sum(d * p for d, p in F))
    actual_order = demand.metrics(value, target, altered)
    assert actual_order["orderValid"] == 1
    assert actual_order["uncensoredOrderLoss_u9"] == 63
    altered["orders"] = []
    assert demand.metrics(value, target, altered)["orderValid"] == 0
    target["complete"][0] = False
    censored = demand.metrics(value, target, output)
    assert censored["uncensoredCRPS"] is None and censored["uncensoredOrderLoss_u3"] is None
    assert censored["observationNLL"] == pytest.approx(-math.log(0.25))
    altered = copy.deepcopy(output)
    altered["F"][1][1] = -1
    assert demand.metrics(value, target, altered)["distributionValid"] == 0

    row = {"id": "train", "family": "store-a", "component": "retail", "input": value}
    samples = demand.samples([row], {"train": target}, rolling=True)
    assert len(samples) == 27
    first = samples[0]
    assert first["cutoff"] < first["dates"][0]
    changed = copy.deepcopy(row)
    changed["input"]["observations"][-1]["sales"] = 1e6
    assert np.array_equal(
        first["x"], demand.samples([changed], {"train": target}, rolling=True)[0]["x"]
    )
