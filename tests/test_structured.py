import copy
from datetime import date, timedelta

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import ModernBertConfig, ModernBertModel, PreTrainedTokenizerFast

from newsvendor import corpus, demand, sequence
from newsvendor.construction import reference
from newsvendor.io import digest, read
from newsvendor.structured_inputs import (
    MODES,
    best_span,
    prepare,
    research_input,
    span_indices,
    span_value,
)
from newsvendor.structured_labels import objective, research_targets, suite_targets
from newsvendor.structured_model import Router, assemble, compute
from newsvendor.structured_rollout import FIELDS, TEXT, collect
from newsvendor.structured_train import variant
from newsvendor.suite import payload


@pytest.fixture
def tokenizer():
    vocab = {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[UNK]": 3}
    for word in (
        "Order id A123 10 2 8 Refund manager purchase cost price policy request "
        "shipping standard express missing facts USD units box carton . , : =".split()
    ):
        if word not in vocab:
            vocab[word] = len(vocab)
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        unk_token="[UNK]",
    )


@pytest.fixture
def settings():
    return {"maxLength": 96, "queryTokens": 16, "overlap": 8, "chunks": 100, "encoderBatch": 4}


@pytest.fixture
def model(settings):
    torch.manual_seed(42)
    torch.set_num_threads(1)
    config = ModernBertConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=128,
        local_attention=32,
        pad_token_id=0,
        cls_token_id=1,
        sep_token_id=2,
        bos_token_id=1,
        eos_token_id=2,
    )
    config._attn_implementation = "eager"
    return Router(ModernBertModel(config), settings)


@pytest.mark.parametrize(
    "formula,scale,answer,atoms,operation",
    [
        ("2,010/442,262", "percent", 0.45, [2010, 442262], "percent"),
        ("$3,313/$39,784", "percent", 0.08, [3313, 39784], "divide"),
        (" ($321,305-$323,902)/$323,902 ", "percent", -0.8, [321305, 323902], "change"),
        ("6.9 / 5.5 - 1", "percent", 25.45, [6.9, 5.5], "change"),
        ("(166+178)/2", "million", 172, [166, 178], "average"),
        ("(32/80)*100", "percent", 40, [32, 80], "percent"),
        ("100*(32/80)", "percent", 40, [32, 80], "percent"),
        ("100*((90-80)/80)", "percent", 12.5, [90, 80], "change"),
        ("9.2-24.2", "percent", -15, [9.2, 24.2], "subtract"),
        ("30,763-(-8,186)", "thousand", 38949, [30763, -8186], "subtract"),
    ],
)
def test_source_arithmetic_preserves_annotated_units_and_operand_order(
    formula, scale, answer, atoms, operation
):
    from newsvendor.structured_inputs import OPS
    from newsvendor.structured_labels import derivation
    from newsvendor.suite_score import rounded_numeric_exact

    # Distractor and repeated values ensure pointers still refer to observations.
    view = {"atoms": [{"value": value} for value in [9999, *atoms, atoms[0]]]}
    target = derivation(view, formula, scale, answer)
    assert OPS[target["relation"]] == operation
    assert target["operand1"] == [1, 3] and target["operand2"] == [2]
    assert rounded_numeric_exact(compute(operation, atoms), answer)


@pytest.mark.parametrize(
    "formula,scale,answer,atoms",
    [
        ("45/102", "", 44.12, [45, 102]),  # no annotated percent conversion
        ("45/102", "percent", 12, [45, 102]),  # inconsistent answer: no operator search
        ("(10-8)/6", "percent", 33.33, [10, 8, 6]),  # not a supported change pair
        ("(3+4+5)/3", "", 4, [3, 4, 5]),  # unsupported expression remains masked
        ("(10-8)/8", "", 0.25, [10, 8]),  # no raw-change calculator operation
        ("10/0", "percent", 0, [10, 0]),
        ("10/2", "", 5, [10]),  # source operand is absent
        ("1e309-1", "", 0, [float("inf"), 1]),
        ("True+1", "", 2, [1]),
    ],
)
def test_inconsistent_or_unrepresentable_arithmetic_stays_masked(formula, scale, answer, atoms):
    from newsvendor.structured_labels import derivation

    assert (
        derivation({"atoms": [{"value": value} for value in atoms]}, formula, scale, answer) is None
    )


def test_arithmetic_answers_only_change_loss_targets(tokenizer, settings):
    from newsvendor.structured_train import language_view

    row = {
        "id": "source",
        "component": "tatqa",
        "input": payload(
            "What percentage?",
            tables=[{"id": "table", "cells": [["part", "45"], ["total", "102"]]}],
        ),
    }
    label = {
        "action": "answer",
        "answerType": "arithmetic",
        "derivation": "45/102",
        "scale": "percent",
        "answer": 44.12,
    }
    view, target = language_view(
        ("public", row), tokenizer, {"encoder": settings}, {"source": label}, []
    )
    other, invalid = language_view(
        ("public", row), tokenizer, {"encoder": settings}, {"source": {**label, "answer": 10}}, []
    )
    assert view["inputHash"] == other["inputHash"] and view["atoms"] == other["atoms"]
    assert torch.equal(view["batch"]["input_ids"], other["batch"]["input_ids"])
    assert "relation" in target["fields"][0] and "relation" not in invalid["fields"][0]


@pytest.mark.parametrize("answer_type,answers", [("span", ["10"]), ("multi-span", ["10", "2"])])
def test_scalar_pointer_does_not_treat_required_multiple_answers_as_alternatives(
    tokenizer, settings, answer_type, answers
):
    from newsvendor.structured_train import language_view

    row = {
        "id": "source",
        "component": "tatqa",
        "input": payload(
            "Which values?",
            tables=[{"id": "table", "cells": [["cost", "10"], ["price", "2"], ["cost", "10"]]}],
        ),
    }
    label = {
        "action": "answer",
        "answerType": answer_type,
        "derivation": "",
        "scale": "",
        "answer": answers,
    }
    view, target = language_view(
        ("public", row), tokenizer, {"encoder": settings}, {"source": label}, []
    )
    field = target["fields"][0]
    assert "scale" in field and "recovery" in target
    if answer_type == "span":
        assert len(field["spans"]) == 2 and field["mode"] == MODES.index("span")
    else:
        assert "spans" not in field and "mode" not in field
    assert view["inputHash"] == digest(row["input"])


def test_accounting_cell_signs_preserve_evidence_without_negating_footnotes(tokenizer, settings):
    texts = ["(8,186)", "$(793)", "($879)", "$ (1,325)", "adjustments(1)", "—", "$—"]
    value = payload(
        "cost",
        tables=[{"id": "table", "cells": [[text] for text in texts]}],
        documents=[{"id": "doc", "title": "report", "text": "Year (2024)."}],
    )
    view = prepare(value, tokenizer, settings)
    cells = {a["location"]["row"]: a for a in view["atoms"] if a["location"]["kind"] == "cell"}
    assert {row: a["value"] for row, a in cells.items()} == {
        0: -8186,
        1: -793,
        2: -879,
        3: -1325,
        4: 1,
    }
    for row in range(4):
        atom = cells[row]
        start, end = atom["location"]["start"], atom["location"]["end"]
        assert texts[row][start:end].startswith("(") and texts[row][start:end].endswith(")")
        assert span_indices(view, atom["location"]) is not None
    assert next(a["value"] for a in view["atoms"] if a["location"]["kind"] == "document") == 2024


def test_offsets_cover_overlapping_sources_and_ignore_hidden_labels(tokenizer, settings):
    text = "Order id A123. " * 100
    value = payload(
        "Order id",
        documents=[{"id": "doc", "title": "policy", "text": text}],
        tables=[{"id": "table", "cells": [["Refund", "10"], ["cost", "2"]]}],
        history=[{"role": "tool", "text": "Order id A123"}],
    )
    row = {"input": value, "label": "secret", "target": {"answer": "secret"}}
    view = prepare(row, tokenizer, settings)
    other = prepare({**row, "label": "different", "target": {}}, tokenizer, settings)
    assert view["inputHash"] == other["inputHash"]
    assert torch.equal(view["batch"]["input_ids"], other["batch"]["input_ids"])
    expected = {
        tuple(o)
        for o in tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)[
            "offset_mapping"
        ]
    }
    actual = {(o["start"], o["end"]) for o in view["locations"] if o["id"] == "doc"}
    assert actual == expected
    assert len(view["locations"]) == len({digest(v) for v in view["locations"]})
    pair = span_indices(
        view, {"kind": "cell", "id": "table", "row": 0, "column": 1, "start": 0, "end": 2}
    )
    assert span_value(view, *pair)[0] == "10"
    doc = next(i for i, v in enumerate(view["locations"]) if v["kind"] == "document")
    cell = next(i for i, v in enumerate(view["locations"]) if v["kind"] == "cell")
    with pytest.raises(ValueError, match="Invalid source span"):
        span_value(view, doc, cell)


def test_packed_queries_preserve_sources_and_receive_gradients(tokenizer, settings, model):
    value = payload("Order id", documents=[{"id": "doc", "title": "policy", "text": "10 2"}])
    fields = [
        {
            "id": str(i),
            "name": "Refund",
            "description": "Order id",
            "choices": ["standard", "express"],
        }
        for i in range(12)
    ]
    separate = prepare(value, tokenizer, settings, fields=fields)
    packed = prepare(value, tokenizer, {**settings, "packedQueries": True}, fields=fields)
    assert packed["locations"] == separate["locations"]
    assert packed["atoms"] == separate["atoms"]
    assert packed["choiceQueries"] == separate["choiceQueries"]
    assert len(packed["batch"]["input_ids"]) < len(separate["batch"]["input_ids"])
    assert len(packed["queryPositions"]) == len(separate["queryPositions"])
    output = model(packed)
    assert output["mode"].shape[0] == len(fields)
    output["mode"].square().sum().backward()
    assert model.encoder.embeddings.tok_embeddings.weight.grad.abs().sum() > 0


def test_packed_sources_keep_all_positions_cells_and_history(tokenizer, settings, model):
    value = payload(
        "Refund",
        documents=[{"id": "doc", "title": "policy", "text": "Refund 10 2 " * 20}],
        history=[{"role": "user", "text": "Order id A123"} for _ in range(8)],
        tables=[{"id": "table", "cells": [["cost", "10"], ["Refund", "2"]]}],
    )
    settings = {**settings, "chunkTokens": 24, "overlap": 8, "packedQueries": True}
    separate = prepare(value, tokenizer, settings)
    packed = prepare(value, tokenizer, {**settings, "packedSources": True})
    assert packed["locations"] == separate["locations"]
    assert packed["atoms"] == separate["atoms"]
    assert packed["encodedTokens"] < separate["encodedTokens"]
    for index, (seq, position) in enumerate(packed["positions"]):
        loc = packed["locations"][index]
        source = packed["sourceIndex"][(loc["kind"], loc["id"], loc.get("row"), loc.get("column"))]
        expected = tokenizer(source["text"][loc["start"] : loc["end"]], add_special_tokens=False)[
            "input_ids"
        ]
        assert int(packed["batch"]["input_ids"][seq, position]) == expected[0]
    output = model(packed)
    output["start"].square().mean().backward()
    assert model.encoder.embeddings.tok_embeddings.weight.grad.abs().sum() > 0


@pytest.mark.parametrize(
    "case_batch,workers,split,needed,distill_weight",
    [
        (1, 0, False, False, 0),
        (2, 2, False, False, 0),
        (2, 2, True, False, 0),
        (2, 2, False, True, 0),
        (2, 2, False, False, 4),
        (2, 2, False, True, 4),
    ],
)
def test_interrupted_language_training_resumes_exactly(
    tmp_path,
    tokenizer,
    settings,
    model,
    case_batch,
    workers,
    split,
    needed,
    distill_weight,
    monkeypatch,
):
    from newsvendor.structured_progress import Progress
    from newsvendor.structured_train import train_language
    from newsvendor.train import seed

    config = variant(read("configs/structured.json"), "base")
    config["encoder"].update(settings)
    config["training"].update(
        epochs=2,
        accumulation=2,
        checkpointEvery=4,
        progressEvery=4,
        caseBatch=case_batch,
        prefetchWorkers=workers,
        distillWeight=distill_weight,
    )
    model.config["splitLongBatches"] = split
    config["encoder"]["splitLongBatches"] = split
    if split:
        model.config["checkpointTokenLimit"] = 0.05
        config["encoder"]["checkpointTokenLimit"] = 0.05
    if needed:
        config["training"]["retrievalTraining"] = "needed"
        config["encoder"].update(
            packedSources=True, chunkTokens=4, overlap=0, documentTokens=2, batchFusion=True
        )
        model.config.update(config["encoder"])
    config["output"] = str(tmp_path)
    rows = [
        (
            "public",
            {
                "id": str(i),
                "component": "cuad",
                "split": "train",
                "input": payload(
                    "Refund",
                    documents=(
                        [
                            {"id": "other", "title": "Refund", "text": "Refund policy"},
                            {"id": "doc", "title": "cost", "text": "10 2"},
                        ]
                        if needed
                        else [{"id": "doc", "title": "policy", "text": "Refund 10"}]
                    ),
                ),
            },
        )
        for i in range(8)
    ]
    labels = {
        str(i): {
            "action": "answer",
            "spans": [{"document": "doc", "start": 0, "end": 2 if needed else 9}],
        }
        for i in range(8)
    }
    initial = copy.deepcopy(model.state_dict())
    if distill_weight:
        from newsvendor import structured_train

        parent = copy.deepcopy(model)
        config["warmStart"] = "immutable-test-parent.pt"

        def initialize_teacher(config):
            seed(987)  # Parent loading must not change student or sampling RNG streams.
            return copy.deepcopy(parent), tokenizer, {"fileHash": "test-parent"}

        monkeypatch.setattr(structured_train, "initialize", initialize_teacher)
    seed(42)
    expected = train_language(model, tokenizer, config, rows, rows[:1], labels, [])
    weights = copy.deepcopy(model.state_dict())
    model.load_state_dict(initial)
    seed(42)
    progress = Progress(config, {"snapshot": "test"}, "cpu", None)
    checkpoint = progress.checkpoint

    def interrupt(*args, **kwargs):
        checkpoint(*args, **kwargs)
        raise InterruptedError("Test interrupted at an optimizer boundary")

    progress.checkpoint = interrupt
    with pytest.raises(InterruptedError):
        train_language(model, tokenizer, config, rows, rows[:1], labels, [], progress)
    progress.checkpoint = checkpoint
    model.load_state_dict(initial)
    seed(123)
    actual = train_language(model, tokenizer, config, rows, rows[:1], labels, [], progress)
    assert actual == expected
    assert all(torch.equal(weights[k], v) for k, v in model.state_dict().items())
    resumed = train_language(model, tokenizer, config, rows, rows[:1], labels, [], progress)
    assert resumed == expected


@pytest.mark.parametrize("padding_ratio", [None, 1.25])
def test_batched_cases_keep_pointers_and_gradients_separate(
    tokenizer, settings, model, padding_ratio
):
    views = [
        prepare(
            payload(request, documents=[{"id": "d", "title": "policy", "text": text}]),
            tokenizer,
            settings,
        )
        for request, text in (("Refund", "10 2"), ("Order id", "A123 8"))
    ]
    model.config["bucketLengths"] = True
    model.config["batchFusion"] = True
    model.config["paddingRatio"] = padding_ratio
    separate = [model(v) for v in views]
    batched = model(views)
    for a, b in zip(separate, batched, strict=True):
        for name in a:
            assert torch.allclose(a[name], b[name], atol=2e-6, rtol=2e-5), name
    model.zero_grad(set_to_none=True)
    sum(o["start"].sum() for o in batched).backward()
    combined = model.project.weight.grad.clone()
    model.zero_grad(set_to_none=True)
    for view in views:
        model(view)["start"].sum().backward()
    assert torch.allclose(combined, model.project.weight.grad, atol=2e-5, rtol=2e-5)


def test_shared_schema_queries_preserve_gradients_and_refresh_after_updates(
    tokenizer, settings, model
):
    settings = {**settings, "packedQueries": True, "packedSources": True, "sharedQueries": True}
    views = [
        prepare(
            payload("Refund", documents=[{"id": "d", "title": "policy", "text": text}]),
            tokenizer,
            settings,
        )
        for text in ("Refund 10 2", "Refund 8")
    ]
    model.config.update(settings, batchFusion=True, sharedQueries=False)
    separate = model(views)
    sum(o["mode"].square().sum() for o in separate).backward()
    expected = model.encoder.embeddings.tok_embeddings.weight.grad.clone()
    model.zero_grad(set_to_none=True)
    model.config["sharedQueries"] = True
    shared = model(views)
    for a, b in zip(separate, shared, strict=True):
        for key in a:
            assert torch.allclose(a[key], b[key], atol=2e-5, rtol=2e-5), key
    sum(o["mode"].square().sum() for o in shared).backward()
    assert torch.allclose(
        expected, model.encoder.embeddings.tok_embeddings.weight.grad, atol=2e-5, rtol=2e-5
    )
    assert model.last_batch["sharedSequences"] > 0
    model.zero_grad(set_to_none=True)
    encoded = model.encode_batch(views)[0][1]
    before = encoded.detach().clone()
    encoded.square().sum().backward()
    optimizer = model.optimizer({"encoderLr": 0.001, "headLr": 0.001})
    optimizer.step()
    after = model.encode_batch(views)[0][1].detach()
    assert not torch.equal(before, after)


def test_padding_keeps_valid_token_outputs_and_backward(tokenizer, settings, model):
    view = prepare(
        payload("Refund", documents=[{"id": "d", "title": "policy", "text": "10 2"}]),
        tokenizer,
        settings,
    )
    plain = model(view)
    model.config["padMultiple"] = 32
    padded = model(view)
    for key in plain:
        assert torch.allclose(plain[key], padded[key], atol=2e-6, rtol=2e-5), key
    model.zero_grad(set_to_none=True)
    padded["start"].sum().backward()
    assert model.encoder.embeddings.tok_embeddings.weight.grad.abs().sum() > 0


def test_padding_buckets_keep_every_sequence_output_and_gradient(tokenizer, settings, model):
    config = dict(
        settings,
        packedQueries=True,
        packedSources=True,
        sharedQueries=True,
        encoderBatch=16,
        padMultiple=8,
        bucketLengths=True,
        batchFusion=True,
    )
    views = [
        prepare(
            payload("Refund", documents=[{"id": "d", "title": "policy", "text": text}]),
            tokenizer,
            config,
        )
        for text in ("10 " * 45, "8 2")
    ]
    model.config.update(config)
    sequences, _ = model.encoder_sequences(views)
    before = sum(len(items) * width for items, width in model.encoder_batches(sequences))
    original = model(views)
    sum(o["mode"].square().sum() for o in original).backward()
    gradients = {k: p.grad.clone() for k, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    model.config["paddingRatio"] = 1.25
    batches = model.encoder_batches(sequences)
    assert [item for items, _ in batches for item in items] == sequences
    assert sum(len(items) * width for items, width in batches) < before
    bucketed = model(views)
    for a, b in zip(original, bucketed, strict=True):
        for key in a:
            assert torch.allclose(a[key], b[key], atol=2e-5, rtol=2e-5), key
    sum(o["mode"].square().sum() for o in bucketed).backward()
    for key, parameter in model.named_parameters():
        if key in gradients:
            assert torch.allclose(gradients[key], parameter.grad, atol=2e-5, rtol=2e-5), key


def test_checkpoint_continuation_preserves_optimizer_and_rejects_input_changes(tmp_path, model):
    from newsvendor.structured_progress import Progress

    old = variant(read("configs/structured.json"), "base")
    old["output"] = str(tmp_path / "old")
    parent = Progress(old, {"snapshot": "test"}, "cpu", None)
    parent.update("language", processed=4)
    optimizer = model.optimizer(old["training"])
    parent.checkpoint(
        "language-progress", model, {"epoch": 0, "offset": 4, "order": list(range(8))}, optimizer
    )
    fast = copy.deepcopy(old)
    fast["output"] = str(tmp_path / "fast")
    fast["encoder"].update(gradientCheckpointing=False, bucketLengths=True, paddingRatio=1.25)
    fast["training"].update(caseBatch=2, prefetchWorkers=2)
    child = Progress(fast, {"snapshot": "test"}, "cpu", None)
    path = parent.directory / "language-progress.pt"
    child.import_language(path)
    restored = child.restore("language-progress", model, optimizer)
    assert restored == {"epoch": 0, "offset": 4, "order": list(range(8))}
    assert read(child.directory / "continuation.json")["identity"] == parent.identity
    third_config = copy.deepcopy(fast)
    third_config["output"] = str(tmp_path / "third")
    third = Progress(third_config, {"snapshot": "test"}, "cpu", None)
    third.import_language(child.directory / "language-progress.pt")
    chained = read(third.directory / "continuation.json")
    prior = read(child.directory / "continuation.json")
    assert chained["parent"] == prior
    assert chained["previousSeconds"] > prior["previousSeconds"]
    bad = copy.deepcopy(fast)
    bad["output"] = str(tmp_path / "bad")
    bad["encoder"]["maxLength"] += 1
    rejected = Progress(bad, {"snapshot": "test"}, "cpu", None)
    with pytest.raises(ValueError, match="objective or inputs"):
        rejected.import_language(path)


def test_family_comparison_requires_alignment_and_bootstraps_source_groups():
    from newsvendor.structured_compare import means, paired

    a = [
        {"key": "a", "family": "g1", "metrics": {"loss": 2.0}},
        {"key": "b", "family": "g1", "metrics": {"loss": 4.0}},
        {"key": "c", "family": "g2", "metrics": {"loss": 10.0}},
    ]
    b = [r | {"metrics": {"loss": r["metrics"]["loss"] - 1}} for r in a]
    result = paired(a, b)["loss"]
    assert result["familyMeanDifference"] == 1 and result["ci95"] == [1, 1]
    assert result["families"] == 2 and result["pairedCases"] == 3
    assert means(a)["loss"]["familyMean"] == 6.5
    with pytest.raises(ValueError, match="cases differ"):
        paired(a, b[:-1])


def test_backbone_size_does_not_change_shared_head_initialization(settings):
    weights = []
    for hidden in (32, 64):
        encoder = ModernBertModel(
            ModernBertConfig(
                vocab_size=64,
                hidden_size=hidden,
                intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=4,
                pad_token_id=0,
                bos_token_id=1,
                cls_token_id=1,
                eos_token_id=2,
                sep_token_id=2,
            )
        )
        torch.manual_seed(42)
        router = Router(encoder, settings)
        weights.append(
            {
                k: v
                for k, v in router.state_dict().items()
                if not k.startswith(("encoder.", "project."))
            }
        )
    assert all(torch.equal(weights[0][k], weights[1][k]) for k in weights[0])


def test_shared_heads_backpropagate_to_modernbert_and_preserve_masked_annotations(
    tokenizer, settings, model
):
    value = payload("Refund", documents=[{"id": "doc", "title": "policy", "text": "Refund 10"}])
    view = prepare(value, tokenizer, settings)
    targets = suite_targets(
        view, "cuad", {"action": "answer", "spans": [{"document": "doc", "start": 0, "end": 9}]}
    )
    output = model(view)
    loss, heads = objective(output, targets)
    assert "type" not in heads and "state" not in heads
    assert targets["fields"][0]["spans"]
    loss.backward()
    assert (
        sum(float(p.grad.abs().sum()) for p in model.encoder.parameters() if p.grad is not None) > 0
    )
    assert model.heads["kind"].layers[0].weight.grad is None
    optimizer = model.optimizer({"encoderLr": 1e-5, "headLr": 1e-3})
    assert [g["lr"] for g in optimizer.param_groups] == [1e-5, 1e-3]
    assert all(p.requires_grad for p in model.encoder.parameters())


def test_internal_arguments_and_question_target_require_no_helper(tokenizer, settings, model):
    value = payload(
        "Order id",
        history=[{"role": "user", "text": "Order id A123"}],
        tools=[
            {"id": "ship", "description": "shipping", "argumentSlots": ["order_id", "shipping"]}
        ],
    )
    view = prepare(value, tokenizer, settings)
    target = suite_targets(
        view, "abcd", {"action": "call_tool", "tool": "ship", "arguments": ["A123", "unobserved"]}
    )
    assert target["fields"][1]["spans"]
    assert target["fields"][2] == {}
    output = model(view)
    for name in ("mode", "state", "evidence", "start", "end", "recovery", "question"):
        output[name] = torch.zeros_like(output[name])
    output["state"][:, 0] = 10
    output["mode"][:, MODES.index("span")] = 10
    start, end = target["fields"][1]["spans"][0]
    output["start"][1, start] = 20
    output["end"][1, end] = 20
    output["mode"][2] = 0
    output["mode"][2, MODES.index("missing")] = 10
    action = next(i for i, a in enumerate(view["actions"]) if a["id"] == "call_tool:ship")
    output["recovery"][action] = 20
    result = assemble(view, output, use_value=False)
    assert result["fields"][1]["value"] == "A123"
    assert result["action"] == "ask" and result["question"]["field"] == "ship:shipping"
    output["mode"][2] = 0
    output["mode"][2, MODES.index("span")] = 10
    output["start"][2, start], output["end"][2, end] = 20, 20
    result = assemble(view, output, use_value=False)
    assert result["action"] == "call_tool"
    assert result["arguments"] == {"order_id": "A123", "shipping": "A123"}


def test_ordered_operands_and_no_value_ablation(tokenizer, settings, model):
    view = prepare(
        payload("Refund", documents=[{"id": "d", "title": "Refund", "text": "10 2"}]),
        tokenizer,
        settings,
    )
    assert compute("subtract", [10, 2]) == 8 and compute("subtract", [2, 10]) == -8
    assert compute("divide", [10, 2]) == 5
    with pytest.raises(ValueError, match="Zero divisor"):
        compute("divide", [10, 0])
    with pytest.raises(ValueError, match="Nonfinite"):
        compute("multiply", [1e308, 1e308])
    output = model(view)
    target = {"fields": [], "values": [[1.0, 0.3]], "valueIndices": [0], "recovery": 0}
    loss, heads = objective(output, target, no_value=True)
    assert "value" not in heads
    loss.backward()
    assert model.heads["value"].layers[0].weight.grad is None
    output["recovery"] = torch.arange(len(view["actions"]), dtype=torch.float32)
    before = assemble(view, output, no_value=True)["action"]
    output["value"] = output["value"].flip(0) * 100
    assert assemble(view, output, no_value=True)["action"] == before
    config = read("configs/structured.json")
    base, large, ablated = [variant(config, name) for name in ("base", "large", "no_value")]
    assert base["training"] == large["training"] == ablated["training"]
    assert base["encoder"] == ablated["encoder"]
    assert base["encoder"]["maxLength"] == large["encoder"]["maxLength"]


def test_paired_initialization_shares_heads_and_imports_only_demand(
    settings, tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from newsvendor import structured_train

    class Encoder(torch.nn.Module):
        def __init__(self, width, value=0.25):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=width)
            self.weight = torch.nn.Parameter(torch.full((width,), value))

    original = Router(Encoder(32, 0.9), settings)
    with torch.no_grad():
        original.demand.gru.weight_ih_l0.fill_(0.75)
        original.heads["kind"].layers[0].weight.fill_(0.8)
    parent_config = {"encoder": {**settings, "model": "test-base", "revision": "pinned"}}
    path = tmp_path / "parent.pt"
    torch.save(
        {
            "config": parent_config,
            "datasetHashes": {"data": "fixed"},
            "weights": original.state_dict(),
        },
        path,
    )
    monkeypatch.setattr(structured_train, "dataset_hashes", lambda config: {"data": "fixed"})
    monkeypatch.setattr(
        structured_train,
        "load_backbone",
        lambda config: (None, Encoder(32 if config["model"] == "test-base" else 48)),
    )
    config = {**parent_config, "seed": 42, "warmStart": None, "demandCheckpoint": str(path)}
    base, _, before = structured_train.initialize(config)
    large, _, after = structured_train.initialize(
        {**config, "encoder": {**config["encoder"], "model": "test-large"}}
    )
    assert before["sharedHeadsHash"] == after["sharedHeadsHash"]
    assert before["demandWeightsHash"] == after["demandWeightsHash"]
    assert before["mode"] == "pretrained_encoder_and_fresh_heads" and "checkpoint" not in before
    assert torch.equal(base.demand.gru.weight_ih_l0, original.demand.gru.weight_ih_l0)
    assert base.encoder.weight.eq(0.25).all() and large.encoder.weight.eq(0.25).all()
    assert not torch.equal(
        base.heads["kind"].layers[0].weight, original.heads["kind"].layers[0].weight
    )
    assert base.project.weight.shape != large.project.weight.shape
    warm = {**config, "warmStart": str(path), "demandCheckpoint": None}
    restored, _, record = structured_train.initialize(warm)
    assert record["mode"] == "warm_start"
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in original.state_dict().items())
    with pytest.raises(ValueError, match="backbone identity"):
        structured_train.initialize(
            {**warm, "encoder": {**warm["encoder"], "revision": "different"}}
        )


def test_standalone_demand_checkpoint_initialization(model, settings, tmp_path, monkeypatch):
    from newsvendor import structured_train

    config = {
        "encoder": {**settings, "model": "test-base", "revision": "pinned"},
        "seed": 42,
        "demandCheckpoint": str(tmp_path / "demand.pt"),
    }
    with torch.no_grad():
        model.demand.gru.weight_ih_l0.fill_(0.125)
    payload = {
        "schema": structured_train.DEMAND_SCHEMA,
        "config": config,
        "datasetHashes": {"snapshot": "fixed"},
        "weights": {"demand." + k: v for k, v in model.demand.state_dict().items()},
    }
    torch.save(payload, config["demandCheckpoint"])
    monkeypatch.setattr(structured_train, "dataset_hashes", lambda c: {"snapshot": "fixed"})
    monkeypatch.setattr(
        structured_train, "load_backbone", lambda c: (None, copy.deepcopy(model.encoder))
    )
    restored, _, parent = structured_train.initialize(config)
    assert parent["mode"] == "pretrained_encoder_and_fresh_heads"
    assert all(
        torch.equal(v, restored.demand.state_dict()[k])
        for k, v in model.demand.state_dict().items()
    )
    with pytest.raises(ValueError, match="demand-only checkpoint cannot warm-start"):
        structured_train.initialize(
            {**config, "demandCheckpoint": None, "warmStart": config["demandCheckpoint"]}
        )
    payload["weights"]["encoder.unexpected"] = torch.ones(1)
    torch.save(payload, config["demandCheckpoint"])
    with pytest.raises(ValueError, match="unrelated weights"):
        structured_train.initialize(config)


def test_standalone_demand_rejects_changed_source_before_loading_encoder(tmp_path, monkeypatch):
    from newsvendor import structured_train

    path = tmp_path / "demand.pt"
    torch.save(
        {
            "schema": structured_train.DEMAND_SCHEMA,
            "config": {},
            "datasetHashes": {"snapshot": "original"},
            "weights": {},
        },
        path,
    )
    monkeypatch.setattr(structured_train, "dataset_hashes", lambda c: {"snapshot": "changed"})
    monkeypatch.setattr(
        structured_train,
        "load_backbone",
        lambda c: pytest.fail("No encoder download on invalid data"),
    )
    with pytest.raises(ValueError, match="Parent checkpoint data changed"):
        structured_train.initialize({"demandCheckpoint": str(path)})


def series(n=35, start="2024-01-01"):
    return [
        {
            "date": (date.fromisoformat(start) + timedelta(days=i)).isoformat(),
            "sales": float(1 + i % 3),
            "stockoutHours": int(i == 30),
        }
        for i in range(n)
    ]


def forecast_input():
    value = copy.deepcopy(
        next(
            e["input"]
            for e in corpus.generate(read("configs/full.json"))
            if e["split"] == "train" and e["scenario"] == "sufficient"
        )
    )
    value["observations"] = series()
    value["task"].update(
        forecast={"start": "2024-02-05", "days": 7, "unit": "globally-normalized-sales"},
        quantityUnit="globally-normalized-sales",
        period="2024-02-05/2024-02-11",
        bounds=[0, 100],
    )
    value["historySource"] = {
        "id": "observed-series",
        "sku": value["task"]["sku"],
        "unit": value["task"]["quantityUnit"],
    }
    for doc in value["docs"]:
        doc["period"] = value["task"]["period"]
    return value


def observed_fields(value):
    """Controlled pointer outputs to isolate demand/solver/response integration."""
    from newsvendor.construction import candidates, matches

    fields = []
    for slot in corpus.SLOTS:
        eligible = [e for e in candidates(value, slot) if matches(e["doc"], slot)]
        expression = (
            max(eligible, key=lambda e: (e["doc"]["version"], len(e["args"]))) if eligible else None
        )
        fields.append(
            {
                "name": slot,
                "type": "preference" if slot == "b" else "fact",
                "value": expression["value"] if expression else None,
                "state": "verified" if expression else "unconfirmed",
                "expression": {
                    "op": expression["op"],
                    "operands": [
                        {"id": expression["doc"]["id"], "start": a["span"][0], "end": a["span"][1]}
                        for a in expression["args"]
                    ],
                }
                if expression
                else None,
            }
        )
    return fields


def test_learned_forecast_changes_actual_research_order(model, tokenizer, settings, monkeypatch):
    from newsvendor import structured_rollout
    from newsvendor.optimizer import optimal

    value = forecast_input()
    monkeypatch.setattr(structured_rollout, "extract", lambda *args: observed_fields(value))
    router = structured_rollout.ResearchRouter(model, tokenizer, settings)
    before = router.construct(value)
    assert before["valid"] and before["types"]["F"] == "estimate"
    assert before["omega"][0]["F"] == before["forecast"]["F"]
    assert before["q"] == optimal(before["omega"][0], value["task"]["bounds"])["q"]
    assert before["forecast"]["source"]["historyHash"] == digest(value["observations"])
    with torch.no_grad():
        # Change only the learned demand heads; language predictions stay identical.
        model.demand.family.layers[-1].weight.zero_()
        model.demand.family.layers[-1].bias.copy_(torch.tensor([0.0, 10.0, 10.0]))
        model.demand.params["truncated_normal"].layers[-1].weight.zero_()
        model.demand.params["truncated_normal"].layers[-1].bias.copy_(
            torch.tensor([-10.0, 3.0, -4.0])
        )
    after = router.construct(value)
    assert after["valid"] and after["values"] == before["values"]
    assert after["q"] != pytest.approx(before["q"])
    assert after["q"] == optimal(after["omega"][0], value["task"]["bounds"])["q"]


def test_forecast_manager_response_updates_own_state(model, tokenizer, settings, monkeypatch):
    from newsvendor import structured_rollout

    value = forecast_input()
    value["docs"] = [d for d in value["docs"] if d["title"] != "Manager decision"]
    monkeypatch.setattr(structured_rollout, "extract", lambda *args: observed_fields(value))
    router = structured_rollout.ResearchRouter(model, tokenizer, settings)
    before = router.construct(value)
    assert not before["valid"] and before["missing"] == ["b"]
    assert "b" in router.allowed(value, before) and "handoff" not in router.allowed(value, before)
    forecast = before["forecast"]
    value = corpus.outcome(value, "b", 7)
    after = router.construct(value)
    assert after["valid"] and after["values"]["b"] == 7
    assert after["links"]["b"] == "response-b"
    assert after["forecast"] == forecast
    assert "handoff" in router.allowed(value, after) and "b" not in router.allowed(value, after)
    assert set(router.allowed(value, after)) == {"handoff", "hold"}


def test_core_publishes_forecast_and_accepted_parameters_with_raw_diagnostics(
    model, tokenizer, settings, monkeypatch
):
    from newsvendor import structured_rollout

    value = forecast_input()
    predicted = observed_fields(value)
    predicted.append(
        {
            "field": "F",
            "name": "F",
            "state": "verified",
            "type": "fact",
            "value": 99999,
            "mode": "span",
        }
    )
    monkeypatch.setattr(structured_rollout, "extract", lambda *args: copy.deepcopy(predicted))
    router = structured_rollout.ResearchRouter(model, tokenizer, settings)
    state = router.construct(value)
    field = next(f for f in state["fields"] if f["name"] == "F")
    assert field["value"] == state["forecast"]["distribution"] and field["type"] == "estimate"
    assert field["evidence"][0]["kind"] == "series"
    assert field["evidence"][0]["historyHash"] == digest(value["observations"])
    assert next(f for f in state["rawFields"] if f["name"] == "F")["value"] == 99999
    assert state["valid"]
    for f in state["fields"]:
        if f["name"] in corpus.SLOTS:
            assert f["value"] == state["values"][f["name"]] and f["scale"] == ""


def test_core_preserves_preference_requirement_before_received_manager_choice(
    model, tokenizer, settings, monkeypatch
):
    from newsvendor import structured_retail, structured_rollout

    value = forecast_input()

    def wrong_type(*args):
        fields = observed_fields(value)
        next(f for f in fields if f["name"] == "b")["type"] = "fact"
        return fields

    monkeypatch.setattr(structured_rollout, "extract", wrong_type)
    router = structured_rollout.ResearchRouter(model, tokenizer, settings)
    before = router.construct(value)
    assert not before["valid"] and before["state"]["b"] == "unconfirmed"
    assert "b" not in before["values"]
    assert next(f for f in before["fields"] if f["name"] == "b")["value"] is None
    value = corpus.outcome(value, "b", 7)
    after = router.construct(value)
    assert after["valid"] and after["values"]["b"] == 7 and after["types"]["b"] == "preference"
    assert after["copiedResponses"]["b"]["historyIndex"] == 0
    measured = structured_retail.parameter_metrics(value, after)
    assert measured["rawTypeAccuracy"] == 0.75 and measured["typeAccuracy"] == 1


@pytest.mark.parametrize("defect", ["answer", "sku", "period", "incomplete", "newer_source"])
def test_core_cannot_copy_a_reply_without_matching_current_evidence(
    model, tokenizer, settings, monkeypatch, defect
):
    from newsvendor import structured_rollout

    value = corpus.outcome(forecast_input(), "c", 7)
    doc = next(d for d in value["docs"] if d["id"] == "response-c")
    if defect == "answer":
        value["history"][-1]["answer"] = 99
    elif defect == "sku":
        doc["sku"] = "another-sku"
    elif defect == "period":
        doc["period"] = "another-period"
    elif defect == "incomplete":
        doc["complete"] = False
    else:
        replacement = corpus.answer_doc(value, "c", 11, "new-quote")
        replacement["version"] = 4
        value["docs"].append(replacement)

    def missing(*args):
        fields = observed_fields(value)
        next(f for f in fields if f["name"] == "c").update(
            value=None, state="verified", mode="missing"
        )
        return fields

    monkeypatch.setattr(structured_rollout, "extract", missing)
    state = structured_rollout.ResearchRouter(model, tokenizer, settings).construct(value)
    assert "c" not in state["values"] and "c" not in state["copiedResponses"]
    assert state["state"]["c"] == "unconfirmed" and not state["valid"]
    assert next(f for f in state["rawFields"] if f["name"] == "c")["state"] == "verified"


def test_core_rejects_refund_as_purchase_cost_and_recovers_from_typed_reply(
    model, tokenizer, settings, monkeypatch
):
    from newsvendor import structured_retail, structured_rollout
    from newsvendor.construction import candidates

    value = forecast_input()

    def wrong_amount(*args):
        fields = observed_fields(value)
        expr = next(
            e
            for e in candidates(value, "c")
            if e["doc"]["title"] == "Current return contract"
            and e["op"] == "copy"
            and "refund" in e["args"][0]["label"].lower()
        )
        next(f for f in fields if f["name"] == "c").update(
            value=expr["value"],
            expression={
                "op": "copy",
                "operands": [
                    {
                        "id": expr["doc"]["id"],
                        "start": expr["args"][0]["span"][0],
                        "end": expr["args"][0]["span"][1],
                    }
                ],
            },
        )
        return fields

    monkeypatch.setattr(structured_rollout, "extract", wrong_amount)
    router = structured_rollout.ResearchRouter(model, tokenizer, settings)
    before = router.construct(value)
    assert "c" not in before["values"] and not before["valid"]
    assert before["state"]["c"] == "unconfirmed"
    value = corpus.outcome(value, "c", 40)
    value = corpus.outcome(value, "c", None)
    after = router.construct(value)
    assert after["values"]["c"] == 40 and after["links"]["c"] == "response-c"
    measured = structured_retail.parameter_metrics(value, after)
    assert measured["parameterAccuracy"] == 1 and measured["rawParameterAccuracy"] == 0.75


def test_core_retrieval_requires_observed_unread_chunks(model, tokenizer, settings, monkeypatch):
    from newsvendor import structured_rollout

    value = forecast_input()
    value["docs"] = [d for d in value["docs"] if d["title"] != "Manager decision"]
    monkeypatch.setattr(structured_rollout, "extract", lambda *args: observed_fields(value))
    complete = structured_rollout.ResearchRouter(model, tokenizer, settings)
    state = complete.construct(value)
    assert state["retrieval"]["unreadChunks"] == 0
    assert "retrieve" not in complete.allowed(value, state) and "b" in complete.allowed(
        value, state
    )
    limited = structured_rollout.ResearchRouter(model, tokenizer, {**settings, "chunks": 1})
    state = limited.construct(value)
    assert state["retrieval"]["unreadChunks"] > 0 and "retrieve" in limited.allowed(value, state)


def test_core_keeps_grounded_values_after_a_different_manager_reply(
    model, tokenizer, settings, monkeypatch
):
    from newsvendor import (
        structured_forecast,
        structured_retail,
        structured_rollout,
        structured_train,
    )

    value = forecast_input()
    cost = next(f["value"] for f in observed_fields(value) if f["name"] == "c")
    value["docs"] = [d for d in value["docs"] if d["title"] != "Purchase quotation"]

    def extracted(*args):
        fields = observed_fields(value)
        if value["history"]:
            next(f for f in fields if f["name"] == "v").update(
                value=None, mode="missing", state="unconfirmed"
            )
        return fields

    monkeypatch.setattr(structured_rollout, "extract", extracted)
    router = structured_rollout.ResearchRouter(model, tokenizer, settings)
    router.cache_states()
    before = router.construct(value)
    assert before["missing"] == ["c"]
    memory = structured_forecast.remember(value, before)
    value = corpus.outcome(value, "c", cost)
    without_memory = router.construct(value)
    assert without_memory["missing"] == ["v"]
    value["memory"] = memory
    after = router.construct(value)
    assert after["valid"] and after["values"]["v"] == before["values"]["v"]
    assert after["expressions"]["v"] == before["expressions"]["v"]
    field = next(f for f in after["fields"] if f["name"] == "v")
    assert field["copiedFromState"] == before["sourceHashes"]["v"] and field["evidence"]
    assert next(f for f in after["rawFields"] if f["name"] == "v")["value"] is None
    assert "v" not in router.allowed(value, after)
    measured = structured_retail.parameter_metrics(value, after)
    assert measured["parameterAccuracy"] == 1 and measured["rawParameterAccuracy"] == 0.75
    value.pop("memory")
    public, trace = structured_train.infer(
        model, tokenizer, value, {"encoder": settings}, linked=True, state=before
    )
    assert public["constructedState"]["values"] == after["values"]
    assert public["constructedState"]["q"] == after["q"] and trace[0]["memoryHash"] == digest(
        memory
    )


@pytest.mark.parametrize("slot", ["c", "p", "v", "b"])
def test_core_preserves_accepted_source_with_noncanonical_title(
    model, tokenizer, settings, monkeypatch, slot
):
    from newsvendor import structured_forecast, structured_rollout

    value = forecast_input()
    predicted = observed_fields(value)
    field = next(f for f in predicted if f["name"] == slot)
    source = field["expression"]["operands"][0]["id"]
    next(d for d in value["docs"] if d["id"] == source)["title"] = "Commercial terms, section 8"
    monkeypatch.setattr(structured_rollout, "extract", lambda *args: copy.deepcopy(predicted))
    router = structured_rollout.ResearchRouter(model, tokenizer, settings)
    before = router.construct(value)
    assert before["valid"] and before["values"][slot] == field["value"]
    memory = structured_forecast.remember(value, before)
    reply_slot = "c" if slot == "b" else "b"
    value = corpus.outcome(value, reply_slot, before["values"][reply_slot])
    value["memory"] = memory
    field.update(value=None, mode="missing", state="unconfirmed")
    after = router.construct(value)
    assert after["valid"] and after["values"][slot] == before["values"][slot]
    assert slot in after["copiedMemory"]
    assert after["sourceHashes"][slot] == before["sourceHashes"][slot]
    assert next(f for f in after["rawFields"] if f["name"] == slot)["value"] is None
    assert slot not in router.allowed(value, after)


@pytest.mark.parametrize(
    "defect",
    [
        "value",
        "source_hash",
        "sku",
        "period",
        "changed_source",
        "incomplete",
        "newer_source",
        "source_conflict",
        "predicted_conflict",
    ],
)
def test_core_rejects_stale_or_ungrounded_memory(model, tokenizer, settings, monkeypatch, defect):
    from newsvendor import structured_forecast, structured_rollout

    value = forecast_input()
    monkeypatch.setattr(structured_rollout, "extract", lambda *args: observed_fields(value))
    router = structured_rollout.ResearchRouter(model, tokenizer, settings)
    before = router.construct(value)
    value["memory"] = structured_forecast.remember(value, before)
    item = next(m for m in value["memory"] if m["slot"] == "v")
    doc = next(d for d in value["docs"] if d["id"] == before["links"]["v"])
    if defect == "value":
        item["value"] += 1
    elif defect == "source_hash":
        item["sourceHash"] = "wrong-source"
    elif defect in ("sku", "period"):
        value["task"][defect] = "another-scope"
        assert structured_forecast.remember(value, before) == []
    elif defect == "changed_source":
        doc["text"] += " Updated source."
        assert all(m["slot"] != "v" for m in structured_forecast.remember(value, before))
    elif defect == "incomplete":
        doc["complete"] = False
    elif defect in ("newer_source", "source_conflict"):
        replacement = corpus.answer_doc(value, "v", item["value"] + 1, "replacement-return")
        replacement["version"] = doc["version"] + (defect == "newer_source")
        value["docs"].append(replacement)

    def missing(*args):
        fields = observed_fields(value)
        next(f for f in fields if f["name"] == "v").update(
            value=None,
            mode="conflict" if defect == "predicted_conflict" else "missing",
            state="conflict" if defect == "predicted_conflict" else "unconfirmed",
        )
        return fields

    monkeypatch.setattr(structured_rollout, "extract", missing)
    after = router.construct(value)
    assert "v" not in after["values"] and "v" not in after["copiedMemory"]
    assert not after["valid"] and "v" in after["missing"]


@pytest.mark.parametrize("change", ["newer_value", "newer_same_value", "same_version_conflict"])
def test_core_rejects_newly_selected_stale_or_conflicting_evidence_and_requests_recovery(
    model, tokenizer, settings, monkeypatch, change
):
    from newsvendor import structured_rollout

    value = forecast_input()
    value["task"]["costs"]["c"] = 1.0
    predicted = observed_fields(value)
    selected = next(f for f in predicted if f["name"] == "c")
    source = next(
        d for d in value["docs"] if d["id"] == selected["expression"]["operands"][0]["id"]
    )
    amount = selected["value"] + (change != "newer_same_value")
    replacement = corpus.answer_doc(value, "c", amount, "updated-quote")
    replacement["version"] = source["version"] + (change != "same_version_conflict")
    value["docs"].append(replacement)
    monkeypatch.setattr(structured_rollout, "extract", lambda *args: copy.deepcopy(predicted))
    router = structured_rollout.ResearchRouter(model, tokenizer, settings)
    state = router.construct(value)
    assert "c" not in state["values"] and state["q"] is None and not state["valid"]
    expected = "conflict" if change == "same_version_conflict" else "unconfirmed"
    assert state["state"]["c"] == expected
    assert "c" in router.allowed(value, state) and "handoff" not in router.allowed(value, state)
    raw = next(f for f in state["rawFields"] if f["name"] == "c")
    assert raw == selected  # Rejecting stale evidence is not an improved raw prediction.
    final = next(f for f in state["fields"] if f["name"] == "c")
    assert final["value"] is None and final["reason"].endswith("source-c")
    # A later observed reply with current evidence can recover even if extraction stays stale.
    answered = corpus.outcome(value, "c", amount)
    reply = next(d for d in answered["docs"] if d["id"] == "response-c")
    reply["version"] = replacement["version"] + 1
    recovered = router.construct(answered)
    assert recovered["values"]["c"] == amount and recovered["state"]["c"] == "verified"
    assert recovered["links"]["c"] == "response-c" and "c" in recovered["copiedResponses"]
    assert not any(e.endswith("-c") for e in recovered["errors"])


def test_core_current_evidence_ignores_foreign_scope_and_other_parameter_versions(
    model, tokenizer, settings, monkeypatch
):
    from newsvendor import structured_rollout

    value = forecast_input()
    predicted = observed_fields(value)
    selected = next(f for f in predicted if f["name"] == "c")
    source = next(
        d for d in value["docs"] if d["id"] == selected["expression"]["operands"][0]["id"]
    )
    source["title"] = "Vendor document 17"
    for key in ("sku", "period"):
        foreign = corpus.answer_doc(value, "c", selected["value"] + 3, "foreign-" + key)
        foreign["version"], foreign[key] = 100, "other-scope"
        value["docs"].append(foreign)
    for doc in value["docs"]:
        if doc["title"] == "Sales price list":
            doc["version"] = 200
    monkeypatch.setattr(structured_rollout, "extract", lambda *args: copy.deepcopy(predicted))
    state = structured_rollout.ResearchRouter(model, tokenizer, settings).construct(value)
    assert state["values"]["c"] == selected["value"] and state["links"]["c"] == source["id"]
    assert state["state"]["c"] == "verified" and state["valid"]


def test_core_memory_does_not_turn_a_fact_into_a_manager_preference(
    model, tokenizer, settings, monkeypatch
):
    from newsvendor import structured_forecast, structured_rollout

    value = forecast_input()
    monkeypatch.setattr(structured_rollout, "extract", lambda *args: observed_fields(value))
    router = structured_rollout.ResearchRouter(model, tokenizer, settings)
    before = router.construct(value)
    value["memory"] = structured_forecast.remember(value, before)
    next(m for m in value["memory"] if m["slot"] == "b")["type"] = "fact"

    def missing(*args):
        fields = observed_fields(value)
        next(f for f in fields if f["name"] == "b").update(value=None, state="unconfirmed")
        return fields

    monkeypatch.setattr(structured_rollout, "extract", missing)
    after = router.construct(value)
    assert "b" not in after["values"] and "b" in router.allowed(value, after)


def test_public_linked_inference_uses_same_validated_forecast_and_questions(
    model, tokenizer, settings, monkeypatch
):
    from newsvendor import structured_rollout, structured_train

    value = forecast_input()
    monkeypatch.setattr(structured_rollout, "extract", lambda *args: observed_fields(value))
    expected = structured_rollout.ResearchRouter(model, tokenizer, settings).decision(value)
    actual, trace = structured_train.infer(
        model, tokenizer, value, {"encoder": settings, "noValue": False}, linked=True
    )
    assert actual == expected and trace[0]["historyHash"] == digest(value["observations"])
    with pytest.raises(ValueError, match="explicit Newsvendor task"):
        structured_train.infer(
            model,
            tokenizer,
            payload("Forecast next 7 daily sales", observations=series()),
            {"encoder": settings, "noValue": False},
            linked=True,
        )


@pytest.mark.parametrize(
    "defect", ["sku", "unit", "future", "gap", "horizon", "period", "stockout"]
)
def test_forecast_rejects_mismatched_or_future_history(model, defect):
    from newsvendor.structured_forecast import predict

    value = forecast_input()
    if defect == "sku":
        value["historySource"]["sku"] = "another-sku"
    elif defect == "unit":
        value["task"]["quantityUnit"] = "individual-items"
    elif defect == "future":
        value["observations"][-1]["date"] = "2024-02-05"
    elif defect == "gap":
        value["observations"].pop(4)
    elif defect == "horizon":
        value["task"]["forecast"]["days"] = 28
    elif defect == "period":
        value["task"]["period"] = "2024-02-05/2024-02-12"
    else:
        value["observations"][0].pop("stockoutHours")
    with pytest.raises(ValueError):
        predict(model, value)


def test_forecast_does_not_read_hidden_distribution_or_response_model(
    model, tokenizer, settings, monkeypatch
):
    from newsvendor import structured_rollout

    value = forecast_input()
    value["docs"] = [d for d in value["docs"] if d["title"] != "Manager decision"]
    monkeypatch.setattr(structured_rollout, "extract", lambda *args: observed_fields(value))
    router = structured_rollout.ResearchRouter(model, tokenizer, settings)
    before = router.construct(value)
    allowed = router.allowed(value, before)
    view = router.view(value, before)
    value["task"].update(allowed={"F": [[[-1, 2]]]}, rho={"b": 0}, partial={"b": 1})
    value["gold"] = {"theta": {"F": [[99999, 1]], "b": 99999}}
    after = router.construct(value)
    assert after == before
    assert router.allowed(value, after) == allowed
    assert torch.equal(view["batch"]["input_ids"], router.view(value, after)["batch"]["input_ids"])


def test_forecast_cache_tracks_numeric_history_after_compact_text_summary(
    model, tokenizer, settings, monkeypatch
):
    from newsvendor import structured_rollout

    value = forecast_input()
    monkeypatch.setattr(structured_rollout, "extract", lambda *args: observed_fields(value))
    router = structured_rollout.ResearchRouter(model, tokenizer, settings)
    router.cache_states()
    before = router.construct(value)
    text = research_input(value)
    value["observations"][0]["sales"] *= 100
    assert text == research_input(value)  # The demand encoder receives the complete observations.
    after = router.construct(value)
    assert before["forecast"]["source"]["historyHash"] != after["forecast"]["source"]["historyHash"]
    assert before["forecast"]["F"] != after["forecast"]["F"]


def test_forecast_invalid_binding_never_falls_back_to_hidden_candidates(
    model, tokenizer, settings, monkeypatch
):
    from newsvendor import structured_rollout

    value = forecast_input()
    value["historySource"]["sku"] = "another-sku"
    monkeypatch.setattr(structured_rollout, "extract", lambda *args: observed_fields(value))
    router = structured_rollout.ResearchRouter(model, tokenizer, settings)
    result = router.construct(value)
    assert not result["valid"] and result["q"] is None and result["omega"] == []
    assert result["forecast"] is None and result["missing"] == ["F"]
    assert "handoff" not in router.allowed(value, result)


def retail_forecast_cases():
    from newsvendor import structured_retail

    rows = [
        {
            "id": "retail-fixture",
            "family": "sales-family",
            "split": "train",
            "component": "retail",
            "input": payload("Forecast next 7 daily sales", observations=series()),
        }
    ]
    labels = {
        rows[0]["id"]: {
            "dates": [(date(2024, 2, 5) + timedelta(days=i)).isoformat() for i in range(7)],
            "answer": [2.0] * 7,
            "complete": [True] * 7,
        }
    }
    return rows, labels, structured_retail.cases(rows)


def test_retail_targets_never_change_observed_inputs_or_family_split():
    from newsvendor import structured_retail

    rows, labels, episodes = retail_forecast_cases()
    original = structured_retail.attach_targets(episodes, rows, labels)
    labels[rows[0]["id"]]["answer"] = [1000.0] * 7
    altered = structured_retail.attach_targets(episodes, rows, labels)
    assert [e["input"] for e in original] == [e["input"] for e in altered]
    assert original[-1]["target"] != altered[-1]["target"]
    assert all(e["family"] == rows[0]["family"] and e["split"] == "train" for e in episodes)
    for episode in original:
        assert episode["input"]["observations"][-1]["date"] < episode["target"]["dates"][0]
    with pytest.raises(ValueError, match="source-group leakage"):
        structured_retail.cases(rows + [{**rows[0], "id": "other", "split": "dev"}])


def test_pack_quantity_is_count_and_reference_cost_is_actual_division():
    from newsvendor.construction import atoms, parameter_record

    value = forecast_input()
    quotation = next(d for d in value["docs"] if d["title"] == "Purchase quotation")
    quotation["text"] = "Pack price = 216.305; units per pack = 10."
    assert [a["unit"] for a in atoms(quotation)] == ["currency/pack", "count"]
    assert parameter_record(value)["values"]["c"] == pytest.approx(21.6305)
    quotation["text"] = "Delivery days = 10; identifier = 216305."
    assert all(a["unit"] == "currency/unknown" for a in atoms(quotation))
    assert "c" not in parameter_record(value)["values"]


def test_retail_numeric_annotations_are_checked_independently_of_parser():
    from newsvendor import structured_retail

    rows, labels, episodes = retail_forecast_cases()
    episodes[0]["financialAnnotations"]["c"] += 1
    with pytest.raises(ValueError, match="document arithmetic"):
        structured_retail.attach_targets(episodes, rows, labels)


def test_research_backward_without_auxiliary_tool_modules(model, tokenizer, settings):
    from newsvendor.structured_train import language_backward, language_view

    _, _, episodes = retail_forecast_cases()
    case = ("research", episodes[0])
    config = {"encoder": settings, "training": {}, "noValue": False}
    view, target = language_view(case, tokenizer, config, {}, [])
    target.update(caseIndex=0, retrievalRound=1)
    model.train()
    _, _, workload = language_backward(model, [(view, target)], tokenizer, config, [case], {}, [])
    assert workload["retrievals"] == 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())


@pytest.mark.parametrize("slot", ["c", "p", "v", "b"])
def test_every_financial_request_updates_only_its_own_document(slot):
    from newsvendor.construction import parameter_record

    value = forecast_input()
    before = parameter_record(value)["values"]
    after = parameter_record(corpus.outcome(value, slot, 7))["values"]
    assert after[slot] == 7
    assert {k: v for k, v in before.items() if k != slot} == {
        k: v for k, v in after.items() if k != slot
    }


def test_retail_rollout_uses_response_then_scores_actual_quantity(
    model, tokenizer, settings, monkeypatch
):
    from newsvendor import structured_retail, structured_rollout

    rows, labels, episodes = retail_forecast_cases()
    episode = next(
        e
        for e in structured_retail.attach_targets(episodes, rows, labels)
        if e["cutoffIndex"] == 35 and e["scenario"] == "missing_b"
    )
    router = structured_rollout.ResearchRouter(model, tokenizer, settings)
    original = router.construct

    def controlled(value):
        monkeypatch.setattr(structured_rollout, "extract", lambda *args: observed_fields(value))
        return original(value)

    monkeypatch.setattr(router, "construct", controlled)
    result = structured_rollout.rollout(episode, router, explore=True)
    assert [e["action"] for e in result["events"]] == ["b", "handoff"]
    assert result["necessaryRequests"] == result["recoveredAfterRequest"] == 1
    assert result["unnecessaryRequests"] == result["unresolvedAfterRequest"] == 0
    assert result["parameters"]["parameterAccuracy"] == 1
    assert not result["falseHandoff"] and result["outcomeComplete"]
    final = result["events"][-1]["state"]
    v, q, y = final["values"], result["q"], sum(episode["target"]["answer"])
    actual_loss = (v["c"] - v["v"]) * max(q - y, 0) + (v["p"] - v["c"] + v["b"]) * max(y - q, 0)
    assert result["economicLoss"] == pytest.approx(actual_loss)
    assert result["total"] == pytest.approx(actual_loss + episode["input"]["task"]["costs"]["b"])
    assert result["forecastMetrics"]["distributionValid"] == 1
    assert result["q"] == final["q"]
    assert final["omega"][0]["F"] == final["forecast"]["F"]
    censored = copy.deepcopy(episode)
    censored["target"]["complete"][0] = False
    score = structured_retail.evaluate(censored, result["events"][-1]["input"], final, "handoff")
    assert score["economicLoss"] is None and score["terminalLoss"] is None
    assert score["censoredOrderLossLowerBound"] >= 0
    with pytest.raises(ValueError, match="complete observed demand"):
        structured_rollout.collect([censored], router)


def test_retail_rollout_carries_own_grounded_values_between_requests(
    model, tokenizer, settings, monkeypatch
):
    from newsvendor import structured_retail, structured_rollout

    rows, labels, episodes = retail_forecast_cases()
    episode = next(
        e
        for e in structured_retail.attach_targets(episodes, rows, labels)
        if e["cutoffIndex"] == 35 and e["scenario"] == "missing_c"
    )
    router = structured_rollout.ResearchRouter(model, tokenizer, settings)
    original = router.construct

    def controlled(value):
        def extracted(*args):
            fields = observed_fields(value)
            if value["history"]:
                next(f for f in fields if f["name"] == "v").update(value=None, state="unconfirmed")
            return fields

        monkeypatch.setattr(structured_rollout, "extract", extracted)
        return original(value)

    monkeypatch.setattr(router, "construct", controlled)
    result = structured_rollout.rollout(episode, router, explore=True)
    assert [e["action"] for e in result["events"]] == ["c", "handoff"]
    assert result["necessaryRequests"] == result["recoveredAfterRequest"] == 1
    assert result["unnecessaryRequests"] == result["unresolvedAfterRequest"] == 0
    initial, final = result["events"]
    assert {m["slot"] for m in final["input"]["memory"]} == set(initial["state"]["values"])
    assert final["state"]["values"]["v"] == initial["state"]["values"]["v"]
    assert result["parameters"]["rawParameterAccuracy"] == 0.75
    assert result["parameters"]["parameterAccuracy"] == 1 and not result["falseHandoff"]


def test_daily_and_weekly_observations_are_separate_and_feature_masks_are_observed_only():
    history = series()
    row = {
        "id": "r",
        "component": "retail",
        "family": "g",
        "split": "train",
        "input": payload("next 7-day", observations=history),
    }
    labels = {
        "r": {
            "answer": [2.0] * 7,
            "dates": [
                (date.fromisoformat(history[-1]["date"]) + timedelta(days=i)).isoformat()
                for i in range(1, 8)
            ],
            "complete": [True] * 7,
        }
    }
    samples = sequence.samples([row], labels)
    daily = next(r for r in samples if r["cutoff"] == history[27]["date"] and r["horizon"] == 1)
    week = next(r for r in samples if r["cutoff"] == history[27]["date"] and r["horizon"] == 7)
    assert not daily["censored"] and week["censored"]
    assert len(daily["dates"]) == 1 and len(week["dates"]) == 7
    assert torch.equal(daily["sequence"], week["sequence"])
    other = copy.deepcopy(labels)
    other["r"]["answer"] = [99.0] * 7
    altered = sequence.samples([row], other)
    assert all(
        torch.equal(a["sequence"], b["sequence"]) for a, b in zip(samples, altered, strict=True)
    )
    sequence_tensor, _ = sequence.features(history)
    assert sequence_tensor.shape == (35, sequence.INPUT_DIM)
    # Missing weather/hourly data has explicit zero masks.
    assert sequence_tensor[:, len(sequence.SCALARS) + 48 + 2].eq(0).all()


def test_source_crossfit_selects_loss_scores_and_numerical_parameters():
    model = sequence.DemandEncoder()
    seq, scale = sequence.features(series())
    training = [
        {
            "sequence": seq,
            "id": f"r{i}",
            "family": f"g{i}",
            "split": "train",
            "horizon": h,
            "y": float(i + 1),
            "censored": i % 2 == 0,
            "cutoff": "2024-02-04",
            "dates": ["2024-02-05"],
            "scale": scale * h,
            "historyHash": digest(i),
        }
        for i in range(3)
        for h in (1, 7)
    ]
    dev = [dict(training[0], family="dev", split="dev", censored=False)]
    dev[0]["horizon"] = 7
    config = {"epochs": 1, "selectorEpochs": 1, "folds": 3, "lr": 0.001}
    report, raw = sequence.cross_fit(model, training, dev, config, 42)
    assert len(raw) == len(training)
    assert all(set(r["fitFamilies"]).isdisjoint(r["heldFamilies"]) for r in report["folds"])
    assert all(len(r["familyNLL"]) == 3 for r in raw)
    with pytest.raises(ValueError, match="Train only"):
        sequence.cross_fit(model, training + dev, dev, config, 42)
    value = payload("next 7-day", observations=series())
    prediction = sequence.predict(model, value)
    dist = prediction["distribution"]
    assert "familyProbabilities" not in dist
    assert dist["family"] == min(dist["familyScores"], key=dist["familyScores"].get)
    metrics = demand.metrics(
        value,
        {
            "answer": [2] * 7,
            "complete": [True] * 7,
            "dates": [(date(2024, 2, 5) + timedelta(days=i)).isoformat() for i in range(7)],
        },
        prediction,
    )
    assert metrics["distributionValid"] == metrics["orderValid"] == 1
    assert torch.isfinite(torch.tensor(metrics["observationNLL"]))


def test_research_type_preference_and_rollout_values_use_actual_terminal_loss(tokenizer, settings):
    episodes = corpus.generate(read("configs/full.json"))
    episode = next(e for e in episodes if e["split"] == "train" and e["scenario"] == "sufficient")
    view = prepare(
        research_input(episode["input"]),
        tokenizer,
        settings,
        fields=FIELDS,
        actions=[{"id": a, "text": TEXT[a]} for a in TEXT],
    )
    target = research_targets(view, episode["input"])
    assert target["fields"][3]["type"] == corpus.KINDS.index("preference")

    class WrongRouter:
        def construct(self, value):
            state = reference(value)
            state["q"] = value["task"]["bounds"][1]
            state["gamma"] = 0.0  # A mistaken self-estimate must not become the value target.
            return state

        def choose(self, value, state):
            return "handoff"

    rows, measured = collect([episode], WrongRouter())
    handoff = next(r for r in measured if r["forcedAction"] == "handoff")
    assert handoff["terminalLoss"] > 0 and handoff["constructorState"]["gamma"] == 0
    assert rows[0]["values"][rows[0]["actions"].index("handoff")][0] == pytest.approx(
        handoff["terminalLoss"] / episode["input"]["task"]["hold"]
    )
    with pytest.raises(ValueError, match="Train/Dev"):
        collect([dict(episode, split="test")], WrongRouter())


def test_enum_types_confirmation_and_explicit_link_mask(tokenizer, settings, model):
    value = payload(
        "Order id",
        history=[{"role": "user", "text": "Order id A123"}],
        tools=[
            {
                "id": "ship",
                "description": "shipping",
                "requiresConfirmation": True,
                "parameters": {
                    "properties": {"speed": {"type": "integer", "enum": [1, 2]}},
                    "required": ["speed"],
                },
            }
        ],
    )
    view = prepare(value, tokenizer, settings)
    output = model(view)
    output["mode"] = torch.zeros_like(output["mode"])
    output["mode"][:, MODES.index("choice")] = 10
    output["state"] = torch.zeros_like(output["state"])
    output["state"][:, 0] = 10
    output["choice"] = torch.zeros_like(output["choice"])
    output["choice"][1, 1] = 10
    output["recovery"] = torch.zeros_like(output["recovery"])
    output["recovery"][-1] = 10
    result = assemble(view, output, use_value=False)
    assert result["action"] == "confirm" and result["arguments"] == {"speed": 2}
    view["confirmations"] = [digest({"tool": "ship", "arguments": {"speed": 2}})]
    assert assemble(view, output, use_value=False)["action"] == "call_tool"
    output["choice"][1] = torch.tensor([20.0, 0.0])
    assert assemble(view, output, use_value=False)["action"] == "confirm"
    # Observed source question text supervises a question field, never a missing-state label.
    value = payload(
        "Order id",
        history=[{"role": "user", "text": "Order id A123"}],
        tools=[{"id": "ship", "description": "shipping", "argumentSlots": ["order_id"]}],
    )
    view = prepare(value, tokenizer, settings)
    target = suite_targets(view, "abcd", {"action": "speak", "answer": "What is your order id?"})
    assert target["question"] == [1] and target["fields"][1] == {}
    value["observations"] = series()
    plain = prepare(value, tokenizer, settings)
    model.zero_grad(set_to_none=True)
    model(plain)["fieldState"].square().mean().backward()
    assert model.demand.gru.weight_ih_l0.grad is None
    linked = prepare(value, tokenizer, settings, linked=True)
    model.zero_grad(set_to_none=True)
    model(linked)["fieldState"].square().mean().backward()
    assert model.demand.gru.weight_ih_l0.grad.abs().sum() > 0


def test_training_stages_and_checkpoint_roundtrip(
    tokenizer, settings, model, tmp_path, monkeypatch
):
    from newsvendor import structured_train
    from newsvendor.io import write

    episodes = corpus.generate(read("configs/full.json"))
    chosen = [
        next(e for e in episodes if e["split"] == split and e["scenario"] == "sufficient")
        for split in ("train", "dev")
    ]
    config = variant(read("configs/structured.json"), "base")
    config["encoder"] = settings
    config["dataset"] = str(tmp_path / "dataset")
    manifest = tmp_path / "dataset" / "manifest.json"
    write(manifest, {"inputHash": digest([e["input"] for e in chosen])})
    config["expansion"] = None
    config["output"] = str(tmp_path)
    config["training"]["epochs"] = 1
    config["policy"].update(iterations=1, epochs=1)
    train, dev = [[("research", e)] for e in chosen]
    report = structured_train.train_language(model, tokenizer, config, train, dev, {}, [])
    assert report["selectedEpoch"] == 1
    report["policy"] = structured_train.train_policy(model, tokenizer, config, chosen)
    assert report["policy"][0]["ownStateTargets"] > 0
    assert "value" in report["policy"][0]["headLosses"]
    path = tmp_path / "model.pt"
    structured_train.save(path, model, config, report)
    monkeypatch.setattr(
        structured_train, "load_backbone", lambda config: (tokenizer, copy.deepcopy(model.encoder))
    )
    loaded, _, saved, metadata = structured_train.load(path)
    assert saved == config and metadata == report
    assert all(torch.equal(v, loaded.state_dict()[k]) for k, v in model.state_dict().items())
    write(manifest, {"inputHash": "changed"})
    with pytest.raises(ValueError, match="Checkpoint data changed"):
        structured_train.load(path)


def test_tool_actions_modes_and_unannotated_state_are_constrained(tokenizer, settings, model):
    from newsvendor.corpus import STATUSES
    from newsvendor.structured_model import extract

    value = payload(
        "Order id",
        history=[{"role": "user", "text": "Order id A123"}],
        tools=[
            {
                "id": "ship",
                "description": "shipping",
                "parameters": {
                    "properties": {"order_id": {"type": "string"}},
                    "required": ["order_id"],
                },
            }
        ],
    )
    view = prepare(value, tokenizer, settings)
    output = model(view)
    for i, action in enumerate(view["actions"]):
        if action["id"] in ("answer", "hold"):
            assert output["recovery"][i] < -1e8
    assert output["mode"][1, MODES.index("compute")] < -1e8
    output["mode"][:, MODES.index("span")] = 100
    output["state"][:] = 0
    output["state"][:, STATUSES.index("unavailable")] = 100
    output["recovery"][:] = 0
    output["recovery"][0] = 1000  # Even manually malformed scores cannot call an unallowed action.
    output["recovery"][-1] = 100
    a123 = next(
        i
        for i, loc in enumerate(view["locations"])
        if loc["kind"] == "history" and loc["start"] == 9
    )
    output["start"][:] = 0
    output["end"][:] = 0
    output["start"][1, a123] = output["end"][1, a123] = 100
    prediction = assemble(view, output, use_value=False)
    assert prediction["action"] == "call_tool" and prediction["arguments"] == {"order_id": "A123"}
    assert prediction["fields"][1]["state"] == "candidate"
    view["fields"][1]["stateRequired"] = True
    assert (
        extract(view, output)[1]["value"] is None
    )  # Explicit research/state supervision remains active.
    assert assemble(view, output, use_value=False)["action"] == "ask"
    output["mode"][1] = 0
    output["mode"][1, MODES.index("missing")] = 100
    view["fields"][1]["stateRequired"] = False
    assert assemble(view, output, use_value=False)["action"] == "ask"


def test_replay_rejects_heldout_and_retention_guards_each_component():
    from newsvendor.structured_policy import Replay, retained

    rows = [
        ("public", {"id": str(i), "family": str(i), "split": "train", "component": "abcd"})
        for i in range(4)
    ]
    labels = {str(i): {"action": "call_tool" if i % 2 else "speak"} for i in range(4)}
    sampled = Replay(rows, labels, 42).sample(8)
    assert len(sampled) == 8 and {labels[r["id"]]["action"] for _, r in sampled} == {
        "speak",
        "call_tool",
    }
    with pytest.raises(ValueError, match="Train only"):
        Replay([("public", {**rows[0][1], "split": "test"})], labels, 42)
    baseline = {"abcd": {"toolExact": 0.625}, "cuad": {"groundedScore": 0.5}}
    assert not retained(baseline, baseline)
    assert (
        retained(baseline, {"abcd": {"toolExact": 0.9}, "cuad": {"groundedScore": 0.4}})[0][
            "component"
        ]
        == "cuad"
    )


def test_span_contract_cannot_select_arithmetic_and_replay_anchor_has_gradients(
    tokenizer, settings, model
):
    from newsvendor.structured_policy import distillation

    value = payload(
        "Highlight the parts about Refund",
        documents=[{"id": "doc", "title": "policy", "text": "Refund 10 2"}],
    )
    view = prepare(value, tokenizer, settings)
    output = model(view)
    assert output["mode"][0, MODES.index("compute")] < -1e8
    teacher = {k: v.detach().clone() for k, v in output.items()}
    assert distillation(output, teacher, view, "cuad").item() < 1e-6
    teacher["start"][0, 0] += 5
    loss = distillation(output, teacher, view, "cuad")
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    assert model.encoder.embeddings.tok_embeddings.weight.grad.abs().sum() > 0


def test_language_distillation_preserves_public_retrieval_gradients_and_skips_research(
    model, tokenizer, settings
):
    from newsvendor.structured_train import language_backward, language_view, teacher_predictions

    config = {"encoder": settings, "training": {}, "noValue": False}
    episode = next(e for e in corpus.generate(read("configs/full.json")) if e["split"] == "train")
    row = {
        "id": "contract",
        "component": "cuad",
        "split": "train",
        "input": payload(
            "Refund",
            documents=[
                {"id": "irrelevant", "title": "Refund", "text": "Refund policy"},
                {"id": "source", "title": "cost", "text": "10 2"},
            ],
        ),
    }
    config["encoder"] = dict(
        settings, packedSources=True, chunkTokens=4, overlap=0, documentTokens=2
    )
    model.config.update(config["encoder"])
    train = [("research", episode), ("public", row)]
    labels = {
        "contract": {"action": "answer", "spans": [{"document": "source", "start": 0, "end": 2}]}
    }
    prepared = []
    for index, case in enumerate(train):
        view, target = language_view(case, tokenizer, config, labels, [])
        target.update(caseIndex=index, retrievalRound=1)
        prepared.append((view, target))
    assert prepared[1][1]["needsRetrieval"]
    teacher = copy.deepcopy(model).requires_grad_(False).eval()
    with torch.no_grad():
        model.pointers["start"].weight.add_(0.03)
    model.train().zero_grad(set_to_none=True)
    baseline, _, _ = language_backward(model, prepared, tokenizer, config, train, labels, [])
    before = model.encoder.embeddings.tok_embeddings.weight.grad.clone()
    model.zero_grad(set_to_none=True)
    values, measurements, cost = language_backward(
        model,
        prepared,
        tokenizer,
        config,
        train,
        labels,
        [],
        teacher=teacher,
        distill_weight=4,
    )
    anchors = [r["distillation"] for r in measurements if "distillation" in r]
    assert len(anchors) == 2 and max(anchors) > 0  # Initial public view and observed retrieval.
    assert values == baseline and cost["teacherTokens"] > 0
    assert not torch.equal(before, model.encoder.embeddings.tok_embeddings.weight.grad)
    assert all(p.grad is None for p in teacher.parameters())
    views, targets = zip(*prepared, strict=True)
    with pytest.raises(ValueError, match="Train only"):
        teacher_predictions(teacher, views, targets, [train[0], ("public", dict(row, split="dev"))])


def test_policy_cannot_publish_economic_improvement_with_tool_regression(
    tokenizer, settings, model, tmp_path, monkeypatch
):
    from newsvendor import structured_policy

    episodes = corpus.generate(read("configs/full.json"))
    chosen = [
        next(e for e in episodes if e["split"] == split and e["scenario"] == "sufficient")
        for split in ("train", "dev")
    ]
    config = variant(read("configs/structured.json"), "base")
    config["encoder"] = settings
    config["output"] = str(tmp_path)
    config["policy"].update(iterations=1, epochs=1)
    values = iter([{"abcd": {"toolExact": 0.625}}, {"abcd": {"toolExact": 0.0}}])
    costs = iter([100.0, 1.0])
    monkeypatch.setattr(structured_policy, "public_validation", lambda *args: next(values))
    monkeypatch.setattr(structured_policy, "rollout", lambda *args: {"total": next(costs)})
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    reports = structured_policy.fit(model, tokenizer, config, chosen)
    epoch = reports[0]["epochs"][0]
    assert epoch["devActualTotalLoss"] == 1 and not epoch["accepted"]
    assert epoch["retentionFailures"][0]["component"] == "abcd"
    assert reports[0]["selected"]["iteration"] is None
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in initial.items())


def test_common_warm_start_preserves_provenance_and_rejects_changed_inputs(tmp_path, model):
    from newsvendor.structured_progress import Progress

    config = variant(read("configs/l40s-efficient.json"), "base")
    config["output"] = str(tmp_path / "parent")
    hashes = {"snapshot": "fixed"}
    parent = Progress(config, hashes, "cpu", None)
    parent.checkpoint(
        "common", model, {"language": {"selectedEpoch": 3}, "demand": {}, "config": config}
    )
    child_config = copy.deepcopy(config)
    child_config["output"] = str(tmp_path / "child")
    child_config["policy"]["replaySize"] = 8
    child = Progress(child_config, hashes, "cpu", None)
    state = child.import_common(tmp_path / "parent/common.pt", model)
    assert state["commonLineage"]["identity"] == parent.identity
    assert state["commonLineage"]["checkpointHash"]
    assert state["config"] == child_config
    bad = copy.deepcopy(child_config)
    bad["output"] = str(tmp_path / "bad")
    bad["encoder"]["documentTokens"] *= 2
    with pytest.raises(ValueError, match="encoder"):
        Progress(bad, hashes, "cpu", None).import_common(tmp_path / "parent/common.pt", model)
    bad["output"] = str(tmp_path / "data-changed")
    with pytest.raises(ValueError, match="data/scope"):
        Progress(bad, {"snapshot": "changed"}, "cpu", None).import_common(
            tmp_path / "parent/common.pt", model
        )


def test_expansion_audit_rejects_protected_input_even_with_new_source_keys(tmp_path):
    from test_suite import pack

    from newsvendor import structured_data
    from newsvendor.io import jsonl, write

    base, extension = tmp_path / "base", tmp_path / "extension"
    old, labels, _ = pack(base)
    protected = next(r for r in old if r["id"] == "abcd:dev")
    doc = protected["input"]["documents"][0]
    key = digest(doc)
    row = copy.deepcopy(protected)
    row.update(id="abcd:extra", family="new-family", split="train", documentRefs=[key])
    row["input"]["documents"] = []
    label = {
        "id": row["id"],
        "officialSplit": "train",
        "leakageKeys": ["new-key"],
        "target": next(r["target"] for r in labels if r["id"] == "abcd:dev"),
    }
    docs = {key: doc}
    manifest = {
        "inputHash": digest([row]),
        "labelHash": digest([label]),
        "documentHash": digest(docs),
        "baseManifest": digest(read(base / "manifest.json")),
        "protectedSnapshotHash": digest([r for r in old if r["split"] != "train"]),
    }
    jsonl(extension / "inputs.jsonl", [row])
    jsonl(extension / "labels.jsonl", [label])
    write(extension / "documents.json", docs)
    write(extension / "manifest.json", manifest)
    with pytest.raises(ValueError, match="Expanded input overlap"):
        structured_data.audit(base, extension)


def test_positional_value_filling_does_not_require_every_ontology_slot(tokenizer, settings, model):
    value = payload(
        "Order id",
        history=[{"role": "user", "text": "Order id A123"}],
        tools=[
            {
                "id": "lookup",
                "description": "Order lookup",
                "argumentSlots": ["customer_name", "account_id"],
            }
        ],
    )
    view = prepare(value, tokenizer, dict(settings, positionalTools=True))
    target = suite_targets(
        view, "abcd", {"action": "call_tool", "tool": "lookup", "arguments": ["A123"]}
    )
    assert target["fields"][1]["use"] == 1 and target["fields"][2] == {"use": 0}
    output = model(view)
    for name in ("mode", "state", "recovery", "start", "end", "use"):
        output[name] = torch.zeros_like(output[name])
    output["state"][:, 0] = 10
    output["mode"][:, MODES.index("span")] = 10
    output["use"][1, 1] = 10
    start, end = target["fields"][1]["spans"][0]
    output["start"][1, start], output["end"][1, end] = 20, 20
    output["recovery"][-1] = 20
    result = assemble(view, output, use_value=False)
    assert result["action"] == "call_tool" and result["arguments"] == ["A123"]
    assert result["argumentFields"] == {"argument_1": "A123"}


def test_span_decoder_matches_brute_force_inside_source_boundaries(tokenizer, settings):
    value = payload("Order id", documents=[{"id": "d", "title": "policy", "text": "10 2 8 Refund"}])
    view = prepare(value, tokenizer, settings)
    torch.manual_seed(43)
    start, end = torch.randn(2, len(view["locations"]))
    actual = best_span(view, start, end, max_tokens=3)
    allowed = []
    for i in range(len(view["locations"])):
        for j in range(i, min(i + 3, len(view["locations"]))):
            try:
                span_value(view, i, j)
            except ValueError:
                continue
            allowed.append((float(start[i]) + float(end[j]), i, j))
    expected = max(allowed, key=lambda v: (v[0], -v[1], -v[2]))[1:]
    assert actual == expected


def test_retail_controller_never_calls_generation_helper_and_binds_approval_to_args(
    model, tokenizer, monkeypatch
):
    from types import SimpleNamespace

    from newsvendor import structured_orders
    from newsvendor.order_benchmark import decide
    from newsvendor.orders import catalog

    args = {"order_id": "order-1", "reason": "no longer needed"}
    prediction = {"action": "confirm", "tool": "cancel_pending_order", "arguments": args}
    seen = []

    def inference(model, tokenizer, value, config, **kwargs):
        seen.append(value)
        return copy.deepcopy(prediction), []

    monkeypatch.setattr(structured_orders, "infer", inference)
    actor = structured_orders.Actor(model, tokenizer, {})

    class NoHelper:
        def generate(self, messages):
            raise AssertionError("Controller must not generate assistant text")

    public = {"policy": "Confirm all changes.", "tools": catalog()}
    history = [{"role": "user", "content": "Please cancel order-1."}]
    environment = SimpleNamespace(confirmed=None, private_goal="hidden-goal")
    response, _ = decide("structured", public, history, environment, NoHelper(), actor, None)
    assert response["action"] == "confirm"

    assert "hidden-goal" not in str(seen)
    environment.confirmed = digest({"tool": prediction["tool"], "arguments": args})
    response, record = decide("structured", public, history, environment, NoHelper(), actor, None)
    assert response["action"] == "call_tool" and record["outputTokens"] == 0
    prediction["arguments"] = {**args, "order_id": "order-2"}
    response, _ = decide("structured", public, history, environment, NoHelper(), actor, None)
    assert response["action"] == "confirm"


@pytest.mark.parametrize("structured", [False, True])
def test_logged_replay_isolates_conversations_and_preserves_legacy_inputs(
    model, tokenizer, monkeypatch, structured
):
    from newsvendor import structured_tool_eval

    rows, labels, seen = [], {}, []
    for conversation in ("a", "b"):
        for turn in (1, 2):
            key = f"abcd:{conversation}:{turn}"
            rows.append(
                {
                    "id": key,
                    "family": "shared-source",
                    "component": "abcd",
                    "split": "test",
                    "input": payload(key, history=[{"role": "user", "text": "Order"}] * turn),
                }
            )
            labels[key] = {"action": "speak", "answer": "Order"}

    def inference(model, tokenizer, value, config, collection, state):
        seen.append((value["request"], copy.deepcopy(state)))
        return {"action": "speak", "memory": [value["request"]]}, []

    monkeypatch.setattr(structured_tool_eval, "infer", inference)
    result = structured_tool_eval.conversation_replay(
        model, tokenizer, {"encoder": {"structuredTools": structured}}, rows[::-1], labels, {}
    )
    assert [key for key, _ in seen] == [r["id"] for r in rows]
    assert all(r["family"] == "shared-source" for r in result)
    if structured:
        assert seen[0][1] is None and seen[2][1] is None
        assert seen[1][1]["memory"] == ["abcd:a:1"]
        assert seen[3][1]["memory"] == ["abcd:b:1"]
    else:
        assert all(state is None for _, state in seen)


def test_lookup_rollout_consumes_budget_and_records_actual_recovery_cost():
    from newsvendor.structured_rollout import rollout

    episode = next(
        e
        for e in corpus.generate(read("configs/full.json"))
        if e["split"] == "train" and e["scenario"] == "sufficient"
    )

    class Reader:
        config = {"retrievalCost": 2.0}

        def construct(self, value):
            state = reference(value)
            if not any(h["action"] == "retrieve" for h in value["history"]):
                state.update(valid=False, values={}, q=None)
            return state

        def allowed(self, value, state):
            return ["handoff", "hold"] if state["valid"] else ["retrieve", "hold"]

        def choose(self, value, state):
            return "handoff" if state["valid"] else "retrieve"

    record = rollout(episode, Reader())
    assert record["result"] == "handoff" and record["requestCost"] == 2.0
    assert record["events"][1]["input"]["remaining"] == episode["input"]["remaining"] - 1
    assert record["events"][0]["observedResponse"]["sourceHashes"]


def test_ontology_candidates_are_open_but_named_enums_and_types_are_strict():
    from newsvendor.structured_inputs import schemas
    from newsvendor.structured_tools import validate

    definitions = read("configs/abcd-schema-v2.json")["fields"]
    spec = schemas(
        [
            {
                "id": "enter-details",
                "description": "enter",
                "argumentSlots": ["details_slotval", "email"],
            }
        ],
        definitions,
        True,
    )[0]
    assert validate("person@example.com", spec) == "person@example.com"
    assert (
        validate("renew subscription", dict(spec, choices=["cancel shipment"]))
        == "renew subscription"
    )
    with pytest.raises(ValueError, match="outside enum"):
        validate("express", {"valueType": "string", "choices": ["standard"]})
    with pytest.raises(ValueError, match="Fractional"):
        validate("1.5", {"valueType": "integer"})
    with pytest.raises(ValueError, match="integer"):
        validate(True, {"valueType": "integer"})
    assert validate("2", {"valueType": "integer"}) == 2
    assert validate("false", {"valueType": "boolean"}) is False


def test_state_copy_uses_original_evidence_and_rejects_forged_values(tokenizer, settings, model):
    from newsvendor.structured_model import extract
    from newsvendor.structured_tool_heads import ToolHeads
    from newsvendor.structured_tools import memory

    config = dict(settings, structuredTools=True)
    model.tools = ToolHeads()
    value = payload(
        "Order id",
        history=[{"role": "user", "text": "Order id A123"}],
        tools=[{"id": "lookup", "description": "lookup", "argumentSlots": ["order_id"]}],
    )
    view = prepare(value, tokenizer, config)
    entity = view["entities"][0]
    old = [
        {
            "field": "lookup:order_id",
            "name": "order_id",
            "value": "A123",
            "evidence": entity["evidence"],
            "use": True,
        }
    ]
    stored = memory(old)
    copied = prepare(value, tokenizer, config, state={"memory": stored})
    own = next(i for i, e in enumerate(copied["entities"]) if e["origin"] == "state")
    output = model(copied)
    output["mode"][1] = -100
    output["mode"][1, MODES.index("entity")] = 100
    output["entity"][1] = -100
    output["entity"][1, own] = 100
    field = extract(copied, output)[1]
    assert field["value"] == "A123" and field["evidence"] == entity["evidence"]
    assert field["source"] == "state"
    forged = [dict(stored[0], value="SECRET")]
    rejected = prepare(value, tokenizer, config, state={"memory": forged})
    assert all(e["value"] != "SECRET" for e in rejected["entities"])
    assert memory([dict(field, mode="conflict", value=None)], stored) == []


def test_memory_accepts_only_selected_tool_and_checks_current_span(tokenizer, settings, model):
    from newsvendor.structured_model import extract
    from newsvendor.structured_tool_heads import ToolHeads
    from newsvendor.structured_tools import copy_agrees, remember

    evidence = [{"kind": "history", "id": "0", "start": 0, "end": 5}]
    fields = [
        {"field": name, "tool": name, "value": "Order", "evidence": evidence}
        for name in ("lookup", "update", None)
    ]
    assert [f["field"] for f in remember({"tool": "lookup", "fields": fields})] == ["lookup"]
    assert remember({"action": "respond", "fields": fields}) == []
    entity = {"origin": "state", "role": "name", "value": "Order"}
    assert not copy_agrees(entity, "Order", "id")
    assert not copy_agrees(entity, "Order id", "name")
    assert copy_agrees(entity, "Order", "name")
    config = dict(settings, structuredTools=True, copyAgreement=True)
    model.tools = ToolHeads()
    value = payload(
        "Order id",
        history=[{"role": "user", "text": "Order id A123"}],
        tools=[{"id": "lookup", "description": "lookup", "argumentSlots": ["order_id"]}],
    )
    view = prepare(value, tokenizer, config)
    output = model(view)
    output["mode"][1] = -100
    output["mode"][1, MODES.index("entity")] = 100
    output["entity"][1] = -100
    output["entity"][1, 0] = 100
    span = next(
        i
        for i, loc in enumerate(view["locations"])
        if loc["kind"] == "history" and loc["start"] == 0
    )
    output["start"][1] = output["end"][1] = -100
    output["start"][1, span] = output["end"][1, span] = 100
    field = extract(view, output)[1]
    assert field["value"] == "Order" and field["mode"] == "span"
    assert field["copyRejected"] == "current_field_disagrees"


def test_procedure_hints_cannot_evict_recent_history(tokenizer, settings):
    value = payload("Order id", history=[{"role": "user", "text": "Order id A123"}])
    base = prepare(value, tokenizer, settings)
    hints = prepare(value, tokenizer, settings, state={"procedureIds": ["x" * 1000]})
    assert torch.equal(base["batch"]["input_ids"], hints["batch"]["input_ids"])


def test_call_metrics_penalize_unnecessary_calls_and_weights_preserve_prior():
    from newsvendor.structured_metrics import tool_metrics
    from newsvendor.structured_train import action_weights, conversation_successors

    row = {"component": "abcd"}
    target = {"action": "speak", "answer": "Order id"}
    correct = tool_metrics(row, target, {"action": "speak", "answer": "Order id"})
    wrong = tool_metrics(row, target, {"action": "call_tool", "tool": "lookup", "arguments": []})
    assert correct["argumentsDecisionAccuracy"] == 1 and wrong["argumentsDecisionAccuracy"] == 0
    counts = {"speak": 900, "lookup": 90, "update": 10}
    weights = action_weights(counts, True)
    assert weights["speak"] == 1 and sum(
        weights[k] * counts[k] for k in ("lookup", "update")
    ) == pytest.approx(100)
    cases = [
        (
            "public",
            {
                "id": f"c:{i}",
                "component": "abcd",
                "split": "train",
                "input": {"history": list(range(i))},
            },
        )
        for i in (1, 2, 3)
    ]
    assert conversation_successors(cases) == {0: 1, 1: 2}
    cases[1][1]["split"] = "test"
    with pytest.raises(ValueError, match="Train only"):
        conversation_successors(cases)


def test_call_selection_uses_all_decisions_in_each_validation_schedule():
    from newsvendor.structured_train import tool_selection_key

    high_recall = {
        "observableToolAndArgumentsExact": 0.9,
        "toolExact": 0.95,
        "actionAccuracy": 0.7,
        "argumentsDecisionAccuracy": 0.6,
        "toolDecisionAccuracy": 0.65,
    }
    precise = {
        "observableToolAndArgumentsExact": 0.8,
        "toolExact": 0.85,
        "actionAccuracy": 0.9,
        "argumentsDecisionAccuracy": 0.85,
        "toolDecisionAccuracy": 0.87,
    }
    for interval in (0, 2048):
        config = {"training": {"selection": "calls", "validationEvery": interval}}
        assert tool_selection_key(precise, 2.0, config) > tool_selection_key(
            high_recall, 1.0, config
        )


def test_goal_selection_cannot_hide_poor_precision_with_recall():
    from newsvendor.structured_train import tool_selection_key

    config = {"training": {"selection": "goal"}}
    a = {
        "toolExact": 0.99,
        "observableToolAndArgumentsExact": 0.95,
        "correctCall": 0.2,
        "predictedCall": 0.4,
    }
    b = {
        "toolExact": 0.9,
        "observableToolAndArgumentsExact": 0.9,
        "correctCall": 0.27,
        "predictedCall": 0.3,
    }
    assert tool_selection_key(b, 3.0, config) > tool_selection_key(a, 1.0, config)


@pytest.mark.parametrize("residual,auxiliary", [(False, False), (True, False), (True, True)])
def test_contextual_controller_keeps_spans_and_trains_encoder(
    tokenizer, settings, model, residual, auxiliary
):
    from newsvendor.structured_control import Controller, selection

    model.control = Controller(residual, auxiliary)
    value = payload(
        "Order id",
        history=[{"role": "user", "text": "Order id A123"}],
        tools=[
            {"id": "lookup", "description": "Order lookup", "argumentSlots": ["order_id"]},
            {"id": "refund", "description": "Refund Order", "argumentSlots": ["order_id"]},
        ],
    )
    config = dict(settings, dialogueController=True, controllerLength=256)
    view = prepare({"input": value, "target": {"tool": "SECRET"}}, tokenizer, config)
    other = prepare({"input": value, "target": {"tool": "different"}}, tokenizer, config)
    plain = prepare(value, tokenizer, settings)
    assert view["locations"] == plain["locations"]
    assert torch.equal(view["batch"]["input_ids"], other["batch"]["input_ids"])
    target = suite_targets(
        view, "abcd", {"action": "call_tool", "tool": "lookup", "arguments": ["A123"]}
    )
    output = model(view)
    loss, measured = objective(output, target)
    assert {"callGate", "control"} <= measured.keys()
    assert ("controlContext" in measured) == auxiliary
    assert ("callGateContext" in measured) == auxiliary
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())
    assert all(
        any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())
        for head in (model.control.call, model.control.action)
    )
    call = next(i for i, a in enumerate(view["actions"]) if a["id"] == "call_tool:lookup")
    output["callGate"] = torch.tensor([-10.0, 10.0])
    output["controlRecovery"] = torch.ones(len(view["actions"]))
    output["controlRecovery"][output["callMask"].bool()] = -2
    output["controlRecovery"][call] = -1
    assert (
        selection(output, [i for i, allowed in enumerate(view["allowedActions"]) if allowed])
        == call
    )


def test_zero_controller_correction_preserves_learned_distribution():
    from newsvendor.structured_control import Controller, selection

    control = Controller(residual=True)
    for head in (control.action, control.call):
        torch.nn.init.zeros_(head.layers[-1].weight)
        torch.nn.init.zeros_(head.layers[-1].bias)
    view = {
        "controllerStart": 0,
        "actions": [{"id": a} for a in ("respond", "call_tool:a", "call_tool:b")],
    }
    prior = torch.tensor([0.0, 3.0, -1.0], requires_grad=True)
    output = {"recovery": prior}
    control(view, torch.randn(4, 256), output)
    assert selection(output, [0, 1, 2]) == 1
    assert selection(output, [0]) == 0
    assert torch.equal(output["controlRecovery"], prior)
    assert torch.allclose(output["callGate"].softmax(0)[1], prior.softmax(0)[1:].sum())
    # The call decision still trains the existing scores, including the non-call path.
    torch.nn.functional.cross_entropy(output["callGate"][None], torch.tensor([0])).backward()
    assert prior.grad[0] < 0 and torch.all(prior.grad[1:] > 0)


def test_workflow_targets_follow_trace_without_assuming_step_order():
    from newsvendor.structured_procedures import STAGES, workflow_targets

    def turn(speaker, tool=None, flow="refund"):
        return {"speaker": speaker, "targets": [flow, None, tool]}

    procedures = {"shipping/refund": {"steps": ["lookup", "refund", "lookup"]}}
    mapping = {"refund": "shipping/refund", "other": "other"}
    turns = [
        turn("agent"),
        turn("action", "refund"),
        turn("action", "lookup"),
        turn("agent"),
        turn("action", "extra"),
        turn("agent"),
    ]
    assert workflow_targets(turns, mapping, procedures) == [
        [1],
        [1],
        [0, 2],
        None,
        None,
        [STAGES - 1],
    ]
    assert workflow_targets(
        [turn("agent"), turn("action", "lookup", "other")], mapping, procedures
    ) == [None, None]


def test_workflow_future_targets_are_official_train_only(tmp_path, monkeypatch):
    import gzip
    import json

    from newsvendor.io import write
    from newsvendor.structured_procedures import annotations

    monkeypatch.chdir(tmp_path)
    text = json.dumps({"Shipping": {"subflows": {"Refund": {"actions": [{"button": "Lookup"}]}}}})
    turns = [
        {"speaker": "agent", "targets": ["refund", None, None, []]},
        {"speaker": "action", "targets": ["refund", None, "lookup", ["Alice Smith"]]},
    ]
    scenario = {"personal": {"customer_name": "Alice Smith"}}
    raw = gzip.compress(
        json.dumps(
            {
                "train": [{"convo_id": i, "delexed": turns, "scenario": scenario} for i in (1, 3)],
                "dev": [{"convo_id": 2, "delexed": turns, "scenario": scenario}],
            }
        ).encode()
    )
    config = {
        "repo": "fixture/repo",
        "revision": "pinned",
        "files": [{"path": "abcd_v1.1.json.gz", "sha256": digest(raw)}],
    }
    write("configs/complementary.json", {"sources": {"abcd": config}})
    write("schema.json", {"procedureMap": {"refund": "shipping/refund"}})
    url = "https://raw.githubusercontent.com/fixture/repo/pinned/abcd_v1.1.json.gz"
    path = tmp_path / "data/raw/complementary/abcd" / (digest(url)[:20] + ".raw")
    path.parent.mkdir(parents=True)
    path.write_bytes(raw)
    rows = [
        {
            "id": f"abcd:{i}:{turn}",
            "component": "abcd",
            "split": split,
            "input": {
                "documents": [{"text": text}],
                "tools": [{"id": "lookup", "argumentSlots": ["customer_name", "account_id"]}],
            },
        }
        for i, split in ((1, "train"), (2, "train"), (3, "dev"))
        for turn in (0, 1)
    ]
    assert annotations(rows, "schema.json", True) == {
        "abcd:1:0": {"procedure": "shipping/refund", "workflowNodes": [0]},
        "abcd:1:1": {"procedure": "shipping/refund", "workflowNodes": [0]},
        "abcd:2:0": {"procedure": "shipping/refund"},
        "abcd:2:1": {"procedure": "shipping/refund"},
    }
    result = annotations(rows, "schema.json", True, True)
    assert result["abcd:1:1"]["argumentRoles"] == [["customer_name"]]
    assert all("argumentRoles" not in v for k, v in result.items() if k != "abcd:1:1")


def test_source_role_labels_keep_ambiguity_and_mask_undeclared_values():
    from newsvendor.structured_procedures import argument_roles

    scenario = {
        "personal": {
            "customer_name": "Alice Smith",
            "username": "Alice Smith",
            "email": "a@example.com",
        },
        "order": {"account_id": "A123"},
        "product": {"names": ["white shirt"]},
    }
    assert argument_roles(
        ["ALICE SMITH", "A123", "a@example.com", "missing", ""],
        scenario,
        ["customer_name", "username", "account_id"],
    ) == [
        ["customer_name", "username"],
        ["account_id"],
        [],
        [],
        [],
    ]


def test_source_roles_add_missing_loss_without_narrowing_enum_labels(tokenizer, settings, model):
    from newsvendor.structured_tool_heads import ToolHeads

    model.tools = ToolHeads()
    value = payload(
        "Order id",
        history=[{"role": "user", "text": "Order id A123"}],
        tools=[
            {
                "id": "lookup",
                "description": "Order",
                "argumentSlots": ["customer_name", "account_id"],
            }
        ],
    )
    config = dict(settings, structuredTools=True, positionalTools=True)
    label = {
        "action": "call_tool",
        "tool": "lookup",
        "arguments": ["A123"],
        "argumentRoles": [["account_id"]],
    }
    view = prepare({"input": value, "target": label}, tokenizer, config)
    other = prepare(
        {"input": value, "target": dict(label, argumentRoles=[["customer_name"]])},
        tokenizer,
        config,
    )
    assert torch.equal(view["batch"]["input_ids"], other["batch"]["input_ids"])
    target = suite_targets(view, "abcd", label)
    index = next(i for i, field in enumerate(view["fields"]) if field.get("tool") == "lookup")
    assert target["fields"][index]["role"] == [view["roles"].index("account_id")]
    bare = suite_targets(view, "abcd", {k: v for k, v in label.items() if k != "argumentRoles"})
    assert "role" not in bare["fields"][index]
    loss, measured = objective(model(view), target)
    assert "role" in measured
    loss.backward()
    assert model.tools.role.weight.grad.abs().sum() > 0
    # A shared enum value does not become a unique action role merely because it
    # also appears in a customer's source scenario.
    view["fields"][index]["roles"] = [
        {"name": name, "choices": ["credit card"]} for name in ("customer_name", "account_id")
    ]
    label.update(arguments=["credit card"], argumentRoles=[["account_id"]])
    assert set(suite_targets(view, "abcd", label)["fields"][index]["role"]) == {
        view["roles"].index(name) for name in ("customer_name", "account_id")
    }


def test_workflow_conditional_nodes_learn_and_preserve_evidence(tokenizer, settings, model):
    import json

    from newsvendor.structured_control import Controller
    from newsvendor.structured_tool_heads import ToolHeads

    text = json.dumps(
        {
            "Shipping": {
                "subflows": {
                    "Refund": {"actions": [{"button": "Lookup"}, {"button": "Refund"}]},
                    "Change": {"actions": [{"button": "Update"}]},
                }
            }
        }
    )
    value = payload(
        "Order id",
        documents=[{"id": "policy", "title": "policy", "text": text}],
        history=[{"role": "user", "text": "Order id A123"}],
        tools=[{"id": "lookup", "description": "Order id", "argumentSlots": ["order_id"]}],
    )
    config = dict(
        settings,
        structuredTools=True,
        procedureContext=True,
        workflowProgress=True,
        dialogueController=True,
        controllerLength=1024,
        controllerPolicyTokens=384,
    )
    model.tools, model.control = ToolHeads(), Controller()
    target = {"action": "speak", "procedure": "shipping/refund", "workflowNodes": [1]}
    view = prepare({"input": value, "target": target}, tokenizer, config)
    other = prepare({"input": value, "target": dict(target, workflowNodes=[15])}, tokenizer, config)
    plain = prepare(value, tokenizer, dict(config, workflowProgress=False))
    assert torch.equal(view["batch"]["input_ids"], other["batch"]["input_ids"])
    assert view["locations"] == plain["locations"]
    output = model(view)
    assert output["stage"].shape == (2, 16)
    assert output["stage"][0, 2:15].max() < -1e8
    assert output["stage"][1, 1:15].max() < -1e8
    targets = suite_targets(view, "abcd", target)
    # Isolate node likelihood: gradients must reach the stage head and encoder
    # without relying on a reranker loss to supply them.
    loss, measured = objective(output, {k: targets[k] for k in ("procedure", "workflowNodes")})
    assert "stage" in measured
    loss.backward()
    for module in (model.encoder, model.tools.stage):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    masked = suite_targets(view, "abcd", {"action": "speak", "procedure": "shipping/refund"})
    assert "stage" not in objective(output, masked)[1]
    output["procedure"] = torch.tensor([100.0, -100.0])
    output["stage"] = output["stage"].detach().clone()
    output["stage"][0, 1] = 100
    result = assemble(view, output, use_value=False)
    node = result["progress"]["nextNodes"][0]
    assert node["tool"] == "refund" and node["node"] == 1
    source = node["source"]
    assert text[source["start"] : source["end"]].startswith('"Refund":')


def test_workflow_loss_marginalizes_duplicate_public_nodes():
    logits = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    loss, measured = objective(
        {"stage": logits, "procedure": torch.tensor([1.0, 0.0])},
        {"procedure": [0], "workflowNodes": [0, 2]},
    )
    expected = -logits[0].log_softmax(-1)[[0, 2]].logsumexp(0)
    assert measured["stage"] == pytest.approx(float(expected.detach()))
    loss.backward()
    assert logits.grad[1].count_nonzero() == 0 and logits.grad[0, 1] > 0


@pytest.mark.parametrize(
    "upgrade",
    [
        "controllerResidual",
        "controllerAuxiliary",
        "procedureContext",
        "rerankSupervision",
        "workflowProgress",
        "sourceRoleSupervision",
        "dialogueWorkflow",
    ],
)
def test_controller_upgrade_keeps_optimizer_and_revalidates_parent(tmp_path, model, upgrade):
    from newsvendor.structured_control import Controller
    from newsvendor.structured_progress import Progress

    model.control = Controller()
    old = variant(read("configs/structured.json"), "base")
    old["output"] = str(tmp_path / "parent")
    old["encoder"]["dialogueController"] = True
    if upgrade == "workflowProgress":
        old["encoder"].update(structuredTools=True, procedureContext=True, rerankSupervision=True)
    if upgrade == "dialogueWorkflow":
        old["encoder"].update(structuredTools=True, dialogueState=True, workflowProgress=True)
        model.control = Controller(history=True)
    if upgrade == "sourceRoleSupervision":
        old["encoder"]["structuredTools"] = True
    if upgrade == "rerankSupervision":
        old["encoder"]["structuredTools"] = True
        old["encoder"]["rerank"] = False
    if upgrade == "controllerAuxiliary":
        old["encoder"]["controllerResidual"] = True
    parent = Progress(old, {"snapshot": "fixture"}, "cpu", None)
    parent.update("language", processed=4)
    optimizer = model.optimizer(old["training"])
    model.control.call.layers[0].weight.square().sum().backward()
    optimizer.step()
    moments = copy.deepcopy(optimizer.state_dict())
    state = {"epoch": 0, "offset": 4, "order": [2, 0, 1, 3], "saved": model.state_dict()}
    parent.checkpoint("language-progress", model, state, optimizer)
    new = copy.deepcopy(old)
    new["output"] = str(tmp_path / "child")
    new["training" if upgrade == "sourceRoleSupervision" else "encoder"][upgrade] = True
    if upgrade == "procedureContext":
        new["encoder"].update(controllerLength=3072, controllerPolicyTokens=1536)
    if upgrade == "rerankSupervision":
        new["encoder"]["rerank"] = True
    child = Progress(new, {"snapshot": "fixture"}, "cpu", None)
    child.import_language(parent.directory / "language-progress.pt")
    restored = child.restore("language-progress", model, optimizer)
    assert restored["validateBest"] and restored["order"] == state["order"]
    assert read(child.directory / "continuation.json")[upgrade + "Upgrade"]
    for key, values in moments["state"].items():
        for name, value in values.items():
            assert torch.equal(value, optimizer.state_dict()["state"][key][name])
    bad = copy.deepcopy(new)
    bad["output"] = str(tmp_path / "bad")
    bad["encoder"]["controllerLength"] = 4096
    rejected = Progress(bad, {"snapshot": "fixture"}, "cpu", None)
    with pytest.raises(ValueError, match="objective or inputs"):
        rejected.import_language(parent.directory / "language-progress.pt")


def test_context_supervision_learns_even_when_prior_already_fits():
    context = torch.tensor([0.0, -1.0, 1.0], requires_grad=True)
    gate = torch.tensor([2.0, -2.0], requires_grad=True)
    output = {
        "controlRecovery": torch.tensor([-100.0, 100.0, -100.0]),
        "callGate": torch.tensor([-100.0, 100.0]),
        "callMask": torch.tensor([False, True, True]),
        "contextRecovery": context,
        "contextCallGate": gate,
    }
    loss, measured = objective(output, {"control": 1})
    assert measured["control"] < 1e-6 and measured["callGate"] < 1e-6
    assert measured["controlContext"] > 1 and measured["callGateContext"] > 1
    loss.backward()
    assert context.grad[1] < 0 and context.grad[2] > 0
    assert gate.grad[1] < 0 and gate.grad[0] > 0


def test_public_policy_conditions_survive_controller_encoding(tokenizer, settings):
    import json

    from newsvendor.structured_control import encode
    from newsvendor.structured_procedures import catalog, observed_catalog

    text = json.dumps(
        {
            "Shipping": {
                "subflows": {
                    "Refund": {
                        "instructions": ["Choose an appropriate option."],
                        "actions": [
                            {
                                "button": "N/A",
                                "text": "Ask before acting",
                                "subtext": ["Confirm policy first."],
                            },
                            {
                                "button": "Lookup",
                                "text": "Option 1",
                                "subtext": ["If missing, request manager."],
                            },
                            {
                                "button": "Refund",
                                "text": "Option 2",
                                "subtext": ["If confirmed, refund purchase."],
                            },
                        ],
                    }
                }
            }
        }
    )
    value = payload(
        "Order id",
        documents=[{"id": "policy", "title": "policy", "text": text}],
        history=[{"role": "user", "text": "Order A123"}],
        tools=[{"id": "lookup", "description": "Order", "argumentSlots": ["order_id"]}],
    )
    legacy, current = catalog(text)[0], observed_catalog(value, True)[0]
    assert " Steps: " in legacy["text"] and " Steps: " not in current["text"]
    assert [a["tool"] for a in current["actions"]] == [None, "lookup", "refund"]
    assert "Ask before acting Confirm policy first." in current["context"]
    assert "Option 2 If confirmed, refund purchase." in current["context"]
    assert text[current["start"] : current["end"]].startswith('"Refund":')
    config = dict(
        settings,
        structuredTools=True,
        dialogueController=True,
        procedureContext=True,
        controllerLength=3072,
        controllerPolicyTokens=1536,
    )
    view = prepare({"input": value, "target": {"procedure": "SECRET"}}, tokenizer, config)
    other = prepare({"input": value, "target": {"procedure": "different"}}, tokenizer, config)
    plain = prepare(
        value, tokenizer, dict(config, dialogueController=False, procedureContext=False)
    )
    assert torch.equal(view["batch"]["input_ids"], other["batch"]["input_ids"])
    assert view["locations"] == plain["locations"]
    seq = view["queryPositions"][view["controllerStart"]][0]
    assert tokenizer.convert_tokens_to_ids("manager") in view["batch"]["input_ids"][seq]
    with pytest.raises(ValueError, match="conditions exceed"):
        encode(
            value,
            view["actions"],
            view["procedures"],
            tokenizer,
            dict(config, controllerPolicyTokens=2),
        )


def test_action_options_do_not_become_count_indexed_transitions(tokenizer, settings, model):
    import json

    from newsvendor.structured_tool_heads import ToolHeads

    text = json.dumps(
        {
            "Shipping": {
                "subflows": {
                    "Refund": {
                        "instructions": ["Choose an option"],
                        "actions": [
                            {"button": "Lookup", "text": "Option 1"},
                            {"button": "Refund", "text": "Option 2"},
                        ],
                    }
                }
            }
        }
    )
    value = payload(
        "Order id",
        documents=[{"id": "policy", "title": "policy", "text": text}],
        history=[{"role": "user", "text": "Order A123"}],
        tools=[{"id": "lookup", "description": "Order", "argumentSlots": ["order_id"]}],
    )
    model.tools = ToolHeads()
    view = prepare(value, tokenizer, dict(settings, structuredTools=True, procedureContext=True))
    captured = []
    hook = model.tools.rerank.register_forward_pre_hook(
        lambda module, args: captured.append(args[0].detach())
    )
    try:
        model(view)
        model(dict(view, procedureContext=False))
    finally:
        hook.remove()
    assert captured[0][:, 257].count_nonzero() == 0
    assert captured[1][:, 257].count_nonzero() > 0


def test_performance_goal_requires_every_threshold_and_strict_loss_bound():
    from newsvendor.structured_metrics import performance_goal

    goal = {
        "toolExact": 0.9,
        "observableToolAndArgumentsExact": 0.9,
        "callPrecision": 0.9,
        "generatedTestTotalLossBelow": 240.35,
    }
    public = {
        "toolExact": 0.9,
        "observableToolAndArgumentsExact": 0.9,
        "correctCall": 0.27,
        "predictedCall": 0.3,
    }
    assert performance_goal(public, 240.34, goal)["passed"]
    assert performance_goal(public, 240.35, goal)["failures"] == ["generatedTotalLoss"]
    assert performance_goal(dict(public, predictedCall=0.4), 100.0, goal)["failures"] == [
        "callPrecision"
    ]


def test_controller_does_not_replace_other_public_task_decisions(tokenizer, settings, model):
    from newsvendor.structured_control import Controller

    model.control = Controller()
    value = payload(
        "Am I eligible?",
        history=[{"role": "user", "text": "Order id A123"}],
        tools=[{"id": "retrieve_rules", "description": "Find rules", "argumentSlots": ["query"]}],
    )
    view = prepare(value, tokenizer, dict(settings, dialogueController=True, controllerLength=256))
    assert view["controllerStart"] is None and "callGate" not in model(view)


def test_construction_replay_rejects_heldout_and_task_mixing_keeps_sources(tmp_path):
    from newsvendor.io import jsonl
    from newsvendor.structured_train import balance_cases, replay_construction

    episodes = [
        {
            "id": "r",
            "family": "f",
            "scenario": "sufficient",
            "split": "train",
            "input": {"observed": 1},
        }
    ]
    path = tmp_path / "states.jsonl"
    row = {"id": "r", "family": "f", "split": "train", "input": {"observed": 2}}
    jsonl(path, [row, row])
    replay = replay_construction([str(path)], episodes)
    assert len(replay) == 1 and replay[0][1]["input"] == row["input"]
    cases = replay + [("public", {"id": str(i), "component": "abcd"}) for i in range(6)]
    cases += [("public", {"id": "contract", "component": "cuad"})]
    mixed = balance_cases(cases, {"mix": {"tools": 0.6, "research": 0.3, "public": 0.1}})
    assert len(mixed) == 10 and sum(mode == "research" for mode, _ in mixed) == 3
    assert {r["id"] for _, r in mixed} == {r["id"] for _, r in cases}
    jsonl(path, [dict(row, split="test")])
    with pytest.raises(ValueError, match="Train only"):
        replay_construction([str(path)], episodes)


def test_document_and_research_mixture_preserves_every_case_without_tools():
    from newsvendor.structured_train import balance_cases

    cases = [("research", {"id": "r"})] + [
        ("public", {"id": f"d{i}", "component": "cuad"}) for i in range(3)
    ]
    mixed = balance_cases(cases, {"mix": {"research": 0.5, "public": 0.5}})
    assert len(mixed) == 6 and sum(mode == "research" for mode, _ in mixed) == 3
    assert {row["id"] for _, row in mixed} == {row["id"] for _, row in cases}
    with pytest.raises(ValueError, match="omitted"):
        balance_cases(
            cases + [("public", {"id": "tool", "component": "abcd"})],
            {"mix": {"research": 0.5, "public": 0.5}},
        )


@pytest.mark.parametrize("no_value", [False, True])
def test_policy_fitting_updates_only_selected_head(
    tokenizer, settings, model, tmp_path, monkeypatch, no_value
):
    from newsvendor import structured_critic, structured_policy
    from newsvendor.structured_progress import Progress

    config = variant(read("configs/l40s-tools-v4.json"), "base")
    config.update(encoder=settings, output=str(tmp_path), noValue=no_value)
    config["policy"].update(iterations=1, epochs=2, validationEvery=1, batchSize=2)
    config["policy"]["trainable"] = "recovery" if no_value else "value"
    episodes = corpus.generate(read("configs/full.json"))
    chosen = [
        next(e for e in episodes if e["split"] == split and e["scenario"] == "sufficient")
        for split in ("train", "dev")
    ]
    costs = iter([100.0, 5.0, 3.0])
    monkeypatch.setattr(
        structured_policy, "public_validation", lambda *args: {"abcd": {"toolExact": 0.8}}
    )
    monkeypatch.setattr(structured_critic, "rollout", lambda *args: {"total": next(costs)})
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    progress = Progress(config, {}, "cpu", None)
    reports = structured_policy.fit(model, tokenizer, config, chosen, progress)
    assert reports[0]["devActualTotalLoss"] == 3 and reports[0]["frozenWeightsUnchanged"]
    name = "recovery" if no_value else "value"
    assert all(
        torch.equal(v, model.state_dict()[k])
        for k, v in initial.items()
        if not k.startswith(f"heads.{name}.")
    )
    assert any(
        not torch.equal(v, model.state_dict()[k])
        for k, v in initial.items()
        if k.startswith(f"heads.{name}.")
    )
    candidate = torch.load(tmp_path / "policy-candidate-0-1.pt", weights_only=True)
    base_path = tmp_path / candidate["state"]["base"]["path"]
    assert candidate["state"]["format"] == "policy-head-v1"
    assert digest(base_path.read_bytes()) == candidate["state"]["base"]["hash"]
    base = torch.load(base_path, weights_only=True)
    restored = dict(base["weights"])
    restored.update({f"heads.{name}.{k}": v for k, v in candidate["weights"].items()})
    assert all(torch.equal(v, restored[k]) for k, v in model.state_dict().items())
    assert set(candidate["weights"]) == set(model.heads[name].state_dict())
    assert candidate["optimizer"]["state"]
    assert reports[0]["checkpointBase"] == candidate["state"]["base"]
    original_base_hash = digest(base_path.read_bytes())
    torch.rand(7)  # Continuing work changes RNG and the selected head, not the fixed base.
    costs = iter([3.0, 2.0, 1.0])
    structured_policy.fit(model, tokenizer, config, chosen, progress)
    assert digest(base_path.read_bytes()) == original_base_hash
    continued = torch.load(tmp_path / "policy-candidate-0-1.pt", weights_only=True)
    restored = dict(base["weights"])
    restored.update({f"heads.{name}.{k}": v for k, v in continued["weights"].items()})
    assert all(torch.equal(v, restored[k]) for k, v in model.state_dict().items())
    if no_value:
        assert "no economic targets" in reports[0]["target"]
        from newsvendor.io import lines

        assert all(
            "values" not in row and "valueIndices" not in row
            for row in lines(tmp_path / "recovery-targets-0.jsonl")
        )


def test_recovery_targets_do_not_depend_on_hidden_economic_outcomes(
    model, tokenizer, settings, monkeypatch
):
    from newsvendor import structured_retail, structured_rollout

    rows, labels, episodes = retail_forecast_cases()
    episode = next(
        e
        for e in structured_retail.attach_targets(episodes, rows, labels)
        if e["cutoffIndex"] == 35 and e["scenario"] == "sufficient"
    )
    monkeypatch.setattr(
        structured_rollout, "extract", lambda *args: observed_fields(episode["input"])
    )
    router = structured_rollout.ResearchRouter(model, tokenizer, settings, no_value=True)
    targets, raw = structured_rollout.collect([episode], router, with_values=False)
    changed = copy.deepcopy(episode)
    changed["target"]["answer"] = [n + 100 for n in changed["target"]["answer"]]
    other, measured = structured_rollout.collect([changed], router, with_values=False)
    assert targets == other
    assert all("values" not in t and "valueIndices" not in t for t in targets)
    assert len(raw) == len(measured) == 1
    assert raw[0]["total"] != measured[0]["total"]
    assert all("forcedAction" not in r for r in raw)


@pytest.mark.parametrize("rerank,supervise", [(True, False), (False, True)])
def test_procedure_and_reranker_learn_without_gold_inputs(
    tokenizer, settings, model, rerank, supervise
):
    import json

    from newsvendor.structured_procedures import catalog, visible
    from newsvendor.structured_tool_heads import ToolHeads

    text = json.dumps(
        {
            "Shipping": {
                "subflows": {
                    "Refund": {
                        "instructions": ["Refund Order id"],
                        "actions": [{"button": "Lookup"}],
                    },
                    "Change": {
                        "instructions": ["shipping standard"],
                        "actions": [{"button": "Update"}],
                    },
                }
            }
        }
    )
    value = payload(
        "Order id",
        documents=[{"id": "policy", "title": "policy", "text": text}],
        history=[{"role": "user", "text": "Order id person@example.com A123"}],
        tools=[
            {"id": "lookup", "description": "Order id", "argumentSlots": ["email", "account_id"]}
        ],
    )
    config = dict(
        settings,
        structuredTools=True,
        positionalTools=True,
        rerank=rerank,
        rerankSupervision=supervise,
    )
    model.tools = ToolHeads()
    view = prepare({"input": value, "target": {"procedure": "SECRET"}}, tokenizer, config)
    other = prepare({"input": value, "target": {"procedure": "different"}}, tokenizer, config)
    assert torch.equal(view["batch"]["input_ids"], other["batch"]["input_ids"])
    assert len(catalog(text)) == 2 and visible(view, view["procedures"][0])
    target = suite_targets(
        view,
        "abcd",
        {
            "action": "call_tool",
            "tool": "lookup",
            "arguments": ["person@example.com"],
            "procedure": "shipping/refund",
            "stage": 1,
        },
    )
    output = model(view)
    loss, heads = objective(output, target)
    assert {"entity", "role", "procedure", "stage", "baseRecovery", "recovery"} <= heads.keys()
    loss.backward()
    for module in (
        model.encoder,
        model.tools.entity,
        model.tools.role,
        model.tools.procedure,
        model.tools.stage,
        model.tools.rerank,
    ):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    # Same weights, inputs and base scores; only the added candidate score is disabled.
    model.eval()
    model.tools.rerank[-1].bias.data.fill_(1)
    on = model(dict(view, rerank=True))
    off = model(dict(view, rerank=False))
    assert torch.equal(on["baseRecovery"], off["baseRecovery"])
    assert torch.equal(off["recovery"], off["baseRecovery"])
    assert torch.allclose(on["recovery"], on["baseRecovery"] + on["rerankDelta"])
    if supervise:
        assert torch.equal(on["rerankRecovery"], off["rerankRecovery"])
        # The decoder toggle changes the ablation, not which branch gets supervised.
        assert torch.allclose(objective(on, target)[0], objective(off, target)[0])


def test_positional_gap_does_not_shift_arguments(tokenizer, settings, model):
    value = payload(
        "Order id",
        history=[{"role": "user", "text": "Order id A123"}],
        tools=[
            {
                "id": "lookup",
                "description": "Order lookup",
                "argumentSlots": ["customer_name", "account_id"],
            }
        ],
    )
    view = prepare(value, tokenizer, dict(settings, positionalTools=True))
    output = model(view)
    output["use"][:] = 0
    output["use"][2, 1] = 100
    output["mode"][:] = 0
    output["mode"][:, MODES.index("span")] = 100
    output["recovery"][:] = 0
    output["recovery"][-1] = 100
    result = assemble(view, output, use_value=False)
    assert result["action"] == "ask" and "arguments" not in result


def test_own_policy_collection_records_real_invalid_handoff_penalty():
    from newsvendor.structured_rollout import rollout

    episodes = corpus.generate(read("configs/full.json"))
    episode = next(e for e in episodes if e["split"] == "train" and e["scenario"] == "sufficient")

    class WrongState:
        def construct(self, value):
            state = reference(value)
            state["omega"] = []
            return state

        def allowed(self, value, state):
            return ["hold", "handoff"]

        def choose(self, value, state):
            return "handoff"

    result = rollout(episode, WrongState())
    assert result["falseHandoff"] and result["terminalLoss"] >= episode["input"]["task"]["hold"]
    assert (
        result["total"]
        == result["economicLoss"] + result["invalidHandoffPenalty"] + result["requestCost"]
    )
    targets, raw = collect([episode], WrongState(), policy="own")
    assert targets[0]["state"]["omega"] == []
    assert any(r["invalidHandoffPenalty"] > 0 for r in raw)


def test_tool_dev_plateau_stops_with_resumable_selected_weights(
    tokenizer, settings, model, tmp_path, monkeypatch
):
    from newsvendor import structured_policy
    from newsvendor.structured_progress import Progress
    from newsvendor.structured_train import train_language

    config = variant(read("configs/structured.json"), "base")
    config["encoder"] = settings
    config["output"] = str(tmp_path)
    config["training"].update(
        epochs=4,
        accumulation=2,
        caseBatch=2,
        checkpointEvery=2,
        selection="tools",
        validationEvery=2,
        minCases=4,
        patience=1,
    )
    episodes = corpus.generate(read("configs/full.json"))
    train = [("research", e) for e in episodes if e["split"] == "train"][:6]
    dev = [("research", e) for e in episodes if e["split"] == "dev"][:1]
    monkeypatch.setattr(
        structured_policy,
        "public_validation",
        lambda *args: {
            "abcd": {
                "observableToolAndArgumentsExact": 0.5,
                "toolExact": 0.5,
                "actionAccuracy": 0.5,
            }
        },
    )
    progress = Progress(config, {"snapshot": "fixture"}, "cpu", None)
    report = train_language(model, tokenizer, config, train, dev, {}, [], progress)
    assert report["earlyStopped"] and report["checks"][-1]["seenCases"] == 4
    weights = {k: v.clone() for k, v in model.state_dict().items()}
    restored = train_language(model, tokenizer, config, train, dev, {}, [], progress)
    assert restored == report
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in weights.items())


@pytest.mark.parametrize("validation_every", [0, 1])
def test_research_selection_preserves_values_and_evidence_when_state_score_improves(
    tokenizer, settings, model, tmp_path, monkeypatch, validation_every
):
    from newsvendor import structured_train
    from newsvendor.structured_progress import Progress

    config = variant(read("configs/structured.json"), "base")
    config.update(encoder=settings, output=str(tmp_path))
    config["training"].update(
        epochs=2,
        accumulation=1,
        caseBatch=1,
        checkpointEvery=1,
        selection="research",
        validationEvery=validation_every,
        validateInitial=True,
        patience=100,
    )
    episodes = corpus.generate(read("configs/full.json"))
    train = [("research", e) for e in episodes if e["split"] == "train"][:1]
    dev = [("research", e) for e in episodes if e["split"] == "dev"][:1]
    baseline = {
        "parameterAccuracy": 1.0,
        "rawStateAccuracy": 3156 / 3168,
        "rawTypeAccuracy": 1.0,
        "evidenceAccuracy": 1.0,
        "allParametersCorrect": 1.0,
    }
    first = baseline | {"rawStateAccuracy": 3161 / 3168}
    second = baseline | {
        "parameterAccuracy": 3165 / 3168,
        "rawStateAccuracy": 3164 / 3168,
        "evidenceAccuracy": 0.9988847583643122,
        "allParametersCorrect": 789 / 792,
    }
    assert structured_train.tool_selection_key(second, 0.1, config) > (
        structured_train.tool_selection_key(first, 0.8, config)
    )
    metrics = iter([baseline, first, second])
    weights = []

    def measure(*args, **kwargs):
        current = next(metrics)
        weights.append({k: v.clone() for k, v in model.state_dict().items()})
        return structured_train.tool_selection_key(current, 0.5, config), 0.5, current

    monkeypatch.setattr(structured_train, "tool_selection", measure)
    progress = Progress(config, {"snapshot": "fixture"}, "cpu", None)
    report = structured_train.train_language(model, tokenizer, config, train, dev, {}, [], progress)
    assert report["selectedEpoch"] == 1
    rejected = report["checks"][-1] if validation_every else report["epochs"][-1]
    assert not rejected["accepted"]
    assert {r["metric"] for r in rejected["retentionFailures"]} == {
        "parameterAccuracy",
        "evidenceAccuracy",
    }
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in weights[1].items())
    restored = structured_train.train_language(
        model, tokenizer, config, train, dev, {}, [], progress
    )
    assert restored == report
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in weights[1].items())


@pytest.mark.parametrize("goal_stop", [False, True])
def test_language_budget_and_goal_stopping(
    tokenizer, settings, model, tmp_path, monkeypatch, goal_stop
):
    from newsvendor import structured_policy
    from newsvendor.structured_train import train_language

    config = variant(read("configs/structured.json"), "base")
    config.update(encoder=settings, output=str(tmp_path))
    config["training"].update(
        epochs=4,
        accumulation=2,
        caseBatch=2,
        checkpointEvery=2,
        selection="goal",
        validationEvery=2,
        minCases=100,
        maxCases=8,
        patience=100,
        stopAtGoalChecks=2 if goal_stop else 0,
        minGoalCases=4,
    )
    config["performanceGoal"] = {
        "toolExact": 0.9,
        "observableToolAndArgumentsExact": 0.9,
        "callPrecision": 0.9,
        "generatedTestTotalLossBelow": 240.35,
    }
    episodes = corpus.generate(read("configs/full.json"))
    train = [("research", e) for e in episodes if e["split"] == "train"][:6]
    dev = [("research", e) for e in episodes if e["split"] == "dev"][:1]
    monkeypatch.setattr(
        structured_policy,
        "public_validation",
        lambda *args: {
            "abcd": {
                "observableToolAndArgumentsExact": 0.95,
                "toolExact": 0.95,
                "correctCall": 0.285,
                "predictedCall": 0.3,
            }
        },
    )
    report = train_language(model, tokenizer, config, train, dev, {}, [])
    last = report["checks"][-1]
    assert last["seenCases"] == (4 if goal_stop else 8)
    assert last["goalStopped"] == goal_stop and last["budgetStopped"] != goal_stop


def test_unused_field_conflict_does_not_erase_active_memory():
    from newsvendor.structured_tools import memory

    previous = [
        {"field": "lookup:argument_1", "name": "argument_1", "value": "A123", "evidence": []}
    ]
    field = dict(previous[0], use=False, mode="conflict", value=None)
    assert memory([field], previous) == previous
    assert memory([dict(field, use=True)], previous) == []


def test_missing_question_is_restricted_to_selected_tool(tokenizer, settings, model):
    value = payload(
        "Order id",
        history=[{"role": "user", "text": "Order id"}],
        tools=[{"id": "lookup", "description": "lookup", "argumentSlots": ["order_id"]}],
    )
    fields = [
        {"id": "answer", "name": "answer", "description": "answer"},
        {
            "id": "lookup:order_id",
            "name": "order_id",
            "description": "Order id",
            "tool": "lookup",
            "required": True,
        },
        {
            "id": "question:order_id",
            "name": "order_id",
            "description": "Order id",
            "questionField": True,
        },
        {"id": "question:email", "name": "email", "description": "email", "questionField": True},
    ]
    view = prepare(value, tokenizer, settings, fields=fields)
    out = model(view)
    out["mode"][:] = 0
    out["mode"][:, MODES.index("missing")] = 100
    out["question"][:] = 0
    out["question"][2, 0] = 10
    out["question"][3, 0] = 100
    out["recovery"][:] = 0
    out["recovery"][-1] = 100
    result = assemble(view, out, use_value=False)
    assert result["action"] == "ask" and result["question"]["field"] == "question:order_id"


def test_computed_memory_replays_only_verified_original_expression():
    from newsvendor.structured_tools import entities, memory

    raw = [
        {"kind": "cell", "id": "table", "row": 0, "column": i, "text": text}
        for i, text in enumerate(["10", "2"])
    ]
    spans = [
        {k: v for k, v in r.items() if k != "text"} | {"start": 0, "end": len(r["text"])}
        for r in raw
    ]
    field = {
        "field": "refund:amount",
        "name": "amount",
        "value": 8.0,
        "evidence": spans,
        "expression": {"op": "subtract", "operands": spans},
        "mode": "compute",
    }
    state = {"memory": memory([field])}
    copied = entities(raw, state)
    assert len(copied) == 1 and copied[0]["value"] == 8
    assert copied[0]["expression"] == field["expression"]
    assert not entities(raw, {"memory": memory([dict(field, value=99)])})
    changed = copy.deepcopy(raw)
    changed[1]["text"] = "3"
    assert not entities(changed, state)


def test_candidate_value_does_not_hide_learned_question_target(tokenizer, settings, model):
    value = payload(
        "Order id",
        history=[{"role": "user", "text": "Order id A123"}],
        tools=[{"id": "lookup", "description": "lookup", "argumentSlots": ["order_id"]}],
    )
    view = prepare(value, tokenizer, settings)
    out = model(view)
    out["mode"][:] = 0
    out["mode"][:, MODES.index("span")] = 100
    out["mode"][0, MODES.index("missing")] = 200
    out["question"][:] = 0
    out["question"][1, 0] = 100
    out["recovery"][:] = 0
    out["recovery"][next(i for i, a in enumerate(view["actions"]) if a["id"] == "ask")] = 100
    result = assemble(view, out, use_value=False)
    assert result["fields"][1]["value"] is not None
    assert result["fields"][1]["state"] == "candidate"
    assert result["question"]["field"] == "lookup:order_id"


def test_question_only_outputs_never_become_copied_values():
    from newsvendor.structured_tools import memory

    field = {
        "field": "question:email",
        "name": "email",
        "value": "untrained pointer",
        "evidence": [{"kind": "history", "id": "0", "start": 0, "end": 17}],
        "queryOnly": True,
    }
    assert memory([field]) == []


def test_context_rerank_corrects_call_and_tool_without_changing_prior():
    from newsvendor.structured_control import selection
    from newsvendor.structured_tool_heads import ToolHeads

    tools = ToolHeads()
    actions = torch.zeros(4, 256)
    queries = torch.zeros(5, 256, requires_grad=True)
    features = torch.zeros(4, 6)
    features[2, 1] = 1
    features[3, 1] = 100
    view = {"controllerStart": 0, "rerank": True, "allowedActions": [True, True, True, False]}
    scores, gate = torch.tensor([0.0, 2.0, 0.0, 1000.0]), torch.log(torch.tensor([0.75, 0.25]))

    def output():
        return {
            "controlRecovery": scores,
            "callGate": gate,
            "callMask": torch.tensor([False, True, True, True]),
            "rerankFeatures": features,
        }

    zero = output()
    tools.rerank_context(view, actions, queries, zero)
    assert torch.allclose(zero["callGate"].softmax(0), gate.softmax(0))
    assert selection(zero, [0, 1, 2]) == selection(output(), [0, 1, 2]) == 0
    assert torch.equal(zero["baseControlRecovery"], scores)
    with torch.no_grad():
        tools.rerank[0].weight.zero_()
        tools.rerank[0].bias.zero_()
        tools.rerank[0].weight[0, 257] = 4
        tools.rerank[0].weight[0, 0] = 1
        tools.rerank[-1].weight[0, 0] = 1
    on, off = output(), output()
    tools.rerank_context(view, actions, queries, on)
    tools.rerank_context(dict(view, rerank=False), actions, queries, off)
    assert selection(on, [0, 1, 2]) == 2
    assert selection(off, [0, 1, 2]) == 0
    assert on["rerankJoint"][3] < -1e8  # forbidden candidates never gain probability
    assert torch.equal(on["rerankJoint"], off["rerankJoint"])
    assert torch.equal(off["controlRecovery"], scores)
    loss = -on["rerankJoint"].log_softmax(0)[2]
    loss.backward()
    assert queries.grad[2, 0] < 0  # final ranking uses the contextual candidate representation
    assert torch.isfinite(queries.grad).all()


@pytest.mark.parametrize("policy", [False, True])
def test_context_rerank_loss_and_encoder_gradients_survive_ablation(
    tokenizer, settings, model, policy
):
    import json

    from newsvendor.structured_control import Controller
    from newsvendor.structured_tool_heads import ToolHeads

    config = dict(
        settings,
        structuredTools=True,
        dialogueController=True,
        contextRerank=True,
        controllerLength=1024,
        controllerPolicyTokens=384,
    )
    model.tools, model.control = ToolHeads(), Controller(residual=True, auxiliary=True)
    documents = (
        [
            {
                "id": "policy",
                "title": "policy",
                "text": json.dumps(
                    {
                        "Shipping": {
                            "subflows": {
                                "Lookup": {
                                    "instructions": ["Order id"],
                                    "actions": [{"button": "Lookup"}],
                                }
                            }
                        }
                    }
                ),
            }
        ]
        if policy
        else []
    )
    value = payload(
        "Order id",
        documents=documents,
        history=[{"role": "user", "text": "Order id A123"}],
        tools=[
            {"id": "lookup", "description": "Order lookup", "argumentSlots": ["order_id"]},
            {"id": "refund", "description": "Refund Order", "argumentSlots": ["order_id"]},
        ],
    )
    view = prepare({"input": value, "target": {"tool": "SECRET"}}, tokenizer, config)
    other = prepare({"input": value, "target": {"tool": "different"}}, tokenizer, config)
    assert torch.equal(view["batch"]["input_ids"], other["batch"]["input_ids"])
    target = suite_targets(
        view, "abcd", {"action": "call_tool", "tool": "lookup", "arguments": ["A123"]}
    )
    model.eval()
    on, off = model(dict(view, rerank=True)), model(dict(view, rerank=False))
    loss_on, metrics_on = objective(on, target)
    loss_off, metrics_off = objective(off, target)
    assert "rerank" in metrics_on and "baseRecovery" not in metrics_on
    assert torch.equal(on["rerankJoint"], off["rerankJoint"])
    assert torch.equal(on["baseControlRecovery"], off["controlRecovery"])
    assert torch.allclose(on["callGate"].softmax(0), off["callGate"].softmax(0), atol=1e-7)
    assert torch.allclose(loss_on, loss_off) and metrics_on == metrics_off
    for enabled, output in ((True, on), (False, off)):
        decoded = assemble(dict(view, rerank=enabled), output, use_value=False)
        assert decoded["actionScoreSpace"] == "joint_call_and_conditional_tool"
        allowed = [i for i, permit in enumerate(view["allowedActions"]) if permit]
        assert len(decoded["actionScores"]) == len(allowed)
        for record, index in zip(decoded["actionScores"], allowed, strict=True):
            assert record["base"] == float(output["rerankPrior"][index].detach())
            assert record["reranked"] == pytest.approx(float(output["rerankJoint"][index].detach()))
    loss_off.backward()
    for module in (model.encoder, model.control.action, model.control.call, model.tools.rerank):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    from newsvendor.structured_train import language_backward

    following = dict(
        value, history=value["history"] + [{"role": "tool", "text": "Order id A123 found"}]
    )
    train = [
        ("public", {"id": key, "component": "abcd", "split": "train", "input": observed})
        for key, observed in (("first", value), ("next", following))
    ]
    labels = {
        "first": {"action": "call_tool", "tool": "lookup", "arguments": ["A123"]},
        "next": {"action": "speak", "answer": "Order found"},
    }
    model.train().zero_grad(set_to_none=True)
    target.update(caseIndex=0, retrievalRound=1, nextIndex=1)
    training_config = {"encoder": config, "training": {"nextTurnTraining": True}, "noValue": False}
    values, heads, cost = language_backward(
        model, [(view, target)], tokenizer, training_config, train, labels, []
    )
    assert len(values) == 1 and len(heads) == 2 and cost["retrievals"] == 1
    assert all("rerank" in item for item in heads)


def test_dialogue_turns_reuse_tokens_preserve_order_and_truncation(tokenizer, settings):
    from newsvendor.structured_control import encode

    history = [
        {"role": "user", "text": "Order id A123"},
        {"role": "tool", "text": "Order " * 200},
        {"role": "assistant", "text": "Refund request"},
        {"role": "user", "text": "standard"},
    ]
    value = payload(
        "Order",
        history=history,
        tools=[{"id": "lookup", "description": "Order", "argumentSlots": ["order_id"]}],
    )
    config = dict(settings, structuredTools=True, dialogueController=True, controllerLength=256)
    plain = prepare(value, tokenizer, config)
    view = prepare(
        {"input": value, "target": {"pastTools": [{"historyIndex": 1, "tool": "SECRET"}]}},
        tokenizer,
        dict(config, dialogueState=True),
    )
    assert torch.equal(plain["batch"]["input_ids"], view["batch"]["input_ids"])
    assert plain["encodedTokens"] == view["encodedTokens"]
    assert plain["locations"] == view["locations"]
    assert view["controllerTruncated"]
    assert [t["historyIndex"] for t in view["controllerTurns"]] == list(range(4))
    assert view["controllerTurns"][1]["partial"]
    encoded = encode(value, view["actions"], [], tokenizer, config, turns=True)
    for i, turn in enumerate(view["controllerTurns"]):
        seq, lo, hi = view["queryPositions"][view["historyStart"] + i]
        assert (lo, hi) == (turn["start"], turn["end"])
        assert 0 < lo < hi <= len(encoded[0])
        assert all(token != tokenizer.sep_token_id for token in encoded[0][lo:hi])
    assert all(
        a < b
        for a, b in zip(
            [t["start"] for t in encoded[3]], [t["end"] for t in encoded[3]], strict=True
        )
    )


def test_dialogue_state_is_predicted_and_trains_through_real_followup(tokenizer, settings, model):
    from newsvendor.structured_control import Controller
    from newsvendor.structured_tool_heads import ToolHeads
    from newsvendor.structured_train import language_backward

    config = dict(
        settings,
        structuredTools=True,
        dialogueController=True,
        dialogueState=True,
        contextRerank=True,
        controllerLength=1024,
        controllerPolicyTokens=384,
    )
    model.tools = ToolHeads()
    model.control = Controller(residual=True, auxiliary=True, history=True)
    value = payload(
        "Refund Order",
        history=[
            {"role": "user", "text": "Order id A123"},
            {"role": "tool", "text": "Order A123"},
            {"role": "user", "text": "Refund"},
        ],
        tools=[
            {"id": name, "description": name, "argumentSlots": ["order_id"]}
            for name in ("lookup", "refund")
        ],
    )
    label = {
        "action": "call_tool",
        "tool": "refund",
        "arguments": ["A123"],
        "pastTools": [{"historyIndex": 1, "tool": "lookup"}],
    }
    view = prepare({"input": value, "target": label}, tokenizer, config)
    changed = prepare(
        {"input": value, "target": dict(label, pastTools=[{"historyIndex": 1, "tool": "refund"}])},
        tokenizer,
        config,
    )
    assert view["controllerTurns"] == changed["controllerTurns"]
    assert torch.equal(view["batch"]["input_ids"], changed["batch"]["input_ids"])
    target = suite_targets(view, "abcd", label)
    assert len(target["pastTools"]) == 1
    model.eval()
    plain = prepare(value, tokenizer, dict(config, dialogueState=False))
    initial, legacy = model(view), model(plain)
    assert torch.allclose(initial["controlRecovery"], legacy["controlRecovery"], atol=1e-6)
    assert torch.allclose(initial["callGate"], legacy["callGate"], atol=1e-6)
    assert torch.equal(initial["pastTools"], model(changed)["pastTools"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss, measured = objective(initial, target)
    assert "pastTools" in measured
    loss.backward()
    for module in (model.encoder, model.control.history.tool, model.control.history.project):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    optimizer.step()
    model.train().zero_grad(set_to_none=True)
    following = dict(
        value, history=value["history"] + [{"role": "tool", "text": "Refund Order A123"}]
    )
    train = [
        ("public", {"id": key, "component": "abcd", "split": "train", "input": observed})
        for key, observed in (("first", value), ("next", following))
    ]
    labels = {
        "first": label,
        "next": {
            "action": "speak",
            "answer": "Refund",
            "pastTools": [
                {"historyIndex": 1, "tool": "lookup"},
                {"historyIndex": 3, "tool": "refund"},
            ],
        },
    }
    target.update(caseIndex=0, retrievalRound=1, nextIndex=1)
    values, heads, cost = language_backward(
        model,
        [(view, target)],
        tokenizer,
        {"encoder": config, "training": {"nextTurnTraining": True}, "noValue": False},
        train,
        labels,
        [],
    )
    assert len(values) == 1 and len(heads) == 2 and cost["retrievals"] == 1
    assert all("pastTools" in item for item in heads)
    assert model.control.history.sequence.weight_ih_l0.grad.abs().sum() > 0
    result = assemble(view, model(view), use_value=False)
    assert len(result["observedActions"]) == 1
    observed = result["observedActions"][0]
    assert observed["historyIndex"] == 1 and observed["type"] == "prediction"
    assert observed["source"] == {"kind": "history", "id": "1"}


def test_past_tool_labels_exclude_current_future_and_official_nontrain(tmp_path, monkeypatch):
    import gzip
    import json

    from newsvendor.io import write
    from newsvendor.structured_procedures import annotations

    monkeypatch.chdir(tmp_path)
    turns = [
        {"speaker": role, "targets": ["refund", None, tool, []]}
        for role, tool in (
            ("customer", None),
            ("action", "lookup"),
            ("agent", None),
            ("action", "refund"),
        )
    ]
    original = [
        [role, f"Order {i}"] for i, role in enumerate(("customer", "action", "agent", "action"))
    ]
    conversations = [{"convo_id": i, "delexed": turns, "original": original} for i in (1, 2, 3)]
    raw = gzip.compress(
        json.dumps(
            {"train": [conversations[0], conversations[2]], "dev": [conversations[1]]}
        ).encode()
    )
    source = {
        "repo": "fixture/repo",
        "revision": "pinned",
        "files": [{"path": "abcd_v1.1.json.gz", "sha256": digest(raw)}],
    }
    write("configs/complementary.json", {"sources": {"abcd": source}})
    write("schema.json", {"procedureMap": {"refund": "shipping/refund"}})
    url = "https://raw.githubusercontent.com/fixture/repo/pinned/abcd_v1.1.json.gz"
    path = tmp_path / "data/raw/complementary/abcd" / (digest(url)[:20] + ".raw")
    path.parent.mkdir(parents=True)
    path.write_bytes(raw)
    roles = {"customer": "user", "action": "tool", "agent": "assistant"}
    history = [{"role": roles[r], "text": t} for r, t in original]
    rows = [
        {
            "id": f"abcd:{i}:{turn}",
            "component": "abcd",
            "split": split,
            "input": {"history": history[:turn], "tools": [{"id": "lookup"}, {"id": "refund"}]},
        }
        for i, split in ((1, "train"), (2, "train"), (3, "dev"))
        for turn in range(4)
    ]
    labels = annotations(rows, "schema.json", history=True)
    assert "pastTools" not in labels["abcd:1:1"]  # current action is unseen
    assert labels["abcd:1:2"]["pastTools"] == [{"historyIndex": 1, "tool": "lookup"}]
    assert labels["abcd:1:3"]["pastTools"] == [{"historyIndex": 1, "tool": "lookup"}]
    assert all(
        "pastTools" not in target for key, target in labels.items() if key.startswith("abcd:2:")
    )
    assert not any(key.startswith("abcd:3:") for key in labels)
    rows[2]["input"]["history"] = history[:3]
    with pytest.raises(ValueError, match="observed Train history"):
        annotations(rows, "schema.json", history=True)


@pytest.mark.parametrize("rerank", [False, True])
def test_dialogue_state_receives_workflow_gradients_and_is_computed_once(
    tokenizer, settings, model, rerank
):
    import json

    from newsvendor.structured_control import Controller
    from newsvendor.structured_tool_heads import ToolHeads

    config = dict(
        settings,
        structuredTools=True,
        dialogueController=True,
        dialogueState=True,
        dialogueWorkflow=True,
        workflowProgress=True,
        procedureContext=True,
        contextRerank=True,
        controllerLength=1024,
        controllerPolicyTokens=384,
        rerank=rerank,
    )
    model.tools = ToolHeads()
    model.control = Controller(residual=True, auxiliary=True, history=True)
    policy = json.dumps(
        {
            "Shipping": {
                "subflows": {
                    "Refund": {
                        "instructions": ["Order refund"],
                        "actions": [{"button": "Lookup"}, {"button": "Refund"}],
                    }
                }
            }
        }
    )
    value = payload(
        "Refund Order",
        documents=[{"id": "policy", "title": "policy", "text": policy}],
        history=[
            {"role": "user", "text": "Order id A123"},
            {"role": "tool", "text": "Order found"},
            {"role": "user", "text": "Refund"},
        ],
        tools=[
            {"id": name, "description": name, "argumentSlots": ["order_id"]}
            for name in ("lookup", "refund")
        ],
    )
    view = prepare(value, tokenizer, config)
    legacy = prepare(value, tokenizer, dict(config, dialogueWorkflow=False))
    target = suite_targets(
        view,
        "abcd",
        {
            "action": "call_tool",
            "tool": "refund",
            "arguments": ["A123"],
            "procedure": "shipping/refund",
            "workflowNodes": [1],
        },
    )
    target = {k: target[k] for k in ("procedure", "workflowNodes")}
    assert torch.equal(view["batch"]["input_ids"], legacy["batch"]["input_ids"])
    assert (
        view["locations"] == legacy["locations"]
        and view["encodedTokens"] == legacy["encodedTokens"]
    )
    calls = []
    hook = model.control.history.register_forward_hook(lambda *_: calls.append(1))
    model.eval()
    initial, before = model(view), model(legacy)
    hook.remove()
    assert len(calls) == 2  # one shared recurrent state per forward, including reranking
    for key in ("procedure", "stage", "controlRecovery", "callGate", "recovery", "start", "end"):
        assert torch.allclose(initial[key], before[key], atol=1e-6)
    assert initial["dialogueState"].abs().sum() == 0
    optimizer = torch.optim.AdamW(model.control.history.parameters(), lr=0.001)
    model.zero_grad(set_to_none=True)
    loss, _ = objective(initial, target)
    loss.backward()
    assert model.control.history.project.weight.grad.abs().sum() > 0
    optimizer.step()
    model.zero_grad(set_to_none=True)
    loss, _ = objective(model(view), target)
    loss.backward()
    for module in (
        model.control.history.sequence,
        model.control.history.tool,
        model.tools.stage,
        model.encoder,
    ):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    model.zero_grad(set_to_none=True)
    loss, _ = objective(model(legacy), target)
    loss.backward()
    assert all(p.grad is None for p in model.control.history.parameters())


@pytest.mark.parametrize(
    "remaining,history,deadline,future", [(2, 0, 3, True), (1, 0, 3, False), (2, 2, 3, False)]
)
def test_observable_action_costs_and_remaining_loss_gradients(remaining, history, deadline, future):
    from newsvendor.structured_value import constrain, known_costs

    actions = ["hold", "handoff", "v", "retrieve"]
    costs = known_costs(
        actions,
        hold=100.0,
        costs={"v": 12.0},
        remaining=remaining,
        history_length=history,
        deadline=deadline,
        retrieval_cost=1.0,
    )
    logits = torch.zeros((4, 2), requires_grad=True)
    prediction = constrain(torch.nn.functional.softplus(logits), costs)
    assert prediction[0].tolist() == [1.0, 0.0]
    assert prediction[1, 1] == 0
    assert prediction[2, 1] >= 0.12 and prediction[3, 1] >= 0.01
    if not future:
        assert prediction[2, 1].item() == pytest.approx(0.12)
        assert prediction[3, 1].item() == pytest.approx(0.01)
    prediction.sum().backward()
    assert logits.grad[0].count_nonzero() == 0 and logits.grad[1, 1] == 0
    assert logits.grad[1:, 0].gt(0).all()
    assert logits.grad[2:, 1].gt(0).all().item() == future


def test_value_costs_shared_by_forward_cached_training_and_decoder(tokenizer, settings, model):
    from newsvendor.structured_critic import value_loss
    from newsvendor.structured_rollout import ResearchRouter
    from newsvendor.structured_value import constrain

    episode = next(
        e for e in corpus.generate(read("configs/full.json")) if e["scenario"] == "missing_contract"
    )
    value = copy.deepcopy(episode["input"])
    state = reference(value)
    router = ResearchRouter(model, tokenizer, dict(settings, exactActionCosts=True))
    view = router.view(value, state)
    changed = copy.deepcopy(value)
    changed["task"]["rho"] = {k: 0.2 for k in changed["task"]["rho"]}
    changed["task"]["partial"] = {k: 0.9 for k in changed["task"]["partial"]}
    changed["gold"] = {"theta": "never read"}
    other = router.view(changed, state)
    assert view["valueCosts"] == other["valueCosts"]
    assert torch.equal(view["batch"]["input_ids"], other["batch"]["input_ids"])
    model.eval()
    output = model(view)
    cached = constrain(
        torch.nn.functional.softplus(model.heads["value"](output["actionState"]).float()),
        view["valueCosts"],
    )
    torch.testing.assert_close(output["value"], cached)
    prediction = assemble(view, output)
    assert prediction["action"] == view["actions"][int(cached.sum(-1).argmin())]["id"]
    target = cached.detach().clone()
    for i, action in enumerate(view["actions"]):
        if action["id"] != "hold":
            target[i, 0] += 0.25
    loss = value_loss(model.heads["value"], [(output["actionState"], target, view["valueCosts"])])
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.heads["value"].parameters()
    )
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())
    legacy = ResearchRouter(model, tokenizer, settings).view(value, state)
    assert "valueCosts" not in legacy
    plain = torch.ones((2, 2))
    assert constrain(plain) is plain


def test_numeric_state_uses_own_predictions_and_public_costs_only():
    from newsvendor.structured_value import STATE_FEATURES, state_features

    episode = next(e for e in corpus.generate(read("configs/full.json")) if e["split"] == "train")
    value = copy.deepcopy(episode["input"])
    state = reference(value)
    actions = ["hold", "handoff", "v"]
    original = state_features(value, state, actions)
    poisoned = copy.deepcopy(value)
    poisoned["task"]["rho"] = {"v": 0}
    poisoned["task"]["partial"] = {"v": 1}
    poisoned["gold"] = {"theta": "not observable"}
    assert state_features(poisoned, state, actions) == original
    numeric = dict(zip(STATE_FEATURES, original[0], strict=True))
    assert numeric["valid"] == float(state["valid"])
    changed = copy.deepcopy(state)
    changed["q"] += 1
    changed["values"]["p"] *= 2
    changed["types"]["b"] = "fact"
    other = dict(zip(STATE_FEATURES, state_features(value, changed, actions)[0], strict=True))
    assert other["q"] != numeric["q"] and other["p_value"] != numeric["p_value"]
    assert other["b_type_fact"] != numeric["b_type_fact"]
    missing = copy.deepcopy(state)
    missing["values"].pop("b", None)
    absent = state_features(value, missing, actions)
    missing["values"]["b"] = 0.0
    zero = state_features(value, missing, actions)
    assert absent[0][STATE_FEATURES.index("b_present")] == 0
    assert zero[0][STATE_FEATURES.index("b_present")] == 1
    assert all(len(row) == len(STATE_FEATURES) for row in original)


def test_numeric_state_is_invariant_to_money_units():
    from newsvendor.structured_value import state_features

    episode = next(e for e in corpus.generate(read("configs/full.json")) if e["split"] == "train")
    value = copy.deepcopy(episode["input"])
    state = reference(value)
    actions = ["hold", "handoff", "v", "retrieve"]
    original = state_features(value, state, actions, retrieval_cost=1)
    for key in ("hold", "tolerance"):
        if value["task"][key] is not None:
            value["task"][key] *= 10
    value["task"]["costs"] = {k: 10 * v for k, v in value["task"]["costs"].items()}
    state["values"] = {k: 10 * v for k, v in state["values"].items()}
    for key in ("gamma", "expectedCost"):
        if state.get(key) is not None:
            state[key] *= 10
    converted = state_features(value, state, actions, retrieval_cost=10)
    torch.testing.assert_close(
        torch.tensor(original), torch.tensor(converted), rtol=1e-6, atol=1e-7
    )


def test_value_features_preserve_observed_ranges_without_hidden_labels():
    from newsvendor.structured_value import VALUE_FEATURES, value_features

    episode = next(
        e
        for e in corpus.generate(read("configs/full.json"))
        if e["split"] == "train" and e["scenario"] == "unset_preference"
    )
    value = copy.deepcopy(episode["input"])
    state = reference(value)
    actions = ["hold", "handoff", "b", "retrieve"]
    original = value_features(value, state, actions)
    poisoned = copy.deepcopy(value)
    poisoned["task"]["rho"], poisoned["task"]["partial"] = {}, {}
    poisoned["gold"] = {"theta": "not observable"}
    assert value_features(poisoned, state, actions) == original
    assert len(original[0]) == len(VALUE_FEATURES)
    current = dict(zip(VALUE_FEATURES, original[0], strict=True))
    assert current["b_range_present"] == 1
    assert current["b_range_max"] > current["b_range_min"]
    changed = copy.deepcopy(state)
    changed["omega"][-1]["b"] *= 2
    assert value_features(value, changed, actions) != original
    for key in ("hold", "tolerance"):
        if value["task"][key] is not None:
            value["task"][key] *= 10
    value["task"]["costs"] = {k: v * 10 for k, v in value["task"]["costs"].items()}
    state["values"] = {k: v * 10 for k, v in state["values"].items()}
    for theta in state["omega"]:
        for key in corpus.SLOTS:
            theta[key] *= 10
    for key in ("gamma", "expectedCost"):
        if state.get(key) is not None:
            state[key] *= 10
    converted = value_features(value, state, actions, retrieval_cost=10)
    torch.testing.assert_close(torch.tensor(original), torch.tensor(converted))


@pytest.mark.parametrize("value_input", ["state", "encoded_state"])
@pytest.mark.parametrize("batch_fusion", [False, True])
def test_value_input_separates_encoder_drift_and_preserves_constructor(
    tokenizer, settings, model, value_input, batch_fusion
):
    from newsvendor.structured_critic import value_loss
    from newsvendor.structured_rollout import ResearchRouter
    from newsvendor.structured_train import import_core
    from newsvendor.structured_value import VALUE_FEATURES, constrain

    config = {
        **settings,
        "valueInput": value_input,
        "batchFusion": batch_fusion,
        "exactActionCosts": True,
        "queryTokens": 1,
    }
    current = Router(copy.deepcopy(model.encoder), config).eval()
    with pytest.raises(RuntimeError, match="size mismatch"):
        import_core(current, model.state_dict())
    import_core(current, model.state_dict(), reset_value=True)
    for name, old in model.state_dict().items():
        if not name.startswith("heads.value."):
            assert torch.equal(current.state_dict()[name], old)
    episode = next(e for e in corpus.generate(read("configs/full.json")) if e["split"] == "train")
    router = ResearchRouter(current, tokenizer, config)
    view = router.view(episode["input"], reference(episode["input"]))
    views = [view, copy.deepcopy(view)]
    outputs = current(views)
    dimension = len(VALUE_FEATURES) + (256 if value_input == "encoded_state" else 0)
    for output in outputs:
        assert output["valueState"].shape[1] == dimension
        cached = constrain(
            torch.nn.functional.softplus(current.heads["value"](output["valueState"])),
            view["valueCosts"],
        )
        torch.testing.assert_close(cached, output["value"], rtol=0, atol=0)
    before = outputs[0]["value"].detach().clone()
    with torch.no_grad():
        current.project.weight.add_(torch.randn_like(current.project.weight) * 0.2)
    changed = current(views)[0]
    assert not torch.equal(outputs[0]["fieldState"], changed["fieldState"])
    assert torch.equal(before, changed["value"]) == (value_input == "state")
    target = changed["value"].detach() + 0.25
    value_loss(
        current.heads["value"], [(changed["valueState"], target, view["valueCosts"])]
    ).backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in current.heads["value"].parameters()
    )
    encoder_gradient = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in current.encoder.parameters()
    )
    assert encoder_gradient == (value_input == "encoded_state")
    current.zero_grad(set_to_none=True)
    objective(current(view), research_targets(view, episode["input"]))[0].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in current.encoder.parameters())


def test_state_value_decision_matches_forward_without_reencoding(tokenizer, settings, model):
    from newsvendor.structured_rollout import ResearchRouter
    from newsvendor.structured_train import import_core
    from newsvendor.structured_value import VALUE_FEATURES

    config = {**settings, "valueInput": "state", "exactActionCosts": True}
    current = Router(copy.deepcopy(model.encoder), config).eval()
    import_core(current, model.state_dict(), reset_value=True)
    value = next(
        e["input"] for e in corpus.generate(read("configs/full.json")) if e["split"] == "train"
    )
    state = reference(value)
    router = ResearchRouter(current, tokenizer, config)
    state["fields"] = observed_fields(value)
    view = router.view(value, state)
    expected = assemble(view, current(view))

    def forbidden(*args):
        raise AssertionError("Decision unexpectedly reencoded the document")

    hook = current.encoder.register_forward_pre_hook(forbidden)
    result = router.decision(value, state)
    hook.remove()
    assert result["action"] == expected["action"]
    assert result["policyMode"] == expected["policyMode"] == "value"
    for actual, old in zip(result["actionValues"], expected["actionValues"], strict=True):
        assert actual["residualLoss"] == old["residualLoss"] * value["task"]["hold"]
        assert actual["requestCost"] == old["requestCost"] * value["task"]["hold"]
    with pytest.raises(ValueError, match="Missing constructed"):
        current.state_values({k: v for k, v in view.items() if k != "valueFeatures"})
    changed = copy.deepcopy(view)
    column = VALUE_FEATURES.index("p_value")
    for before, after in zip(view["valueFeatures"], changed["valueFeatures"], strict=True):
        before[column], after[column] = 0.5, 0.501
    with torch.no_grad():
        current.heads["value"].layers[0].weight[:, column] = 1
    with torch.autocast("cpu", dtype=torch.bfloat16):
        a, b = current.state_values(view), current.state_values(changed)
    assert a.dtype == torch.float32 and not torch.equal(a, b)


@pytest.mark.parametrize("batch_fusion", [False, True])
def test_numeric_state_survives_prefix_truncation_and_cached_value_training(
    tokenizer, settings, model, batch_fusion
):
    from newsvendor.structured_critic import value_loss
    from newsvendor.structured_rollout import ResearchRouter
    from newsvendor.structured_train import import_core
    from newsvendor.structured_value import STATE_FEATURES, constrain

    config = {
        **settings,
        "queryTokens": 1,
        "numericState": True,
        "batchFusion": batch_fusion,
        "exactActionCosts": True,
    }
    current = Router(copy.deepcopy(model.encoder), config)
    import_core(current, model.state_dict())
    assert current.heads["value"].layers[0].weight[:, 256:].count_nonzero() == 0
    episode = next(e for e in corpus.generate(read("configs/full.json")) if e["split"] == "train")
    value = episode["input"]
    state = reference(value)
    router = ResearchRouter(current, tokenizer, config)
    before = router.view(value, state)
    changed = copy.deepcopy(state)
    changed["q"] += 1
    after = router.view(value, changed)
    assert torch.equal(before["batch"]["input_ids"], after["batch"]["input_ids"])
    assert before["economicFeatures"] != after["economicFeatures"]
    current.eval()
    results = current([before, after])
    for view, output in zip((before, after), results, strict=True):
        assert output["actionState"].shape[1] == 256 + len(STATE_FEATURES)
        cached = constrain(
            torch.nn.functional.softplus(current.heads["value"](output["actionState"])),
            view["valueCosts"],
        )
        torch.testing.assert_close(cached, output["value"])
    # Zero extension retains the old head's function and construction weights.
    reference_scores = constrain(
        torch.nn.functional.softplus(model.heads["value"](results[0]["actionState"][:, :256])),
        before["valueCosts"],
    )
    torch.testing.assert_close(results[0]["value"], reference_scores, rtol=1e-5, atol=1e-6)
    targets = results[0]["value"].detach().clone()
    targets[0, 0] += 0.5
    value_loss(
        current.heads["value"], [(results[0]["actionState"], targets, before["valueCosts"])]
    ).backward()
    assert current.heads["value"].layers[0].weight.grad[:, 256:].abs().sum() > 0
    # Once learned, the numeric state can affect a decision even when text is identical.
    with torch.no_grad():
        current.heads["value"].layers[0].weight[:, 256:] += 0.1
    new_scores = current([before, after])
    assert not torch.equal(new_scores[0]["value"], new_scores[1]["value"])
    for name, old in model.state_dict().items():
        if not name.startswith(("heads.value.", "heads.recovery.")):
            assert torch.equal(old, current.state_dict()[name])


@pytest.mark.parametrize("numeric_state", [False, True])
@pytest.mark.parametrize("batch_fusion", [False, True])
def test_action_fp32_preserves_numeric_inputs_and_matches_cached_training(
    tokenizer, settings, model, numeric_state, batch_fusion
):
    from newsvendor.structured_rollout import ResearchRouter
    from newsvendor.structured_train import import_core
    from newsvendor.structured_value import STATE_FEATURES, constrain

    config = {
        **settings,
        "queryTokens": 1,
        "numericState": numeric_state,
        "actionPrecision": "float32",
        "exactActionCosts": True,
    }
    current = Router(copy.deepcopy(model.encoder), config).eval()
    import_core(current, model.state_dict())
    episode = next(e for e in corpus.generate(read("configs/full.json")) if e["split"] == "train")
    router = ResearchRouter(current, tokenizer, config)
    before = router.view(episode["input"], reference(episode["input"]))
    after = copy.deepcopy(before)
    column = STATE_FEATURES.index("p_value")
    if numeric_state:
        for old, new in zip(before["economicFeatures"], after["economicFeatures"], strict=True):
            old[column], new[column] = 0.5, 0.501
        assert torch.tensor(0.5).bfloat16() == torch.tensor(0.501).bfloat16()
        with torch.no_grad():
            current.heads["value"].layers[0].weight[:, 256 + column] = 1
    views = [before, after]
    encoded = [(t.bfloat16(), q.bfloat16()) for t, q in current.encode_batch(views)]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        outputs = (
            current.finish_batch(views, encoded)
            if batch_fusion
            else [current.finish(v, *e) for v, e in zip(views, encoded, strict=True)]
        )
    for view, out in zip(views, outputs, strict=True):
        assert out["actionState"].dtype == out["recovery"].dtype == torch.float32
        assert out["type"].dtype == torch.bfloat16
        cached = constrain(
            torch.nn.functional.softplus(current.heads["value"](out["actionState"])),
            view["valueCosts"],
        )
        torch.testing.assert_close(out["value"], cached, rtol=0, atol=0)
        recovery = current.heads["recovery"](out["actionState"]).flatten()
        allowed = torch.tensor(view["allowedActions"])
        torch.testing.assert_close(out["recovery"], recovery.masked_fill(~allowed, -1e9))
        if numeric_state:
            torch.testing.assert_close(
                out["actionState"][:, 256:], torch.tensor(view["economicFeatures"]), rtol=0, atol=0
            )
    if numeric_state:
        assert not torch.equal(outputs[0]["value"], outputs[1]["value"])
    sum(out["value"].sum() for out in outputs).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in current.encoder.parameters())
    if numeric_state:
        assert current.heads["value"].layers[0].weight.grad[:, 256 + column].abs().sum() > 0


def test_generated_cost_rejects_pointer_to_selling_price(tokenizer, settings, model, monkeypatch):
    from newsvendor import structured_rollout
    from newsvendor.construction import candidates

    episode = next(
        e
        for e in corpus.generate(read("configs/full.json"))
        if e["split"] == "train" and e["scenario"] == "sufficient"
    )
    value = episode["input"]
    price = next(
        e for e in candidates(value, "p") if e["doc"]["id"] == "price" and e["op"] == "copy"
    )
    loc = {
        "kind": "document",
        "id": price["doc"]["id"],
        "start": price["args"][0]["span"][0],
        "end": price["args"][0]["span"][1],
    }
    fields = [
        {
            "name": s,
            "field": s,
            "state": "verified",
            "type": "fact",
            "value": price["value"],
            "expression": {"op": "copy", "operands": [loc]},
        }
        for s in ("c", "p")
    ]
    monkeypatch.setattr(structured_rollout, "extract", lambda *args: fields)
    router = structured_rollout.ResearchRouter(model, tokenizer, settings)
    state = router.construct(value)
    assert "c" not in state["values"] and state["values"]["p"] == price["value"]
    assert "invalid-expression-c" in state["errors"] and not state["valid"]


def test_generated_retrieval_requires_unread_chunks(tokenizer, settings, model, monkeypatch):
    from newsvendor import structured_rollout

    value = next(
        e["input"]
        for e in corpus.generate(read("configs/full.json"))
        if e["split"] == "train" and e["scenario"] == "sufficient"
    )
    fields = [
        {"name": s, "field": s, "state": "unconfirmed", "type": "fact", "value": None}
        for s in ("c", "p", "v", "b", "F")
    ]
    monkeypatch.setattr(structured_rollout, "extract", lambda *args: fields)
    router = structured_rollout.ResearchRouter(model, tokenizer, {**settings, "chunks": 1000})
    state = router.construct(value)
    assert state["retrieval"]["unreadChunks"] == 0
    assert "retrieve" not in router.allowed(value, state)
    state["retrieval"]["unreadChunks"] = 2
    assert "retrieve" in router.allowed(value, state)


def test_generated_model_and_labels_ignore_unobserved_response_availability(
    tokenizer, settings, model, monkeypatch
):
    from newsvendor import structured_rollout

    episode = next(
        e
        for e in corpus.generate(read("configs/full.json"))
        if e["split"] == "train" and e["scenario"] == "unavailable"
    )
    value = copy.deepcopy(episode["input"])
    value["task"]["rho"] = {a: 0 for a in value["task"]["costs"]}
    monkeypatch.setattr(structured_rollout, "extract", lambda *args: observed_fields(value))
    router = structured_rollout.ResearchRouter(model, tokenizer, settings)
    before = router.construct(value)
    view = router.view(value, before)
    target = research_targets(view, value)
    assert "b" in router.allowed(value, before)
    assert target["fields"][3]["state"] == corpus.STATUSES.index("unconfirmed")
    value["task"]["rho"] = {a: 1 for a in value["task"]["costs"]}
    value["task"]["partial"] = {a: 0.9 for a in value["task"]["costs"]}
    value["gold"] = {"theta": "not an input"}
    after = router.construct(value)
    assert before == after
    other = router.view(value, after)
    assert view["actions"] == other["actions"]
    assert torch.equal(view["batch"]["input_ids"], other["batch"]["input_ids"])
    assert target == research_targets(other, value)
    answered = corpus.outcome(value, "b", None)
    assert research_targets(view, answered)["fields"][3]["state"] == corpus.STATUSES.index(
        "unavailable"
    )


@pytest.mark.parametrize("defect", [None, "source", "scope"])
def test_generated_state_preserves_grounding_after_manager_reply(
    tokenizer, settings, model, monkeypatch, defect
):
    from newsvendor import structured_forecast, structured_rollout

    value = copy.deepcopy(
        next(
            e["input"]
            for e in corpus.generate(read("configs/full.json"))
            if e["split"] == "train" and e["scenario"] == "sufficient"
        )
    )
    monkeypatch.setattr(structured_rollout, "extract", lambda *args: observed_fields(value))
    router = structured_rollout.ResearchRouter(model, tokenizer, settings)
    before = router.construct(value)
    memory = structured_forecast.remember(value, before)
    value = corpus.outcome(value, "b", before["values"]["b"])
    value["memory"] = memory
    if defect == "source":
        next(d for d in value["docs"] if d["id"] == before["links"]["v"])["text"] += " Modified."
    elif defect == "scope":
        value["task"]["sku"] = "another-sku"

    def missing(*args):
        fields = observed_fields(value)
        for field in fields:
            if field["name"] in ("v", "b"):
                field.update(value=None, state="unconfirmed", mode="missing")
        return fields

    monkeypatch.setattr(structured_rollout, "extract", missing)
    after = router.construct(value)
    if defect is None:
        assert after["values"] == before["values"] and after["valid"]
        assert after["copiedMemory"]["v"] and after["copiedResponses"]["b"]
        assert next(f for f in after["rawFields"] if f["name"] == "v")["value"] is None
        assert after["fields"][-1]["evidence"] and after["fields"][-1]["type"] == "estimate"
    else:
        assert "v" not in after["values"]
        assert "v" not in after["copiedMemory"]
    if defect == "scope":
        assert "b" not in after["copiedResponses"]


def test_research_cases_use_actual_bounded_replies_and_reject_test():
    from newsvendor.structured_train import research_cases

    episode = copy.deepcopy(
        next(
            e
            for e in corpus.generate(read("configs/full.json"))
            if e["split"] == "train" and e["scenario"] == "unavailable"
        )
    )
    episode["input"]["task"]["rho"] = {a: 0 for a in episode["input"]["task"]["costs"]}
    original = copy.deepcopy(episode)
    cases = research_cases([episode])
    assert episode == original and cases[0][1] is episode
    assert len(cases) > 1
    for mode, case in cases[1:]:
        assert mode == "research" and case["family"] == episode["family"]
        assert all(h["answer"] == "no_response" for h in case["input"]["history"])
        assert len(case["input"]["history"]) <= min(
            episode["input"]["remaining"], episode["input"]["task"]["deadline"]
        )
    with pytest.raises(ValueError, match="Train/Dev only"):
        research_cases([{**episode, "split": "test"}])


def test_task_scope_survives_long_state_and_history_without_changing_pointer_sources(
    tokenizer, settings
):
    episode = corpus.generate(read("configs/full.json"))[0]
    value = copy.deepcopy(episode["input"])
    value["task"].update(sku="SKU-001", period="2024-05-09/2024-05-15")
    tokenizer.add_tokens([value["task"]["sku"], value["task"]["period"]])
    value["history"] = [{"action": "request_b", "text": "shipping " * 200}]
    baseline = research_input(value)
    assert baseline == research_input(value, {})
    configured = settings | {"scopePrefix": True}
    scoped = research_input(value, configured)
    assert scoped.pop("taskScope") == {k: value["task"][k] for k in ("sku", "period")}
    assert scoped == baseline
    scoped = research_input(value, configured)
    state = {"description": "shipping " * 200}
    original = prepare(baseline, tokenizer, settings, fields=FIELDS, state=state)
    changed = prepare(scoped, tokenizer, configured, fields=FIELDS, state=state)
    scope_ids = [tokenizer.convert_tokens_to_ids(value["task"][k]) for k in ("sku", "period")]
    before = original["batch"]["input_ids"][0, 1 : 1 + settings["queryTokens"]].tolist()
    after = changed["batch"]["input_ids"][0, 1 : 1 + settings["queryTokens"]].tolist()
    assert all(token not in before for token in scope_ids)
    assert all(token in after for token in scope_ids)
    assert changed["locations"] == original["locations"]
    hidden = copy.deepcopy(value)
    hidden["task"].update(rho={"b": 0.99}, partial={"b": 0.01})
    hidden["gold"] = {"theta": {"b": 9999}}
    assert research_input(hidden, configured) == scoped


@pytest.mark.parametrize("scope_prefix", [False, True])
def test_research_field_training_matches_constructor_inputs(
    tokenizer, settings, model, monkeypatch, scope_prefix
):
    from newsvendor import structured_rollout
    from newsvendor.structured_train import language_view

    settings["scopePrefix"] = scope_prefix
    episode = next(
        e
        for e in corpus.generate(read("configs/full.json"))
        if e["split"] == "train" and e["scenario"] == "sufficient"
    )
    observed = []

    def capture(*args, **kwargs):
        view = prepare(*args, **kwargs)
        observed.append(view)
        return view

    monkeypatch.setattr(structured_rollout, "prepare", capture)
    monkeypatch.setattr(
        structured_rollout, "extract", lambda *args: observed_fields(episode["input"])
    )
    structured_rollout.ResearchRouter(model, tokenizer, settings).construct(episode["input"])
    training, targets = language_view(
        ("research", episode), tokenizer, {"encoder": settings}, {}, []
    )
    inference = observed[0]
    assert set(targets) == {"fields"}
    assert training["actions"] == inference["actions"]
    assert training["queryPositions"] == inference["queryPositions"]
    assert torch.equal(training["batch"]["input_ids"], inference["batch"]["input_ids"])
    assert torch.equal(training["batch"]["attention_mask"], inference["batch"]["attention_mask"])


def test_policy_selection_keeps_benchmarks_separate_and_is_unit_invariant():
    from newsvendor.structured_critic import selection_score

    episodes = [
        {"input": {"task": {"hold": 100, "forecast": {}}}},
        {"input": {"task": {"hold": 1000}}},
    ]
    measured = [{"total": 20}, {"total": 500}]
    score, groups = selection_score(episodes, measured, "benchmark_normalized")
    assert score == pytest.approx(0.35)
    assert groups["retail"]["meanTotal"] == 20 and groups["generated"]["meanTotal"] == 500
    repeated, _ = selection_score(
        [episodes[0]] * 3 + [episodes[1]], [measured[0]] * 3 + [measured[1]], "benchmark_normalized"
    )
    assert repeated == pytest.approx(score)
    episodes[1]["input"]["task"]["hold"] *= 10
    measured[1]["total"] *= 10
    assert selection_score(episodes, measured, "benchmark_normalized")[0] == score
    with pytest.raises(ValueError, match="Mixed benchmarks"):
        selection_score(episodes, measured)


def test_portable_bundle_loads_exactly_without_training_data_or_network(
    model, tokenizer, settings, tmp_path, monkeypatch
):
    from newsvendor import bundle, structured_train

    model.eval()
    settings["attention"] = "eager"
    checkpoint = tmp_path / "original.pt"
    torch.save(model.state_dict(), checkpoint)
    directory = tmp_path / "bundle"
    config = {"encoder": settings, "noValue": False, "dataset": "absent-training-data"}
    manifest = bundle.export_bundle(
        directory, model, tokenizer, config, {"checkpointHash": bundle.file_hash(checkpoint)}
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Inference must use only local bundle assets")

    monkeypatch.setattr(structured_train, "dataset_hashes", forbidden)
    monkeypatch.setattr(structured_train, "load_backbone", forbidden)
    monkeypatch.setattr("socket.create_connection", forbidden)
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.chdir(empty)
    restored, other_tokenizer, other_config, observed = bundle.load_bundle(directory)
    assert observed == manifest and set(other_config) == {"encoder", "noValue"}
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items())
    value = payload(request="purchase cost 10", documents=[], tables=[], tools=[], history=[])
    actions = [{"id": "hold", "text": TEXT["hold"]}]
    first = prepare(value, tokenizer, settings, fields=FIELDS, actions=actions)
    second = prepare(
        value, other_tokenizer, other_config["encoder"], fields=FIELDS, actions=actions
    )
    assert torch.equal(first["batch"]["input_ids"], second["batch"]["input_ids"])
    with torch.inference_mode():
        left, right = model(first), restored(second)
    for key, tensor in left.items():
        if isinstance(tensor, torch.Tensor):
            assert torch.equal(tensor, right[key]), key


@pytest.mark.parametrize("asset", ["model.pt", "encoder/tokenizer.json"])
def test_portable_bundle_rejects_modified_assets_before_loading(
    model, tokenizer, settings, tmp_path, asset
):
    from newsvendor import bundle

    directory = tmp_path / "bundle"
    bundle.export_bundle(
        directory, model, tokenizer, {"encoder": settings}, {"checkpointHash": "fixture"}
    )
    path = directory / asset
    changed = bytearray(path.read_bytes())
    changed[len(changed) // 2] ^= 1
    path.write_bytes(changed)
    with pytest.raises(ValueError, match="file hash changed"):
        bundle.load_bundle(directory)


def test_portable_bundle_rejects_extra_tokenizer_assets(model, tokenizer, settings, tmp_path):
    from newsvendor import bundle

    directory = tmp_path / "bundle"
    bundle.export_bundle(
        directory, model, tokenizer, {"encoder": settings}, {"checkpointHash": "fixture"}
    )
    (directory / "encoder/added_tokens.json").write_text('{"changed": 999}')
    with pytest.raises(ValueError, match="Unregistered bundle files"):
        bundle.load_bundle(directory)
