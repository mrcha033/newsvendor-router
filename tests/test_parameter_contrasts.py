import copy
import importlib.util
import random
import re
from datetime import date
from pathlib import Path

import pytest

from newsvendor import corpus
from newsvendor.construction import parameter_record
from newsvendor.io import read
from newsvendor.structured_inputs import research_input

spec = importlib.util.spec_from_file_location(
    "prepare_parameter_contrasts",
    Path(__file__).resolve().parents[1] / "scripts/prepare_parameter_contrasts.py",
)
contrasts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contrasts)


@pytest.fixture
def row():
    return next(
        r
        for r in corpus.generate(read("configs/full.json"))
        if r["split"] == "train" and r["scenario"] == "sufficient"
    )


@pytest.mark.parametrize("condition", contrasts.CONDITIONS)
def test_contrasts_preserve_scope_and_arithmetic_with_opaque_source_ids(row, condition):
    before = copy.deepcopy(row)
    result = contrasts.contrast(row, condition, 97)
    assert row == before
    original = parameter_record(row["input"])
    actual = parameter_record(result["input"])
    assert actual["state"] == original["state"]
    assert actual["types"] == original["types"]
    assert all(len(d["id"]) == 16 for d in result["input"]["docs"])
    if condition == "newer_version":
        assert all(actual["values"][s] != original["values"][s] for s in corpus.SLOTS)
    else:
        assert actual["values"] == pytest.approx(original["values"])
    assert result["family"] == row["family"] and result["split"] == "train"


def test_contrast_generation_ignores_hidden_truth_probabilities_and_model_guesses(row):
    changed = copy.deepcopy(row)
    changed.update(gold={"theta": {"v": 9999}}, responses={"v": 9999}, total=-1000)
    changed["input"]["task"].update(rho={"v": 0}, partial={"v": 1})
    changed["input"]["memory"] = {"v": 9999, "state": "verified"}
    for condition in contrasts.CONDITIONS:
        left, right = [contrasts.contrast(r, condition, 97) for r in (row, changed)]
        assert left == right
        assert research_input(left["input"]) == research_input(right["input"])


@pytest.mark.parametrize("split", ["dev", "test"])
def test_contrasts_reject_evaluation_rows(row, split):
    with pytest.raises(ValueError, match="Train only"):
        contrasts.contrast({**row, "split": split}, "wrong_sku", 97)


def test_contrasts_do_not_fill_a_missing_parameter_or_invent_manager_response(row):
    row = copy.deepcopy(row)
    row["input"]["docs"] = [d for d in row["input"]["docs"] if d["role"] != "manager"]
    for condition in contrasts.CONDITIONS:
        result = contrasts.contrast(row, condition, 97)
        record = parameter_record(result["input"])
        assert "b" not in record["values"]
        assert record["state"]["b"] == "unconfirmed"
        assert not result["input"]["history"]


def test_contrast_holdout_keeps_source_families_together(row):
    rows = [{**row, "id": str(i), "family": f"source-{i // 3}"} for i in range(30)]
    held = contrasts.partition(rows, 0.2, 101)
    assert len(held) == 2
    changed = [{**r, "gold": {"theta": {"p": i}}} for i, r in enumerate(rows)]
    assert contrasts.partition(changed, 0.2, 101) == held
    with pytest.raises(ValueError, match="Train only"):
        contrasts.partition([{**rows[0], "split": "test"}], 0.2, 101)


@pytest.mark.parametrize("value", ["retail:80:117", "SKU-0123456789"])
def test_foreign_sku_preserves_the_identifier_format(value):
    changed = contrasts.alternate_scope(value, "sku", random.Random(109))
    assert changed != value
    pattern = r"retail:\d+:\d+" if value.startswith("retail:") else r"SKU-[0-9a-f]{10}"
    assert re.fullmatch(pattern, changed)
    assert "other" not in changed and "distractor" not in changed


def test_foreign_period_is_a_real_date_interval_with_the_same_duration():
    current = "2024-05-09/2024-05-15"
    changed = contrasts.alternate_scope(current, "period", random.Random(109))
    assert changed != current
    start, end = [date.fromisoformat(s) for s in changed.split("/")]
    assert (end - start).days == 6
