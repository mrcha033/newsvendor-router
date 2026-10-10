"""Recovery sampling depends on observed Train responses, never future outcomes."""

import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
spec = importlib.util.spec_from_file_location(
    "study_recovery_sampling",
    Path(__file__).resolve().parents[1] / "scripts/study_recovery_sampling.py",
)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def row(family, history=(), benchmark="retail", split="train"):
    return {
        "family": family,
        "benchmark": benchmark,
        "split": split,
        "input": {"history": list(history)},
    }


def test_sampling_ignores_targets_and_unselected_sources():
    rows = [row("a"), row("b", [{"action": "v", "answer": "no_response"}])]
    before = study.sampling_probabilities(rows, [0, 1], 0.5)
    altered = copy.deepcopy(rows)
    altered[0].update(values=[[999, 2]], falseHandoffs=[1], gold={"rho": 0})
    altered.append(row("held", [{"action": "v", "answer": None}], split="dev"))
    assert np.array_equal(before, study.sampling_probabilities(altered, [0, 1], 0.5))
    assert rows[0] == row("a")


def test_balancing_preserves_equal_family_mass_within_observed_strata():
    rows = [row("a"), row("a"), row("b"), row("c", benchmark="generated")]
    weights = study.sampling_probabilities(rows, [0, 1, 2, 3], 1)
    assert weights.tolist() == [0.125, 0.125, 0.25, 0.5]
    assert study.sampling_probabilities(rows, [0, 1, 2, 3], 0).tolist() == [0.25] * 4


@pytest.mark.parametrize("split", ["dev", "test"])
def test_sampling_rejects_non_train_sources(split):
    with pytest.raises(ValueError, match="Train sources"):
        study.sampling_probabilities([row("x", split=split)], [0], 0.5)


def test_retrieval_is_not_a_failed_manager_response():
    value = {"history": [{"action": "retrieve", "answer": None}]}
    assert study.response_group(value) == "initial"
    value["history"].append({"action": "b", "answer": 0})
    assert study.response_group(value) == "answered"
    value["history"].append({"action": "v", "answer": "partial"})
    assert study.response_group(value) == "partial"
    value["history"].append({"action": "p", "answer": "no_response"})
    assert study.response_group(value) == "unanswered"
