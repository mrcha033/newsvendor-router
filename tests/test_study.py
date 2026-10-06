import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "repair_study", Path(__file__).resolve().parents[1] / "scripts/run_repair_study.py"
)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def test_bootstrap_does_not_count_initialization_seeds_as_source_families():
    first = {42: {"a": 10, "b": 20}, 43: {"a": 14, "b": 24}}
    second = {42: {"a": 5, "b": 5}, 43: {"a": 5, "b": 5}}
    result = study.contrast(first, second, 100)
    assert result["families"] == 2 and result["difference"] == 12
    assert result["seedDifferences"] == {42: 10, 43: 14}
    with pytest.raises(ValueError, match="Seed sets differ"):
        study.contrast(first, {42: second[42]})


def test_bootstrap_requires_matching_families():
    with pytest.raises(ValueError, match="Source families differ"):
        study.contrast({42: {"a": 10}}, {42: {"b": 5}})
