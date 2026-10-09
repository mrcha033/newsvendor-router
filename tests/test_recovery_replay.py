"""Train replay selection preserves observation identity and source boundaries."""

import copy
import importlib.util
import sys
from pathlib import Path

import pytest

from newsvendor.io import digest, jsonl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
spec = importlib.util.spec_from_file_location(
    "study_recovery_replay",
    Path(__file__).resolve().parents[1] / "scripts/study_recovery_replay.py",
)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def records():
    value = {"task": {"sku": "a", "period": "week"}, "docs": [], "history": []}
    after = copy.deepcopy(value)
    after["history"] = [{"action": "v", "answer": "no_response"}]
    episode = {"id": "case", "family": "source", "split": "train", "input": value}
    row = episode | {"total": 123, "events": [{"input": value}, {"input": after}]}
    return episode, row


def test_replay_uses_all_observed_states_without_outcome_selection(tmp_path):
    episode, row = records()
    path = tmp_path / "raw.jsonl"
    jsonl(path, [row, row])
    original = digest(row)
    selected = study.replay_states([path], [episode])
    assert len(selected) == 2
    assert [r["supervised"] for r in selected] == [True, False]
    row["total"] = -99999
    row["gold"] = {"theta": 99999}
    jsonl(path, [row, row])
    assert study.replay_states([path], [episode]) == selected
    assert "gold" not in selected[0] and "total" not in selected[0]
    assert original != digest(row)
    # The same continuation can later be a supervised root; keep one state and its membership.
    jsonl(path, [row, row | {"events": row["events"][1:]}])
    promoted = study.replay_states([path], [episode])
    assert len(promoted) == 2 and all(r["supervised"] for r in promoted)


@pytest.mark.parametrize("split", ["dev", "test"])
def test_replay_refuses_held_out_data(tmp_path, split):
    episode, row = records()
    path = tmp_path / "raw.jsonl"
    jsonl(path, [row | {"split": split}])
    with pytest.raises(ValueError, match="Train rollouts"):
        study.replay_states([path], [episode])
    with pytest.raises(ValueError, match="Train episodes"):
        study.replay_states([path], [episode | {"split": split}])


@pytest.mark.parametrize("change", ["family", "id", "task"])
def test_replay_requires_registered_source_and_observed_task(tmp_path, change):
    episode, row = records()
    if change == "task":
        row = copy.deepcopy(row)
        row["events"][0]["input"]["task"]["sku"] = "different"
    else:
        row[change] = "different"
    path = tmp_path / "raw.jsonl"
    jsonl(path, [row])
    with pytest.raises(ValueError, match="source differs|task changed"):
        study.replay_states([path], [episode])
