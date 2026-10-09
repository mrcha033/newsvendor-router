import copy
import importlib.util
from pathlib import Path

import pytest

from newsvendor import corpus
from newsvendor.io import jsonl, read
from newsvendor.structured_inputs import research_input
from newsvendor.structured_train import research_cases

spec = importlib.util.spec_from_file_location(
    "study_parameter_state",
    Path(__file__).resolve().parents[1] / "scripts/study_parameter_state.py",
)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def empty_input():
    return {
        "task": {
            "sku": "one",
            "period": "next-day",
            "bounds": [0, 100],
            "hold": 100,
            "costs": {"c": 1, "p": 1, "v": 1, "b": 1},
            "tolerance": None,
            "deadline": 3,
            "decision": "minimax",
        },
        "docs": [],
        "observations": [],
        "history": [],
        "remaining": 3,
    }


def test_replayed_state_targets_follow_the_observed_response_field(tmp_path):
    value = empty_input()
    absent = corpus.outcome(value, "b", None)
    partial = corpus.outcome(absent, "v", "partial")
    row = {
        "id": "one",
        "family": "source",
        "split": "train",
        "total": 10,
        "events": [{"input": v} for v in (value, absent, partial)],
    }
    path = tmp_path / "rollouts.jsonl"
    jsonl(path, [row, row])
    original = study.observed_states([path], {"one": "source"})
    assert len(original) == 3
    targets = [[corpus.STATUSES[t] for t in r["target"]] for r in original]
    assert targets == [
        ["unconfirmed"] * 4,
        ["unconfirmed", "unconfirmed", "unconfirmed", "unavailable"],
        ["unconfirmed", "unconfirmed", "candidate", "unavailable"],
    ]
    changed = copy.deepcopy(row)
    changed.update(total=-10000, gold={"theta": {"c": 8000}})
    for event in changed["events"]:
        event["input"]["task"].update(rho={"b": 0}, partial={"v": 1})
        event["input"]["memory"] = {"hiddenAnswer": 999}
    jsonl(path, [changed, changed])
    assert original == study.observed_states([path], {"one": "source"})


@pytest.mark.parametrize("defect", ["test", "family", "id"])
def test_replay_rejects_unregistered_or_evaluation_sources(tmp_path, defect):
    row = {"id": "one", "family": "source", "split": "train", "events": []}
    row[{"test": "split", "family": "family", "id": "id"}[defect]] = "other"
    path = tmp_path / "rows.jsonl"
    jsonl(path, [row])
    with pytest.raises(ValueError, match="registered Train"):
        study.observed_states([path], {"one": "source"})


def test_head_selection_keeps_entire_sources_together_without_target_selection():
    rows = [
        {"split": "train", "benchmark": b, "family": f"{b}:{i}", "target": [j]}
        for b in ("generated", "retail")
        for i in range(10)
        for j in range(3)
    ]
    fit, held = study.source_partition(rows, 0.2, 73)
    fit_families = {rows[i]["family"] for i in fit}
    held_families = {rows[i]["family"] for i in held}
    assert len(fit_families) == 16 and len(held_families) == 4
    assert not fit_families & held_families
    assert sorted(fit + held) == list(range(len(rows)))
    changed = [{**r, "target": [-1000]} for r in rows]
    assert (fit, held) == study.source_partition(changed, 0.2, 73)
    changed[0]["split"] = "dev"
    with pytest.raises(ValueError, match="Train only"):
        study.source_partition(changed, 0.2, 73)


def test_core_language_cases_include_actual_train_replay_without_changing_dev(tmp_path):
    episodes = corpus.generate(read("configs/full.json"))
    train = next(e for e in episodes if e["split"] == "train" and e["scenario"] == "sufficient")
    dev = next(e for e in episodes if e["split"] == "dev" and e["scenario"] == "sufficient")
    value = copy.deepcopy(train["input"])
    value["docs"] = [d for d in value["docs"] if d["title"] != "Manager decision"]
    value = corpus.outcome(value, "b", None)
    row = {**{k: train[k] for k in ("id", "family", "split")}, "input": value}
    duplicate = copy.deepcopy(row)
    duplicate["input"]["memory"] = {"ignored": "old hypothesis"}
    duplicate["input"]["task"]["rho"] = {"b": 1}
    path = tmp_path / "states.jsonl"
    jsonl(path, [row, duplicate])
    initial = research_cases([train, dev])
    replayed = research_cases([train, dev], replay=[path])
    assert replayed[:-1] == initial
    assert replayed[-1][1]["split"] == "train"
    assert research_input(replayed[-1][1]["input"]) == research_input(value)
    assert replayed[-1][1]["input"]["history"][-1] == {"action": "b", "answer": "no_response"}
    bad = {**row, "id": dev["id"], "family": dev["family"], "split": "dev"}
    jsonl(path, [bad])
    with pytest.raises(ValueError, match="Train only"):
        research_cases([train, dev], replay=[path])
