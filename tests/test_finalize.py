import copy
import importlib.util
import tarfile
from pathlib import Path

import pytest

from newsvendor.io import digest, read, write

SPEC = importlib.util.spec_from_file_location(
    "finalize_tools", Path(__file__).parents[1] / "scripts/finalize_tools.py"
)
finalize = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(finalize)


def test_final_evaluation_refuses_live_training(tmp_path, monkeypatch):
    write(tmp_path / "job.json", {"pid": 123, "stage": "dev_goal_not_met"})
    monkeypatch.setattr(finalize, "running", lambda pid: True)
    with pytest.raises(ValueError, match="still running"):
        finalize.freeze(tmp_path)
    assert not (tmp_path / "final-evaluation-freeze.json").exists()


def test_dev_failure_can_be_frozen_but_incomplete_or_changed_runs_cannot(tmp_path, monkeypatch):
    from newsvendor import cli, structured_train

    monkeypatch.setattr(finalize, "running", lambda pid: False)
    monkeypatch.setattr(cli, "provenance", lambda config: {"sourceHash": "pinned"})
    monkeypatch.setattr(structured_train, "dataset_hashes", lambda config: {"snapshot": "fixed"})
    baseline = tmp_path / "baseline.pt"
    baseline.write_bytes(b"prior model")
    config = {
        "encoder": {"rerank": True},
        "performanceGoal": {"toolExact": 0.9},
        "comparisonBaseline": str(baseline),
    }
    identity = {
        "sourceHash": "pinned",
        "datasetHashes": {"snapshot": "fixed"},
        "configHash": digest(config),
        "limit": None,
    }
    write(tmp_path / "config.json", config)
    write(tmp_path / "run.json", {"identity": identity})
    write(
        tmp_path / "training.json",
        {
            "config": config,
            "limitedCasesPerComponent": None,
            "language": {"checks": []},
            "policy": [{"selected": 1}],
        },
    )
    for name in ("language-done.pt", "common.pt", "policy-done.pt", "model.pt"):
        (tmp_path / name).write_bytes(name.encode())
    write(tmp_path / "job.json", {"pid": 123, "stage": "failed"})
    with pytest.raises(ValueError, match="did not finish"):
        finalize.freeze(tmp_path)
    write(tmp_path / "job.json", {"pid": 123, "stage": "dev_goal_not_met"})
    _, frozen = finalize.freeze(tmp_path)
    assert read(tmp_path / "final-evaluation-freeze.json") == frozen
    assert finalize.freeze(tmp_path)[1] == frozen
    (tmp_path / "policy-done.pt").write_bytes(b"another selected checkpoint")
    with pytest.raises(ValueError, match="Frozen checkpoint"):
        finalize.freeze(tmp_path)


def test_final_goal_uses_configured_toggle_and_validates_all_comparisons():
    config = {
        "encoder": {"rerank": False},
        "performanceGoal": {
            "toolExact": 0.9,
            "observableToolAndArgumentsExact": 0.9,
            "callPrecision": 0.9,
            "generatedTestTotalLossBelow": 240.35,
        },
    }
    frozen = {
        "selectedRerank": False,
        "files": {"common.pt": "common", "policy-done.pt": "policy"},
        "baseline": {"checkpoint": "prior.pt", "hash": "prior"},
        "identity": {"sourceHash": "pinned"},
    }
    passing = {
        "toolExact": 1.0,
        "observableToolAndArgumentsExact": 1.0,
        "predictedCall": 0.3,
        "correctCall": 0.3,
    }
    failing = dict(passing, toolExact=0.6)
    conditions = {}
    for name in (
        "decoder_fixed",
        "common_rerank_off",
        "common_rerank_on",
        "policy_rerank_off",
        "policy_rerank_on",
        "policy_recovery_only",
    ):
        conditions[name] = {
            "checkpoint": "prior.pt" if name == "decoder_fixed" else "current.pt",
            "checkpointHash": "prior"
            if name == "decoder_fixed"
            else "common"
            if name.startswith("common_")
            else "policy",
            "public": {
                k: {"mean": v}
                for k, v in (failing if name == "policy_rerank_off" else passing).items()
            },
            "research": {"0.0": {"total": {"mean": 100.0}}},
        }
    comparison = {
        "status": "complete",
        "conditions": conditions,
        "goalTest": {"passed": True},
        "trainingProvenance": {"sourceHash": "pinned"},
        "evaluationProvenance": {"sourceHash": "pinned"},
    }
    result = finalize.result(comparison, config, frozen)
    assert result["status"] == "goal_not_met" and not result["goalTest"]["passed"]
    missing = copy.deepcopy(comparison)
    del missing["conditions"]["policy_rerank_on"]
    with pytest.raises(ValueError, match="Missing declared"):
        finalize.result(missing, config, frozen)
    altered = copy.deepcopy(comparison)
    altered["conditions"]["policy_rerank_off"]["checkpointHash"] = "different"
    with pytest.raises(ValueError, match="Evaluated checkpoint"):
        finalize.result(altered, config, frozen)


def test_partial_evaluation_measurements_survive_a_retry(tmp_path):
    write(tmp_path / "tool-comparison.json", {"status": "running"})
    raw = tmp_path / "comparison" / "partial.jsonl"
    raw.parent.mkdir()
    raw.write_text('{"measurement": 1}\n')
    archive = finalize.preserve_partial(tmp_path)
    raw.write_text('{"measurement": 2}\n')
    with tarfile.open(archive) as saved:
        assert saved.extractfile("comparison/partial.jsonl").read() == b'{"measurement": 1}\n'
