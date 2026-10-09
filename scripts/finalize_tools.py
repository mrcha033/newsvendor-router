"""Freeze a completed Train/Dev run and evaluate its fixed Test configuration."""

import argparse
import fcntl
import hashlib
import os
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from newsvendor.io import digest, read, require  # noqa: E402


def running(pid):
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        return state != "Z"
    except FileNotFoundError:
        return False


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def freeze(directory):
    from newsvendor.cli import provenance
    from newsvendor.structured_progress import atomic_json
    from newsvendor.structured_train import dataset_hashes

    directory = Path(directory)
    job = read(directory / "job.json")
    require(not running(job["pid"]), "Training process is still running")
    require(job["stage"] in {"complete", "test_goal_not_met", "dev_goal_not_met"},
            "Training/selection did not finish")
    names = ("config.json", "run.json", "training.json", "language-done.pt", "common.pt",
             "policy-done.pt", "model.pt")
    require(all((directory / name).is_file() for name in names), "Missing completed training artifacts")
    config, run, report = [read(directory / name) for name in names[:3]]
    identity = run["identity"]
    require(identity["limit"] is None and report["limitedCasesPerComponent"] is None,
            "A limited run cannot establish the full goal")
    require(digest(config) == identity["configHash"] and config == report["config"], "Run config changed")
    require(dataset_hashes(config) == identity["datasetHashes"], "Run dataset identity changed")
    require(provenance(config)["sourceHash"] == identity["sourceHash"], "Use the archived training source")
    require(config.get("performanceGoal") and report.get("language") and report.get("policy"),
            "Missing completed selection or declared goal")
    baseline = config.get("comparisonBaseline") or str(Path(config["warmStart"]).parent / "model.pt")
    frozen = {"identity": identity, "files": {name: file_hash(directory / name) for name in names},
              "baseline": {"checkpoint": baseline, "hash": file_hash(baseline)},
              "selectedRerank": config["encoder"].get("rerank", True),
              "scope": "Checkpoint/configuration frozen after Train/Dev selection and before final Test evaluation"}
    path = directory / "final-evaluation-freeze.json"
    if path.exists():
        require(read(path) == frozen, "Frozen checkpoint or selection changed")
    else:
        atomic_json(path, frozen)
    return config, frozen


def result(comparison, config, frozen):
    from newsvendor.structured_metrics import performance_goal

    require(comparison.get("status") == "complete", "Incomplete paired evaluation")
    conditions = comparison["conditions"]
    expected = {"decoder_fixed", "common_rerank_off", "common_rerank_on", "policy_rerank_off",
                "policy_rerank_on", "policy_recovery_only"}
    if config["encoder"].get("dialogueController"):
        expected.add("policy_controller_off")
    require(expected <= conditions.keys(), "Missing declared comparison conditions")
    require(all(comparison[k]["sourceHash"] == frozen["identity"]["sourceHash"]
                for k in ("trainingProvenance", "evaluationProvenance")), "Comparison source differs")
    for name, item in conditions.items():
        if name == "decoder_fixed":
            require(item["checkpoint"] == frozen["baseline"]["checkpoint"]
                    and item["checkpointHash"] == frozen["baseline"]["hash"], "Comparison baseline changed")
            continue
        filename = "common.pt" if name.startswith("common_") else "policy-done.pt"
        require(item["checkpointHash"] == frozen["files"][filename], "Evaluated checkpoint differs from frozen checkpoint")
    selected = "policy_rerank_on" if frozen["selectedRerank"] else "policy_rerank_off"
    condition = conditions[selected]
    # Recompute the declared goal for the configured branch; do not select the
    # better Test toggle or trust a cached pass/fail flag.
    goal = performance_goal({k: v["mean"] for k, v in condition["public"].items()},
                            condition["research"]["0.0"]["total"]["mean"], config["performanceGoal"])
    return {"status": "goal_met" if goal["passed"] else "goal_not_met",
            "selectedCondition": selected, "goalTest": goal,
            "freezeHash": digest(frozen)}


def preserve_partial(directory):
    path = directory / f"partial-evaluation-{time.time_ns()}.tar.gz"
    with tarfile.open(path, "x:gz") as archive:
        for name in ("tool-comparison.json", "comparison"):
            if (directory / name).exists():
                archive.add(directory / name, arcname=name)
    return path


def evaluate(directory, wait):
    job = read(directory / "job.json")
    if wait:
        print({"stage": "waiting_for_training", "trainingPid": job["pid"], "pid": os.getpid()}, flush=True)
        while running(job["pid"]):
            time.sleep(5)
    config, frozen = freeze(directory)
    import torch

    from newsvendor.structured_progress import atomic_json
    from newsvendor.structured_tool_eval import run

    require(torch.cuda.device_count() == 1 and torch.cuda.get_device_name() == "NVIDIA L40S",
            "Exactly one visible L40S required")
    torch.set_num_threads(4)
    path = directory / "tool-comparison.json"
    comparison = read(path) if path.exists() else {}
    if comparison.get("status") != "complete":
        if comparison or (directory / "comparison").exists():
            print({"preservedPartialEvaluation": str(preserve_partial(directory))}, flush=True)
        print({"stage": "final_evaluation", "freezeHash": digest(frozen)}, flush=True)
        comparison = run(config, directory)
    outcome = result(comparison, config, frozen)
    atomic_json(directory / "final-evaluation.json", outcome)
    print(outcome, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--wait", action="store_true", help="Wait for this run's actual process to exit")
    args = parser.parse_args()
    os.chdir(ROOT)
    directory = Path(args.run)
    # The kernel releases this lock on exit; a leftover file never establishes liveness.
    with (directory / "final-evaluation.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another final evaluator owns this run") from error
        lock.seek(0)
        lock.truncate()
        lock.write(str(os.getpid()))
        lock.flush()
        evaluate(directory, args.wait)


if __name__ == "__main__":
    main()
