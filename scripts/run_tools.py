"""Full base tool/state/procedure fine-tuning and paired L40S evaluations."""

import argparse
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/l40s-tools-v4.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continuation", help="Explicit optimizer checkpoint with preserved parent lineage")
    args = parser.parse_args()
    os.chdir(ROOT)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    from newsvendor.io import read, require
    from newsvendor.structured_progress import atomic_json
    from newsvendor.structured_tool_eval import run as compare
    from newsvendor.structured_train import run, variant

    require(
        torch.cuda.is_available()
        and torch.cuda.device_count() == 1
        and torch.cuda.get_device_name() == "NVIDIA L40S",
        "Exactly one visible L40S required",
    )
    torch.set_num_threads(4)
    config = variant(read(args.config), "base")
    path = Path(config["output"])
    path.mkdir(parents=True, exist_ok=True)
    status = {"pid": os.getpid(), "started": datetime.now(UTC).isoformat(), "stage": "training"}
    atomic_json(path / "job.json", status)
    try:
        if args.resume and (path / "training.json").exists():
            config = read(path / "config.json")
            report = read(path / "training.json")
        else:
            report = run(config, "cuda:0", resume=args.resume, continuation=args.continuation)
            config = report["config"]
        if report.get("goalDevelopment", {}).get("passed") is False:
            status.update(stage="dev_goal_not_met", development=report["goalDevelopment"], finished=datetime.now(UTC).isoformat())
            atomic_json(path / "job.json", status)
            return
        status["stage"] = "comparison"
        atomic_json(path / "job.json", status)
        comparison = compare(config, path)
        passed = comparison.get("goalTest", {}).get("passed", True)
        status.update(stage="complete" if passed else "test_goal_not_met", finished=datetime.now(UTC).isoformat())
        if "goalTest" in comparison:
            status["goalTest"] = comparison["goalTest"]
        atomic_json(path / "job.json", status)
    except Exception as error:
        status.update(stage="failed", error=str(error), finished=datetime.now(UTC).isoformat())
        atomic_json(path / "job.json", status)
        raise


if __name__ == "__main__":
    main()
