# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["torch==2.14.1", "transformers==5.18.0", "numpy==2.5.3", "pydantic==2.12.5"]
# [tool.uv.sources]
# torch = { index = "cuda" }
# [[tool.uv.index]]
# name = "cuda"
# url = "https://download.pytorch.org/whl/cu130"
# explicit = true
# ///
"""Train the paired base/no_value and large comparison exclusively on one L40S."""

import argparse
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/l40s-efficient.json")
    parser.add_argument("--gpu", help="L40S GPU UUID; defaults to the first installed L40S")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continuation", help="Recorded base optimizer checkpoint to continue")
    parser.add_argument("--common-root", help="Explicit completed language/demand warm-start root")
    parser.add_argument(
        "--limit", type=int, help="Explicit diagnostic cap, never a full experiment"
    )
    args = parser.parse_args()
    devices = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,uuid", "--format=csv,noheader"], text=True
    )
    available = {
        uuid.strip(): name.strip()
        for name, uuid in (line.split(",") for line in devices.splitlines())
    }
    gpu = args.gpu or next((k for k, v in available.items() if v == "NVIDIA L40S"), None)
    if gpu not in available or available[gpu] != "NVIDIA L40S":
        raise ValueError("Select an installed NVIDIA L40S UUID")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.chdir(ROOT)
    import torch

    from newsvendor.io import digest, read, require, write
    from newsvendor.structured_compare import compare, markdown
    from newsvendor.structured_progress import atomic_json
    from newsvendor.structured_train import run, variant

    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "CUDA L40S is required")
    require(torch.cuda.get_device_name(0) == "NVIDIA L40S", "Unexpected visible GPU")
    require(args.limit is None or args.limit > 0, "--limit must be positive")
    config = read(args.config)
    reports = []
    directory = Path(config["output"])
    directory.mkdir(parents=True, exist_ok=True)
    batch = {
        "pid": os.getpid(),
        "gpuUuid": gpu,
        "configHash": digest(config),
        "seed": args.seed,
        "limit": args.limit,
        "started": datetime.now(UTC).isoformat(),
        "status": "running",
        "completed": [],
    }
    for name in ("base", "no_value", "large"):
        batch["variant"] = name
        atomic_json(directory / "batch.json", batch)
        current = variant(config, name, args.seed)
        report = Path(current["output"]) / "training.json"
        if args.resume and report.exists():
            reports.append(read(report))
            batch["completed"].append(name)
            continue
        common = Path(config["output"]) / "base" / str(args.seed) / "common.pt"
        try:
            result = run(
                current,
                "cuda:0",
                args.limit,
                args.resume,
                reuse_common=(
                    str(common) if name == "no_value" else
                    str(Path(args.common_root) / name / str(args.seed) / "common.pt")
                    if args.common_root else None
                ),
                continuation=args.continuation if name == "base" else None,
            )
        except Exception as error:
            batch.update(status="failed", error=str(error), finished=datetime.now(UTC).isoformat())
            atomic_json(directory / "batch.json", batch)
            raise
        reports.append(result)
        batch["completed"].append(name)
    comparison = compare(reports)
    write(Path(config["output"]) / f"comparison-{args.seed}.json", comparison)
    (Path(config["output"]) / f"comparison-{args.seed}.md").write_text(markdown(comparison))
    batch.update(status="complete", finished=datetime.now(UTC).isoformat())
    atomic_json(directory / "batch.json", batch)


if __name__ == "__main__":
    main()
