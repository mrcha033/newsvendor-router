# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["torch==2.14.1", "transformers==5.18.0", "numpy==2.5.3", "pydantic==2.12.5"]
# ///
"""Run the remaining local GPU inference without the project's CPU-only torch index."""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("components", "orders", "all"), default="all")
    parser.add_argument("--max-calls", type=int)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate captured data and portable models without model inference",
    )
    args = parser.parse_args()
    os.chdir(ROOT)
    from newsvendor.io import read, require, write
    from newsvendor.native_model import load_portable
    from newsvendor.snapshot import restore

    require(args.max_calls is None or args.max_calls > 0, "--max-calls must be positive")
    checks = restore()
    config, orders = read("configs/native.json"), read("configs/orders.json")
    for value in config["seeds"]:
        _, metadata = load_portable("models/native", value)
        require(
            metadata["datasetHash"] == read(Path(config["dataset"]) / "manifest.json")["inputHash"],
            "Model dataset hash mismatch",
        )
    if args.check:
        print(
            {
                "checks": checks,
                "model": config["baseline"],
                "pending": ["native typed/agent inference", "retail dialogue rollouts"],
            }
        )
        return
    from newsvendor.native_benchmark import Generator
    from newsvendor.native_benchmark import run as components
    from newsvendor.order_benchmark import run as retail

    generator = Generator(config["baseline"], args.max_calls)
    if args.stage in {"components", "all"}:
        components(config, generator)
    if args.stage in {"orders", "all"}:
        retail(config, orders, generator)
    write(
        "results/gpu-run.json",
        {
            "stage": args.stage,
            "newGenerations": generator.calls,
            "model": config["baseline"],
            "scope": "Native components and simulated retail; no real procurement monetary labels",
        },
    )


if __name__ == "__main__":
    main()
