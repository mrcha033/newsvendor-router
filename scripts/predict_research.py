"""Run the structured Newsvendor model from a verified local inference bundle."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from newsvendor.bundle import load_bundle
from newsvendor.io import digest, jsonl, lines, require
from newsvendor.structured_inputs import research_input
from newsvendor.structured_rollout import ResearchRouter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--inputs", required=True, help="JSONL records with observed input fields")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    require(not Path(args.output).exists(), "Preserve existing predictions")
    require(args.threads > 0, "Threads must be positive")
    torch.set_num_threads(args.threads)
    model, tokenizer, config, manifest = load_bundle(args.bundle, args.device)
    router = ResearchRouter(model, tokenizer, config["encoder"], config["noValue"])
    results = []
    for index, row in enumerate(lines(args.inputs)):
        value = row["input"]
        results.append(
            {
                "id": row.get("id", str(index)),
                "inputHash": digest(research_input(value, config["encoder"])),
                "prediction": router.decision(value),
            }
        )
    jsonl(args.output, results)
    print({"cases": len(results), "output": args.output, "weightsHash": manifest["weightsHash"]})


if __name__ == "__main__":
    main()
