# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["torch==2.14.1", "transformers==5.18.0", "numpy==2.5.3", "pydantic==2.12.5"]
# ///
"""Train/evaluate the ModernBERT base, large and no_value models without Qwen helpers."""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("check", "expand", "smoke", "train", "evaluate", "predict", "orders"),
        default="check",
    )
    parser.add_argument("--config", default="configs/structured.json")
    parser.add_argument("--variant", choices=("base", "large", "no_value", "all"), default="base")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reuse-common", help="Base common.pt for paired no_value policy training")
    parser.add_argument("--inputs", help="Observed public JSONL to predict without task adapters")
    parser.add_argument("--expansion")
    parser.add_argument(
        "--limit", type=int, help="Recorded diagnostic limit per component/scenario"
    )
    args = parser.parse_args()
    os.chdir(ROOT)
    import torch

    from newsvendor import corpus, structured_data, structured_train
    from newsvendor.io import read, require, write

    require(args.limit is None or args.limit > 0, "--limit must be positive")
    config = read(args.config)
    if (
        config["dataset"] == "data/processed/complementary"
        and not Path(config["dataset"], "manifest.json").exists()
    ):
        from newsvendor.snapshot import restore

        restore()
    if args.expansion:
        config["expansion"] = args.expansion
    if args.stage == "expand":
        print(structured_data.expand(config, args.expansion or "data/processed/structured-train"))
        return
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.stage in ("train", "smoke"):
        seeds = (
            config["seeds"]
            if args.all_seeds
            else [args.seed if args.seed is not None else config["seed"]]
        )
        for value in seeds:
            for name in ("base", "large", "no_value") if args.variant == "all" else (args.variant,):
                current = structured_train.variant(config, name, value)
                if args.stage == "smoke":
                    from newsvendor.structured_smoke import run

                    result = run(current, device)
                else:
                    result = structured_train.run(
                        current, device, args.limit, args.resume, args.reuse_common
                    )
                print(
                    {
                        "variant": name,
                        "output": current["output"],
                        "trainableParameters": result["trainableParameters"],
                    }
                )
        return
    rows, labels, collection = structured_data.load(config)
    episodes = corpus.generate(read(config["researchConfig"]))
    if args.stage == "check":
        print(
            {
                "publicCases": len(rows),
                "research": corpus.audit(episodes),
                "expansion": config["expansion"],
                "helpers": [],
                "variants": ["base", "large", "no_value"],
            }
        )
        return
    if args.stage in ("evaluate", "predict", "orders"):
        require(args.checkpoint, "Evaluation requires --checkpoint")
        model, tokenizer, saved, report = structured_train.load(args.checkpoint, device)
        if args.stage == "orders":
            from newsvendor.native_benchmark import Generator
            from newsvendor.structured_orders import run

            generator = Generator(read("configs/native.json")["baseline"])
            print(run(model, tokenizer, saved, generator, read("configs/orders.json"), args.limit))
            return
        if args.stage == "predict":
            from newsvendor.io import jsonl, lines
            from newsvendor.suite import public_input

            require(args.inputs, "Prediction requires --inputs")
            predictions = []
            for row in lines(args.inputs):
                observed = row.get("input", row)
                linked = "forecast" in observed.get("task", {})
                output, trace = structured_train.infer(
                    model, tokenizer, observed if linked else public_input(row), saved, collection, adapt=False, linked=linked
                )
                predictions.append({"id": row["id"], "prediction": output, "trace": trace})
            path = Path(saved["output"]) / "predictions.jsonl"
            jsonl(path, predictions)
            print({"cases": len(predictions), "output": str(path)})
            return
        require(
            structured_train.dataset_hashes(config) == structured_train.dataset_hashes(saved),
            "Evaluation dataset differs from checkpoint",
        )
        result = structured_train.evaluate(
            model, tokenizer, saved, rows, labels, collection, episodes, limit=args.limit
        )
        write(Path(saved["output"]) / "evaluation.json", result)
        print(result)
        return


if __name__ == "__main__":
    main()
