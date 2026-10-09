"""Audit observed request-cost visibility in saved Train own-state policy inputs."""

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    import torch
    from transformers import AutoTokenizer

    from newsvendor.io import digest, lines, read, require, write
    from newsvendor.structured_inputs import source_tokens
    from newsvendor.structured_rollout import ResearchRouter

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    require(not Path(args.output).exists(), "Preserve previous input audit")
    torch.set_num_threads(1)
    config = read(args.config)
    encoder = config["encoder"]
    tokenizer = AutoTokenizer.from_pretrained(encoder["model"], revision=encoder["revision"],
        cache_dir=".cache/torch-models", local_files_only=True, trust_remote_code=False)
    router = ResearchRouter(None, tokenizer, encoder)
    records = []
    for path in args.targets:
        for number, row in enumerate(lines(path)):
            require(row["split"] == "train", "Input audit accepts only Train rollout states")
            # Only observed inputs and the model's constructed state enter view().
            view = router.view(row["input"], row["state"])
            request = next(s for s in view["sources"] if s["kind"] == "request")
            text = request["text"]
            marker = '"costs": ' + json.dumps(row["input"]["task"]["costs"], sort_keys=True)
            start = text.index(marker)
            expected = [(a, b) for a, b in source_tokens(tokenizer, text)["offset_mapping"]
                        if a < start + len(marker) and b > start]
            visible = {(p["start"], p["end"]) for p in view["locations"] if p["kind"] == "request"}
            covered = sum(p in visible for p in expected)
            entry = {"path": path, "row": number, "id": row["id"], "family": row["family"],
                     "inputHash": view["inputHash"], "queryTruncated": view["queryTruncated"],
                     "costTokens": len(expected), "visibleCostTokens": covered}
            # A deterministic limited perturbation checks tensor conditioning,
            # not the desired decision or the learned economic value response.
            if len(records) < 64:
                changed = copy.deepcopy(row["input"])
                changed["task"]["costs"] = {k: 123.456 + 3 * v for k, v in changed["task"]["costs"].items()}
                other = router.view(changed, row["state"])
                require(view["actions"] == other["actions"], "Cost perturbation changed allowed actions")
                entry["perturbedInputHash"] = other["inputHash"]
                entry["encoderTokensChanged"] = not torch.equal(view["batch"]["input_ids"], other["batch"]["input_ids"])
            records.append(entry)
    require(records, "No Train states")
    summary = {"trainStates": len(records),
               "truncatedQueryPrefixes": sum(r["queryTruncated"] for r in records),
               "statesWithAllRequestCostsVisible": sum(r["costTokens"] == r["visibleCostTokens"] for r in records),
               "perturbedStates": sum("encoderTokensChanged" in r for r in records),
               "changedEncoderInputs": sum(r.get("encoderTokensChanged", False) for r in records)}
    write(args.output, {
        "scope": "Read-only Train own-state input audit. Short prefix truncation is distinguished from the separate request source. Counterfactual request costs are observed-input perturbations for tensor sensitivity only, not effectiveness, target labels, or training data.",
        "config": args.config, "configHash": digest(config), "scriptHash": digest(Path(__file__).read_bytes()),
        "targetFileHashes": {p: digest(Path(p).read_bytes()) for p in args.targets},
        "sourceHashes": {str(p): digest(p.read_bytes()) for p in sorted(Path("src/newsvendor").glob("*.py"))},
        "summary": summary, "records": records,
    })
    print(summary)


if __name__ == "__main__":
    main()
