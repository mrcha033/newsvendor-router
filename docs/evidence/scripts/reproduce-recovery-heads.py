"""Verify the published cached-head Train screen; this does not rerun the encoder or policy."""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def main():
    import torch
    from evaluate_responses import file_hash
    from study_recovery_replay import measure

    from newsvendor.cli import provenance
    from newsvendor.heads import Head
    from newsvendor.io import lines, read, require, write

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoded", required=True)
    parser.add_argument("--numeric", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    require(
        torch.cuda.device_count() == 1 and "L40S" in torch.cuda.get_device_name(0),
        "One L40S required",
    )
    require(bool(os.environ.get("CUDA_VISIBLE_DEVICES")), "Choose the visible L40S")
    require(not Path(args.output).exists(), "Preserve prior reproduction")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    encoded, numeric = Path(args.encoded), Path(args.numeric)
    original = read(encoded / "report.json")
    require(
        file_hash(encoded / "targets.jsonl") == original["rawHashes"]["targets.jsonl"],
        "Targets changed",
    )
    rows = lines(encoded / "targets.jsonl")
    families = set(original["partitions"]["held"]["families"])
    held = [i for i, row in enumerate(rows) if row["family"] in families]
    result = {"scope": __doc__, "testUsed": False, "devUsed": False, "results": {}}
    for name, directory in (("encoded", encoded), ("numeric", numeric)):
        report = read(directory / "report.json")
        require(report["source"]["sourceHash"] == provenance({})["sourceHash"], "Source changed")
        require(
            file_hash(directory / "features.pt") == report["rawHashes"]["features.pt"],
            "Features changed",
        )
        payload = torch.load(directory / "features.pt", weights_only=True, map_location="cpu")
        features = [(x.cuda(), y.cuda(), costs) for x, y, costs in payload["features"]]
        weights = payload["initial"]
        head = (
            Head(weights["layers.0.weight"].shape[1], weights["layers.0.weight"].shape[0], 2)
            .cuda()
            .eval()
        )
        head.load_state_dict(weights)
        require(
            measure(head, features, rows, held) == report["initial"], "Initial cached scores differ"
        )
        records = {"initial": True}
        for condition, expected in report["results"].items():
            path = directory / f"{condition}.pt"
            require(file_hash(path) == report["rawHashes"][path.name], "Candidate weights changed")
            candidate = torch.load(path, weights_only=True, map_location="cpu")
            head.load_state_dict(candidate["weights"])
            require(
                measure(head, features, rows, held) == expected["selected"],
                "Selected cached scores differ",
            )
            records[condition] = True
        result["results"][name] = records
    result["heldStates"] = len(held)
    result["comparedStateDecisions"] = sum(len(r) for r in result["results"].values()) * len(held)
    write(args.output, result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
