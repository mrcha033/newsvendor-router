"""Compose the accepted recovery head with the published forecast model for offline inference."""

import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from newsvendor.bundle import export_bundle, file_hash, load_bundle
from newsvendor.cli import provenance
from newsvendor.io import read, require
from newsvendor.structured_model import Router
from newsvendor.structured_tool_eval import weights_hash
from newsvendor.structured_train import import_core


def compose(bundle, study, dev, output):
    study, dev = Path(study), Path(dev)
    fitted, report = read(study / "report.json"), read(dev / "report.json")
    require(report["eligibleForAdoption"], "Registered Dev gates failed")
    require(report["primary"] == "scoped-42", "Fixed primary differs")
    require(all(all(g.values()) for g in report["gates"].values()), "A required seed failed")
    require(
        provenance({})["sourceHash"]
        == report["source"]["sourceHash"]
        == fitted["source"]["sourceHash"],
        "Evaluated package source differs",
    )
    path = study / "refit-42.pt"
    candidate = report["results"]["scoped-42"]
    require(
        file_hash(path) == candidate["headHash"] == fitted["rawHashes"][path.name],
        "Registered head changed",
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    require(payload["selectedEpoch"] == candidate["selectedEpoch"], "Fixed epoch differs")
    parent, tokenizer, config, manifest = load_bundle(bundle)
    require(manifest["weightsHash"] == fitted["plan"]["parentWeightsHash"], "Wrong parent")
    config = copy.deepcopy(config)
    require(not config["noValue"], "Learned parent required")
    config["encoder"].update(numericState=True, actionPrecision="float32", recoveryResidual=True)
    model = Router(parent.encoder, config["encoder"])
    import_core(model, parent.state_dict())
    model.heads["value"].load_state_dict(payload["weights"], strict=True)
    model.requires_grad_(False).eval()
    require(
        weights_hash(model.heads["value"].state_dict()) == fitted["refit"]["42"]["weightsHash"],
        "Selected head tensors differ",
    )
    require(
        weights_hash(model.heads["value"].base.state_dict()) == fitted["baseHeadHash"],
        "Frozen prior differs",
    )

    def core(m):
        return {
            k: v
            for k, v in m.state_dict().items()
            if not k.startswith(("heads.value.", "heads.recovery."))
        }

    require(weights_hash(core(model)) == weights_hash(core(parent)), "Constructor or GRU changed")
    source = provenance(config) | {
        "checkpointHash": file_hash(path),
        "parentWeightsHash": manifest["weightsHash"],
        "parentManifestHash": file_hash(Path(bundle) / "manifest.json"),
        "studyReportHash": file_hash(study / "report.json"),
        "devReportHash": file_hash(dev / "report.json"),
        "selectedEpoch": payload["selectedEpoch"],
        "selectedSeed": 42,
        "testUsed": False,
        "parameters": sum(p.numel() for p in model.parameters()),
        "frozenCoreWeightsHash": weights_hash(core(model)),
        "scope": "Fixed recovery-scoped v2/42 on the published forecast-v1 model; controlled Dev selection, no new Test result.",
    }
    return export_bundle(output, model, tokenizer, config, source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--study", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    require(args.threads > 0, "Threads must be positive")
    require(not Path(args.output).exists(), "Preserve existing bundle")
    torch.set_num_threads(args.threads)
    manifest = compose(args.bundle, args.study, args.dev, args.output)
    print({"output": args.output, "weightsHash": manifest["weightsHash"]}, flush=True)


if __name__ == "__main__":
    main()
