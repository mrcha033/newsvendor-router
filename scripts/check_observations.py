"""Check quadrature and unchanged aggregate scoring using Train observations only."""

import argparse
import sys
import tarfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from newsvendor import demand, sequence
from newsvendor.io import digest, lines, read, require, write
from newsvendor.structured_tool_eval import weights_hash
from newsvendor.train import seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default="results/l40s-demand-expanded-v2/42")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    reference = Path(args.reference)
    report = read(reference / "report.json")
    config = read(reference / "config.json")
    rows = [
        r
        for r in lines(Path(config["dataset"]) / "inputs.jsonl")
        if r["component"] == "retail" and r["split"] == "train"
    ]
    ids = {r["id"] for r in rows}
    labels = {
        r["id"]: r["target"]
        for r in lines(Path(config["dataset"]) / "labels.jsonl")
        if r["id"] in ids
    }
    samples = sequence.samples(rows, labels)
    with tarfile.open(reference / "source.tar.gz") as archive:
        source = archive.extractfile("src/newsvendor/sequence.py").read()
        demand_source = archive.extractfile("src/newsvendor/demand.py").read()
    require(
        digest(source) == report["provenance"]["sources"]["src/newsvendor/sequence.py"],
        "Archived reference source changed",
    )
    old = types.ModuleType("newsvendor._previous_sequence")
    old.__package__ = "newsvendor"
    exec(compile(source, "<archived sequence.py>", "exec"), old.__dict__)
    old_demand = types.ModuleType("newsvendor._previous_demand")
    old_demand.__package__ = "newsvendor"
    require(
        digest(demand_source) == report["provenance"]["sources"]["src/newsvendor/demand.py"],
        "Archived demand source changed",
    )
    exec(compile(demand_source, "<archived demand.py>", "exec"), old_demand.__dict__)
    old.demand = old_demand
    previous = old.samples(rows, labels)
    require(len(samples) == len(previous), "Changed Train origins")
    for a, b in zip(samples, previous, strict=True):
        require(all(a[k] == b[k] for k in b if k != "sequence"), "Changed Train targets")
        require(torch.equal(a["sequence"], b["sequence"]), "Changed Train features")
    rng = np.random.default_rng(53)
    batch = [samples[i] for i in sorted(rng.choice(len(samples), 256, replace=False))]
    seed(42)
    model = sequence.DemandEncoder()
    initial = weights_hash(model.state_dict())
    require(initial == report["conditions"]["daily10"]["initialHash"], "Initialization changed")
    result = {
        "scope": "Train-only numerical check, not effectiveness evidence",
        "testUsed": False,
        "devUsed": False,
        "trainOrigins": len(samples),
        "sampledOrigins": len(batch),
        "samplingSeed": 53,
        "initialHash": initial,
        "featuresAndTargetsEqual": True,
        "reference": str(reference),
        "referenceSourceHash": digest(source),
        "scriptHash": digest(Path(__file__).read_bytes()),
        "conditions": {},
    }
    for condition in ("initial", "trained_aggregate"):
        if condition == "trained_aggregate":
            checkpoint = reference / "daily10/demand.pt"
            require(
                digest(checkpoint.read_bytes())
                == report["conditions"]["daily10"]["rawHashes"]["demand.pt"],
                "Reference weights changed",
            )
            model.load_state_dict(
                torch.load(checkpoint, weights_only=True, map_location="cpu")["weights"]
            )
        out = model([r["sequence"] for r in batch], [r["horizon"] for r in batch], daily=True)
        old_loss = old.losses(out, batch)
        current_loss = sequence.losses(out, batch)
        torch.testing.assert_close(old_loss, current_loss, rtol=0, atol=0)
        old_grad = torch.autograd.grad(
            old_loss.sum(), tuple(out["raw"].values()), retain_graph=True
        )
        new_grad = torch.autograd.grad(
            current_loss.sum(), tuple(out["raw"].values()), retain_graph=True
        )
        for a, b in zip(old_grad, new_grad, strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        losses, gradients = {}, {}
        for nodes in (64, 128, 256):
            values = sequence.losses(
                out, batch, {"observation": "daily_allocation", "observationNodes": nodes}
            )
            grad = torch.autograd.grad(values.sum(), tuple(out["raw"].values()), retain_graph=True)
            require(
                torch.isfinite(values).all() and all(torch.isfinite(g).all() for g in grad),
                "Nonfinite quadrature",
            )
            losses[nodes] = values.detach()
            gradients[nodes] = torch.cat([g.flatten() for g in grad]).detach()
        result["conditions"][condition] = {
            "aggregateLossAndGradientEqual": True,
            "quadrature": {
                str(n): {
                    "maxAbsoluteNLLDifferenceFrom256": float((losses[n] - losses[256]).abs().max()),
                    "meanAbsoluteNLLDifferenceFrom256": float(
                        (losses[n] - losses[256]).abs().mean()
                    ),
                    "relativeGradientNormDifferenceFrom256": float(
                        (gradients[n] - gradients[256]).norm() / gradients[256].norm()
                    ),
                }
                for n in (64, 128)
            },
            "familyMeanJointNLL": dict(
                zip(demand.FAMILIES, losses[256].mean(0).tolist(), strict=True)
            ),
        }
    require(not Path(args.output).exists(), "Preserve prior numerical records")
    write(args.output, result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
