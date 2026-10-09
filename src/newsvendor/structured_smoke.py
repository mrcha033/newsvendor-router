"""Numerical/gradient smoke using the pinned pretrained encoder, not effectiveness evidence."""

import time
from pathlib import Path

import torch

from . import corpus, sequence
from .cli import provenance
from .io import read, require, write
from .structured_inputs import prepare, research_input
from .structured_labels import objective, research_targets, suite_targets
from .structured_model import Router, load_backbone
from .structured_rollout import FIELDS, TEXT
from .train import seed


def run(config, device="cpu"):
    seed(config["seed"])
    torch.set_num_threads(4)
    tokenizer, encoder = load_backbone(config["encoder"])
    seed(config["seed"])
    model = Router(encoder, config["encoder"]).to(device).train()
    episodes = corpus.generate(read(config["researchConfig"]))
    episode = next(e for e in episodes if e["split"] == "train" and e["scenario"] == "sufficient")
    settings = {
        **config["encoder"],
        "maxLength": 128,
        "queryTokens": 32,
        "overlap": 16,
        "chunks": 16,
    }
    view = prepare(
        research_input(episode["input"]),
        tokenizer,
        settings,
        fields=FIELDS,
        actions=[{"id": a, "text": TEXT[a]} for a in TEXT],
    )
    targets = research_targets(view, episode["input"])
    optimizer = model.optimizer(config["training"])
    started = time.perf_counter()
    output = model(view)
    loss, heads = objective(output, targets, no_value=config["noValue"])
    require(torch.isfinite(loss).item(), "Nonfinite structured smoke objective")
    loss.backward()
    gradients = {
        name: sum(
            float(p.grad.detach().abs().sum()) for p in module.parameters() if p.grad is not None
        )
        for name, module in (("encoder", model.encoder), ("fusion", model.fusion))
    }
    require(all(v > 0 for v in gradients.values()), "Encoder/fusion did not receive gradients")
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
    optimizer.step()
    model.eval()
    with torch.inference_mode():
        final, _ = objective(model(view), targets, no_value=config["noValue"])
    require(torch.isfinite(final).item(), "Nonfinite structured smoke after update")
    from .io import lines

    public_rows = lines(Path(config["dataset"]) / "inputs.jsonl")
    labels = {r["id"]: r["target"] for r in lines(Path(config["dataset"]) / "labels.jsonl")}
    case = next(
        r
        for r in public_rows
        if r["component"] == "abcd"
        and r["split"] == "train"
        and labels[r["id"]]["action"] == "call_tool"
        and labels[r["id"]]["tool"] == "pull-up-account"
        and labels[r["id"]]["argumentsObservable"]
    )
    abcd_view = prepare(case, tokenizer, {**settings, "chunks": 2})
    abcd_targets = suite_targets(abcd_view, "abcd", labels[case["id"]])
    model.train()
    optimizer.zero_grad(set_to_none=True)
    abcd_loss, abcd_heads = objective(model(abcd_view), abcd_targets, no_value=config["noValue"])
    require(torch.isfinite(abcd_loss).item(), "Nonfinite source ABCD objective")
    abcd_loss.backward()
    use_gradient = sum(
        float(p.grad.abs().sum()) for p in model.heads["use"].parameters() if p.grad is not None
    )
    require(use_gradient > 0, "ABCD optional-argument head did not receive gradients")
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
    optimizer.step()
    rows = [r for r in public_rows if r["split"] == "train" and r["component"] == "retail"]
    series = rows[0]["input"]["observations"]
    seq, _ = sequence.features(series)
    model.demand.train()
    out = model.demand([seq, seq], [1, 7])
    observations = [{"y": 0.0, "censored": False}, {"y": 2.0, "censored": True}]
    daily = sequence.losses(out, observations).mean()
    optimizer.zero_grad(set_to_none=True)
    daily.backward()
    require(torch.isfinite(daily).item(), "Nonfinite mixed-horizon demand smoke")
    require(
        sum(float(p.grad.abs().sum()) for p in model.demand.gru.parameters() if p.grad is not None)
        > 0,
        "GRU did not receive gradients",
    )
    optimizer.step()
    forecasts = {str(h): sequence.predict(model.demand, rows[0]["input"], h) for h in (1, 7)}
    report = {
        "scope": "Pinned-encoder gradient and numerical checks; not effectiveness evidence",
        "trainableParameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "encoderParameters": sum(p.numel() for p in model.encoder.parameters()),
        "initialLoss": float(loss.detach()),
        "afterUpdateLoss": float(final),
        "headLosses": heads,
        "gradients": gradients,
        "abcd": {
            "id": case["id"],
            "inputHash": abcd_view["inputHash"],
            "annotatedArguments": len(labels[case["id"]]["arguments"]),
            "headLosses": abcd_heads,
            "useGradient": use_gradient,
        },
        "demandLoss": float(daily.detach()),
        "forecasts": forecasts,
        "selectedChunks": view["selectedChunks"],
        "sourceTokens": len(view["locations"]),
        "sourceSplits": corpus.audit(episodes),
        "seconds": time.perf_counter() - started,
        "provenance": provenance(config),
        "device": device,
        "peakGpuBytes": torch.cuda.max_memory_allocated() if device.startswith("cuda") else None,
    }
    write(Path(config["output"]) / "smoke.json", report)
    return report
