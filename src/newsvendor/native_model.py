"""Fit native evidence/state/action heads on original public-data annotations, without rollout teachers."""

import math
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from .heads import Head, train
from .io import digest, jsonl, lines, read, require, write
from .native_inputs import prediction, prepare
from .suite import check
from .suite_score import evidence_f1, exact, text_f1
from .train import seed

STATES = ["entailment", "contradiction", "not_mentioned"]
RULES = ["Yes", "No", "Irrelevant", "ask"]
SCALES = ["", "percent", "thousand", "million", "billion"]
SCHEMA = "native-router-v1"


def embeddings(texts, config, path=".cache/native-embeddings.pt"):
    cache = torch.load(path, map_location="cpu", weights_only=True) if Path(path).exists() else {}
    values = cache.get("values", {}) if cache.get("configHash") == digest(config) else {}
    missing = sorted(t for t in set(texts) if digest(t) not in values)
    if missing:
        torch.set_num_threads(4)
        tokenizer = AutoTokenizer.from_pretrained(
            config["model"],
            revision=config["revision"],
            cache_dir=".cache/torch-models",
            token=False,
            trust_remote_code=False,
        )
        # Full tokenization is intentional; bounded chunks are passed to the encoder below.
        tokenizer.model_max_length = 10**9
        encoder = (
            AutoModel.from_pretrained(
                config["model"],
                revision=config["revision"],
                cache_dir=".cache/torch-models",
                token=False,
                trust_remote_code=False,
                use_safetensors=True,
            )
            .eval()
            .requires_grad_(False)
        )
        width = config["maxLength"] - tokenizer.num_special_tokens_to_add()
        chunks, totals, counts = [], {}, defaultdict(int)

        def flush():
            if not chunks:
                return
            encoded = tokenizer.pad([c[1] for c in chunks], padding=True, return_tensors="pt")
            with torch.inference_mode():
                hidden = encoder(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1)
                vectors = ((hidden * mask).sum(1) / mask.sum(1)).cpu().numpy()
            for (key, _, weight), vector in zip(chunks, vectors, strict=True):
                totals[key] = totals.get(key, 0) + vector * weight
                counts[key] += weight
            chunks.clear()

        for i, text in enumerate(missing):
            ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
            for start in range(0, max(1, len(ids)), width):
                part = ids[start : start + width]
                # This pinned encoder uses BERT's CLS/SEP layout. Transformers 5 removed
                # prepare_for_model; construct the same public token layout explicitly.
                tokens = [tokenizer.cls_token_id, *part, tokenizer.sep_token_id]
                chunks.append(
                    (
                        digest(text),
                        {"input_ids": tokens, "attention_mask": [1] * len(tokens)},
                        max(1, len(part)),
                    )
                )
                if len(chunks) == 32:
                    flush()
            if (i + 1) % 200 == 0:
                print(
                    f"Native encoder: {i + 1}/{len(missing)} texts; no token truncation", flush=True
                )
        flush()
        for key, total in totals.items():
            v = total / counts[key]
            v = v / max(np.linalg.norm(v), 1e-12)
            require(len(v) == config["dim"], "Encoder dimension mismatch")
            values[key] = v.astype(np.float32).tolist()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"configHash": digest(config), "config": config, "values": values}, path)
    return {t: np.asarray(values[digest(t)], dtype=np.float32) for t in set(texts)}


def feature(view, item, cache):
    q = cache[view["context"]]
    if view["public"]["history"]:
        q = (q + cache[view["public"]["history"][-1]["text"]]) / 2
    d = cache[item["text"]]
    text = view["context"].casefold()
    a, b = set(re.findall(r"\w+", text)), set(re.findall(r"\w+", item["text"].casefold()))
    extra = np.zeros(32, dtype=np.float32)
    extra[:8] = [
        float(q @ d),
        len(a & b) / max(1, len(a)),
        len(a & b) / max(1, len(b)),
        min(len(item["text"]), 2000) / 2000,
        len(view["public"]["history"]) / 30,
        item.get("action") == "call_tool",
        item.get("action") == "speak",
        min(len(item.get("arguments", [])), 8) / 8,
    ]
    for i, word in enumerate(
        ("difference", "sum", "average", "ratio", "percent", "change", "literal", "span")
    ):
        extra[8 + i] = word in item.get("op", "")
    for i, word in enumerate(
        ("before", "after", "change", "percentage", "average", "total", "how many", "missing")
    ):
        extra[16 + i] = word in text
    return np.concatenate((q, d, extra))


def state_feature(view, cache):
    text = "\n".join(f["text"] for f in view["fragments"]) or view["context"]
    return feature(view, {"text": text}, cache)


def forecast_features(view):
    rows = view["public"]["observations"]
    sales = np.asarray([r["sales"] for r in rows], dtype=np.float32)
    scale = max(float(sales.mean()), 0.1)
    normalized = sales / scale
    x = list(normalized[-14:])
    x += [float(normalized[-n:].mean()) for n in (7, 14, 30, 60)]
    x += [float(normalized[-n:].std()) for n in (7, 14, 30, 60)]
    x += [sum(r["stockoutHours"] > 0 for r in rows[-n:]) / n for n in (7, 14, 30, 60)]
    x += [float(np.mean([r.get("discount", 1) for r in rows[-7:]])), math.log1p(scale)]
    return np.asarray(x, dtype=np.float32), scale


def build(config):
    check(config["dataset"])
    rows = lines(Path(config["dataset"]) / "inputs.jsonl")
    collection = lines(Path(config["dataset"]) / "collection.jsonl")
    views = {r["id"]: prepare(r, collection, config) for r in rows}
    require(
        all(v["indexedCharacters"] == v["sourceCharacters"] for v in views.values()),
        "Source text lost before retrieval",
    )
    texts = set()
    for view in views.values():
        texts.add(view["context"])
        texts.update(h["text"] for h in view["public"]["history"][-1:])
        texts.add("\n".join(f["text"] for f in view["fragments"]) or view["context"])
        texts.update(c["text"] for c in view["candidates"])
    cache = embeddings(texts, config["encoder"])
    for view in views.values():
        view["x"] = state_feature(view, cache)
        view["xs"] = [feature(view, c, cache) for c in view["candidates"]]
    return rows, views, cache


def targets(rows, views, labels):
    result, coverage = defaultdict(list), defaultdict(lambda: {"cases": 0, "representable": 0})
    for row in rows:
        v, target = views[row["id"]], labels[row["id"]]
        task, x = v["task"], v["x"]
        candidates = v["candidates"]
        if task in {"cuad", "contractnli"}:
            label = (
                int(target["action"] == "answer")
                if task == "cuad"
                else STATES.index(target["answer"])
            )
            result[task + "_state"].append({"x": x, "y": label})
            for c, f in zip(candidates, v["xs"], strict=True):
                overlap = evidence_f1(c["evidence"], target["spans"], v["public"]["documents"])
                result["evidence"].append({"x": f, "y": int(overlap > 0.05)})
        elif task == "orsharc":
            label = "ask" if target["action"] == "ask" else target["answer"].title()
            result["rule_state"].append({"x": x, "y": RULES.index(label)})
        elif task == "retail":
            features, scale = forecast_features(v)
            result["forecast"].append({"x": features, "y": [n / scale for n in target["answer"]]})
        else:
            if task == "abcd":
                result["abcd_state"].append({"x": x, "y": int(target["action"] == "call_tool")})
                matches = [
                    float(
                        c["action"] == target["action"]
                        and (c["action"] != "call_tool" or c["tool"] == target["tool"])
                    )
                    for c in candidates
                ]
            else:
                result["scale"].append({"x": x, "y": SCALES.index(target["scale"])})
                matches = [float(exact(c["answer"], target["answer"])) for c in candidates]
                if not max(matches, default=0) and target["answerType"] in {"span", "multi-span"}:
                    matches = [text_f1(c["answer"], target["answer"]) for c in candidates]
            coverage[task]["cases"] += 1
            if max(matches, default=0) >= 0.5:
                coverage[task]["representable"] += 1
                if task == "abcd" and target["action"] != "call_tool":
                    continue
                xs = v["xs"][1:] if task == "abcd" else v["xs"]
                eligible = matches[1:] if task == "abcd" else matches
                result[task + "_relation"].append(
                    {
                        "xs": xs,
                        "y": int(np.argmax(eligible)),
                        "values": [0.0] * len(xs),
                        "scale": 1.0,
                        "value": 0.0,
                    }
                )
    return dict(result), dict(coverage)


def infer(view, model):
    task, candidates = view["task"], view["candidates"]
    x = view["x"]
    if task in {"cuad", "contractnli"}:
        state = int(model[task + "_state"].scores([x])[0].argmax())
        absent = state == (0 if task == "cuad" else 2)
        output = {"action": "abstain" if task == "cuad" and absent else "answer", "evidence": []}
        if task == "contractnli":
            output["answer"] = STATES[state]
        if not absent:
            scores = model["evidence"].scores(view["xs"])
            probs = torch.as_tensor(scores).softmax(-1).numpy()[:, 1]
            selected = sorted(range(len(probs)), key=lambda i: -probs[i])[:3]
            output["evidence"] = [
                s for i in selected if probs[i] >= 0.5 for s in candidates[i]["evidence"]
            ]
        return output
    if task == "orsharc":
        selected = RULES[int(model["rule_state"].scores([x])[0].argmax())]
        output = prediction(
            candidates[3]
            if selected == "ask" and len(candidates) > 3
            else candidates[RULES.index(selected) if selected != "ask" else 0]
        )
        output["retrieved"] = view["retrieved"]
        return output
    if task == "retail":
        features, scale = forecast_features(view)
        return {
            "action": "answer",
            "answer": (np.maximum(model["forecast"].scores([features])[0], 0) * scale).tolist(),
        }
    if task == "abcd":
        state = int(model["abcd_state"].scores([x])[0].argmax())
        index = (
            int(model["abcd_relation"].scores(view["xs"][1:]).ravel().argmax()) + 1 if state else 0
        )
    else:
        index = int(model[task + "_relation"].scores(view["xs"]).ravel().argmax())
    output = prediction(candidates[index])
    if task == "tatqa":
        output["scale"] = SCALES[int(model["scale"].scores([x])[0].argmax())]
    return output


def fit(config, rows, views, value):
    seed(value)
    ids = {r["id"] for r in rows if r["split"] in {"train", "dev"}}
    labels = {
        r["id"]: r["target"]
        for r in lines(Path(config["dataset"]) / "labels.jsonl")
        if r["id"] in ids
    }
    training, coverage = targets([r for r in rows if r["split"] == "train"], views, labels)
    development, devcoverage = targets([r for r in rows if r["split"] == "dev"], views, labels)
    dimensions = {
        "cuad_state": (config["encoder"]["dim"] * 2 + 32, 2),
        "contractnli_state": (config["encoder"]["dim"] * 2 + 32, 3),
        "rule_state": (config["encoder"]["dim"] * 2 + 32, 4),
        "evidence": (config["encoder"]["dim"] * 2 + 32, 2),
        "abcd_relation": (config["encoder"]["dim"] * 2 + 32, 1),
        "abcd_state": (config["encoder"]["dim"] * 2 + 32, 2),
        "tatqa_relation": (config["encoder"]["dim"] * 2 + 32, 1),
        "scale": (config["encoder"]["dim"] * 2 + 32, 5),
        "forecast": (28, 7),
    }
    model, losses = {}, {}
    started = time.perf_counter()
    for name, (dim, output) in dimensions.items():
        require(training.get(name) and development.get(name), "Missing native supervision: " + name)
        head = Head(dim, 32, output)
        mode = (
            "choice" if name.endswith("_relation") else "value" if name == "forecast" else "class"
        )
        losses[name] = train(
            head, training[name], config["epochs"], value, mode=mode, valid=development[name]
        )
        model[name] = head
        print(f"Native seed {value}: {name}, {losses[name]['rows']} training rows", flush=True)
    directory = Path(config["output"]) / str(value)
    directory.mkdir(parents=True, exist_ok=True)
    training_info = {
        "schema": SCHEMA,
        "config": config,
        "seed": value,
        "dimensions": dimensions,
        "losses": losses,
        "candidateCoverageTrain": coverage,
        "candidateCoverageDev": devcoverage,
        "fitFamilies": sorted({r["family"] for r in rows if r["split"] == "train"}),
        "devFamilies": sorted({r["family"] for r in rows if r["split"] == "dev"}),
        "testFamilies": sorted({r["family"] for r in rows if r["split"] == "test"}),
        "supervisedIdsHash": digest(sorted(ids)),
        "encoderHash": digest(config["encoder"]),
        "wallSeconds": time.perf_counter() - started,
        "unsupportedHeads": ["economic-action-value", "fact-versus-preference-type"],
        "scope": "Native component head adaptation; no fabricated economic action-value targets",
        "evaluationStatus": "Exploratory adapter evaluation; initial test measurements were inspected during development",
    }
    require(
        set(training_info["fitFamilies"]).isdisjoint(training_info["testFamilies"]),
        "Native fit/test leakage",
    )
    torch.save(
        {"weights": {k: h.state_dict() for k, h in model.items()}, "metadata": training_info},
        directory / "model.pt",
    )
    write(directory / "training.json", training_info)
    return model


def run(config):
    from .cli import provenance
    from .suite_score import score

    rows, views, _ = build(config)
    test = [r for r in rows if r["split"] == "test"]
    records = []
    for value in config["seeds"]:
        model = fit(config, rows, views, value)
        predictions, traces = [], []
        for row in test:
            stamp = time.perf_counter()
            output = infer(views[row["id"]], model)
            predictions.append({"id": row["id"], "prediction": output})
            traces.append(
                {
                    "id": row["id"],
                    "inputHash": views[row["id"]]["inputHash"],
                    "viewHash": views[row["id"]]["viewHash"],
                    "latencyMs": (time.perf_counter() - stamp) * 1000,
                }
            )
        directory = Path(config["output"]) / str(value)
        report = score(config["dataset"], predictions, "test", directory)
        jsonl(directory / "traces.jsonl", traces)
        write(directory / "provenance.json", provenance(config))
        restored = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
        loaded = {
            k: Head(dim, 32, output)
            for k, (dim, output) in restored["metadata"]["dimensions"].items()
        }
        for name, head in loaded.items():
            head.load_state_dict(restored["weights"][name])
            head.eval()
        require(
            predictions
            == [{"id": r["id"], "prediction": infer(views[r["id"]], loaded)} for r in test],
            "Native checkpoint replay differs",
        )
        records.append(
            {
                "seed": value,
                "components": report["components"],
                "predictionHash": report["predictionHash"],
            }
        )
    write(
        Path(config["output"]) / "runs.json",
        {
            "runs": records,
            "datasetHash": read(Path(config["dataset"]) / "manifest.json")["inputHash"],
            "scope": "Native components; not economic action-value efficacy",
        },
    )
    print(
        {"output": config["output"], "seeds": config["seeds"], "testCases": len(test)}, flush=True
    )
    export(config)


def load_portable(directory, value=42):
    directory = Path(directory)
    metadata = read(directory / f"{value}.json")
    require(
        digest((directory / f"{value}.npz").read_bytes()) == metadata["weightsHash"],
        "Native weights changed",
    )
    heads = {k: Head(dim, 32, output) for k, (dim, output) in metadata["dimensions"].items()}
    with np.load(directory / f"{value}.npz", allow_pickle=False) as arrays:
        for name, head in heads.items():
            head.load_state_dict(
                {k: torch.from_numpy(arrays[name + "/" + k].copy()) for k in head.state_dict()}
            )
            head.eval()
    return heads, metadata


def export(config):
    from .cli import provenance

    destination = Path("models/native")
    destination.mkdir(parents=True, exist_ok=True)
    manifest = read(Path(config["dataset"]) / "manifest.json")
    pipeline = {
        p.name: digest(p.read_bytes())
        for p in (
            Path(__file__),
            Path(__file__).with_name("native_inputs.py"),
            Path(__file__).with_name("heads.py"),
            Path(__file__).with_name("suite_score.py"),
        )
    }
    for value in config["seeds"]:
        source = Path(config["output"]) / str(value)
        checkpoint = torch.load(source / "model.pt", map_location="cpu", weights_only=True)
        weights = {
            name + "/" + key: tensor.numpy()
            for name, params in checkpoint["weights"].items()
            for key, tensor in params.items()
        }
        np.savez_compressed(destination / f"{value}.npz", **weights)
        write(
            destination / f"{value}.json",
            {
                **checkpoint["metadata"],
                "datasetHash": manifest["inputHash"],
                "labelHash": manifest["labelHash"],
                "pipeline": pipeline,
                "weightsHash": digest((destination / f"{value}.npz").read_bytes()),
                "provenance": provenance(config),
            },
        )
        for stem in ("predictions", "measurements", "traces"):
            jsonl(destination / f"{value}.{stem}.jsonl", lines(source / f"{stem}.jsonl"))
        write(destination / f"{value}.metrics.json", read(source / "metrics.json"))
        loaded, _ = load_portable(destination, value)
        require(
            all(
                np.array_equal(loaded[k].state_dict()[n].numpy(), tensor.numpy())
                for k, params in checkpoint["weights"].items()
                for n, tensor in params.items()
            ),
            "Portable model differs",
        )
    write(destination / "runs.json", read(Path(config["output"]) / "runs.json"))
