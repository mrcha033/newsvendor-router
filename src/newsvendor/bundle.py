"""Portable inference weights with pinned architecture, tokenizer and provenance."""

import hashlib
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from .io import read, require, write
from .structured_model import Router
from .structured_tool_eval import weights_hash

SCHEMA = "newsvendor-inference-v1"


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def export_bundle(directory, model, tokenizer, config, source):
    root = Path(directory)
    require(not root.exists(), "Preserve existing inference bundle")
    require(source.get("checkpointHash"), "Missing source checkpoint identity")
    root.mkdir(parents=True)
    assets = root / "encoder"
    model.encoder.config.save_pretrained(assets)
    tokenizer.save_pretrained(assets)
    weights = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    inference = {"encoder": config["encoder"], "noValue": config.get("noValue", False)}
    torch.save({"schema": SCHEMA, "config": inference, "weights": weights}, root / "model.pt")
    files = {
        str(p.relative_to(root)): {"sha256": file_hash(p), "bytes": p.stat().st_size}
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }
    manifest = {
        "schema": SCHEMA,
        "config": inference,
        "source": source,
        "weightsHash": weights_hash(weights),
        "files": files,
    }
    write(root / "manifest.json", manifest)
    return manifest


def load_bundle(directory, device="cpu"):
    """Load local assets without training data, Hub access or pretrained weight downloads."""
    root = Path(directory).resolve()
    manifest = read(root / "manifest.json")
    require(manifest.get("schema") == SCHEMA, "Wrong inference bundle schema")
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    require(actual == set(manifest["files"]) | {"manifest.json"}, "Unregistered bundle files")
    for name, record in manifest["files"].items():
        path = root / name
        require(
            path.resolve().is_relative_to(root) and not path.is_symlink(), "Invalid bundle path"
        )
        require(
            path.is_file() and path.stat().st_size == record["bytes"], "Bundle file size changed"
        )
        require(file_hash(path) == record["sha256"], "Bundle file hash changed: " + name)
    require(
        "model.pt" in manifest["files"] and "encoder/config.json" in manifest["files"],
        "Missing inference assets",
    )
    payload = torch.load(root / "model.pt", map_location="cpu", weights_only=True, mmap=True)
    require(
        payload["schema"] == SCHEMA and payload["config"] == manifest["config"],
        "Bundle config changed",
    )
    require(weights_hash(payload["weights"]) == manifest["weightsHash"], "Bundle tensors changed")
    config = payload["config"]
    assets = root / "encoder"
    tokenizer = AutoTokenizer.from_pretrained(
        assets, local_files_only=True, trust_remote_code=False
    )
    architecture = AutoConfig.from_pretrained(
        assets, local_files_only=True, trust_remote_code=False
    )
    encoder = AutoModel.from_config(
        architecture,
        attn_implementation=config["encoder"].get("attention", "sdpa"),
        trust_remote_code=False,
    )
    model = Router(encoder, config["encoder"])
    model.load_state_dict(payload["weights"], strict=True)
    require(
        weights_hash(model.state_dict()) == manifest["weightsHash"],
        "Loaded tensor types or values changed",
    )
    return model.to(device).eval(), tokenizer, config, manifest
