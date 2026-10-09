"""Train expansion with immutable snapshot Dev/Test and connected-prefix exclusions."""

import gzip
import json
from collections import Counter
from pathlib import Path

from .io import digest, jsonl, lines, read, require, write
from .suite import abcd, assets, check


def expand(config, output):
    transform = digest(
        {
            str(p): digest(p.read_bytes())
            for p in (Path(__file__), Path(__file__).with_name("suite.py"))
        }
    )
    directory = Path(config["dataset"])
    check(directory)
    source_config = read("configs/complementary.json")["sources"]["abcd"]
    contents, captured = assets("abcd", source_config)
    original = json.loads(gzip.decompress(contents["abcd_v1.1.json.gz"]))
    # Reuse the original adapter, but supply only original Train to construction.
    training = original["train"]
    contents = {
        **contents,
        "abcd_v1.1.json.gz": gzip.compress(json.dumps({"train": training}).encode(), mtime=0),
    }
    records, _ = abcd(contents, {**source_config, "seed": 53, "groups": len(training)})
    old = lines(directory / "inputs.jsonl")
    labels = {r["id"]: r for r in lines(directory / "labels.jsonl")}
    old_abcd = [r for r in old if r["component"] == "abcd"]
    existing = {r["id"] for r in old}
    parent = {}

    def root(key):
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def join(keys):
        first = root(keys[0])
        for key in keys[1:]:
            other = root(key)
            if first != other:
                parent[other] = first

    def keys(record):
        return record["keys"] + ["input:" + digest(record["input"]["history"])]

    for r in records:
        join(keys(r))
    for r in old_abcd:
        join(labels[r["id"]]["leakageKeys"] + ["input:" + digest(r["input"]["history"])])
    protected, families = set(), {}
    for r in old_abcd:
        group = root(labels[r["id"]]["leakageKeys"][0])
        if r["split"] != "train":
            protected.add(group)
        else:
            families.setdefault(group, set()).add(r["family"])
    additions, new_labels, excluded = [], [], Counter()
    documents = {}
    for r in records:
        group = root(r["keys"][0])
        reason = (
            "existing"
            if r["id"] in existing
            else "protected_source_or_prefix"
            if group in protected
            else "bridges_snapshot_families"
            if len(families.get(group, set())) > 1
            else None
        )
        if reason:
            excluded[reason] += 1
            continue
        family = next(iter(families[group])) if group in families else digest(["abcd-train", group])
        refs = []
        value = dict(r["input"])
        for doc in value["documents"]:
            key = digest(doc)
            documents[key] = doc
            refs.append(key)
        value["documents"] = []
        additions.append(
            {
                "id": r["id"],
                "component": "abcd",
                "family": family,
                "split": "train",
                "input": value,
                "documentRefs": refs,
            }
        )
        new_labels.append(
            {
                "id": r["id"],
                "target": r["target"],
                "officialSplit": "train",
                "leakageKeys": r["keys"],
            }
        )
    output = Path(output)
    jsonl(output / "inputs.jsonl", additions)
    jsonl(output / "labels.jsonl", new_labels)
    write(output / "documents.json", documents)
    report = {
        "schema": "structured-train-expansion-v1",
        "transformHash": transform,
        "baseManifest": digest(read(directory / "manifest.json")),
        "source": {**source_config, "captured": captured},
        "originalTrainConversations": len(training),
        "addedCases": len(additions),
        "addedConversations": len({r["id"].split(":")[1] for r in additions}),
        "addedToolEvents": sum(r["target"]["action"] == "call_tool" for r in new_labels),
        "exclusions": dict(excluded),
        "inputHash": digest(additions),
        "labelHash": digest(new_labels),
        "documentHash": digest(documents),
        "protectedSnapshotHash": digest([r for r in old if r["split"] != "train"]),
    }
    write(output / "manifest.json", report)
    audit(directory, output)
    return report


def audit(base, expansion):
    base, expansion = Path(base), Path(expansion)
    check(base)
    rows, labels = lines(expansion / "inputs.jsonl"), lines(expansion / "labels.jsonl")
    documents, manifest = read(expansion / "documents.json"), read(expansion / "manifest.json")
    for name, values in (("inputHash", rows), ("labelHash", labels), ("documentHash", documents)):
        require(digest(values) == manifest[name], "Expansion hash mismatch: " + name)
    require(
        manifest["baseManifest"] == digest(read(base / "manifest.json")), "Base snapshot changed"
    )
    old, old_labels = lines(base / "inputs.jsonl"), lines(base / "labels.jsonl")
    require(
        digest([r for r in old if r["split"] != "train"]) == manifest["protectedSnapshotHash"],
        "Protected Dev/Test changed",
    )
    keys, ids = {}, {r["id"] for r in old}
    fingerprints = {}

    def fingerprint(value, refs):
        return digest({**value, "documents": refs})

    for row, label in zip(old, old_labels, strict=True):
        value = row["input"]
        fp = fingerprint(value, [digest(d) for d in value["documents"]])
        fingerprints[fp] = row["split"]
        for key in [row["family"], *label["leakageKeys"]]:
            keys[key] = row["split"]
    for row, label in zip(rows, labels, strict=True):
        require(
            row["split"] == label["officialSplit"] == "train"
            and row["id"] == label["id"]
            and row["id"] not in ids,
            "Invalid expanded Train id/split",
        )
        ids.add(row["id"])
        require(set(row["documentRefs"]) <= set(documents), "Missing shared document")
        require(
            len(row["input"]["history"]) == label["target"]["prefixLength"],
            "Expanded ABCD contains future turns",
        )
        fp = fingerprint(row["input"], row["documentRefs"])
        require(fingerprints.get(fp, "train") == "train", "Expanded input overlap")
        fingerprints[fp] = "train"
        for key in [row["family"], *label["leakageKeys"]]:
            require(keys.get(key, "train") == "train", "Expanded source-group leakage")
            keys[key] = "train"
    return {"addedCases": len(rows), "sourceOverlaps": 0, "protectedSnapshotUnchanged": True}


def load(config):
    directory = Path(config["dataset"])
    check(directory)
    rows = lines(directory / "inputs.jsonl")
    labels = {r["id"]: r["target"] for r in lines(directory / "labels.jsonl")}
    extension = config.get("expansion")
    if extension:
        audit(directory, extension)
        extension = Path(extension)
        docs = read(extension / "documents.json")
        for row in lines(extension / "inputs.jsonl"):
            row["input"]["documents"] = [docs[k] for k in row.pop("documentRefs")]
            rows.append(row)
        labels.update({r["id"]: r["target"] for r in lines(extension / "labels.jsonl")})
    return rows, labels, lines(directory / "collection.jsonl")
