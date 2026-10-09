"""Audit past tool-result annotations in official Train, without constructing model inputs."""

import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def records(path):
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main():
    from newsvendor.io import digest, read, require, write
    from newsvendor.suite import canonical

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = read(args.config)
    require(not Path(args.output).exists(), "Preserve earlier audit")
    allowed, tools = {}, None
    for directory in (config["dataset"], config["expansion"]):
        for row in records(Path(directory) / "inputs.jsonl"):
            if row["split"] != "train" or row["component"] != "abcd":
                continue
            names = {t["id"] for t in row["input"]["tools"]}
            require(tools is None or names == tools, "Observed Train tool catalogs differ")
            tools = names
            allowed[row["id"]] = (len(row["input"]["history"]), digest(row["input"]["history"]))
    source = read("configs/complementary.json")["sources"]["abcd"]
    item = next(f for f in source["files"] if f["path"].endswith("abcd_v1.1.json.gz"))
    url = f"https://raw.githubusercontent.com/{source['repo']}/{source['revision']}/{item['path']}"
    path = Path("data/raw/complementary/abcd") / (digest(url)[:20] + ".raw")
    data = path.read_bytes()
    require(digest(data) == item["sha256"], "Source changed")
    # Official non-Train annotations are never iterated or used as audit targets.
    conversations = json.loads(gzip.decompress(data))["train"]
    counts, per_tool = Counter(), Counter()
    annotations, unique_events, texts = [], {}, defaultdict(Counter)
    roles = {"customer": "user", "agent": "assistant", "action": "tool"}
    for conversation in conversations:
        observed, past = [], []
        require(len(conversation["original"]) == len(conversation["delexed"]), "Turn alignment differs")
        for index, (original, turn) in enumerate(zip(conversation["original"], conversation["delexed"], strict=True)):
            require(original[0] == turn["speaker"], "Speaker alignment differs")
            key = f"abcd:{conversation['convo_id']}:{index}"
            if key in allowed:
                require(allowed[key] == (index, digest(observed)), "Source prefix differs from observed Train input")
                counts["officialTrainCases"] += 1
                counts["casesWithObservedTools"] += bool(past)
                counts["pastToolTargets"] += len(past)
                require(all(p["historyIndex"] < index for p in past), "Current/future tool entered past annotations")
                annotations.append({"id": key, "past": list(past)})
                for previous in past:
                    event = conversation["convo_id"], previous["historyIndex"]
                    unique_events[event] = previous["tool"]
                    per_tool[previous["tool"]] += 1
            observed.append({"role": roles[original[0]], "text": original[1]})
            if turn["speaker"] == "action":
                tool = turn["targets"][2]
                if tool not in tools:
                    counts["sourceActionsOutsideCatalog"] += 1
                    continue
                past.append({"historyIndex": index, "tool": tool})
    require({a["id"] for a in annotations} <= allowed.keys(), "Annotations escaped Train")
    for conversation in conversations:
        for index, original in enumerate(conversation["original"]):
            event = conversation["convo_id"], index
            if event in unique_events:
                texts[canonical(original[1])][unique_events[event]] += 1
    counts.update(allowedTrainCases=len(allowed), excludedNonOfficialTrainCases=len(allowed) - len(annotations),
                  uniquePastToolEvents=len(unique_events), uniqueObservedTexts=len(texts),
                  textsWithMultipleToolLabels=sum(len(t) > 1 for t in texts.values()),
                  eventsInAmbiguousTexts=sum(sum(t.values()) for t in texts.values() if len(t) > 1))
    write(args.output, {
        "scope": "Read-only official-Train annotation coverage audit. Every label refers to an already observed tool-result turn strictly before its Train prefix. No model features, training targets in active jobs, parameters, Dev or Test results changed. Exact-text ambiguity is a dataset property, not a measured model accuracy.",
        "scriptHash": digest(Path(__file__).read_bytes()), "source": {"path": str(path), "hash": digest(data), "revision": source["revision"]},
        "config": args.config, "configHash": digest(config), "counts": dict(counts),
        "annotationHash": digest(sorted(annotations, key=lambda a: a["id"])), "perToolTargets": dict(per_tool),
        "ambiguousTexts": [{"textHash": digest(text), "labels": dict(labels)} for text, labels in texts.items() if len(labels) > 1],
    })
    print(dict(counts))


if __name__ == "__main__":
    main()
