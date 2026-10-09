"""Score saved Dev past-tool predictions; annotations never enter model inputs."""

import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    from newsvendor.io import digest, lines, read, require, write

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--dataset", default="data/processed/complementary")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    require(not Path(args.output).exists(), "Preserve previous diagnosis")
    # Freeze observed inputs and saved predictions before reading scoring labels.
    rows = {r["id"]: r for r in lines(Path(args.dataset) / "inputs.jsonl")
            if r["component"] == "abcd" and r["split"] == "dev"}
    predictions = lines(args.predictions)
    raw_hash = digest(Path(args.predictions).read_bytes())
    require(len(predictions) == len(rows) and {r["id"] for r in predictions} == rows.keys(),
            "Diagnosis requires the complete fixed Dev ABCD predictions")
    saved = {r["id"]: r["prediction"].get("observedActions", []) for r in predictions}
    for key, actions in saved.items():
        history = rows[key]["input"]["history"]
        require(len({p["historyIndex"] for p in actions}) == len(actions), "Repeated history prediction")
        for action in actions:
            i = action["historyIndex"]
            require(0 <= i < len(history) and history[i]["role"] == "tool", "Unobserved tool prediction")
            require(action["source"] == {"kind": "history", "id": str(i)}, "Prediction source differs")
            require(action["type"] == "prediction", "Past tool is not self-predicted")
    source = read("configs/complementary.json")["sources"]["abcd"]
    item = next(f for f in source["files"] if f["path"].endswith("abcd_v1.1.json.gz"))
    url = f"https://raw.githubusercontent.com/{source['repo']}/{source['revision']}/{item['path']}"
    path = Path("data/raw/complementary/abcd") / (digest(url)[:20] + ".raw")
    data = path.read_bytes()
    require(digest(data) == item["sha256"], "Scoring source changed")
    raw = json.loads(gzip.decompress(data))
    selected = {key.split(":")[1] for key in rows}
    conversations = {str(c["convo_id"]): c for group in raw.values() for c in group
                     if str(c["convo_id"]) in selected}
    require(conversations.keys() == selected, "Dev conversation source missing")
    counts, tools, records = Counter(), defaultdict(Counter), []
    role_names = {"customer": "user", "agent": "assistant", "action": "tool"}
    unique = set()
    for key, row in rows.items():
        conversation = conversations[key.split(":")[1]]
        history = row["input"]["history"]
        observed = [{"role": role_names[r], "text": t} for r, t in conversation["original"][:len(history)]]
        require(observed == history, "Scoring source prefix differs")
        names = {t["id"] for t in row["input"]["tools"]}
        predicted = {p["historyIndex"]: p for p in saved[key]}
        truth = {}
        for index, turn in enumerate(conversation["delexed"][:len(history)]):
            require(role_names[turn["speaker"]] == history[index]["role"], "Source speaker alignment differs")
            if turn["speaker"] == "action" and turn["targets"][2] in names:
                truth[index] = turn["targets"][2]
        counts["cases"] += 1
        counts["casesWithPastTools"] += bool(truth)
        counts["observedPastTools"] += len(truth)
        counts["predictedPastTools"] += len(predicted)
        for index, tool in truth.items():
            prediction = predicted.get(index)
            correct = prediction is not None and prediction["tool"] == tool
            unique.add((conversation["convo_id"], index))
            counts["representedPastTools"] += prediction is not None
            counts["correctPastTools"] += correct
            tools[tool]["observed"] += 1
            tools[tool]["represented"] += prediction is not None
            tools[tool]["correct"] += correct
            if index == max(truth):
                counts["lastPastTools"] += 1
                counts["lastPastToolsCorrect"] += correct
            records.append({"id": key, "family": row["family"], "historyIndex": index,
                            "goldToolForScoringOnly": tool, "prediction": prediction, "correct": correct})
    counts["uniqueObservedToolEvents"] = len(unique)
    write(args.output, {
        "scope": "Post-hoc complete fixed-Dev past-tool recognition diagnosis. Saved predictions were read and hashed before labels. Only source turns strictly before each observed prefix are scoring targets. No current/future action target, Test evaluation, input change, or training change. Repeated events across prefixes are reported separately from unique events.",
        "raw": args.predictions, "rawHash": raw_hash, "scriptHash": digest(Path(__file__).read_bytes()),
        "source": {"path": str(path), "hash": digest(data), "revision": source["revision"]},
        "counts": dict(counts), "perTool": dict(tools), "records": records,
        "accuracyOnRepresented": counts["correctPastTools"] / max(1, counts["representedPastTools"]),
        "accuracyOnObserved": counts["correctPastTools"] / max(1, counts["observedPastTools"]),
        "lastPastToolAccuracy": counts["lastPastToolsCorrect"] / max(1, counts["lastPastTools"]),
    })
    print(dict(counts))


if __name__ == "__main__":
    main()
