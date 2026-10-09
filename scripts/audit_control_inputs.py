"""Audit actual controller policy inputs against saved Dev replay traces."""

import argparse
import sys
from collections import Counter
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    from newsvendor import structured_control as control
    from newsvendor.cli import provenance
    from newsvendor.io import digest, lines, read, require, write
    from newsvendor.structured_inputs import prepare
    from newsvendor.suite import public_input

    destination = Path(args.output)
    require(not destination.exists(), "Preserve previous audit")
    config = read(Path(args.run) / "config.json")
    source_hash = read(Path(args.run) / "run.json")["identity"]["sourceHash"]
    require(provenance(config)["sourceHash"] == source_hash, "Replay source changed")
    require(config["encoder"].get("procedureContext"), "Full policy context required")
    raw = lines(args.raw)
    saved = {r["key"]: r for r in raw}
    rows = [r for r in lines(Path(config["dataset"]) / "inputs.jsonl")
            if r["id"] in saved]
    require(len(rows) == len(saved) and all(r["split"] == "dev" for r in rows),
            "This audit accepts matching Dev inputs only")
    tokenizer = AutoTokenizer.from_pretrained(config["encoder"]["model"],
                                             revision=config["encoder"]["revision"],
                                             cache_dir=".cache/torch-models",
                                             local_files_only=True)
    rows.sort(key=lambda r: (r["id"].rsplit(":", 1)[0], len(r["input"]["history"]), r["id"]))
    states, histories, inputs = {}, {}, []
    original = control.query_tokens
    for row in rows:
        value = public_input(row)
        conversation = row["id"].rsplit(":", 1)[0]
        history, old = value["history"], histories.get(conversation, [])
        state = states.get(conversation) if history[:len(old)] == old else None
        record, rounds = saved[row["id"]], []
        require(len((state or {}).get("memory", [])) == record["priorOwnMemory"],
                "Own-memory replay differs")
        for trace in record["trace"]:
            encoded_text = []

            def tokens(current_tokenizer, text, captured=encoded_text):
                captured.append(text)
                return original(current_tokenizer, text)

            with patch.object(control, "query_tokens", tokens):
                view = prepare(value, tokenizer, config["encoder"], state=state,
                               round=trace["round"])
            for name in ("indexedChunks", "selectedChunks", "queryTruncated"):
                require(view[name] == trace[name], f"Replay differs: {row['id']} {name}")
            require(len(encoded_text) == len(view["actions"]) + 2,
                    "Unexpected controller tokenization order")
            # This is the actual policy string passed to the controller tokenizer.
            # procedureContext checks that this string fits without truncation.
            policy_text = encoded_text[-2]
            prediction = trace["prediction"]
            rounds.append({"round": trace["round"], "policyHash": digest(policy_text),
                           "policies": [p["id"] for p in view["procedures"]
                                        if p["context"] in policy_text],
                           "learnedProcedures": prediction.get("procedureIds", [])})
            state = {"memory": prediction.get("memory", []),
                     "procedureIds": prediction.get("procedureIds", []),
                     "unresolved": [{"field": f["field"], "state": f["state"]}
                                    for f in prediction["fields"] if f["value"] is None],
                     "question": prediction.get("question", {})}
        inputs.append({"id": row["id"], "rounds": rounds})
        prediction = record["prediction"]
        states[conversation] = {"memory": prediction.get("memory", []),
                                "procedureIds": prediction.get("procedureIds", [])}
        histories[conversation] = history
    # Hidden Dev annotations are opened only after all observable inputs are built.
    annotated = read(args.annotations)["conditions"]["114688"]["records"]
    labels = {r["id"]: r for r in annotated}
    require(set(saved) == labels.keys(), "Annotation IDs differ from audited Dev IDs")
    counts = Counter()
    for record in inputs:
        target = labels[record["id"]]
        gold = target["goldProcedure"]
        first, last = record["rounds"][0], record["rounds"][-1]
        is_call = target["goldAction"] == "call_tool"
        correct = bool(saved[record["id"]]["metrics"].get("toolExact"))
        record.update(goldProcedure=gold, goldCall=is_call, toolCorrect=correct,
                      firstCovered=gold in first["policies"],
                      finalCovered=gold in last["policies"])
        counts["cases"] += 1
        counts["firstCovered"] += record["firstCovered"]
        counts["finalCovered"] += record["finalCovered"]
        counts["multiRound"] += len(record["rounds"]) > 1
        if is_call:
            counts["calls"] += 1
            counts["callFinalCovered"] += record["finalCovered"]
            counts["toolErrors"] += not correct
            counts["toolErrorsWithPolicy"] += not correct and record["finalCovered"]
    report = {"scope": "Actual controller policy tokenization reconstructed from saved Dev own-state traces; no inference, training, or Test use. Labels opened only after input construction.",
              "raw": args.raw, "rawHash": digest(Path(args.raw).read_bytes()),
              "annotationsHash": digest(Path(args.annotations).read_bytes()),
              "configHash": digest(config), "scriptHash": digest(Path(__file__).read_bytes()),
              "sourceHash": source_hash,
              "counts": dict(counts), "records": inputs}
    write(destination, report)
    print(report["counts"])


if __name__ == "__main__":
    main()
