import io
import json
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .io import digest, jsonl, lines, read, require, write


def get(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def download(url, directory):
    path = Path(directory) / (digest(url)[:20] + ".raw")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        content = get(url)
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(content)
        temporary.replace(path)
    content = path.read_bytes()
    return content, {"url": url, "sha256": digest(content), "bytes": len(content)}


def normalize_tat(contexts, source):
    result = []
    for context in contexts:
        table = context.get("table", {}).get("table", [])
        paragraphs = [
            {"id": p.get("uid", str(p["order"])), "order": p["order"], "text": p["text"]}
            for p in context.get("paragraphs", [])
        ]
        for question in context.get("questions", []):
            result.append(
                {
                    "id": question["uid"],
                    "group": digest({"table": table, "paragraphs": paragraphs}),
                    "source": source,
                    "input": {
                        "question": question["question"],
                        "table": table,
                        "paragraphs": paragraphs,
                    },
                    "label": {
                        "answer": question["answer"],
                        "answerType": question.get("answer_type"),
                        "derivation": question.get("derivation"),
                        "mapping": question.get("mapping"),
                        "scale": question.get("scale"),
                    },
                }
            )
    return result


def normalize_sharc(rows, source):
    return [
        {
            "id": r["utterance_id"],
            "group": digest(r["snippet"]),
            "source": source,
            "input": {
                k: r.get(k, []) if k == "history" else r.get(k, "")
                for k in ("question", "snippet", "scenario", "history")
            },
            "label": {
                "answer": r["answer"],
                "evidence": r.get("evidence"),
                "action": r["answer"].lower()
                if r["answer"].lower() in ("yes", "no", "irrelevant")
                else "ask",
            },
        }
        for r in rows
    ]


def select(train, dev, count):
    chosen = train[: count * 3 // 5]
    groups = {r["group"] for r in chosen}
    chosen += [r for r in dev if r["group"] not in groups][: count - len(chosen)]
    require(len(chosen) == count, "Insufficient cases with disjoint train/dev source groups")
    require(len({r["id"] for r in chosen}) == count, "Duplicate case ID")
    return chosen


def split_retail(rows, count):
    groups = {}
    for row in rows:
        key = f"{row['store_id']}:{row['product_id']}"
        groups.setdefault(key, []).append(row)
    selected = []
    for id, daily in groups.items():
        if len(daily) < 60:
            continue
        daily = sorted(daily, key=lambda row: row["dt"])
        require(len({r["dt"] for r in daily}) == len(daily), "Duplicate retail day")
        a, b = len(daily) * 3 // 5, len(daily) * 4 // 5

        def convert(r):
            return {
                "date": r["dt"],
                "sales": r["sale_amount"],
                "stockout": r["stock_hour6_22_cnt"] > 0,
                "stockHours": r["stock_hour6_22_cnt"],
                "hours": r["hours_stock_status"],
                "complete": r["stock_hour6_22_cnt"] == 0,
            }

        selected.append(
            {
                "id": id,
                "units": "globally-normalized-sales",
                "train": list(map(convert, daily[:a])),
                "cal": list(map(convert, daily[a:b])),
                "test": list(map(convert, daily[b:])),
                "latentDemandLabels": False,
            }
        )
        if len(selected) == count:
            break
    return selected


def save_cases(name, cases, meta):
    inputs = [{k: v for k, v in r.items() if k != "label"} for r in cases]
    labels = [{"id": r["id"], "label": r["label"]} for r in cases]
    jsonl(f"data/processed/{name}/inputs.jsonl", inputs)
    jsonl(f"data/processed/{name}/labels.jsonl", labels)
    write(
        f"data/processed/{name}/manifest.json",
        {**meta, "n": len(cases), "inputHash": digest(inputs), "labelHash": digest(labels)},
    )


def fetch(config):
    manifests = {}
    tat = config["tatqa"]
    partitions, sources = [], []
    for file, source in zip(tat["files"], ("official-train", "official-dev"), strict=True):
        url = f"https://raw.githubusercontent.com/{tat['repo']}/{tat['revision']}/{file}"
        content, meta = download(url, "data/raw/tatqa")
        sources.append(meta)
        partitions.append(normalize_tat(json.loads(content), source))
    cases = select(*partitions, tat["limit"])
    manifests["tatqa"] = {
        **tat,
        "sources": sources,
        "dataGap": "Raw schema lacks original report identity; grouping uses complete table/paragraph context.",
    }
    save_cases("tatqa", cases, manifests["tatqa"])
    print(f"TAT-QA: {len(cases)} cases", flush=True)
    sharc = config["sharc"]
    content, meta = download(sharc["url"], "data/raw/sharc")
    partitions = []
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        for kind in ("train", "dev"):
            name = next(
                n
                for n in archive.namelist()
                if n.endswith(f"/sharc_{kind}.json") and not n.startswith("__MACOSX")
            )
            partitions.append(normalize_sharc(json.loads(archive.read(name)), "official-" + kind))
    cases = select(*partitions, sharc["limit"])
    manifests["sharc"] = {**sharc, **meta}
    save_cases("sharc", cases, manifests["sharc"])
    print(f"ShARC: {len(cases)} cases", flush=True)
    retail = config["retail"]
    hub = "https://huggingface.co/api/datasets/" + retail["repo"]
    require(json.loads(get(hub))["sha"] == retail["revision"], "FreshRetail revision changed")
    rows, sources, selected = [], [], []

    def capture(offset):
        query = urllib.parse.urlencode(
            {
                "dataset": retail["repo"],
                "config": "default",
                "split": "train",
                "offset": offset,
                "length": 100,
            }
        )
        content, meta = download(
            "https://datasets-server.huggingface.co/rows?" + query,
            "data/raw/retail/" + retail["revision"],
        )
        parsed = json.loads(content)
        require(isinstance(parsed.get("rows"), list), "Invalid FreshRetail rows")
        return [r["row"] for r in parsed["rows"]], {**meta, "offset": offset}

    with ThreadPoolExecutor(max_workers=4) as pool:
        for offset in range(0, retail["maxRows"], 400):
            for batch, meta in pool.map(
                capture, range(offset, min(offset + 400, retail["maxRows"]), 100)
            ):
                rows.extend(batch)
                sources.append(meta)
            selected = split_retail(rows, retail["series"])
            if offset % 2000 == 0:
                print(f"FreshRetail: {len(selected)}/{retail['series']} series", flush=True)
            if len(selected) == retail["series"]:
                break
    require(json.loads(get(hub))["sha"] == retail["revision"], "FreshRetail changed during capture")
    require(len(selected) == retail["series"], "Insufficient FreshRetail series")
    manifests["retail"] = {
        **retail,
        "sources": sources,
        "series": len(selected),
        "revisionStatus": "Hub SHA matched before/after capture; rows API has no revision pin. Local response hashes identify the exact snapshot.",
        "selection": "First store/product series with at least 60 days in official train ordering",
        "seriesHash": digest(selected),
        "latentDemandLabels": False,
    }
    write("data/processed/retail/series.json", selected)
    write("data/processed/retail/manifest.json", manifests["retail"])
    write("data/processed/manifest.json", manifests)
    print(f"FreshRetail: {len(selected)} series ready", flush=True)
    return check()


def check():
    report = {}
    for name in ("tatqa", "sharc"):
        directory = f"data/processed/{name}"
        inputs, labels = lines(directory + "/inputs.jsonl"), lines(directory + "/labels.jsonl")
        require([r["id"] for r in inputs] == [r["id"] for r in labels], "Input/label ID mismatch")
        manifest = read(directory + "/manifest.json")
        require(
            digest(inputs) == manifest["inputHash"] and digest(labels) == manifest["labelHash"],
            "Processed dataset checksum mismatch",
        )
        for r in inputs:
            require(
                "label" not in r and not ({"answer", "evidence", "derivation"} & r["input"].keys()),
                "Evaluation label leaked into input",
            )
        train = {r["group"] for r in inputs if r["source"] == "official-train"}
        require(
            all(r["group"] not in train for r in inputs if r["source"] == "official-dev"),
            "Source context crosses train/dev",
        )
        report[name] = {
            "n": len(inputs),
            "train": sum(r["source"] == "official-train" for r in inputs),
            "dev": sum(r["source"] == "official-dev" for r in inputs),
        }
    series = read("data/processed/retail/series.json")
    require(
        digest(series) == read("data/processed/retail/manifest.json")["seriesHash"],
        "Processed retail checksum mismatch",
    )
    for s in series:
        require(
            s["train"][-1]["date"] < s["cal"][0]["date"]
            and s["cal"][-1]["date"] < s["test"][0]["date"],
            "Future retail observation leaked",
        )
        require(s["latentDemandLabels"] is False, "Censored sales mislabeled as latent demand")
    report["retail"] = {
        "series": len(series),
        "days": sum(len(s[k]) for s in series for k in ("train", "cal", "test")),
    }
    write("data/processed/check.json", report)
    return report
