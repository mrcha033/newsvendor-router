"""English component workload. Native targets stay separate from raw provider inputs."""

import copy
import gzip
import io
import json
import math
import re
import urllib.parse
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .data import download, get
from .io import digest, jsonl, lines, read, require, write

SCHEMA = "complementary-workload-v1"
PRIVATE = {
    "label",
    "labels",
    "gold",
    "gold_snippet_id",
    "evidence",
    "derivation",
    "mapping",
    "targets",
    "candidates",
    "flow",
    "subflow",
    "future",
    "theta",
    "responseTree",
    "answer",
    "answers",
    "is_impossible",
    "expected",
    "futureResponses",
    "scenario",
}
COMPONENTS = {"cuad", "contractnli", "orsharc", "abcd", "tatqa", "retail"}


def canonical(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


def template(text):
    return re.sub(r"\d+", "0", canonical(text))


def clean(value):
    if isinstance(value, dict):
        require(not PRIVATE.intersection(value), "Private annotation in public payload")
        for item in value.values():
            clean(item)
    elif isinstance(value, list):
        for item in value:
            clean(item)


def payload(request, *, documents=(), tables=(), history=(), observations=(), tools=()):
    value = {
        "schema": SCHEMA,
        "language": "en",
        "request": request,
        "documents": list(documents),
        "tables": list(tables),
        "history": list(history),
        "observations": list(observations),
        "tools": list(tools),
    }
    clean(value)
    return value


def public_input(row):
    value = row["input"]
    require(
        set(value)
        == {
            "schema",
            "language",
            "request",
            "documents",
            "tables",
            "history",
            "observations",
            "tools",
        },
        "Unexpected public input fields",
    )
    require(
        value["schema"] == SCHEMA and value["language"] == "en", "Invalid suite language/schema"
    )
    require(isinstance(value["request"], str) and value["request"].strip(), "Missing raw request")
    allowed = {
        "documents": {"id", "title", "text"},
        "tables": {"id", "cells"},
        "history": {"role", "text"},
        "tools": {"id", "kind", "description", "argumentSlots"},
        "observations": {
            "date",
            "sales",
            "stockoutHours",
            "hourlySales",
            "hourlyStockStatus",
            "discount",
            "holidayFlag",
            "activityFlag",
            "precipitation",
            "temperature",
            "humidity",
            "wind",
        },
    }
    for field, keys in allowed.items():
        require(isinstance(value[field], list), "Public field must be a list: " + field)
        for item in value[field]:
            require(
                isinstance(item, dict) and not set(item) - keys, "Unexpected public field: " + field
            )
    clean(value)
    return copy.deepcopy(value)


def document(id, title, text):
    return {"id": id, "title": title, "text": text}


def record(component, id, source, input, target, keys, text=""):
    return {
        "id": component + ":" + str(id),
        "component": component,
        "officialSplit": source,
        "input": input,
        "target": target,
        "keys": keys,
        "groupText": text,
    }


def assets(name, config):
    require(re.fullmatch(r"[0-9a-f]{40}", config["revision"]), "Pin a commit, not a branch")
    contents, sources = {}, []
    for item in config["files"]:
        url = f"https://raw.githubusercontent.com/{config['repo']}/{config['revision']}/{item['path']}"
        content, meta = download(url, "data/raw/complementary/" + name)
        require(
            digest(content) == item["sha256"], f"Source checksum changed: {name}/{item['path']}"
        )
        contents[Path(item["path"]).name] = content
        sources.append(meta)
    return contents, sources


def cuad(contents, config):
    with zipfile.ZipFile(io.BytesIO(contents["data.zip"])) as archive:
        contracts = json.loads(archive.read("CUADv1.json"))["data"]
    rows = []
    for contract in contracts:
        if not any(term in contract["title"].casefold() for term in config["titleTerms"]):
            continue
        for paragraph in contract["paragraphs"]:
            text = paragraph["context"]
            doc = document("contract", contract["title"], text)
            for q in paragraph["qas"]:
                if q["id"].split("__")[-1] not in config["categories"]:
                    continue
                spans = [
                    {
                        "document": "contract",
                        "start": a["answer_start"],
                        "end": a["answer_start"] + len(a["text"]),
                    }
                    for a in q["answers"]
                ]
                target = {
                    "action": "answer" if spans else "abstain",
                    "spans": spans,
                    "answers": [a["text"] for a in q["answers"]],
                }
                rows.append(
                    record(
                        "cuad",
                        q["id"],
                        "original-full",
                        payload(
                            q["question"]
                            + " Cite character spans; abstain if no such clause is present.",
                            documents=[doc],
                        ),
                        target,
                        ["cuad-contract:" + contract["title"]],
                        text,
                    )
                )
    return rows, []


def contractnli(contents, config):
    rows = []
    choices = {
        "Entailment": "entailment",
        "Contradiction": "contradiction",
        "NotMentioned": "not_mentioned",
    }
    with zipfile.ZipFile(io.BytesIO(contents["contract-nli.zip"])) as archive:
        for split in ("train", "dev", "test"):
            data = json.loads(archive.read(f"contract-nli/{split}.json"))
            for doc in data["documents"]:
                annotations = doc["annotation_sets"][0]["annotations"]
                for id, hypothesis in data["labels"].items():
                    annotation = annotations[id]
                    spans = [
                        {
                            "document": "contract",
                            "start": doc["spans"][i][0],
                            "end": doc["spans"][i][1],
                        }
                        for i in annotation["spans"]
                    ]
                    request = (
                        "Classify the following claim as entailment, contradiction, or "
                        "not_mentioned under this agreement, and cite supporting character "
                        "spans: " + hypothesis["hypothesis"]
                    )
                    rows.append(
                        record(
                            "contractnli",
                            f"{doc['id']}:{id}",
                            split,
                            payload(
                                request,
                                documents=[document("contract", doc["file_name"], doc["text"])],
                            ),
                            {
                                "action": "answer",
                                "answer": choices[annotation["choice"]],
                                "spans": spans,
                            },
                            ["nli-contract:" + str(doc["id"])],
                            doc["text"],
                        )
                    )
    return rows, []


def orsharc(contents, config):
    snippets = json.loads(contents["id2snippet.json"])
    collection = [
        document("rule:" + id, "English government rule", text) for id, text in snippets.items()
    ]
    rows = []
    for split in ("train", "dev", "test"):
        for r in map(json.loads, contents[f"open_retrieval_sharc_{split}.json"].splitlines()):
            history = []
            if r["scenario"]:
                history.append({"role": "user", "text": r["scenario"]})
            for turn in r["history"]:
                history.extend(
                    [
                        {"role": "assistant", "text": turn["follow_up_question"]},
                        {"role": "user", "text": turn["follow_up_answer"]},
                    ]
                )
            answer = r["answer"]
            action = "answer" if answer.casefold() in {"yes", "no", "irrelevant"} else "ask"
            snippet = str(r["gold_snippet_id"])
            url = urllib.parse.urlsplit(r["source_url"])
            page = url.netloc.casefold() + url.path.rstrip("/")
            rows.append(
                record(
                    "orsharc",
                    r["utterance_id"],
                    split,
                    payload(
                        r["question"],
                        history=history,
                        tools=[
                            {
                                "id": "retrieve_rules",
                                "kind": "retrieval",
                                "description": "Search the shared English rules collection by query. "
                                "Return raw documents and their ids; no answers are supplied.",
                            }
                        ],
                    ),
                    {"action": action, "answer": answer, "retrieved": ["rule:" + snippet]},
                    ["rule-page:" + page, "rule-tree:" + str(r["tree_id"])],
                    snippets[snippet],
                )
            )
    return rows, collection


def abcd(contents, config):
    conversations = json.loads(gzip.decompress(contents["abcd_v1.1.json.gz"]))
    guidelines = json.loads(contents["guidelines.json"])
    ontology = json.loads(contents["ontology.json"])
    policy = document(
        "abcd-policy",
        "Complete retailer policies for all workflows",
        json.dumps(guidelines, ensure_ascii=False, indent=2),
    )
    tools = [
        {"id": name, "kind": kind, "description": name.replace("-", " "), "argumentSlots": slots}
        for kind, values in ontology["actions"].items()
        for name, slots in values.items()
    ]
    rows = []
    eligible = [
        (split, convo)
        for split, records in conversations.items()
        for convo in records
        if any(
            role == "customer" and len(re.findall(r"\w+", text)) >= 8
            for role, text in convo["original"]
        )
    ]
    eligible.sort(
        key=lambda item: digest([config["seed"], "abcd-conversation:" + str(item[1]["convo_id"])])
    )
    chosen = {convo["convo_id"] for _, convo in eligible[: config["groups"]]}
    for split, records in conversations.items():
        for convo in records:
            if convo["convo_id"] not in chosen:
                continue
            require(
                len(convo["original"]) == len(convo["delexed"]), "ABCD sequence length mismatch"
            )
            for index, (original, turn) in enumerate(
                zip(convo["original"], convo["delexed"], strict=True)
            ):
                require(original[0] == turn["speaker"], "ABCD speaker alignment mismatch")
                if turn["speaker"] not in {"agent", "action"}:
                    continue
                # turn_count has gaps after source preprocessing; list position is aligned.
                prefix = convo["original"][:index]
                # Remove greeting-only prefixes using observed input, never the next target.
                if not any(
                    role == "customer" and len(re.findall(r"\w+", text)) >= 8
                    for role, text in prefix
                ):
                    continue
                roles = {"customer": "user", "agent": "assistant", "action": "tool"}
                history = [{"role": roles[role], "text": text} for role, text in prefix]
                targets = turn["targets"]
                observed_text = canonical(
                    " ".join(text for _, text in prefix) + " " + policy["text"]
                )
                target = {
                    "action": "call_tool" if turn["speaker"] == "action" else "speak",
                    "tool": targets[2],
                    "arguments": targets[3],
                    "answer": convo["original"][index][1],
                    "prefixLength": index,
                    "argumentsObservable": all(
                        canonical(str(arg)) in observed_text for arg in targets[3]
                    ),
                }
                # Delexicalization is used only for duplicate-prefix grouping, not as input.
                prefix_text = "\n".join(t["text"] for t in convo["delexed"][:index])
                rows.append(
                    record(
                        "abcd",
                        f"{convo['convo_id']}:{index}",
                        split,
                        payload(
                            "Continue this customer interaction with a spoken response or one tool call. "
                            "Use the complete policy and observed conversation; provide ordered arguments "
                            "only when supported by the observed conversation or policy.",
                            documents=[policy],
                            history=history,
                            tools=tools,
                        ),
                        target,
                        [
                            "abcd-conversation:" + str(convo["convo_id"]),
                            "abcd-prefix:" + digest(template(prefix_text)),
                        ],
                    )
                )
    return rows, []


def tatqa(contents, config):
    rows = []
    for split in ("train", "dev"):
        for context in json.loads(contents[f"tatqa_dataset_{split}.json"]):
            table = context["table"]["table"]
            paragraphs = sorted(context["paragraphs"], key=lambda p: p["order"])
            docs = [
                document("paragraph:" + str(p["order"]), "Report paragraph", p["text"])
                for p in paragraphs
            ]
            identity = {"table": table, "paragraphs": [p["text"] for p in paragraphs]}
            text = json.dumps(identity, ensure_ascii=False)
            for q in context["questions"]:
                rows.append(
                    record(
                        "tatqa",
                        q["uid"],
                        split,
                        payload(
                            q["question"] + " Return the answer in the table's displayed units and "
                            "give its scale (empty, percent, thousand, million, or billion).",
                            documents=docs,
                            tables=[{"id": "report-table", "cells": table}],
                        ),
                        {
                            "action": "answer",
                            "answer": q["answer"],
                            "scale": q["scale"],
                            "answerType": q["answer_type"],
                            "derivation": q["derivation"],
                        },
                        ["report-context:" + digest(identity)],
                        text,
                    )
                )
    return rows, []


def retail(contents, config):
    hub = "https://huggingface.co/api/datasets/" + config["repo"]
    require(json.loads(get(hub))["sha"] == config["revision"], "Retail Hub revision changed")

    def capture(offset):
        query = urllib.parse.urlencode(
            {
                "dataset": config["repo"],
                "config": "default",
                "split": "train",
                "offset": offset,
                "length": 100,
            }
        )
        content, meta = download(
            "https://datasets-server.huggingface.co/rows?" + query,
            "data/raw/complementary/retail/" + config["revision"],
        )
        parsed = json.loads(content)
        return (
            [r["row"] for r in parsed["rows"]],
            {**meta, "offset": offset},
            parsed["num_rows_total"],
        )

    _, first_meta, total = capture(0)
    offsets = sorted(
        {(i * total // config["batches"] // 90) * 90 for i in range(config["batches"])}
    )
    sources, groups = [], defaultdict(list)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for batch, meta, size in pool.map(capture, offsets):
            require(size == total, "Retail row count changed during capture")
            sources.append(meta)
            for r in batch:
                groups[(r["store_id"], r["product_id"])].append(r)
    require(first_meta == sources[0], "Retail cached snapshot changed")
    require(json.loads(get(hub))["sha"] == config["revision"], "Retail changed during capture")
    rows = []
    history_days, horizon = config["historyDays"], config["horizonDays"]
    for (store, product), daily in groups.items():
        daily = sorted({r["dt"]: r for r in daily}.values(), key=lambda r: r["dt"])
        if len(daily) < history_days + horizon:
            continue
        observed, future = daily[:history_days], daily[history_days : history_days + horizon]
        observations = [
            {
                "date": r["dt"],
                "sales": r["sale_amount"],
                "stockoutHours": r["stock_hour6_22_cnt"],
                "hourlySales": r["hours_sale"],
                "hourlyStockStatus": r["hours_stock_status"],
                "discount": r["discount"],
                "holidayFlag": r["holiday_flag"],
                "activityFlag": r["activity_flag"],
                "precipitation": r["precpt"],
                "temperature": r["avg_temperature"],
                "humidity": r["avg_humidity"],
                "wind": r["avg_wind_level"],
            }
            for r in observed
        ]
        input = payload(
            f"Predict the next {horizon}-day total demand distribution after {observed[-1]['dt']} "
            f"for store {store}, product {product}. Return F as [[demand, probability], ...] "
            "in globally normalized sales units, with nonnegative demand and probabilities "
            "summing to one. Use sales history and stockout information to estimate demand.",
            observations=observations,
        )
        target = {
            "action": "answer",
            "answer": [r["sale_amount"] for r in future],
            "dates": [r["dt"] for r in future],
            "complete": [r["stock_hour6_22_cnt"] == 0 for r in future],
            "latentDemandLabels": False,
            "cutoff": observed[-1]["dt"],
        }
        rows.append(
            record(
                "retail",
                f"{store}:{product}",
                "train-observations",
                input,
                target,
                [f"retail-store:{store}", f"retail-product:{product}"],
            )
        )
    return rows, sources


def select(rows, config, seed):
    groups = defaultdict(list)
    for r in rows:
        groups[r["keys"][0]].append(r)
    chosen = sorted(groups, key=lambda key: digest([seed, key]))[: config["groups"]]
    require(len(chosen) == config["groups"], "Insufficient source groups")
    return [
        r
        for key in chosen
        for r in sorted(groups[key], key=lambda r: digest([seed, r["id"]]))[: config["perGroup"]]
    ]


def partition(rows, config):
    parent = {}

    def find(key):
        parent.setdefault(key, key)
        if parent[key] != key:
            parent[key] = find(parent[key])
        return parent[key]

    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)

    texts = {}
    for r in rows:
        keys = r["keys"] + ["public-case:" + digest(public_input(r))]
        if r["groupText"]:
            key = "document-template:" + digest(template(r["groupText"]))
            keys.append(key)
            texts.setdefault(key, r["groupText"])
        r["keys"] = keys
        for key in keys[1:]:
            union(keys[0], key)
    shingles = {}
    for key, text in texts.items():
        words = template(text).split()
        shingles[key] = {tuple(words[i : i + 5]) for i in range(len(words) - 4)}
    pairs = 0
    keys = sorted(shingles)
    threshold = config["templateSimilarity"]
    for i, a in enumerate(keys):
        x = shingles[a]
        for b in keys[i + 1 :]:
            y = shingles[b]
            if min(len(x), len(y)) < 25 or min(len(x), len(y)) / max(len(x), len(y)) < threshold:
                continue
            common = len(x & y)
            if common / (len(x) + len(y) - common) >= threshold:
                union(a, b)
                pairs += 1
    identities = defaultdict(set)
    for r in rows:
        identities[find(r["keys"][0])].update(
            key
            for key in r["keys"]
            if not key.startswith(("public-case:", "document-template:", "abcd-prefix:"))
        )
    for r in rows:
        # Stable source identities keep assignments fixed when raw formatting changes.
        r["family"] = digest(sorted(identities[find(r["keys"][0])]))
    families = defaultdict(set)
    for r in rows:
        families[r["component"]].add(r["family"])
    assigned = {}
    for component, values in sorted(families.items()):
        ordered = sorted(values - assigned.keys(), key=lambda f: digest([config["seed"], f]))
        require(len(ordered) >= 5, f"Too few independent families: {component}")
        train = math.floor(len(ordered) * config["splits"]["train"])
        dev = math.floor(len(ordered) * config["splits"]["dev"])
        for i, family in enumerate(ordered):
            assigned[family] = "train" if i < train else "dev" if i < train + dev else "test"
    for r in rows:
        r["split"] = assigned[r["family"]]
    return {
        "nearTemplatePairs": pairs,
        "families": len(assigned),
        "method": "connected source/page/conversation/store/product, exact inputs, "
        "number-normalized documents, and 5-word-shingle Jaccard templates",
        "templateSimilarity": threshold,
    }


def prepare(config):
    require(config["schema"] == SCHEMA and config["language"] == "en", "English suite required")
    require(set(config["sources"]) == COMPONENTS, "Missing complementary component")
    require(
        set(config["splits"]) == {"train", "dev", "test"} and sum(config["splits"].values()) == 1,
        "Invalid split fractions",
    )
    rows, collection, provenance = [], [], {}
    adapters = {
        "cuad": cuad,
        "contractnli": contractnli,
        "orsharc": orsharc,
        "abcd": abcd,
        "tatqa": tatqa,
    }
    for name, source in config["sources"].items():
        if name == "retail":
            records, files = retail({}, source)
            documents = []
        else:
            contents, files = assets(name, source)
            records, documents = adapters[name](contents, {**source, "seed": config["seed"]})
        rows.extend(select(records, source, config["seed"]))
        collection.extend(documents)
        provenance[name] = {**source, "captured": files, "compiledCandidates": len(records)}
        print(f"{name}: {len(rows)} cumulative cases", flush=True)
    grouping = partition(rows, config)
    rows.sort(key=lambda r: r["id"])
    inputs = [{k: r[k] for k in ("id", "component", "family", "split", "input")} for r in rows]
    labels = [
        {
            "id": r["id"],
            "target": r["target"],
            "officialSplit": r["officialSplit"],
            "leakageKeys": r["keys"],
        }
        for r in rows
    ]
    output = Path(config["output"])
    jsonl(output / "inputs.jsonl", inputs)
    jsonl(output / "labels.jsonl", labels)
    jsonl(output / "collection.jsonl", collection)
    manifest = {
        "schema": SCHEMA,
        "language": "en",
        "retailTask": "conditional-total-demand-v1",
        "config": config,
        "configHash": digest(config),
        "transformHash": digest(Path(__file__).read_bytes()),
        "sources": provenance,
        "inputHash": digest(inputs),
        "labelHash": digest(labels),
        "collectionHash": digest(collection),
        "grouping": grouping,
        "scope": "native component diagnostics, not an end-to-end order benchmark",
        "coupledSources": False,
        "newHumanReviewedCases": 0,
        "modelRuns": 0,
        "latentDemandLabels": False,
        "economicActionValueLabels": False,
        "limitations": [
            "Custom source-disjoint splits replace official leaderboard splits.",
            "TAT-QA identifies contexts, not original reports; report-wide disjointness is unknown.",
            "ABCD is roleplay with shared policies, not production logs or unseen-policy evaluation.",
            "ABCD argument scoring is restricted to values observable in the prefix or full policy.",
            "OR-ShARC supplies crowd scenarios and recorded follow-ups, not procurement logs.",
            "The unlabeled rule collection is public to all arms, including held-out rule text.",
            "FreshRetail rows API is not revision-pinned; captured response hashes identify the snapshot.",
            "Recorded follow-up text is a reference, not a unique or causally optimal action.",
            "Template checks cannot guarantee organization-level independence or no pretraining exposure.",
            "No unsupported purchase parameters, question costs, response probabilities or latent F labels.",
        ],
    }
    write(output / "manifest.json", manifest)
    report = check(output)
    manifest["counts"] = report["components"]
    write(output / "manifest.json", manifest)
    return report


def check(directory):
    directory = Path(directory)
    manifest = read(directory / "manifest.json")
    inputs, labels = lines(directory / "inputs.jsonl"), lines(directory / "labels.jsonl")
    collection = lines(directory / "collection.jsonl")
    require(manifest["schema"] == SCHEMA and manifest["language"] == "en", "Invalid manifest")
    for key, values in (
        ("inputHash", inputs),
        ("labelHash", labels),
        ("collectionHash", collection),
    ):
        require(digest(values) == manifest[key], "Dataset checksum mismatch: " + key)
    require(len({r["id"] for r in inputs}) == len(inputs), "Duplicate case id")
    require([r["id"] for r in inputs] == [r["id"] for r in labels], "Input/label id mismatch")
    require(len({d["id"] for d in collection}) == len(collection), "Duplicate retrieval document")
    groups, public_hashes = {}, {}
    for row, label in zip(inputs, labels, strict=True):
        require(
            row["component"] in COMPONENTS and row["split"] in {"train", "dev", "test"},
            "Invalid component or split",
        )
        value = public_input(row)
        for key in [row["family"], *label["leakageKeys"]]:
            require(groups.setdefault(key, row["split"]) == row["split"], "Source/template leakage")
        fingerprint = digest(value)
        require(
            public_hashes.setdefault(fingerprint, row["split"]) == row["split"], "Input overlap"
        )
        target = label["target"]
        docs = {d["id"]: d["text"] for d in value["documents"]}
        for span in target.get("spans", []):
            require(
                span["document"] in docs
                and 0 <= span["start"] < span["end"] <= len(docs[span["document"]]),
                "Invalid target evidence span",
            )
        if row["component"] == "orsharc":
            require(not value["documents"], "Gold rule supplied as a retrieved document")
            require(set(target["retrieved"]) <= {d["id"] for d in collection}, "Missing rule")
        if row["component"] == "abcd":
            require(len(value["history"]) == target["prefixLength"], "ABCD future turn in prefix")
        if row["component"] == "retail":
            dates = [r["date"] for r in value["observations"]]
            require(
                dates == sorted(set(dates))
                and max(dates) == target["cutoff"] < min(target["dates"]),
                "Future retail observation leaked",
            )
            require(target["latentDemandLabels"] is False, "Censored sales labeled as demand")
            require(
                len(target["answer"]) == len(target["complete"]) == len(target["dates"]),
                "Retail horizon mismatch",
            )
    counts = {}
    for component in sorted(COMPONENTS):
        rows = [r for r in inputs if r["component"] == component]
        require(rows, "Missing component: " + component)
        counts[component] = {
            "cases": len(rows),
            "families": len({r["family"] for r in rows}),
            "splits": dict(Counter(r["split"] for r in rows)),
            "splitFamilies": {
                s: len({r["family"] for r in rows if r["split"] == s})
                for s in ("train", "dev", "test")
            },
        }
        require(
            set(counts[component]["splits"]) == {"train", "dev", "test"},
            "A component lacks a held-out split",
        )
    report = {
        "cases": len(inputs),
        "components": counts,
        "sourceOverlaps": 0,
        "inputOverlaps": 0,
        "retrievalDocuments": len(collection),
        "language": "en",
        "scope": manifest["scope"],
        "endToEndOrderBenchmarkReady": False,
    }
    write(directory / "check.json", report)
    return report


def retrieve(collection, query, limit=5):
    """Shared lexical retrieval uses only query and raw collection, never annotations."""
    require(isinstance(limit, int) and 1 <= limit <= 100, "Invalid retrieval limit")
    words = set(canonical(query).split())
    frequencies = [Counter(canonical(d["text"]).split()) for d in collection]
    df = Counter(word for freq in frequencies for word in freq)
    average = sum(sum(f.values()) for f in frequencies) / max(1, len(frequencies))
    hits = []
    for doc, freq in zip(collection, frequencies, strict=True):
        size = sum(freq.values())
        score = sum(
            math.log(1 + (len(collection) - df[w] + 0.5) / (df[w] + 0.5))
            * (freq[w] * 2.2)
            / (freq[w] + 1.2 * (0.25 + 0.75 * size / average))
            for w in words
            if freq[w]
        )
        if score:
            hits.append({**doc, "score": score})
    return sorted(hits, key=lambda r: (-r["score"], r["id"]))[:limit]
