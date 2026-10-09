"""Public, source-preserving fragments and calculator candidates for all model arms."""

import itertools
import math
import re
from collections import Counter
from functools import lru_cache

from .io import digest, require
from .suite import canonical, public_input, retrieve
from .suite_score import normalized, number


def kind(value):
    request = value["request"]
    if request.startswith("Highlight the parts"):
        return "cuad"
    if request.startswith("Classify the following claim"):
        return "contractnli"
    if value["observations"]:
        return "retail"
    if value["tables"]:
        return "tatqa"
    if any(t["id"] == "retrieve_rules" for t in value["tools"]):
        return "orsharc"
    if value["documents"] and not value["history"] and not value["tools"]:
        return "cuad"
    require(bool(value["history"]), "Unrecognized public task")
    return "abcd"


def context(value):
    return (
        value["request"] + "\n" + "\n".join(h["role"] + ": " + h["text"] for h in value["history"])
    )


def fragments(documents, width=500):
    result = []
    for doc in documents:
        text = doc["text"]
        # Fixed boundaries never inspect expert spans or answers. Every source character is indexed.
        start = 0
        while start < len(text):
            end = min(start + width, len(text))
            if end < len(text):
                boundaries = list(re.finditer(r"(?:[.;!?]\s+|\n\s*\n)", text[start:end]))
                if boundaries and boundaries[-1].end() > width // 2:
                    end = start + boundaries[-1].end()
                else:
                    space = text.rfind(" ", start + width // 2, end)
                    if space > start:
                        end = space + 1
            result.append(
                {"document": doc["id"], "start": start, "end": end, "text": text[start:end]}
            )
            start = end
    return result


@lru_cache(maxsize=128)
def rank_index(texts):
    counts = [Counter(canonical(text).split()) for text in texts]
    df = Counter(w for c in counts for w in c)
    return counts, df


def rank(items, query, limit):
    words = set(canonical(query).split())
    counts, df = rank_index(tuple(x["text"] for x in items))
    scored = []
    for i, (item, counts_i) in enumerate(zip(items, counts, strict=True)):
        score = sum(
            math.log(1 + len(items) / (1 + df[w])) * min(counts_i[w], 3)
            for w in words
            if counts_i[w]
        ) / (1 + sum(counts_i.values()) / 200)
        scored.append((score, i, item))
    return [v for _, _, v in sorted(scored, key=lambda x: (-x[0], x[1]))[:limit]]


def argument(slot, value):
    text = "\n".join(h["text"] for h in value["history"])
    patterns = {
        "email": r"[\w.+-]+@[\w.-]+\.[A-Za-z]+",
        "email_address": r"[\w.+-]+@[\w.-]+\.[A-Za-z]+",
        "order_id": r"(?:order\s*(?:id|number)?\s*[:#]?\s*)([A-Za-z0-9-]{5,})",
        "account_id": r"(?:account\s*(?:id|number)?\s*[:#]?\s*)([A-Za-z0-9-]{5,})",
        "zip_code": r"\b\d{5}(?:-\d{4})?\b",
        "phone": r"\b\d{3}[- .]\d{3}[- .]\d{4}\b",
        "customer_name": r"(?:my name is|name\s*:)\s*([A-Za-z]+\s+[A-Za-z]+)",
    }
    if slot in patterns:
        hits = list(re.finditer(patterns[slot], text, re.I))
        if hits:
            hit = hits[-1]
            return hit.group(1) if hit.lastindex else hit.group()
    # No hidden profile values or original tool arguments are consulted.
    return ""


def calculators(value, selected, maximum):
    literals, answers = [], []
    for table in value["tables"]:
        cells = table["cells"]
        for row in cells:
            for j, cell in enumerate(row):
                n = number(cell)
                caption = " | ".join(str(x) for x in [cells[0][j], row[0], cell])
                answers.append({"answer": cell, "text": caption, "op": "literal"})
                if n is not None:
                    literals.append({"answer": n, "text": caption, "op": "literal"})
    for frag in selected:
        answers.append({"answer": frag["text"].strip(), "text": frag["text"], "op": "span"})
        for match in re.finditer(r"(?<!\w)[−-]?\d[\d,]*(?:\.\d+)?%?(?!\w)", frag["text"]):
            n = number(match.group())
            if n is not None:
                literals.append({"answer": n, "text": frag["text"], "op": "literal"})
    relevant = rank(literals, value["request"], 10)
    computed = []
    for a, b in itertools.permutations(relevant, 2):
        x, y = a["answer"], b["answer"]
        for op, result in (
            ("difference", x - y),
            ("sum", x + y),
            ("average", (x + y) / 2),
            ("ratio", x / y if y else None),
            ("percent", 100 * x / y if y else None),
            ("percent change", 100 * (x - y) / y if y else None),
        ):
            if result is not None and math.isfinite(result):
                computed.append(
                    {"answer": result, "text": op + ": " + a["text"] + " ; " + b["text"], "op": op}
                )
    # Keep a mix of direct values and operations without consulting answerType/derivation.
    selected_answers = rank(answers + literals, value["request"], maximum // 2)
    selected_answers += rank(computed, value["request"], maximum - len(selected_answers))
    unique = {}
    for candidate in selected_answers:
        unique.setdefault(normalized(candidate["answer"]), candidate)
    return [{"action": "answer", **c} for c in unique.values()]


def prepare(row, collection, config):
    value = public_input(row)
    task = kind(value)
    query = context(value)
    hits = retrieve(collection, query, 5) if task == "orsharc" else []
    documents = value["documents"] + [{k: h[k] for k in ("id", "title", "text")} for h in hits]
    indexed = fragments(documents)
    selected = rank(indexed, query, config["fragments"])
    candidates = []
    if task in {"cuad", "contractnli"}:
        candidates = [
            {
                "action": "answer",
                "evidence": [{k: f[k] for k in ("document", "start", "end")}],
                "text": f["text"],
            }
            for f in selected
        ]
    elif task == "orsharc":
        candidates = [
            {"action": "answer", "answer": a, "text": a} for a in ("Yes", "No", "Irrelevant")
        ] + [
            {
                "action": "ask",
                "answer": "Does this condition apply: " + f["text"].strip(),
                "text": f["text"],
            }
            for f in selected
        ]
    elif task == "abcd":
        candidates = [
            {
                "action": "speak",
                "answer": "Could you provide the missing details?",
                "text": "Ask the customer for needed details or explain the policy.",
            }
        ]
        candidates += [
            {
                "action": "call_tool",
                "tool": t["id"],
                "arguments": [argument(s, value) for s in t["argumentSlots"]],
                "text": t["description"] + " " + " ".join(t["argumentSlots"]),
            }
            for t in value["tools"]
        ]
    elif task == "tatqa":
        candidates = calculators(value, selected, config["candidates"])
    for i, candidate in enumerate(candidates):
        candidate["id"] = str(i)
    require(task == "retail" or candidates, "Empty public candidate catalog")
    return {
        "task": task,
        "public": value,
        "context": query,
        "fragments": selected,
        "candidates": candidates,
        "retrieved": [h["id"] for h in hits],
        "inputHash": digest(value),
        "viewHash": digest({"public": value, "fragments": selected, "candidates": candidates}),
        "indexedCharacters": sum(f["end"] - f["start"] for f in indexed),
        "sourceCharacters": sum(len(d["text"]) for d in documents),
    }


def prediction(candidate):
    return {k: v for k, v in candidate.items() if k not in {"id", "text", "op"}}
