"""Annotation-free views with reversible token, cell and tool-result locations."""

import json
import re
from bisect import bisect_left, bisect_right
from collections import deque
from functools import lru_cache

import torch

from .io import digest, read, require
from .native_inputs import kind, rank
from .structured_procedures import observed_catalog, prioritize
from .structured_tools import entities, roles
from .suite import clean, public_input, retrieve

OPS = ("copy", "add", "subtract", "multiply", "divide", "average", "percent", "change")
SCALES = ("", "percent", "thousand", "million", "billion")
MODES = ("span", "choice", "entity", "compute", "missing", "conflict")
DECISIONS = ("entailment", "contradiction", "not_mentioned", "Yes", "No", "Irrelevant")


def numeric_spans(source):
    """Retain the sign of a wholly numeric accounting cell and its source span."""
    from .suite_score import number

    text = source["text"]
    accounting = re.fullmatch(
        r"\s*[$£€]?\s*(\(\s*[$£€]?\s*(\d[\d,]*(?:\.\d+)?%?)\s*\))\s*", text
    ) if source["kind"] == "cell" else None
    if accounting:
        value = number(accounting.group(2))
        yield -value if value is not None else None, *accounting.span(1)
        return
    for match in re.finditer(r"(?<!\w)[−-]?\d[\d,]*(?:\.\d+)?%?(?!\w)", text):
        yield number(match.group()), *match.span()


def tool_modes(spec):
    modes = ["span", "entity", "missing", "conflict"]
    if spec.get("choices"):
        modes.append("choice")
    if spec.get("valueType") in ("number", "integer"):
        modes.append("compute")
    return modes


def public_actions(value):
    task = kind(value)
    allowed = {
        "abcd": {"respond", "ask", "confirm", "retrieve"},
        "cuad": {"answer", "retrieve", "hold"},
        "contractnli": {"answer", "retrieve"},
        "orsharc": {"answer", "ask", "retrieve"},
        "tatqa": {"answer", "retrieve"},
        "retail": {"answer", "ask", "confirm", "retrieve", "hold"},
    }[task]
    if task == "abcd":
        allowed |= {"call_tool:" + tool["id"] for tool in value["tools"]}
    return allowed


def schemas(tools, definitions=None, positional=False):
    definitions = definitions or {}
    fields = []
    for tool in tools:
        if tool.get(
            "argumentStyle", "positional" if positional else "named"
        ) == "positional" and not tool.get("parameters"):
            slots = tool.get("argumentSlots", [])
            choices = list(
                dict.fromkeys(
                    v for name in slots for v in definitions.get(name, {}).get("enum", [])
                )
            )
            for position in range(len(slots)):
                fields.append(
                    {
                        "id": tool["id"] + ":argument_" + str(position + 1),
                        "tool": tool["id"],
                        "name": "argument_" + str(position + 1),
                        "context": tool["description"],
                        "description": "Ordered value "
                        + str(position + 1)
                        + " for "
                        + tool["description"]
                        + "; possible fields: "
                        + ", ".join(slots),
                        "choices": choices,
                        "required": False,
                        "valueType": "string",
                        "position": position,
                        "slotHints": slots,
                        "roles": roles(slots, definitions),
                        "closed": False,
                    }
                )
            continue
        properties = tool.get("parameters", {}).get("properties", {})
        required = tool.get("parameters", {}).get("required", tool.get("argumentSlots", []))
        for name in tool.get("argumentSlots", list(properties)):
            spec = properties.get(name, definitions.get(name, {}))
            fields.append(
                {
                    "id": tool["id"] + ":" + name,
                    "tool": tool["id"],
                    "context": tool["description"],
                    "name": name,
                    "description": spec.get("description", name.replace("_", " ")),
                    "choices": spec.get("enum", []),
                    "required": name in required,
                    "valueType": spec.get("type", "string"),
                    "format": spec.get("format"),
                }
            )
    for field in fields:
        field["modes"] = tool_modes(field)
    return fields


def research_input(value):
    """Only currently observed research inputs; environment truth is not accepted here."""
    docs = []
    for doc in value["docs"]:
        docs.append(
            {
                "id": doc["id"],
                "title": doc["title"],
                "text": doc["text"],
                "metadata": {k: doc[k] for k in ("sku", "period", "role", "version", "complete")},
            }
        )
    task = value["task"]
    history = value["observations"]
    if "forecast" in task:
        # Daily/hourly numeric features go to the demand encoder, not thousands
        # of serialized numbers competing with contract evidence for text tokens.
        history = {
            "source": value.get("historySource"),
            "rows": len(history),
            "firstDate": history[0].get("date") if history else None,
            "lastDate": history[-1].get("date") if history else None,
            "forecast": task["forecast"],
        }
    docs.append(
        {
            "id": "observed-demand",
            "title": "Observed demand and stockout history",
            "text": json.dumps(history, sort_keys=True),
        }
    )
    return {
        "request": "Construct purchase cost, selling price, net refund, manager preference and "
        "demand; choose the next action. "
        + json.dumps(
            {
                k: task[k]
                for k in (
                    "sku",
                    "period",
                    "bounds",
                    "hold",
                    "costs",
                    "tolerance",
                    "deadline",
                    "decision",
                )
            }
            | {"remaining": value["remaining"]}
            | ({k: task[k] for k in ("forecast", "quantityUnit")} if "forecast" in task else {}),
            sort_keys=True,
        ),
        "documents": docs,
        "tables": [],
        "tools": [],
        "history": [
            {"role": "tool", "text": json.dumps(h, sort_keys=True)} for h in value["history"]
        ],
        "observations": [],
    }


def sources(value):
    result = []
    for doc in value["documents"]:
        result.append(
            {
                "kind": "document",
                "id": doc["id"],
                "text": doc["text"],
                "title": doc.get("title", ""),
                "metadata": doc.get("metadata", {}),
            }
        )
    for i, turn in enumerate(value["history"]):
        result.append(
            {"kind": "history", "id": str(i), "text": turn["text"], "title": turn["role"]}
        )
    for table in value["tables"]:
        for i, row in enumerate(table["cells"]):
            for j, cell in enumerate(row):
                result.append(
                    {
                        "kind": "cell",
                        "id": table["id"],
                        "row": i,
                        "column": j,
                        "text": str(cell),
                        "title": " | ".join(map(str, [table["cells"][0][j], row[0]])),
                    }
                )
    # The request itself may contain observed values and is a legitimate pointer source.
    result.append({"kind": "request", "id": "request", "text": value["request"], "title": ""})
    return result


def location(source, start, end):
    return {k: source[k] for k in ("kind", "id", "row", "column") if k in source} | {
        "start": start,
        "end": end,
    }


def same_source(a, b):
    return all(a.get(k) == b.get(k) for k in ("kind", "id", "row", "column"))


def source_key(value):
    return value["kind"], value["id"], value.get("row"), value.get("column")


@lru_cache(maxsize=2048)
def source_tokens(tokenizer, text):
    # Cache public tokenization only. Encoder states use current trainable weights.
    return tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)


@lru_cache(maxsize=4096)
def query_tokens(tokenizer, text):
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def span_tokens(view, loc):
    if "spanIndex" not in view:
        index = {}
        for i, token in enumerate(view["locations"]):
            starts, ends, ids = index.setdefault(source_key(token), ([], [], []))
            starts.append(token["start"])
            ends.append(token["end"])
            ids.append(i)
        view["spanIndex"] = index
        view["sourceIndex"] = {source_key(s): s for s in view["sources"]}
    starts, ends, ids = view["spanIndex"].get(source_key(loc), ([], [], []))
    return ids[bisect_right(ends, loc["start"]) : bisect_left(starts, loc["end"])]


def document_chunks(chunks, context, config, round, raw=(), procedures=(), selected_ids=()):
    budget = config.get("documentTokens")
    if budget is None:
        return rank(chunks, context, config.get("chunks", 8) * (round + 1))
    budget *= 2**round
    selected, tokens = [], 0
    ranked = rank(chunks, context, len(chunks))
    if selected_ids:
        ranked = prioritize(ranked, raw, procedures, selected_ids)
    for chunk in ranked:
        if tokens + len(chunk["ids"]) <= budget:
            selected.append(chunk)
            tokens += len(chunk["ids"])
    return selected


def merge_chunks(chunks, raw, tokenizer):
    merged = []
    for chunk in chunks:
        if (
            merged and merged[-1]["source"] == chunk["source"]
            and chunk["tokenStart"] <= merged[-1]["tokenEnd"]
        ):
            first = merged[-1]
            end = max(first["tokenEnd"], chunk["tokenEnd"])
            encoded = source_tokens(tokenizer, raw[chunk["source"]]["text"])
            first.update(
                tokenEnd=end,
                ids=encoded["input_ids"][first["tokenStart"] : end],
                offsets=encoded["offset_mapping"][first["tokenStart"] : end],
            )
        else:
            merged.append(dict(chunk))
    return merged


def prepare(
    value,
    tokenizer,
    config,
    *,
    fields=None,
    actions=None,
    state=None,
    round=0,
    collection=(),
    confirmations=(),
    linked=False,
):
    if "input" in value:
        value = public_input(value)
    clean(value)
    value = {**value, "documents": list(value["documents"])}
    hits = []
    if collection and any(t["id"] == "retrieve_rules" for t in value["tools"]):
        hits = retrieve(collection, value["request"], config.get("retrievalDocuments", 5))
        value["documents"] += [{k: h[k] for k in ("id", "title", "text")} for h in hits]
    definitions = read(config["toolSchema"]) if config.get("toolSchema") else {"fields": {}}
    positional = config.get("positionalTools", False)
    if positional:
        value["tools"] = [
            dict(t, argumentStyle="positional")
            if not t.get("parameters") and "argumentStyle" not in t
            else t
            for t in value["tools"]
        ]
    default_fields = fields is None
    fields = fields or [
        {
            "id": "answer",
            "name": "answer",
            "description": value["request"],
            "choices": [],
            "required": False,
        }
    ] + schemas(value["tools"], definitions["fields"], positional)
    fields = list(fields)
    if default_fields and (
        value["request"].startswith(("Highlight the parts", "Classify the following claim"))
        or any(tool["id"] == "retrieve_rules" for tool in value["tools"])
    ):
        fields[0]["modes"] = ["span", "entity", "missing", "conflict"]
    if positional and any(
        t.get("argumentSlots") and t.get("argumentStyle") == "positional" for t in value["tools"]
    ):
        fields += [
            {
                "id": "question:" + name,
                "name": name,
                "description": "Ask or confirm the observed " + name.replace("_", " "),
                "choices": definitions["fields"].get(name, {}).get("enum", []),
                "required": False,
                "questionField": True,
            }
            for name in definitions.get("names", [])
        ]
    for field in fields:
        if field.get("tool") or field.get("questionField"):
            field.setdefault("valueType", "string")
            field["modes"] = tool_modes(field)
    explicit_actions = actions is not None
    actions = actions or [
        {"id": "answer", "text": "Answer using observed evidence"},
        {"id": "ask", "text": "Ask for a missing field or resolve conflicting evidence"},
        {"id": "confirm", "text": "Request confirmation of the selected field and tool arguments"},
        {"id": "retrieve", "text": "Read additional source chunks"},
        {"id": "hold", "text": "Stop without a supported answer"},
        {"id": "respond", "text": "Respond using an observed policy and a fixed template"},
        *[{"id": "call_tool:" + t["id"], "text": t["description"]} for t in value["tools"]],
    ]
    allowed = {a["id"] for a in actions} if explicit_actions else public_actions(value)
    recent = "\n".join(h["role"] + ": " + h["text"] for h in reversed(value["history"][-4:]))
    # Retrieval hints and copied candidates have dedicated paths. Serializing them
    # before the recent dialogue can evict the user's latest words from the prefix.
    summary = {k: v for k, v in (state or {}).items() if k not in ("memory", "procedureIds")}
    context = json.dumps(summary, sort_keys=True) + "\n" + recent + "\n" + value["request"]
    context_ids = query_tokens(tokenizer, context)
    max_length = config["maxLength"]
    prefix = context_ids[: min(config.get("queryTokens", 128), max_length // 3)]
    header_limit = min(48, max_length // 4)
    width = min(
        config.get("chunkTokens", max_length), max_length - len(prefix) - header_limit - 3
    )
    overlap = config.get("overlap", 64)
    require(0 <= overlap < width, "Chunk overlap exceeds available source tokens")
    raw = sources(value)
    procedures = observed_catalog(value, config.get("procedureContext", False)) if config.get("structuredTools") else []
    chunks = []
    for index, source in enumerate(raw):
        encoded = source_tokens(tokenizer, source["text"])
        ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
        for start in range(0, len(ids), width - overlap):
            end = min(start + width, len(ids))
            chunks.append(
                {
                    "source": index,
                    "tokenStart": start,
                    "tokenEnd": end,
                    "ids": ids[start:end],
                    "offsets": offsets[start:end],
                    "text": source["title"]
                    + " "
                    + source["text"][offsets[start][0] : offsets[end - 1][1]],
                }
            )
            if end == len(ids):
                break
    require(chunks, "Empty source tokens")
    documents = [c for c in chunks if raw[c["source"]]["kind"] == "document"]
    selected = document_chunks(documents, context, config, round, raw, procedures,
                               (state or {}).get("procedureIds", []))
    for source_kind, limit in (
        ("history", config.get("historyChunks", 64)),
        ("cell", config.get("tableChunks", 128)),
        ("request", 4),
    ):
        candidates = [c for c in chunks if raw[c["source"]]["kind"] == source_kind]
        if source_kind == "history":
            candidates = sorted(candidates, key=lambda c: c["source"], reverse=True)
        selected.extend(candidates[: limit * (round + 1)])
    selected.sort(key=lambda c: (c["source"], c["tokenStart"]))
    count = len(selected)
    packed = config.get("packedSources", False)
    if packed:
        selected = merge_chunks(selected, raw, tokenizer)
    sequence_ids, positions, token_locations, span_index = [], [], [], {}
    seen = set()
    for chunk in selected:
        source = raw[chunk["source"]]
        identity = source_key(source)
        starts, ends, token_ids = span_index.setdefault(identity, ([], [], []))
        # Titles and public applicability metadata condition the encoder but are not pointer targets.
        metadata = source.get("metadata", {})
        header_text = " ".join(
            f"{k} {metadata[k]}"
            for k in ("complete", "role", "version", "sku", "period")
            if k in metadata
        )
        title = source["title"]
        if packed:
            title = source["kind"] + " " + source["id"] + " " + title
        header = query_tokens(tokenizer, header_text + " " + title)[:header_limit]
        available = max_length - len(prefix) - len(header) - 3
        for begin in range(0, len(chunk["ids"]), available):
            ids = chunk["ids"][begin : begin + available]
            offsets = chunk["offsets"][begin : begin + available]
            if not packed:
                base = 2 + len(prefix) + len(header)
                sequence_ids.append([
                    tokenizer.cls_token_id, *prefix, *header, tokenizer.sep_token_id,
                    *ids, tokenizer.sep_token_id,
                ])
            else:
                if (
                    not sequence_ids
                    or len(sequence_ids[-1]) + len(header) + len(ids) + 1 > max_length
                ):
                    sequence_ids.append([tokenizer.cls_token_id, *prefix, tokenizer.sep_token_id])
                base = len(sequence_ids[-1]) + len(header)
                sequence_ids[-1].extend([*header, *ids, tokenizer.sep_token_id])
            seq = len(sequence_ids) - 1
            for j, (start, end) in enumerate(offsets):
                key = identity, start, end
                if start == end or key in seen:
                    continue
                seen.add(key)
                positions.append((seq, base + j))
                token_ids.append(len(token_locations))
                starts.append(start)
                ends.append(end)
                token_locations.append(location(source, start, end))
    queries = []
    for field in fields:
        queries.append(
            field["name"] + ": " + field["description"] + "\n" + field.get("context", "")
        )
    queries += [a["text"] for a in actions]
    choices, choice_fields, choice_values = [], [], []
    for i, field in enumerate(fields):
        for choice in field.get("choices", []):
            choices.append(str(choice))
            choice_fields.append(i)
            choice_values.append(choice)
    query_start = len(sequence_ids)
    query_prefix = [] if config.get("sharedQueries", False) else prefix
    unique_choices = list(dict.fromkeys(choices))
    query_positions = []
    role_names = list(dict.fromkeys(r["name"] for f in fields for r in f.get("roles", []))) if config.get("structuredTools") else []
    extra_queries = ["Argument role: " + n.replace("_", " ") for n in role_names]
    procedure_start = len(queries) + len(unique_choices) + len(extra_queries)
    extra_queries += [p["text"] for p in procedures]
    for query in queries + unique_choices + extra_queries:
        qids = query_tokens(tokenizer, query)
        if query in extra_queries and query not in extra_queries[:len(role_names)]:
            qids = qids[:config.get("procedureTokens", 96)]
        qids = qids[: max_length - len(query_prefix) - 3]
        if config.get("packedQueries", False):
            if (
                len(sequence_ids) == query_start
                or len(sequence_ids[-1]) + len(qids) + 1 > max_length
            ):
                sequence_ids.append([tokenizer.cls_token_id, *query_prefix, tokenizer.sep_token_id])
            start = len(sequence_ids[-1])
            sequence_ids[-1].extend([*qids, tokenizer.sep_token_id])
            query_positions.append((len(sequence_ids) - 1, start, start + max(len(qids), 1)))
        else:
            sequence_ids.append(
                [
                    tokenizer.cls_token_id,
                    *query_prefix,
                    tokenizer.sep_token_id,
                    *qids,
                    tokenizer.sep_token_id,
                ]
            )
            query_positions.append((len(sequence_ids) - 1, 0, 1))
    controller_start, controller_truncated = None, False
    history_start, controller_turns = None, []
    if config.get("dialogueController") and value["tools"] and value["history"] and kind(value) == "abcd":
        from .structured_control import encode

        encoded = encode(value, actions, procedures, tokenizer, config, (state or {}).get("procedureIds", []),
                         turns=config.get("dialogueState", False))
        ids, spans, controller_truncated = encoded[:3]
        controller_start = len(query_positions)
        seq = len(sequence_ids)
        sequence_ids.append(ids)
        query_positions += [(seq, lo, hi) for lo, hi in spans] + [(seq, 0, 1)]
        if config.get("dialogueState"):
            controller_turns = encoded[3]
            history_start = len(query_positions)
            query_positions += [(seq, turn["start"], turn["end"]) for turn in controller_turns]
    batch_ids = torch.full(
        (len(sequence_ids), max(map(len, sequence_ids))), tokenizer.pad_token_id, dtype=torch.long
    )
    batch_mask = torch.zeros_like(batch_ids)
    for i, ids in enumerate(sequence_ids):
        batch_ids[i, : len(ids)] = torch.tensor(ids)
        batch_mask[i, : len(ids)] = 1
    batch = {"input_ids": batch_ids, "attention_mask": batch_mask}
    atoms = []
    span_view = {
        "locations": token_locations,
        "sources": raw,
        "spanIndex": span_index,
        "sourceIndex": {source_key(s): s for s in raw},
    }
    for source in raw:
        for val, start, end in numeric_spans(source):
            loc = location(source, start, end)
            indices = span_tokens(span_view, loc)
            if val is not None and indices and span_indices(span_view, loc) is not None:
                atoms.append({"value": val, "location": loc, "tokens": indices})
    observed_entities = []
    if config.get("structuredTools"):
        for entity in entities(raw, state):
            indices = [i for loc in entity["evidence"] for i in span_tokens(span_view, loc)]
            if indices:
                observed_entities.append(entity | {"tokens": sorted(set(indices))})
    return {
        "batch": dict(batch),
        "positions": positions,
        "locations": token_locations,
        "spanIndex": span_index,
        "sourceIndex": span_view["sourceIndex"],
        "queryStart": query_start,
        "queryPositions": query_positions,
        "controllerStart": controller_start,
        "controllerTruncated": controller_truncated,
        "controllerEnabled": config.get("controllerEnabled", True),
        "controllerTurns": controller_turns,
        "dialogueWorkflow": config.get("dialogueWorkflow", False),
        "historyStart": history_start,
        "historyActions": [i for i, action in enumerate(actions) if action["id"].startswith("call_tool:")],
        "contextRerank": config.get("contextRerank", False),
        "fields": fields,
        "actions": actions,
        "allowedActions": [a["id"] in allowed for a in actions],
        "choices": choices,
        "choiceValues": choice_values,
        "choiceQueries": [unique_choices.index(c) for c in choices],
        "choiceFields": choice_fields,
        "atoms": atoms,
        "entities": observed_entities,
        "structuredTools": config.get("structuredTools", False),
        "copyAgreement": config.get("copyAgreement", False),
        "roles": role_names,
        "roleStart": len(queries) + len(unique_choices),
        "procedures": procedures,
        "procedureContext": config.get("procedureContext", False),
        "workflowProgress": config.get("workflowProgress", False),
        "rerankSupervision": config.get("rerankSupervision", False),
        "procedureStart": procedure_start,
        "rerank": config.get("rerank", True),
        "sources": raw,
        "public": value,
        "inputHash": digest(value),
        "retrieved": [h["id"] for h in hits],
        "confirmations": list(confirmations),
        "linked": linked,
        "indexedChunks": len(chunks),
        "selectedChunks": count,
        "encodedTokens": int(batch_mask.sum()),
        "documentTokens": sum(
            1 for loc in token_locations if loc["kind"] == "document"
        ),
        "queryTruncated": len(context_ids) > len(prefix),
    }


def span_indices(view, loc):
    ids = span_tokens(view, loc)
    if not ids:
        return None
    source = view["sourceIndex"][source_key(loc)]
    # Whitespace may be outside tokenizer offsets; unretrieved content cannot bridge a label.
    if (
        source["text"][loc["start"] : view["locations"][ids[0]]["start"]].strip()
        or source["text"][view["locations"][ids[-1]]["end"] : loc["end"]].strip()
    ):
        return None
    for first, last in zip(ids[:-1], ids[1:], strict=True):
        if source["text"][
            view["locations"][first]["end"] : view["locations"][last]["start"]
        ].strip():
            return None
    return ids[0], ids[-1]


def span_value(view, start, end):
    first, last = view["locations"][start], view["locations"][end]
    require(same_source(first, last) and first["start"] <= last["start"], "Invalid source span")
    source = next(s for s in view["sources"] if same_source(s, first))
    loc = first | {"end": last["end"]}
    return source["text"][loc["start"] : loc["end"]], loc


def best_span(view, start, end, max_tokens=512):
    # One CPU transfer per score vector; a sliding maximum avoids quadratic scalar GPU reads.
    start, end = start.detach().float().cpu().tolist(), end.detach().float().cpu().tolist()
    if "segments" not in view:
        segments, begin = [], 0
        locations = view["locations"]
        for i in range(1, len(locations)):
            first, last = locations[i - 1], locations[i]
            split = not same_source(first, last) or last["start"] < first["start"]
            if not split:
                source = next(s for s in view["sources"] if same_source(s, first))
                split = bool(source["text"][first["end"] : last["start"]].strip())
            if split:
                segments.append((begin, i))
                begin = i
        segments.append((begin, len(locations)))
        view["segments"] = segments
    best = None
    for lo, hi in view["segments"]:
        queue = deque()
        for i in range(hi - 1, lo - 1, -1):
            while queue and queue[0] >= i + max_tokens:
                queue.popleft()
            while queue and end[queue[-1]] <= end[i]:
                queue.pop()
            queue.append(i)
            j = queue[0]
            score = start[i] + end[j]
            if best is None or score > best[0] or (score == best[0] and (i, j) < best[1:]):
                best = (score, i, j)
    require(best is not None, "No valid source span")
    return best[1:]
