"""Public policy indexing and Train-only procedure/progress supervision."""

import gzip
import json
from functools import lru_cache
from pathlib import Path

from .io import digest, read, require
from .suite import canonical

STAGES = 16


def tool_name(button):
    name = canonical(button).replace(" ", "-")
    if name in {"boots", "shirt", "jeans", "jacket", "pricing", "timing", "policy", "membership"}:
        return "search-" + name
    return {
        "notify-internal-team": "notify-team",
        "ask-oracle": "ask-the-oracle",
        "membership-privileges": "membership",
        "select-answer": "select-faq",
        "enter-detail": "enter-details",
    }.get(name, name)


@lru_cache(maxsize=16)
def catalog(text):
    """Return original source offsets, not rewritten policy spans."""
    try:
        policy = json.loads(text)
    except (ValueError, TypeError):
        return []
    if not isinstance(policy, dict):
        return []
    result = []
    for flow, body in policy.items():
        if not isinstance(body, dict) or not isinstance(body.get("subflows"), dict):
            continue
        for name, procedure in body["subflows"].items():
            marker = json.dumps(name, ensure_ascii=False) + ":"
            start = text.find(marker)
            if start < 0:
                continue
            begin = start + len(marker)
            while text[begin].isspace():
                begin += 1
            _, length = json.JSONDecoder().raw_decode(text[begin:])
            steps = [
                tool_name(a["button"])
                for a in procedure.get("actions", [])
                if a.get("button") and a["button"] not in ("N/A", "End Conversation")
            ]
            title = flow + ": " + name + ". " + " ".join(procedure.get("instructions", []))
            actions = [
                {
                    "tool": None if a["button"] == "N/A" else tool_name(a["button"]),
                    "text": a.get("text", ""),
                    "subtext": a.get("subtext", []),
                }
                for a in procedure.get("actions", [])
                if a.get("button")
            ]
            result.append(
                {
                    "id": canonical(flow) + "/" + canonical(name),
                    "name": name,
                    "flow": flow,
                    "start": start,
                    "end": begin + length,
                    "steps": steps,
                    # Keep the old representation for pinned checkpoint reproduction.
                    "text": title + " Steps: " + "; ".join(steps),
                    "summary": title + " Actions: " + "; ".join(steps),
                    "actions": actions,
                    # Public policies contain alternatives and conditional branches.
                    # Preserve their own wording, including non-tool communication.
                    "context": title
                    + "\nAction descriptions:\n"
                    + "\n".join(
                        (a["tool"] or "speak") + ": " + a["text"] + " " + " ".join(a["subtext"])
                        for a in actions
                    ),
                }
            )
    return result


def observed_catalog(value, contextual=False):
    return [
        p | {"document": doc["id"], "text": p["summary"] if contextual else p["text"]}
        for doc in value["documents"]
        for p in catalog(doc["text"])
    ]


def workflow_targets(turns, procedure_map, procedures):
    """Next logged public tool node, conditional on the annotated procedure.

    Future turns are Train supervision only. List positions identify alternatives,
    not a mandatory execution order. Unmapped actions have no node target.
    """
    result, upcoming = [None] * len(turns), None
    for i in range(len(turns) - 1, -1, -1):
        turn = turns[i]
        procedure = procedure_map.get(turn["targets"][0])
        if turn["speaker"] == "action":
            upcoming = (procedure, turn["targets"][2])
        if procedure not in procedures:
            continue
        steps = procedures[procedure]["steps"]
        require(len(steps) < STAGES, "Public tool nodes exceed progress head capacity")
        if upcoming is None:
            # This is a property of the recorded trace, not certified task success.
            result[i] = [STAGES - 1]
        elif upcoming[0] == procedure:
            result[i] = [s for s, tool in enumerate(steps) if tool == upcoming[1]] or None
    return result


def argument_roles(arguments, scenario, slots):
    """Match actual Train argument values to the source's named scenario slots.

    This returns loss labels, never candidate values or observable model state.
    Undeclared or ambiguous values remain masked or marginal alternatives.
    """
    values = {**scenario.get("personal", {}), **scenario.get("order", {})}
    values["membership_level"] = values.get("member_level")
    values["product"] = scenario.get("product", {}).get("names", [])
    values["amount"] = scenario.get("product", {}).get("amounts", [])
    matches = {}
    for role in slots:
        value = values.get(role)
        if value is None:
            continue
        for item in value if isinstance(value, list) else [value]:
            text = canonical(str(item))
            if text:
                matches.setdefault(text, set()).add(role)
    return [sorted(matches.get(canonical(str(value)), ())) for value in arguments]


def annotations(rows, schema_path, workflow=False, source_roles=False, history=False):
    """Load targets only for allowed Train IDs; no source labels enter prepare()."""
    allowed = {r["id"] for r in rows if r["split"] == "train" and r["component"] == "abcd"}
    conversations_allowed = {key.split(":")[1] for key in allowed}
    observed = {r["id"]: r["input"]["history"] for r in rows if history and r["id"] in allowed}
    names = {t["id"] for r in rows if history and r["id"] in allowed for t in r["input"]["tools"]}
    schema = read(schema_path)
    config = read("configs/complementary.json")["sources"]["abcd"]
    item = next(f for f in config["files"] if f["path"].endswith("abcd_v1.1.json.gz"))
    url = f"https://raw.githubusercontent.com/{config['repo']}/{config['revision']}/{item['path']}"
    data = (Path("data/raw/complementary/abcd") / (digest(url)[:20] + ".raw")).read_bytes()
    require(digest(data) == item["sha256"], "Procedure annotation source changed")
    raw, result = json.loads(gzip.decompress(data)), {}
    procedures = {}
    tool_slots = {}
    if source_roles:
        for row in rows:
            if row["id"] not in allowed:
                continue
            for tool in row["input"]["tools"]:
                slots = tool.get("argumentSlots", [])
                require(
                    tool_slots.get(tool["id"], slots) == slots,
                    "Train tool role declarations conflict",
                )
                tool_slots[tool["id"]] = slots
    if workflow:
        # Only public policy documents observed in Train supply node identities.
        documents = {
            doc["text"] for row in rows if row["id"] in allowed for doc in row["input"]["documents"]
        }
        for text in sorted(documents):
            for procedure in catalog(text):
                previous = procedures.get(procedure["id"])
                require(
                    previous is None or previous["steps"] == procedure["steps"],
                    "Public procedure node definitions conflict",
                )
                procedures[procedure["id"]] = procedure
        require(procedures, "No observed Train policies for workflow supervision")
    for split, conversations in raw.items():
        for conversation in conversations:
            if str(conversation["convo_id"]) not in conversations_allowed:
                continue
            nodes = (
                workflow_targets(conversation["delexed"], schema["procedureMap"], procedures)
                if workflow and split == "train"
                else None
            )
            stage, past = 0, []
            original = []
            if history and split == "train":
                require(
                    len(conversation["original"]) == len(conversation["delexed"]),
                    "Observed turn alignment changed",
                )
                roles = {"customer": "user", "agent": "assistant", "action": "tool"}
                original = [{"role": roles[t[0]], "text": t[1]} for t in conversation["original"]]
            for i, turn in enumerate(conversation["delexed"]):
                key = f"abcd:{conversation['convo_id']}:{i}"
                if key in allowed:
                    procedure = schema["procedureMap"].get(turn["targets"][0])
                    if procedure:
                        result[key] = {"procedure": procedure}
                        if not workflow:
                            result[key]["stage"] = min(stage, STAGES - 1)
                        elif nodes is not None and nodes[i] is not None:
                            result[key]["workflowNodes"] = nodes[i]
                        if source_roles and split == "train" and turn["speaker"] == "action":
                            roles = argument_roles(
                                turn["targets"][3],
                                conversation["scenario"],
                                tool_slots.get(turn["targets"][2], []),
                            )
                            if any(roles):
                                result[key]["argumentRoles"] = roles
                    if history and split == "train":
                        require(
                            observed[key] == original[:i],
                            "Past tool targets do not match observed Train history",
                        )
                        if past:
                            result.setdefault(key, {})["pastTools"] = list(past)
                if (
                    history
                    and split == "train"
                    and turn["speaker"] == "action"
                    and turn["targets"][2] in names
                ):
                    past.append({"historyIndex": i, "tool": turn["targets"][2]})
                stage += int(turn["speaker"] == "action")
    require(set(result) <= allowed, "Procedure labels escaped Train")
    return result


def prioritize(chunks, raw, procedures, selected):
    regions = [p for p in procedures if p["id"] in selected]

    def overlap(chunk):
        source = raw[chunk["source"]]
        return any(
            p["document"] == source["id"]
            and chunk["offsets"][0][0] < p["end"]
            and chunk["offsets"][-1][1] > p["start"]
            for p in regions
        )

    return sorted(chunks, key=lambda c: not overlap(c))


def visible(view, procedure):
    return any(
        loc["kind"] == "document"
        and loc["id"] == procedure["document"]
        and procedure["start"] <= loc["start"] < procedure["end"]
        for loc in view["locations"]
    )
