"""Workload auditing and a raw-document pilot contract, separate from the legacy generator.

Private annotations are used only for checking fixtures. They never construct provider inputs
or planner states. This module does not claim to run trained adapters on the new schema.
"""

import copy
import math
import re
from collections import Counter
from datetime import datetime, timedelta

from .construction import atoms
from .corpus import audit, generate
from .io import digest, require
from .optimizer import optimal
from .policy import planner

SPLITS = {"train", "dev", "cal", "test", "fixture"}
PRIVATE = {
    "gold",
    "label",
    "labels",
    "theta",
    "environment",
    "expected",
    "caseTags",
    "scenario",
    "omega",
    "responseTree",
    "possibleAnswers",
    "futureResponses",
    "numeric",
    "reviewed",
    "resolves",
    "adjudicatedValues",
}
SCHEMA = "raw-newsvendor-v2"


def guarded(value):
    if isinstance(value, dict):
        require(not PRIVATE.intersection(value), "Private annotations in public input")
        for v in value.values():
            guarded(v)
    elif isinstance(value, list):
        for v in value:
            guarded(v)


def instant(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(result.tzinfo is not None, "Clock must include a timezone")
    return result


def validate_input(input):
    guarded(input)
    require(
        set(input) == {"schema", "task", "docs", "observations", "tools", "history"},
        "Raw input schema",
    )
    require(input["schema"] == SCHEMA, "Unsupported raw input schema")
    task = input["task"]
    require(
        not {"allowed", "rho", "partial"}.intersection(task), "Generator response model in raw task"
    )
    require(
        set(task)
        == {
            "request",
            "sku",
            "period",
            "currency",
            "units",
            "now",
            "cutoff",
            "requestLimit",
            "managerIds",
            "uncertaintyPolicy",
        },
        "Unknown raw task fields",
    )
    require(isinstance(task["requestLimit"], int) and task["requestLimit"] >= 0, "Request limit")
    require(instant(task["now"]) <= instant(task["cutoff"]), "Clock exceeds cutoff")
    ids = [d["id"] for d in input["docs"]]
    require(len(ids) == len(set(ids)), "Duplicate document ID")
    for doc in input["docs"]:
        require(
            set(doc) == {"id", "title", "text", "source"},
            "Semantic document annotations are not raw evidence",
        )
        require(bool(doc["text"]) and bool(doc["source"]), "Missing source text or provenance")
        require(
            not {"sku", "period", "role", "version", "complete"}.intersection(doc["source"]),
            "Inferred document metadata in provenance",
        )
        require(
            set(doc["source"]) == {"channel", "author", "sentAt"},
            "Unknown source provenance fields",
        )
        require(
            instant(doc["source"]["sentAt"]) <= instant(task["now"]),
            "Future document is not observed evidence",
        )
    for row in input["observations"]:
        require("demand" not in row, "Latent demand is not an observed sale")
        require(row["sold"] >= 0 and row["available"] >= 0, "Negative observations")
        require(row["sold"] <= row["available"], "Sales exceed available units")
    tool_ids = [t["id"] for t in input["tools"]]
    require(len(tool_ids) == len(set(tool_ids)), "Duplicate tool ID")
    for tool in input["tools"]:
        require(
            tool["kind"] in {"retrieval", "factual_query", "preference_selection", "escalation"},
            "Tool kind",
        )
        require(
            not {"rho", "partial", "answers", "outcomes"}.intersection(tool),
            "Private response model in tool",
        )
        require(
            set(tool) == {"id", "kind", "owner", "description", "fee", "handlingMinutes"},
            "Unknown tool fields",
        )
        require(math.isfinite(tool["fee"]) and tool["fee"] >= 0, "Invalid tool fee")
        require(
            math.isfinite(tool["handlingMinutes"]) and tool["handlingMinutes"] >= 0,
            "Invalid handling time",
        )


def public_input(row):
    """Only this raw payload goes to constructors and providers; grouping stays with evaluators."""
    require(
        set(row) == {"id", "family", "template", "split", "origin", "input"}, "Public record schema"
    )
    require(row["split"] in SPLITS, "Unknown split")
    require(row["origin"] in {"constructed", "organizational"}, "Unknown source origin")
    validate_input(row["input"])
    return copy.deepcopy(row["input"])


def partition(rows):
    seen, families, templates = set(), {}, {}
    for row in rows:
        public_input(row)
        require(row["id"] not in seen, "Duplicate episode ID")
        seen.add(row["id"])
        for key, owners in (("family", families), ("template", templates)):
            value = row[key]
            require(
                value not in owners or owners[value] == row["split"],
                f"{key} crosses splits: {value}",
            )
            owners[value] = row["split"]
    return {
        "episodes": len(rows),
        "families": len(families),
        "templates": len(templates),
        "counts": dict(Counter(r["split"] for r in rows)),
    }


def observe(input, tool_id, response):
    """Apply one obtained tool result, respecting elapsed time and preserving original sources.

    The environment selects the response outside the provider. A late response adds no evidence;
    a forecast or partial answer remains text, without promotion to true demand or chosen utility.
    """
    validate_input(input)
    require(
        set(response) == {"afterMinutes", "outcome", "docs", "observations"}, "Tool result schema"
    )
    guarded(response)
    tool = next((t for t in input["tools"] if t["id"] == tool_id), None)
    require(tool is not None, "Unknown tool")
    require(input["task"]["requestLimit"] > 0, "Request budget exhausted")
    require(
        response["outcome"] in {"complete", "partial", "no_response"}, "Unknown response outcome"
    )
    delay = response["afterMinutes"]
    require(math.isfinite(delay) and delay >= 0, "Invalid response delay")
    updated = copy.deepcopy(input)
    start, cutoff = instant(input["task"]["now"]), instant(input["task"]["cutoff"])
    require(start < cutoff, "Cannot request after cutoff")
    arrival = start + timedelta(minutes=delay)
    late = arrival > cutoff
    updated["task"]["now"] = min(arrival, cutoff).isoformat()
    updated["task"]["requestLimit"] -= 1
    updated["history"].append(
        {
            "tool": tool_id,
            "outcome": "timeout" if late else response["outcome"],
            "elapsedMinutes": (min(arrival, cutoff) - start).total_seconds() / 60,
            "fee": tool["fee"],
            "handlingMinutes": tool["handlingMinutes"],
        }
    )
    require(
        response["outcome"] != "no_response" or not (response["docs"] or response["observations"]),
        "No-response result contains evidence",
    )
    if not late:
        updated["docs"].extend(copy.deepcopy(response["docs"]))
        updated["observations"].extend(copy.deepcopy(response["observations"]))
    validate_input(updated)
    return updated


def comparison_contract(design):
    require(design["schema"] == SCHEMA, "Design schema")
    names = set()
    for arm in design["comparisons"]:
        require(arm["name"] not in names, "Duplicate comparison name")
        names.add(arm["name"])
        require(
            arm["track"] in {"primary", "upper-bound", "candidate-diagnostic"}, "Comparison track"
        )
        if arm["track"] == "primary":
            require(
                arm["publicView"] == "raw" and arm["stateAccess"] == "own",
                "Primary comparisons must construct their own states from raw evidence",
            )
            require(
                arm["responseModel"] in {"train-estimated", "none"},
                "Exact environment model is an upper-bound condition",
            )
            require(
                all(arm[k] == "shared" for k in ("tools", "validator", "optimizer")),
                "Primary methods need shared tool, validation and optimizer access",
            )
            require(
                arm["candidateSource"] in {"raw-proposed", "shared-unlabeled"},
                "Annotated expression menu in primary comparison",
            )
        require(isinstance(arm["implemented"], bool), "Adapter readiness must be explicit")
    primary = [a for a in design["comparisons"] if a["track"] == "primary"]
    require(bool(primary), "Missing primary comparison")
    return {
        "valid": True,
        "primaryArms": len(primary),
        "allPrimaryAdaptersReady": all(a["implemented"] for a in primary),
    }


def check(rows, annotations, design):
    splits = partition(rows)
    contract = comparison_contract(design)
    require(len(annotations) == len(rows), "Missing annotations")
    by_id = {r["id"]: r for r in rows}
    require(
        len({a["id"] for a in annotations}) == len(annotations)
        and set(by_id) == {a["id"] for a in annotations},
        "Annotation IDs differ",
    )
    numbers, responses, statuses, tags = [], 0, Counter(), Counter()
    for annotation in annotations:
        row = by_id[annotation["id"]]
        require(annotation["scope"] in {"core", "extension", "out-of-scope"}, "Unknown model scope")
        statuses[annotation["scope"]] += 1
        tags.update(annotation["tags"])
        documents = {d["id"]: d for d in row["input"]["docs"]}
        for event in annotation["environment"]:
            require(
                set(event.get("resolves", [])) <= {"c", "p", "v", "b", "F"}, "Unknown resolved slot"
            )
            require(
                event["response"]["outcome"] == "complete" or not event.get("resolves"),
                "Partial response cannot resolve a parameter",
            )
            require(
                set(event.get("adjudicatedValues", {})) <= set(event.get("resolves", [])),
                "Response adjudication needs resolved fields",
            )
            documents.update({d["id"]: d for d in event["response"]["docs"]})
        for field in annotation["expected"]["grounding"].values():
            require(bool(field["evidence"]), "Grounded field needs evidence")
            for source in field["evidence"]:
                require(source["docId"] in documents, "Unknown evidence source")
                lo, hi = source["span"]
                require(
                    0 <= lo < hi <= len(documents[source["docId"]]["text"]),
                    "Evidence span out of bounds",
                )
        # These checks are private numerical fixtures, not constructor predictions.
        if annotation.get("numeric"):
            require(
                annotation["scope"] == "core",
                "Unsupported constraints cannot use the scalar salvage solver",
            )
            n = annotation["numeric"]
            fit = optimal(n["theta"], n["bounds"], integer=n["integer"])
            require(
                abs(fit["q"] - n["expectedQ"]) < 1e-8, f"Numerical fixture differs: {row['id']}"
            )
            numbers.append(
                {"id": row["id"], "q": fit["q"], "cost": fit["cost"], "stage": n["stage"]}
            )
        for event in annotation["environment"]:
            observe(public_input(row), event["tool"], event["response"])
            responses += 1
    reviewed = sum(a["reviewed"] for a in annotations)
    surface = [atoms(d) for r in rows for d in r["input"]["docs"]]
    return {
        "schema": SCHEMA,
        "split": splits,
        "inputHash": digest(rows),
        "annotationHash": digest(annotations),
        "designHash": digest(design),
        "humanReviewed": reviewed,
        "organizationalFamilies": len(
            {r["family"] for r in rows if r["origin"] == "organizational"}
        ),
        "caseScopes": dict(statuses),
        "caseTags": dict(tags),
        "numericCandidateParser": {
            "documents": len(surface),
            "documentsWithAtoms": sum(bool(a) for a in surface),
            "scope": "Syntax coverage only; not trained constructor accuracy.",
        },
        "checkedResponses": responses,
        "numericFixtures": numbers,
        "comparisonContract": contract,
        "dataReviewComplete": bool(rows) and reviewed == len(rows),
        "primaryEvaluationReady": contract["allPrimaryAdaptersReady"]
        and reviewed == len(rows)
        and bool(splits["counts"].get("test"))
        and bool(splits["counts"].get("train")),
        "scope": "Case and protocol smoke validation; no real-work efficacy measurement.",
    }


def controlled(config):
    episodes = generate(config)
    sources = list({e["family"]: e for e in episodes}.values())
    train = {digest(e["gold"]["theta"]) for e in episodes if e["split"] == "train"}
    test = [e for e in episodes if e["split"] == "test"]
    docs = [d for e in episodes for d in e["input"]["docs"]]
    templates = {
        split: {
            re.sub(r"-?\d+(?:\.\d+)?", "#", d["title"] + "\n" + d["text"])
            for e in episodes
            if e["split"] == split
            for d in e["input"]["docs"]
        }
        for split in ("train", "test")
    }
    qs = [optimal(e["gold"]["theta"], e["input"]["task"]["bounds"])["q"] for e in episodes]
    return {
        "scope": "Controlled generator audit; private labels measure dataset diversity, never construct provider inputs.",
        "config": config,
        "inputHash": digest([e["input"] for e in episodes]),
        "labelHash": digest([e["gold"] for e in episodes]),
        "split": audit(episodes),
        "testFamilies": len({e["family"] for e in test}),
        "uniqueTheta": len({digest(e["gold"]["theta"]) for e in episodes}),
        "testThetaSeenTrain": sum(digest(e["gold"]["theta"]) in train for e in test),
        "testEpisodes": len(test),
        "uniquePurchaseSaleRatios": sorted(
            {e["gold"]["theta"]["c"] / e["gold"]["theta"]["p"] for e in episodes}
        ),
        "demandSupportSizes": sorted({len(e["gold"]["theta"]["F"]) for e in episodes}),
        "optimalOrderAtBounds": sum(
            q in e["input"]["task"]["bounds"] for q, e in zip(qs, episodes, strict=True)
        ),
        "scenarioNamesInPublicSku": sum(
            e["scenario"] in e["input"]["task"]["sku"] for e in episodes
        ),
        "annotatedDocuments": sum(
            all(k in d for k in ("sku", "period", "version", "role", "complete")) for d in docs
        ),
        "documents": len(docs),
        "fullyReliableResponses": sum(
            all(v == 1 for v in e["input"]["task"]["rho"].values())
            and all(v == 0 for v in e["input"]["task"]["partial"].values())
            for e in episodes
        ),
        "scenarioCounts": dict(Counter(e["scenario"] for e in episodes)),
        "referenceInitialActionsFamilies": dict(
            Counter(planner(e["input"])["action"] for e in sources)
        ),
        "normalizedDocumentTemplates": {
            "train": len(templates["train"]),
            "test": len(templates["test"]),
            "shared": len(templates["train"] & templates["test"]),
            "normalization": "Numbers replaced; original formatting retained; metadata and family IDs excluded.",
        },
        "humanReviewed": sum(e["gold"]["reviewed"] for e in sources),
        "informationRegimes": {
            "reference": "Semantic-rule state scoring plus known answer sets, response probabilities and generated future documents.",
            "planner": "Own constructed state scoring, but still known answer sets and response dynamics. Control diagnostic only.",
            "learned": "Own-state rollout targets; still controlled known response sets and generator-specific validation.",
            "typed": "Chooses among parser-generated expressions, including computed values; shared reference conflict guard.",
            "external-business": "Receives the full reference-rule state; structured action selection only.",
            "oracle": "Hidden theta and no request costs; evaluation bound only.",
        },
    }
