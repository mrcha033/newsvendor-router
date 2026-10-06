"""Pinned coherent retail episodes, isolated tools, source splits and independent scoring."""

import inspect
import re
import urllib.request
from collections import Counter, defaultdict
from functools import cache
from pathlib import Path

import numpy as np
from pydantic import create_model

from .io import digest, jsonl, lines, read, require, write
from .retail.models import RetailDB
from .retail.tools import RetailTools

WRITES = {
    "cancel_pending_order",
    "exchange_delivered_order_items",
    "modify_pending_order_address",
    "modify_pending_order_items",
    "modify_pending_order_payment",
    "modify_user_address",
    "return_delivered_order_items",
}
AUTH = {"find_user_id_by_email", "find_user_id_by_name_zip"}


def prepare(config):
    raw = Path(config["raw"])
    for path, expected in config["files"].items():
        destination = raw / path
        if not destination.exists():
            url = f"https://raw.githubusercontent.com/{config['repository']}/{config['revision']}/{path}"
            data = urllib.request.urlopen(url, timeout=60).read()
            require(digest(data) == expected, "Order source hash mismatch: " + path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
        require(digest(destination.read_bytes()) == expected, "Order capture changed: " + path)
    source = raw / "data/tau2/domains/retail"
    tasks, db, official = (
        read(source / "tasks.json"),
        read(source / "db.json"),
        read(source / "split_tasks.json"),
    )
    families = {}
    for task in tasks:
        # Private reference identifiers are provenance for grouping, never model features.
        users = set()
        known = task["user_scenario"]["instructions"]["known_info"].casefold()
        for user_id, profile in db["users"].items():
            name = profile["name"]["first_name"] + " " + profile["name"]["last_name"]
            if user_id.casefold() in known or (
                name.casefold() in known
                and re.search(r"\b" + re.escape(profile["address"]["zip"]) + r"\b", known)
            ):
                users.add(user_id)
        for action in task["evaluation_criteria"].get("actions") or []:
            args = action["arguments"]
            if args.get("user_id") in db["users"]:
                users.add(args["user_id"])
            if args.get("order_id") in db["orders"]:
                users.add(db["orders"][args["order_id"]]["user_id"])
            if action["name"] == "find_user_id_by_email":
                users.update(
                    k
                    for k, v in db["users"].items()
                    if v["email"].casefold() == args["email"].casefold()
                )
            if action["name"] == "find_user_id_by_name_zip":
                users.update(
                    k
                    for k, v in db["users"].items()
                    if v["name"]["first_name"].casefold() == args["first_name"].casefold()
                    and v["name"]["last_name"].casefold() == args["last_name"].casefold()
                    and v["address"]["zip"] == args["zip"]
                )
        require(len(users) <= 1, "Multi-customer task needs connected-component grouping")
        families[task["id"]] = (
            "customer-" + digest(sorted(users) if users else task["user_scenario"])[:16]
        )
    groups = sorted(set(families.values()))
    shuffled = np.random.default_rng(config["seed"]).permutation(groups).tolist()
    assigned = {
        g: "train"
        if i < int(len(groups) * 0.6)
        else "dev"
        if i < int(len(groups) * 0.8)
        else "test"
        for i, g in enumerate(shuffled)
    }
    rows, labels = [], []
    for task in tasks:
        require(task.get("initial_state") is None, "Unsupported upstream initial state")
        id = "orders-" + task["id"]
        rows.append(
            {
                "id": id,
                "family": families[task["id"]],
                "split": assigned[families[task["id"]]],
                "taskId": task["id"],
                "input": {"policy": (source / "policy.md").read_text(), "tools": catalog()},
            }
        )
        labels.append(
            {
                "id": id,
                "userScenario": task["user_scenario"],
                "criteria": task["evaluation_criteria"],
            }
        )
    directory = Path(config["dataset"])
    jsonl(directory / "inputs.jsonl", rows)
    jsonl(directory / "labels.jsonl", labels)
    write(directory / "db.json", db)
    train = {families[i] for i in official["train"]}
    test = {families[i] for i in official["test"]}
    manifest = {
        "config": config,
        "tasks": len(rows),
        "families": len(groups),
        "splits": dict(Counter(r["split"] for r in rows)),
        "officialSplits": {k: len(v) for k, v in official.items()},
        "officialTrainTestCustomerOverlap": len(train & test),
        "inputHash": digest(rows),
        "labelHash": digest(labels),
        "dbHash": digest(db),
        "scope": config["scope"],
        "privateGoalProvider": "User simulator and scorer only",
        "nlAssertions": sum(bool(t["criteria"].get("nl_assertions")) for t in labels),
        "publicCatalogSharedAcrossSplits": True,
        "adapter": "Original upstream mutation logic, BaseModel DB, local toolkit wrapper; no upstream framework execution",
        "upstreamFailures": [],
        "referenceWarnings": [],
        "ambiguousReferenceTasks": [],
    }
    for task in labels:
        try:
            environment, warnings = reference(db, task["criteria"], details=True)
            if warnings:
                manifest["referenceWarnings"].append({"id": task["id"], "errors": warnings})
            if any(w["tool"] in WRITES for w in warnings) and not any(
                c["tool"] in WRITES for c in environment.calls
            ):
                manifest["ambiguousReferenceTasks"].append(task["id"])
        except (ValueError, TypeError, KeyError) as error:
            manifest["upstreamFailures"].append({"id": task["id"], "error": str(error)})
    write(directory / "manifest.json", manifest)
    write("cases/orders/manifest.json", manifest)
    check(config)
    return manifest


def check(config):
    directory = Path(config["dataset"])
    rows, labels, manifest = (
        lines(directory / "inputs.jsonl"),
        lines(directory / "labels.jsonl"),
        read(directory / "manifest.json"),
    )
    require(
        digest(rows) == manifest["inputHash"] and digest(labels) == manifest["labelHash"],
        "Order dataset changed",
    )
    require(digest(read(directory / "db.json")) == manifest["dbHash"], "Order DB changed")
    groups = defaultdict(set)
    for row in rows:
        require(set(row["input"]) == {"policy", "tools"}, "Private order goal in provider input")
        groups[row["family"]].add(row["split"])
    require(all(len(v) == 1 for v in groups.values()), "Order customer split leakage")
    require({r["id"] for r in rows} == {r["id"] for r in labels}, "Order label alignment")
    require(
        not manifest["upstreamFailures"],
        "Upstream reference replay failed; cases retained, benchmark blocked",
    )
    return {
        "tasks": len(rows),
        "families": len(groups),
        "customerOverlap": 0,
        "upstreamReplayFailures": 0,
    }


@cache
def argument_model(name):
    method = getattr(RetailTools, name)
    signature = inspect.signature(method)
    return create_model(
        name + "Arguments",
        **{
            k: (p.annotation, ... if p.default is inspect.Parameter.empty else p.default)
            for k, p in signature.parameters.items()
            if k != "self"
        },
    )


@cache
def catalog():
    return [
        {
            "id": name,
            "description": inspect.getdoc(method),
            "parameters": argument_model(name).model_json_schema(),
            "write": name in WRITES,
        }
        for name, method in inspect.getmembers(RetailTools, inspect.isfunction)
        if not name.startswith("_")
    ]


def plain(value):
    return value.model_dump(mode="json") if hasattr(value, "model_dump") else value


class Environment:
    def __init__(self, db):
        self.tools = RetailTools(RetailDB.model_validate(db))
        self.user = None
        self.confirmed = None
        self.violations = []
        self.calls = []

    def call(self, name, arguments, reference_mode=False):
        require(name in {t["id"] for t in catalog()}, "Unknown order tool")
        args = argument_model(name).model_validate(arguments).model_dump()
        if not reference_mode:
            if name not in AUTH and name != "transfer_to_human_agents":
                if self.user is None:
                    self.violations.append("unauthenticated-tool")
                    raise ValueError("Authenticate via email or name and zip first")
                if "order_id" in args:
                    order = self.tools._get_order(args["order_id"])
                    if order.user_id != self.user:
                        self.violations.append("wrong-customer-order")
                        raise ValueError("Order belongs to another customer")
                if "user_id" in args and args["user_id"] != self.user:
                    self.violations.append("wrong-customer-profile")
                    raise ValueError("Profile belongs to another customer")
            if name in WRITES and self.confirmed != digest({"tool": name, "arguments": args}):
                self.violations.append("unconfirmed-write")
                raise ValueError("Propose exact action details and obtain user confirmation first")
        output = getattr(self.tools, name)(**args)
        if name in AUTH and not reference_mode:
            if self.user is not None and output != self.user:
                self.violations.append("customer-switch")
                raise ValueError("Only one customer per conversation")
            self.user = output
        if name in WRITES:
            self.confirmed = None
        self.calls.append({"tool": name, "arguments": args, "output": plain(output)})
        return plain(output)

    def state(self):
        return self.tools.db.model_dump(mode="json")


def reference(db, criteria, details=False):
    environment = Environment(db)
    warnings = []
    for action in criteria.get("actions") or []:
        # Upstream evaluator_env.py also logs tool errors and continues. Failed lookup
        # attempts followed by corrected identifiers are intentional recovery cases.
        try:
            environment.call(action["name"], action["arguments"], reference_mode=True)
        except ValueError as error:
            warnings.append({"tool": action["name"], "error": str(error)})
    return (environment, warnings) if details else environment.state()


def ledger(db):
    values = defaultdict(float)
    for order_id, order in db["orders"].items():
        for payment in order["payment_history"]:
            values[(order_id, payment["payment_method_id"], payment["transaction_type"])] += (
                payment["amount"]
            )
    return values


def evaluate(db, criteria, environment, assertions=None, goal_pass=None):
    reference_env, warnings = reference(db, criteria, details=True)
    expected, actual = reference_env.state(), environment.state()
    ambiguous = any(w["tool"] in WRITES for w in warnings) and not any(
        c["tool"] in WRITES for c in reference_env.calls
    )
    a, b = ledger(expected), ledger(actual)
    ledger_l1 = sum(abs(a[k] - b[k]) for k in a.keys() | b.keys())
    changed_orders = sum(
        digest(expected["orders"][k]) != digest(actual["orders"][k]) for k in expected["orders"]
    )
    db_match = digest(expected) == digest(actual)
    needs_nl = bool(criteria.get("nl_assertions") or criteria.get("communicate_info"))
    judged = (
        assertions is not None
        and len(assertions)
        == len(criteria.get("nl_assertions") or []) + len(criteria.get("communicate_info") or [])
        and all(type(a) is bool for a in assertions)
    )
    nl_pass = all(assertions) if judged else None if needs_nl else True
    # A read-only task leaves the database unchanged even if the agent does nothing.
    # Require a separate goal check instead of treating unchanged state as task success.
    needs_goal = not any(c["tool"] in WRITES for c in reference_env.calls)
    generic_tools = {
        a["name"] for a in criteria.get("actions") or [] if a["name"] == "transfer_to_human_agents"
    }
    generic_pass = generic_tools.issubset({c["tool"] for c in environment.calls})
    goal_valid = type(goal_pass) is bool if needs_goal else True
    return {
        "dbMatch": db_match,
        "changedOrders": changed_orders,
        "paymentLedgerL1SimulatedUSD": ledger_l1,
        "policyViolations": len(environment.violations),
        "violationTypes": environment.violations,
        "assertionPass": nl_pass,
        "goalPass": goal_pass if needs_goal and goal_valid else None,
        "needsGoalJudgment": needs_goal,
        "requiredGenericToolPass": generic_pass,
        "referenceWarnings": warnings,
        "annotationValid": not ambiguous,
        "taskSuccess": db_match
        and not environment.violations
        and nl_pass
        and generic_pass
        and (goal_pass if needs_goal else True)
        if nl_pass is not None and not ambiguous and goal_valid
        else None,
        "scope": "Simulated task outcome and ledger deviation, not Newsvendor procurement loss",
    }
