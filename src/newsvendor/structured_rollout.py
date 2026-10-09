"""Sequential own-state rollouts. Hidden truth is read only by the environment/scorer."""

import copy
import time

import torch

from . import structured_forecast
from .construction import candidates, demand, finish
from .corpus import SLOTS, outcome
from .evaluation import score
from .io import digest, require
from .optimizer import regret
from .policy import actions, response
from .structured_inputs import prepare, research_input
from .structured_model import assemble, extract

FIELDS = [
    {"id": name, "name": name, "description": description, "choices": [], "required": True, "stateRequired": True}
    for name, description in {**SLOTS, "F": "Observed or estimated demand distribution"}.items()
]
TEXT = {
    "hold": "Stop and defer the decision",
    "handoff": "Finish with the computed order",
    "c": "Request the purchase cost and its source",
    "p": "Request the selling price and its source",
    "v": "Request refund and return contract terms",
    "b": "Ask the manager to select a cost",
    "demand": "Request uncensored demand history or a forecast",
    "retrieve": "Read more source chunks to recover the constructed parameters",
}


class ResearchRouter:
    def __init__(self, model, tokenizer, config, no_value=False):
        self.model, self.tokenizer, self.config = model, tokenizer, config
        self.no_value = no_value
        self.cache = None

    def cache_states(self, enabled=True):
        self.cache = {} if enabled else None

    def allowed(self, value, state):
        allowed = (
            structured_forecast.actions(value, state)
            if "forecast" in value["task"]
            else actions(value, state)
        )
        lookups = sum(h["action"] == "retrieve" for h in value["history"])
        if (
            value["remaining"] > 0
            and len(value["history"]) < value["task"]["deadline"]
            and (not state["valid"] or any(s not in state["values"] for s in SLOTS))
            and lookups < self.config.get("retrievals", 2)
            and state.get("retrieval", {}).get("unreadChunks", 0) > 0
        ):
            allowed.append("retrieve")
        return allowed

    @torch.inference_mode()
    def construct(self, value):
        self.model.eval()
        observed = research_input(value)
        key = digest([observed, value["observations"], value.get("historySource"), value.get("memory")])
        if self.cache is not None and key in self.cache:
            return copy.deepcopy(self.cache[key])
        view = prepare(
            observed,
            self.tokenizer,
            self.config,
            fields=FIELDS,
            actions=[{"id": "hold", "text": TEXT["hold"]}],
            round=sum(h["action"] == "retrieve" for h in value["history"]),
        )
        predicted = extract(view, self.model(view))
        record = {k: {} for k in ("values", "links", "state", "types", "expressions")}
        record["errors"] = []
        record["fields"] = predicted
        for field in predicted:
            slot = field["name"]
            record["state"][slot], record["types"][slot] = field["state"], field["type"]
            if slot == "F":
                continue
            if field["state"] == "conflict":
                record["errors"].append("conflict-" + slot)
            if field["value"] is None or field["state"] != "verified":
                continue
            if slot == "b" and field["type"] != "preference":
                record["errors"].append("unselected-preference")
                continue
            # Validate selected operations, operand order, units and public applicability.
            expression = field.get("expression")
            if expression:
                match = next(
                    (
                        e
                        for e in candidates(value, slot)
                        if {"copy": "copy", "divide": "divide", "subtract": "subtract"}.get(e["op"])
                        == expression["op"]
                        and len(e["args"]) == len(expression["operands"])
                        and structured_forecast.compatible(e, slot)
                        and all(
                            a["span"] == [loc["start"], loc["end"]] and e["doc"]["id"] == loc["id"]
                            for a, loc in zip(e["args"], expression["operands"], strict=True)
                        )
                    ),
                    None,
                )
                if match:
                    record["values"][slot] = match["value"]
                    record["links"][slot] = match["doc"]["id"]
                    record["expressions"][slot] = match["id"]
                else:
                    record["errors"].append("invalid-expression-" + slot)
        if "forecast" in value["task"]:
            forecast, error = None, None
            try:
                forecast = structured_forecast.predict(
                    self.model, value, self.config.get("minDemandHistory", 28)
                )
            except (KeyError, TypeError, ValueError, OverflowError) as exception:
                error = str(exception)
            result = structured_forecast.finish(value, record, forecast, error)
        else:
            # Preserve the original generated benchmark and its loss definition.
            Fs = demand(value, record)
            result = finish(value, record, Fs)
        result["retrieval"] = {
            "indexedChunks": view["indexedChunks"], "selectedChunks": view["selectedChunks"],
            "unreadChunks": max(0, view["indexedChunks"] - view["selectedChunks"]),
        }
        if self.cache is not None:
            self.cache[key] = copy.deepcopy(result)
        return result

    def view(self, value, state):
        allowed = self.allowed(value, state)
        summary = {
            k: state[k] for k in ("values", "state", "types", "q", "gamma", "valid", "errors")
        }
        summary["retrievalCost"] = self.config.get("retrievalCost", 1.0)
        if state.get("forecast") is not None:
            summary["demand"] = {
                k: state["forecast"][k] for k in ("distribution", "period")
            }
        if "missing" in state:
            summary["missing"] = state["missing"]
        view = prepare(
            research_input(value),
            self.tokenizer,
            self.config,
            fields=FIELDS,
            actions=[{"id": a, "text": TEXT[a]} for a in allowed],
            state=summary,
            round=sum(h["action"] == "retrieve" for h in value["history"]),
        )
        view["allowReread"] = False
        if self.config.get("exactActionCosts"):
            from .structured_value import known_costs

            task = value["task"]
            view["valueCosts"] = known_costs(
                allowed, hold=task["hold"], costs=task["costs"],
                remaining=value["remaining"], history_length=len(value["history"]),
                deadline=task["deadline"], retrieval_cost=self.config.get("retrievalCost", 1.0),
            )
        return view

    @torch.inference_mode()
    def decision(self, value, state=None):
        self.model.eval()
        state = self.construct(value) if state is None else state
        view = self.view(value, state)
        result = assemble(view, self.model(view), no_value=self.no_value)
        for estimates in result.get("actionValues", []):
            estimates["residualLoss"] *= value["task"]["hold"]
            estimates["requestCost"] *= value["task"]["hold"]
        result["constructedState"] = state
        result["fields"] = state["fields"]
        if result["action"] in (*SLOTS, "demand"):
            slot = "F" if result["action"] == "demand" else result["action"]
            description = "uncensored demand history or a forecast" if slot == "F" else SLOTS[slot]
            result["question"] = {
                "field": slot,
                "reason": next((f.get("reason", f["state"]) for f in state["fields"] if f["name"] == slot),
                               state["state"].get(slot, "unconfirmed")),
                "choices": [],
                "text": "Please provide or confirm " + description + ".",
            }
        self.last_decision = result
        return result

    def choose(self, value, state):
        return self.decision(value, state)["action"]


def allowed_actions(router, value, state):
    return router.allowed(value, state) if hasattr(router, "allowed") else actions(value, state)


def behavior(value, state, router=None):
    """Observable request/checklist behavior, independent of measured terminal losses."""
    allowed = allowed_actions(router, value, state)
    if "retrieve" in allowed and any(s not in state["values"] for s in ("c", "p")):
        return "retrieve"
    return next(
        (a for a in allowed if a not in ("hold", "handoff")),
        "handoff" if "handoff" in allowed else "hold",
    )


def rollout(episode, router, *, value=None, first=None, explore=False, noise=0):
    started = time.perf_counter()
    current = copy.deepcopy(episode["input"] if value is None else value)
    linked = "forecast" in current["task"]
    if linked:
        from . import structured_retail

        require(episode.get("benchmark") == structured_retail.VERSION, "Forecast rollout requires its own outcome evaluator")
    events, cost = [], 0.0
    for step in range(current["remaining"] + 2):
        state = router.construct(current)
        allowed = allowed_actions(router, current, state)
        action = (
            first
            if step == 0 and first is not None
            else (behavior(current, state, router) if explore else router.choose(current, state))
        )
        require(action in allowed, "Rollout selected a forbidden action")
        event = {
            "input": copy.deepcopy(current),
            "state": state,
            "action": action,
            "stateHash": digest(current),
        }
        if (
            not explore
            and not (step == 0 and first is not None)
            and hasattr(router, "last_decision")
        ):
            event["decision"] = router.last_decision
        events.append(event)
        if action in ("hold", "handoff"):
            break
        if linked:
            current = copy.deepcopy(current)
            current["memory"] = structured_forecast.remember(current, state)
        if action == "retrieve":
            cost += router.config.get("retrievalCost", 1.0)
            current = copy.deepcopy(current)
            current["remaining"] -= 1
            observed = {"sourceHashes": {d["id"]: digest(d) for d in current["docs"]}}
            current["history"].append({"action": action, "answer": observed})
            event["observedResponse"] = observed
            continue
        cost += current["task"]["costs"][action]
        observed = (
            structured_retail.response(episode, current, action, noise)
            if linked else response(episode, current, action, noise)
        )
        event["observedResponse"] = observed
        current = outcome(current, action, observed)
    if linked:
        evaluation = structured_retail.evaluate(episode, current, state, action)
        return {
            "id": episode["id"], "family": episode["family"], "split": episode["split"],
            "result": action, "q": state["q"] if action == "handoff" else None,
            "requestCost": cost,
            "total": evaluation["terminalLoss"] + cost if evaluation["terminalLoss"] is not None else None,
            "interactions": sum(e["action"] not in ("hold", "handoff") for e in events),
            **evaluation, **structured_retail.interactions(events),
            "elapsedMs": (time.perf_counter() - started) * 1000, "events": events,
        }
    evaluation = score(episode, current, state, action)
    terminal = (
        regret(state["q"], episode["gold"]["theta"], current["task"]["bounds"])
        if action == "handoff"
        else current["task"]["hold"]
    )
    economic = terminal
    if evaluation["falseHandoff"]:
        terminal = max(terminal, current["task"]["hold"])
    return {
        "id": episode["id"],
        "family": episode["family"],
        "split": episode["split"],
        "result": action,
        "q": state["q"] if action == "handoff" else None,
        "terminalLoss": terminal,
        "economicLoss": economic,
        "invalidHandoffPenalty": terminal - economic,
        "lossDefinition": "Economic regret or hold loss, floored at hold loss for invalid handoff, plus request costs",
        "requestCost": cost,
        "total": terminal + cost,
        "interactions": sum(e["action"] not in ("hold", "handoff") for e in events),
        **evaluation,
        "elapsedMs": (time.perf_counter() - started) * 1000,
        "events": events,
    }


def collect(episodes, router, split="train", noise=0, progress=None, policy="behavior", with_values=True):
    require(
        split in ("train", "dev") and all(e["split"] == split for e in episodes),
        "Rollout supervision may use only its requested Train/Dev partition",
    )
    require(
        not with_values or all("forecast" not in e["input"]["task"] or all(e["target"]["complete"]) for e in episodes),
        "Economic rollout targets require complete observed demand, not censored lower bounds",
    )
    rows, measurements = [], []
    if hasattr(router, "cache_states"):
        router.cache_states()
    for number, episode in enumerate(episodes):
        # Collect actual encountered own-prediction states, including failed construction.
        require(policy in ("behavior", "mixed", "own"), "Unknown collection policy")
        explore = policy == "behavior" or (policy == "mixed" and number % 2 == 0)
        observed = rollout(episode, router, explore=explore, noise=noise)
        if not with_values:
            measurements.append(observed)
        for event in observed["events"]:
            value, state = event["input"], event["state"]
            permitted = allowed_actions(router, value, state)
            outcomes = [
                rollout(episode, router, value=value, first=action, noise=noise)
                for action in permitted
            ] if with_values else []
            scale = value["task"]["hold"]
            rows.append(
                {
                    "id": episode["id"],
                    "family": episode["family"],
                    "split": split,
                    "input": value,
                    "state": state,
                    "actions": permitted,
                    "recovery": permitted.index(behavior(value, state, router)),
                    "scale": scale,
                }
            )
            if with_values:
                rows[-1].update(values=[[r["terminalLoss"] / scale, r["requestCost"] / scale] for r in outcomes],
                                valueIndices=list(range(len(permitted))))
            for action, measured in zip(permitted if with_values else [], outcomes, strict=True):
                measurements.append(
                    {
                        "id": episode["id"],
                        "family": episode["family"],
                        "split": split,
                        "stateHash": event["stateHash"],
                        "constructorState": state,
                        "forcedAction": action,
                        "scale": scale,
                        **measured,
                    }
                )
        if progress and (number + 1) % 10 == 0:
            progress.update(
                "policy_collect",
                split=split,
                processed=number + 1,
                total=len(episodes),
                ownStateTargets=len(rows),
            )
    if hasattr(router, "cache_states"):
        router.cache_states(False)
    return rows, measurements
