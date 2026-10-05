from pathlib import Path

import numpy as np

from .construction import candidates, demand, finish, reference
from .corpus import KINDS, SLOTS, STATUSES
from .io import digest, read, require, write
from .providers import request_many


def questions(input):
    result = {}

    def add(id, instructions, criteria):
        result[id] = {"type": "choice", "instructions": instructions, "criteria": criteria}

    for slot, description in SLOTS.items():
        for i, doc in enumerate(input["docs"]):
            add(
                f"e_{slot}_{i}",
                f"Does document {doc['id']} provide applicable evidence for {description}? Check SKU, period and chosen policy.",
                {
                    "yes": "Applicable supporting evidence",
                    "no": "Not applicable supporting evidence",
                },
            )
        cs = candidates(input, slot)
        criteria = {
            f"g{i}": f"Source {c['doc']['id']}; operation {c['op']}; operands "
            f"{[(a['label'], a['value'], a['span']) for a in c['args']]}; value {c['value']}; unit {c['unit']}"
            for i, c in enumerate(cs)
        }
        add(
            f"g_{slot}",
            f"Choose the evidence expression for {description}. Select none when missing or conflicting.",
            {**criteria, "none": "No supported expression"},
        )
    for slot in (*SLOTS, "F", "C"):
        description = SLOTS.get(
            slot,
            "Empirical demand distribution"
            if slot == "F"
            else "Declared single-period model constraints",
        )
        add(f"t_{slot}", f"Classify {description}.", {k: k for k in KINDS})
        add(
            f"s_{slot}",
            f"Classify the current evidence state of {description}.",
            {s: s for s in STATUSES},
        )
    return result


def probabilities(answer, options, temperature=1):
    raw = answer["probabilities"]
    if raw is None:
        return None
    logits = np.log(np.array([raw[k] for k in options]).clip(1e-12)) / temperature
    exp = np.exp(logits - logits.max())
    return exp / exp.sum()


def decode(input, answers, threshold=0.5, temperatures=None):
    temperatures = temperatures or {}
    record = {k: {} for k in ("values", "links", "state", "types", "expressions", "probabilities")}
    record["errors"] = []
    ref = reference(input)
    for slot in SLOTS:
        kind = answers[f"t_{slot}"]
        status = answers[f"s_{slot}"]
        record["types"][slot], record["state"][slot] = kind["choice"], status["choice"]
        record["probabilities"][slot] = {
            "type": kind["probabilities"],
            "state": status["probabilities"],
        }
        if ref["state"][slot] == "conflict":
            record["state"][slot] = "conflict"
            record["errors"].append("conflict-" + slot)
            continue
        cs = candidates(input, slot)
        scores = {}
        for i, doc in enumerate(input["docs"]):
            answer = answers[f"e_{slot}_{i}"]
            p = probabilities(answer, ("no", "yes"), temperatures.get("evidence", 1))
            # An agent without probabilities supplies only a hard decision; no probability is fabricated.
            scores[doc["id"]] = float(p[1]) if p is not None else int(answer["choice"] == "yes")
        eligible = [c for c in cs if scores[c["doc"]["id"]] >= threshold]
        expression = answers[f"g_{slot}"]
        if expression["choice"] == "none" or not eligible:
            record["state"][slot] = "unconfirmed"
            continue
        raw = expression["probabilities"]
        if raw is None:
            chosen = cs[int(expression["choice"][1:])]
            if chosen not in eligible:
                record["state"][slot] = "unconfirmed"
                continue
        else:
            source = max(eligible, key=lambda c: (scores[c["doc"]["id"]], c["doc"]["id"]))["doc"][
                "id"
            ]
            selected = [(i, c) for i, c in enumerate(cs) if c["doc"]["id"] == source]
            chosen = max(selected, key=lambda pair: raw[f"g{pair[0]}"])[1]
        if slot == "b" and record["types"][slot] != "preference":
            record["state"][slot] = "unconfirmed"
            record["errors"].append("preference-type")
            continue
        record["values"][slot], record["links"][slot] = chosen["value"], chosen["doc"]["id"]
        record["expressions"][slot] = chosen["id"]
    return finish(input, record, demand(input, record))


class TypedBuilder:
    def __init__(self, provider, budget=0, directory=".cache/providers"):
        self.provider, self.budget, self.calls = provider, budget, 0
        self.threshold, self.temperatures = 0.5, {}
        self.identity = {
            "provider": provider.kind,
            "url": provider.url,
            "model": provider.model,
            "prompt": "typed-construction-v1",
            "sourceHash": digest(Path(__file__).read_bytes()),
        }
        self.directory = Path(directory) / digest(self.identity)
        self.snapshots = {}

    def raw(self, input):
        qs = questions(input)
        key = digest({"input": input, "questions": qs, "identity": self.identity})
        if key in self.snapshots:
            return self.snapshots[key]
        path = self.directory / (key + ".json")
        if path.exists():
            saved = read(path)
            require(
                saved["requestHash"] == key and digest(saved["answers"]) == saved["answerHash"],
                "Cached provider output checksum mismatch",
            )
        else:
            require(
                self.calls < self.budget,
                "Provider call budget exhausted; increase --max-calls explicitly or resume from cached outputs",
            )
            self.calls += 1
            answers, trace = request_many(self.provider, input, qs)
            saved = {
                "requestHash": key,
                "answers": answers,
                "answerHash": digest(answers),
                "trace": trace,
            }
            write(path, saved)
        self.snapshots[key] = saved
        return saved

    def __call__(self, input):
        return decode(input, self.raw(input)["answers"], self.threshold, self.temperatures)

    def calibrate(self, episodes):
        rows = []
        for episode in episodes:
            input = episode["input"]
            answers, ref = self.raw(input)["answers"], reference(input)
            for slot in SLOTS:
                for i, doc in enumerate(input["docs"]):
                    answer = answers[f"e_{slot}_{i}"]
                    if answer["probabilities"] is not None:
                        rows.append((answer, int(doc["id"] == ref["links"].get(slot))))
        if not rows:
            return {
                "source": "calibration",
                "rows": 0,
                "probabilityCalibration": "unavailable for hard agent output",
            }
        best = (float("inf"), 1.0)
        for t in np.geomspace(0.25, 8, 60):
            nll = np.mean(
                [-np.log(probabilities(a, ("no", "yes"), t)[y].clip(1e-12)) for a, y in rows]
            )
            if nll < best[0]:
                best = (float(nll), float(t))
        self.temperatures["evidence"] = best[1]
        bestf1 = -1.0
        for threshold in np.linspace(0.1, 0.9, 17):
            predicted = [probabilities(a, ("no", "yes"), best[1])[1] >= threshold for a, _ in rows]
            tp = sum(p and y == 1 for p, (_, y) in zip(predicted, rows, strict=True))
            f1 = 2 * tp / max(1, sum(predicted) + sum(y for _, y in rows))
            if f1 > bestf1:
                self.threshold, bestf1 = float(threshold), float(f1)
        return {
            "source": "calibration only",
            "rows": len(rows),
            "temperature": best[1],
            "nll": best[0],
            "threshold": self.threshold,
            "f1": bestf1,
        }
