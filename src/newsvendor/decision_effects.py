"""Exploratory paired effects; never used by model construction or action selection."""

import math
from collections import defaultdict

import torch

from .io import require


def pair_outcomes(rows, kind, metric):
    """Pair identical observed cases, conditions and outcome availability."""
    grouped = defaultdict(dict)
    for row in rows:
        if row["kind"] != kind:
            continue
        key = row["id"], row["condition"]
        method = row["method"]
        require(method in ("parent", "scoped"), "Unknown treatment arm")
        require(method not in grouped[key], "Duplicate outcome within treatment arm")
        grouped[key][method] = row
    paired = []
    for (case, condition), arms in sorted(grouped.items()):
        require(set(arms) == {"parent", "scoped"}, "Unmatched treatment arms")
        before, after = arms["parent"], arms["scoped"]
        for field in ("family", "period", "complete"):
            require(before[field] == after[field], "Paired provenance or availability changed")
        a, b = before["metrics"].get(metric), after["metrics"].get(metric)
        require((a is None) == (b is None), "Treatment-dependent outcome availability")
        if a is None:
            continue
        require(math.isfinite(a) and math.isfinite(b), "Non-finite outcome")
        paired.append(
            {
                "id": case,
                "condition": condition,
                "family": before["family"],
                "period": before["period"],
                "parent": float(a),
                "scoped": float(b),
                "difference": float(b - a),
            }
        )
    require(paired, "No paired outcomes")
    return paired


def paired_regression(pairs, conditions):
    """OLS on paired differences and pre-existing document-condition indicators.

    Case effects cancel before estimation. This saturated specification equals
    paired cell means; it does not identify an AI-versus-no-AI treatment effect.
    Source deletion is a sensitivity analysis, not a confidence interval.
    """
    require(len(conditions) == len(set(conditions)), "Duplicate condition")
    require(conditions and conditions[0] == "original", "Original must be the reference")
    cases, identities = defaultdict(set), {}
    for row in pairs:
        require(row["condition"] in conditions, "Unregistered document condition")
        require(row["condition"] not in cases[row["id"]], "Duplicate paired outcome")
        identity = row["family"], row["period"]
        require(
            identities.setdefault(row["id"], identity) == identity,
            "Case provenance changed across conditions",
        )
        cases[row["id"]].add(row["condition"])
    require(all(c == set(conditions) for c in cases.values()), "Unbalanced condition support")

    def fit(selected):
        x = torch.tensor(
            [[1.0, *[float(r["condition"] == c) for c in conditions[1:]]] for r in selected],
            dtype=torch.float64,
        )
        y = torch.tensor([r["difference"] for r in selected], dtype=torch.float64)
        require(int(torch.linalg.matrix_rank(x)) == len(conditions), "Rank-deficient design")
        beta = torch.linalg.lstsq(x, y, driver="gelsd").solution
        return beta.tolist(), float(torch.sum((y - x @ beta) ** 2))

    beta, sse = fit(pairs)
    families = sorted({r["family"] for r in pairs})
    require(len(families) >= 2, "Source-deletion analysis requires at least two families")
    deleted = {f: fit([r for r in pairs if r["family"] != f])[0] for f in families}
    cells = {}
    for index, condition in enumerate(conditions):
        selected = [r for r in pairs if r["condition"] == condition]
        groups = defaultdict(list)
        for row in selected:
            groups[row["family"]].append(row["difference"])
        effect = sum(r["difference"] for r in selected) / len(selected)
        fitted = beta[0] + (beta[index] if index else 0.0)
        require(math.isclose(effect, fitted, abs_tol=1e-9), "OLS and paired means disagree")
        leave_one_out = {f: b[0] + (b[index] if index else 0.0) for f, b in deleted.items()}
        cells[condition] = {
            "cases": len(selected),
            "sourceFamilies": len(groups),
            "periods": len({r["period"] for r in selected if r["period"] is not None}),
            "parentMean": sum(r["parent"] for r in selected) / len(selected),
            "scopedMean": sum(r["scoped"] for r in selected) / len(selected),
            "effect": effect,
            "equalSourceMeanEffect": sum(sum(v) / len(v) for v in groups.values()) / len(groups),
            "lower": sum(r["difference"] < -1e-9 for r in selected),
            "unchanged": sum(abs(r["difference"]) <= 1e-9 for r in selected),
            "higher": sum(r["difference"] > 1e-9 for r in selected),
            "sourceEffects": {f: sum(v) / len(v) for f, v in groups.items()},
            "leaveOneSourceOut": leave_one_out,
            "deletionRange": [min(leave_one_out.values()), max(leave_one_out.values())],
        }
    return {
        "formula": "scoped_minus_parent ~ 1 + document_condition (reference: original)",
        "estimand": "Existing decoder modification, not AI adoption or human behavior",
        "coefficients": dict(zip(["intercept", *conditions[1:]], beta, strict=True)),
        "pairedObservations": len(pairs),
        "sourceFamilies": len(families),
        "residualSumSquares": sse,
        "cells": cells,
        "inference": {
            "pValues": None,
            "confidenceIntervals": None,
            "reason": "Already-used Dev with few independent sources; descriptive sensitivity only",
        },
    }


def critical_ratio(c, p, v, b):
    over, under = c - v, p - c + b
    require(all(math.isfinite(x) for x in (c, p, v, b)), "Non-finite economic parameter")
    require(over > 0 and under > 0, "Interior Newsvendor solution required")
    return under / (over + under)


def uniform_risk(q, low, high, c, p, v, b):
    """Analytic continuous-demand check, distinct from the implemented discrete F."""
    require(low <= q <= high and high > low, "Uniform interior domain required")
    return ((c - v) * (q - low) ** 2 + (p - c + b) * (high - q) ** 2) / (2 * (high - low))


def theory_checks():
    """Numerically verify stated analytic identities; no fitted AI or empirical effects."""
    results = []
    for c, p, v, b in ((6.0, 10.0, 1.0, 0.0), (6.0, 10.0, 1.0, 5.0), (4.0, 8.0, 2.0, 2.0)):
        for width in (20.0, 100.0):
            alpha = critical_ratio(c, p, v, b)
            q = width * alpha
            base = uniform_risk(q, 0, width, c, p, v, b)
            gap = min(q, width - q) / 5
            actual = uniform_risk(q + gap, 0, width, c, p, v, b) - base
            expected = (p - v + b) * gap**2 / (2 * width)
            require(math.isclose(actual, expected, abs_tol=1e-9), "Regret curvature mismatch")
            eps = 1e-5
            derivative = (critical_ratio(c, p, v, b + eps) - critical_ratio(c, p, v, b - eps)) / (
                2 * eps
            )
            target = (c - v) / (p - v + b) ** 2
            require(
                math.isclose(derivative, target, rel_tol=1e-8), "Preference sensitivity mismatch"
            )
            results.append(
                {
                    "c": c,
                    "p": p,
                    "v": v,
                    "b": b,
                    "width": width,
                    "qStar": q,
                    "quantityError": gap,
                    "actualRegret": actual,
                    "curvatureRegret": expected,
                    "numericRatioDerivativeB": derivative,
                    "analyticRatioDerivativeB": target,
                }
            )
    return {"scope": "Analytic identity checks only; not measured AI effects", "cases": results}
