"""Conditional total-demand distributions and censored observation likelihoods."""

import math
import re
from datetime import date, timedelta
from statistics import NormalDist

import numpy as np
import torch
from torch.nn import functional as fn

from .io import digest, require
from .optimizer import optimal, risk

FAMILIES = ("truncated_normal", "lognormal", "weibull")
HEADS = ("demand_family", *["demand_" + family for family in FAMILIES])
PARAMETERS = {
    "truncated_normal": ("location", "scale"),
    "lognormal": ("logMean", "logStd"),
    "weibull": ("shape", "scale"),
}
QUADRATURE = 64
COST_RATIOS = (1, 3, 9)


def horizon(value):
    match = re.search(r"next (\d+)[ -]", value["request"])
    require(match is not None and int(match[1]) > 0, "Missing demand forecast horizon")
    return int(match[1])


def features(rows, days):
    require(len(rows) >= 14 and days > 0, "Insufficient demand history")
    sales = np.asarray([r["sales"] for r in rows], dtype=np.float32)
    require(np.isfinite(sales).all() and (sales >= 0).all(), "Invalid sales history")
    daily_scale = max(float(sales.mean()), 0.1)
    normalized = sales / daily_scale
    windows = [rows[-n:] for n in (7, 14, 30, 60)]
    x = list(normalized[-14:])
    x += [float(normalized[-n:].mean()) for n in (7, 14, 30, 60)]
    x += [float(normalized[-n:].std()) for n in (7, 14, 30, 60)]
    x += [sum(r["stockoutHours"] > 0 for r in w) / len(w) for w in windows]
    x += [float(np.mean([r.get("discount", 1) for r in rows[-7:]])), math.log1p(daily_scale)]
    return np.asarray(x, dtype=np.float32), daily_scale * days


def parameters(family, raw):
    if family == "lognormal":
        first = raw[..., 1].clamp(-10, 10)
        second = (fn.softplus(raw[..., 2]) + 0.05).clamp(max=3)
    elif family == "truncated_normal":
        first = fn.softplus(raw[..., 1]).clamp(max=20)
        second = (fn.softplus(raw[..., 2]) + 0.05).clamp(max=3)
    else:
        require(family == "weibull", "Unknown demand family")
        first = (fn.softplus(raw[..., 1]) + 0.2).clamp(max=5)
        second = (fn.softplus(raw[..., 2]) + 0.05).clamp(max=20)
    return first, second


def observation_nll(family, zero_logit, first, second, observed, censored):
    """Density / zero mass for exact observations; survival for stockout lower bounds."""
    log_y = observed.clamp_min(1e-12).log()
    if family == "lognormal":
        z = (log_y - first) / second
        density = -log_y - second.log() - z.square() / 2 - math.log(2 * math.pi) / 2
        tail = torch.special.log_ndtr(-z)
    elif family == "truncated_normal":
        z = (observed - first) / second
        normalizer = torch.special.log_ndtr(first / second)
        density = -second.log() - z.square() / 2 - math.log(2 * math.pi) / 2 - normalizer
        tail = torch.special.log_ndtr(-z) - normalizer
    else:
        require(family == "weibull", "Unknown demand family")
        log_ratio = log_y - second.log()
        power = (first * log_ratio).clamp(max=80).exp()
        density = first.log() - second.log() + (first - 1) * log_ratio - power
        tail = -power
    exact = fn.logsigmoid(-zero_logit) + density
    exact = torch.where(observed == 0, fn.logsigmoid(zero_logit), exact)
    tail = fn.logsigmoid(-zero_logit) + tail
    tail = torch.where(observed == 0, torch.zeros_like(tail), tail)
    return -torch.where(censored, tail, exact)


def nll(family, raw, observed, censored):
    return observation_nll(family, raw[..., 0], *parameters(family, raw), observed, censored)


def samples(rows, labels, *, rolling=False, min_history=28):
    """Only the caller's source split is used; each feature cutoff precedes its outcome."""
    result = []
    for row in rows:
        if row["component"] != "retail":
            continue
        observations = row["input"]["observations"]
        days = horizon(row["input"])
        periods = []
        if rolling:
            for end in range(min_history, len(observations) - days + 1):
                future = observations[end : end + days]
                periods.append(
                    (
                        observations[:end],
                        [r["sales"] for r in future],
                        [r["stockoutHours"] == 0 for r in future],
                        [r["date"] for r in future],
                    )
                )
        target = labels[row["id"]]
        periods.append((observations, target["answer"], target["complete"], target["dates"]))
        for past, sales, complete, dates in periods:
            require(len(sales) == len(complete) == len(dates) == days, "Demand horizon mismatch")
            require(past[-1]["date"] < min(dates), "Future observation in demand features")
            x, scale = features(past, days)
            result.append(
                {
                    "x": x.tolist(),
                    "y": sum(sales) / scale,
                    "censored": not all(complete),
                    "id": row["id"],
                    "family": row["family"],
                    "cutoff": past[-1]["date"],
                    "dates": dates,
                    "sales": sales,
                    "complete": complete,
                    "scale": scale,
                }
            )
    return result


def evaluate(rows, labels, head, min_history=28):
    """Source-held-out rolling forecasts; records retain replayable family and parameters."""
    from collections import defaultdict

    periods = samples(rows, labels, rolling=True, min_history=min_history)
    by_id = {r["id"]: r["input"] for r in rows}
    records = []
    for sample in periods:
        value = dict(by_id[sample["id"]])
        value["observations"] = [r for r in value["observations"] if r["date"] <= sample["cutoff"]]
        value["request"] = re.sub(
            r"after \d{4}-\d{2}-\d{2}", "after " + sample["cutoff"], value["request"]
        )
        output = predict(value, head)
        target = {
            "answer": sample["sales"],
            "complete": sample["complete"],
            "dates": sample["dates"],
        }
        records.append(
            {
                "id": sample["id"],
                "family": sample["family"],
                "cutoff": sample["cutoff"],
                "inputHash": digest(value),
                "observedTotal": sum(sample["sales"]),
                "censored": sample["censored"],
                "metrics": metrics(value, target, output),
                "prediction": {k: v for k, v in output.items() if k != "F"},
            }
        )
    summary = {}
    for name in sorted({k for r in records for k in r["metrics"]}):
        grouped = defaultdict(list)
        for record in records:
            if record["metrics"][name] is not None:
                grouped[record["family"]].append(record["metrics"][name])
        values = [v for group in grouped.values() for v in group]
        means = [sum(group) / len(group) for group in grouped.values()]
        summary[name] = {
            "mean": sum(values) / len(values) if values else None,
            "familyMean": sum(means) / len(means) if means else None,
            "periods": len(values),
            "families": len(means),
        }
    return {
        "periods": len(records),
        "families": len({r["family"] for r in records}),
        "uncensoredPeriods": sum(not r["censored"] for r in records),
        "censoredPeriods": sum(r["censored"] for r in records),
        "familyCounts": {
            family: sum(r["prediction"]["distribution"]["family"] == family for r in records)
            for family in FAMILIES
        },
        "samplesHash": digest(periods),
        "metrics": summary,
        "aggregation": "Overlapping 7-day windows; family means group connected store/product sources",
        "costs": "Surplus coefficient 1; shortage coefficients 1, 3, 9; normalized sales units",
    }, records


def train(training, development, epochs, seed, warmup=10):
    """Learn family routing and family-specific parameters from observation likelihoods."""
    from .heads import Head

    heads = {name: Head(28, 32, 3) for name in HEADS}
    with torch.no_grad():
        heads["demand_family"].layers[-1].weight.zero_()
        heads["demand_family"].layers[-1].bias.zero_()
        initial = {
            "truncated_normal": [-3.0, math.log(math.expm1(1)), -1.0],
            "lognormal": [-3.0, 0.0, -1.0],
            "weibull": [-3.0, math.log(math.expm1(1.3)), math.log(math.expm1(0.95))],
        }
        for family in FAMILIES:
            layer = heads["demand_" + family].layers[-1]
            layer.weight.mul_(0.1)
            layer.bias.copy_(torch.tensor(initial[family]))
    module = torch.nn.ModuleDict(heads)
    optimizer = torch.optim.AdamW(module.parameters(), lr=0.003, weight_decay=1e-4)
    rows = training + development
    x = torch.tensor([r["x"] for r in rows], dtype=torch.float32)
    y = torch.tensor([r["y"] for r in rows], dtype=torch.float32)
    censored = torch.tensor([r["censored"] for r in rows], dtype=torch.bool)
    fit_ids = torch.arange(len(training))
    dev_ids = torch.arange(len(training), len(rows))
    rand = torch.Generator().manual_seed(seed)

    def losses(indices):
        logits = heads["demand_family"](x[indices])
        values = torch.stack(
            [
                nll(family, heads["demand_" + family](x[indices]), y[indices], censored[indices])
                for family in FAMILIES
            ],
            -1,
        )
        return logits, values

    def selected_loss(indices):
        logits, values = losses(indices)
        return values.gather(-1, logits.argmax(-1, keepdim=True)).mean()

    with torch.no_grad():
        start, best = float(selected_loss(fit_ids)), float(selected_loss(dev_ids))
    saved = {k: v.detach().clone() for k, v in module.state_dict().items()}
    selected = 0
    for epoch in range(epochs):
        for ids in fit_ids[torch.randperm(len(fit_ids), generator=rand)].split(128):
            optimizer.zero_grad(set_to_none=True)
            logits, values = losses(ids)
            # Every family keeps learning; routing cannot starve an unselected parameter head.
            objective = values.mean()
            if epoch >= warmup:
                objective = objective + (logits.softmax(-1) * values.detach()).sum(-1).mean()
            require(torch.isfinite(objective).item(), "Nonfinite demand likelihood")
            objective.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), 5)
            optimizer.step()
        with torch.no_grad():
            validation = float(selected_loss(dev_ids))
        if validation < best:
            best, selected = validation, epoch + 1
            saved = {k: v.detach().clone() for k, v in module.state_dict().items()}
    module.load_state_dict(saved)
    module.eval()
    with torch.no_grad():
        final = float(selected_loss(fit_ids))
    return heads, {
        "initial": start,
        "final": final,
        "devLoss": best,
        "selectedEpoch": selected,
        "epochs": epochs,
        "warmupEpochs": warmup,
        "rows": len(training),
        "devRows": len(development),
        "objective": "Uniform family-parameter NLL plus family-probability-weighted detached NLL",
        "selection": "Dev NLL of the actual argmax family and its parameters",
    }


def pmf(distribution):
    """Conditional quantile-bin means preserve the selected distribution's first moment."""
    normal = NormalDist()
    family, params = distribution["family"], distribution["parameters"]
    zero = params["zeroProbability"]
    first, second = (params[k] for k in PARAMETERS[family])
    if family == "lognormal":
        edges = [
            -math.inf,
            *[normal.inv_cdf(i / QUADRATURE) for i in range(1, QUADRATURE)],
            math.inf,
        ]
        moment = math.exp(first + second * second / 2)
        values = []
        for lo, hi in zip(edges[:-1], edges[1:], strict=True):
            values.append(moment * (normal.cdf(second - lo) - normal.cdf(second - hi)) * QUADRATURE)
    elif family == "truncated_normal":
        lower = -first / second
        base, total = normal.cdf(lower), normal.cdf(-lower)
        edges = [
            lower,
            *[normal.inv_cdf(base + total * i / QUADRATURE) for i in range(1, QUADRATURE)],
            math.inf,
        ]

        def density(z):
            return math.exp(-z * z / 2) / math.sqrt(2 * math.pi)

        values = [
            first + second * (density(lo) - density(hi)) * QUADRATURE / total
            for lo, hi in zip(edges[:-1], edges[1:], strict=True)
        ]
    else:
        require(family == "weibull", "Unknown demand family")
        edges = torch.tensor(
            [0.0, *[-math.log1p(-i / QUADRATURE) for i in range(1, QUADRATURE)], math.inf],
            dtype=torch.float64,
        )
        shape = 1 + 1 / first
        survival = torch.special.gammaincc(torch.tensor(shape, dtype=torch.float64), edges)
        values = (
            second * math.exp(math.lgamma(shape)) * (survival[:-1] - survival[1:]) * QUADRATURE
        ).tolist()
    pairs = [
        [0.0, zero],
        *[
            [max(d, 0) * distribution["normalizationScale"], (1 - zero) / QUADRATURE]
            for d in values
        ],
    ]
    total = sum(p for _, p in pairs)
    return [[d, p / total] for d, p in pairs]


def order(F, *, underage=1, overage=1):
    require(underage > 0 and overage > 0, "Positive shortage and surplus costs required")
    theta = {"c": overage, "p": overage + underage, "v": 0, "b": 0, "F": F}
    return optimal(theta, bounds=(0, max(d for d, _ in F)))


def predict(value, heads):
    days = horizon(value)
    x, scale = features(value["observations"], days)
    probabilities = torch.tensor(
        heads["demand_family"].scores([x])[0], dtype=torch.float64
    ).softmax(-1)
    family = FAMILIES[int(probabilities.argmax())]
    raw = torch.tensor(heads["demand_" + family].scores([x])[0], dtype=torch.float64)
    first, second = parameters(family, raw)
    distribution = {
        "family": family,
        "familyProbabilities": dict(zip(FAMILIES, probabilities.tolist(), strict=True)),
        "parameters": {
            "zeroProbability": float(raw[0].sigmoid()),
            PARAMETERS[family][0]: float(first),
            PARAMETERS[family][1]: float(second),
        },
        "normalizationScale": scale,
    }
    F = pmf(distribution)
    start = date.fromisoformat(value["observations"][-1]["date"]) + timedelta(days=1)
    return {
        "action": "answer",
        "F": F,
        "distribution": distribution,
        "period": {
            "start": start.isoformat(),
            "end": (start + timedelta(days=days - 1)).isoformat(),
            "days": days,
            "unit": "globally-normalized-sales",
        },
        "orders": [{"underage": u, "overage": 1, **order(F, underage=u)} for u in COST_RATIOS],
    }


def metrics(value, target, prediction):
    result = {
        "distributionValid": 0.0,
        "orderValid": 0.0,
        "observationNLL": None,
        "uncensoredCRPS": None,
        "uncensoredTotalMAE": None,
        "uncensoredInterval80Coverage": None,
        **{f"uncensoredOrderLoss_u{u}": None for u in COST_RATIOS},
        **{f"censoredOrderLossLowerBound_u{u}": None for u in COST_RATIOS},
    }
    try:
        distribution, F = prediction["distribution"], prediction["F"]
        require(prediction["action"] == "answer", "Invalid forecast action")
        family = distribution["family"]
        require(family in FAMILIES, "Invalid family")
        if "familyScores" in distribution:
            scores = distribution["familyScores"]
            require(set(scores) == set(FAMILIES), "Missing family loss scores")
            values = np.asarray([scores[f] for f in FAMILIES], dtype=float)
            require(np.isfinite(values).all(), "Invalid family loss scores")
            require(
                FAMILIES[int(values.argmin())] == family,
                "Selected family disagrees with loss scores",
            )
        else:
            scores = distribution["familyProbabilities"]
            require(set(scores) == set(FAMILIES), "Missing family probabilities")
            probabilities = np.asarray([scores[f] for f in FAMILIES], dtype=float)
            require(
                np.isfinite(probabilities).all()
                and (probabilities >= 0).all()
                and abs(probabilities.sum() - 1) < 1e-8,
                "Invalid family probabilities",
            )
            require(
                FAMILIES[int(probabilities.argmax())] == family,
                "Selected family disagrees with head",
            )
        params = distribution["parameters"]
        require(set(params) == {"zeroProbability", *PARAMETERS[family]}, "Wrong family parameters")
        zero = params["zeroProbability"]
        first, second = (params[k] for k in PARAMETERS[family])
        require(
            all(math.isfinite(v) for v in (zero, first, second)) and 0 < zero < 1,
            "Invalid parameter values",
        )
        require(0.05 <= second <= (20 if family == "weibull" else 3), "Invalid scale")
        require(
            (-10 <= first <= 10)
            if family == "lognormal"
            else ((0.2 <= first <= 5) if family == "weibull" else (0 <= first <= 20)),
            "Invalid location/shape",
        )
        scale = distribution["normalizationScale"]
        require(math.isfinite(scale) and scale > 0, "Invalid normalization scale")
        require(
            np.asarray(F).shape == (QUADRATURE + 1, 2)
            and np.allclose(F, pmf(distribution), rtol=1e-10, atol=1e-10),
            "F/family/parameter mismatch",
        )
        period = prediction["period"]
        require(
            period["days"] == len(target["dates"]) == horizon(value)
            and period["start"] == target["dates"][0]
            and period["end"] == target["dates"][-1]
            and period["unit"] == "globally-normalized-sales",
            "Wrong forecast period/unit",
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return result
    result["distributionValid"] = 1.0
    observed, censored = sum(target["answer"]), not all(target["complete"])
    loss = observation_nll(
        family,
        torch.tensor(math.log(zero / (1 - zero)), dtype=torch.float64),
        torch.tensor(first, dtype=torch.float64),
        torch.tensor(second, dtype=torch.float64),
        torch.tensor(observed / scale, dtype=torch.float64),
        torch.tensor(censored),
    )
    if observed > 0 and not censored:
        loss = loss + math.log(scale)
    result["observationNLL"] = float(loss)
    supports, probabilities = np.asarray(F).T
    if not censored:
        result["uncensoredTotalMAE"] = abs(float(supports @ probabilities) - observed)
        result["uncensoredCRPS"] = float(
            abs(supports - observed) @ probabilities
            - 0.5
            * (abs(supports[:, None] - supports) * probabilities[:, None] * probabilities).sum()
        )
        quantiles = [
            supports[min(np.searchsorted(probabilities.cumsum(), a), len(F) - 1)]
            for a in (0.1, 0.9)
        ]
        result["uncensoredInterval80Coverage"] = float(quantiles[0] <= observed <= quantiles[1])
    try:
        decisions = prediction["orders"]
        require(
            isinstance(decisions, list) and len(decisions) == len(COST_RATIOS), "Missing orders"
        )
        by_cost = {}
        for decision in decisions:
            u = decision["underage"]
            require(
                u in COST_RATIOS and u not in by_cost and decision["overage"] == 1,
                "Wrong or repeated order cost",
            )
            q, cost = decision["q"], decision["cost"]
            require(
                math.isfinite(q) and q >= 0 and math.isfinite(cost) and cost >= 0,
                "Invalid emitted order",
            )
            expected = risk(q, {"c": 1, "p": u + 1, "v": 0, "b": 0, "F": F})
            require(
                math.isclose(cost, expected, rel_tol=1e-8, abs_tol=1e-8),
                "Reported order risk disagrees with F",
            )
            by_cost[u] = q
    except (KeyError, TypeError, ValueError, OverflowError):
        return result
    result["orderValid"] = 1.0
    for u in COST_RATIOS:
        q = by_cost[u]
        if censored:
            result[f"censoredOrderLossLowerBound_u{u}"] = u * max(observed - q, 0)
        else:
            result[f"uncensoredOrderLoss_u{u}"] = max(q - observed, 0) + u * max(observed - q, 0)
    return result
