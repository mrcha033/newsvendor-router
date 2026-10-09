"""Horizon-conditioned GRU distributions, with source-held-out loss-score supervision."""

import copy
import math
from datetime import date, timedelta

import numpy as np
import torch
from torch import nn
from torch.nn import functional as fn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

from . import demand
from .heads import Head
from .io import digest, require

SCALARS = (
    "sales",
    "stockoutHours",
    "discount",
    "holidayFlag",
    "activityFlag",
    "precipitation",
    "temperature",
    "humidity",
    "wind",
)
INPUT_DIM = 2 * (len(SCALARS) + 48) + 4


def features(rows):
    require(rows, "Missing demand history")
    dates = [r["date"] for r in rows]
    require(dates == sorted(set(dates)), "Demand history must be chronological")
    sales = [float(r["sales"]) for r in rows]
    require(all(math.isfinite(v) and v >= 0 for v in sales), "Invalid sales history")
    scale = max(float(np.mean(sales)), 0.1)
    result = []
    for row in rows:
        values, mask = [], []
        for name in SCALARS:
            present = row.get(name) is not None
            v = float(row[name]) if present else 0.0
            require(math.isfinite(v), "Nonfinite history feature")
            if name == "sales":
                v /= scale
            elif name == "stockoutHours":
                v /= 24
            elif name in ("temperature", "humidity", "precipitation", "wind"):
                v /= 100
            values.append(v)
            mask.append(float(present))
        for name in ("hourlySales", "hourlyStockStatus"):
            hourly = row.get(name)
            require(hourly is None or len(hourly) == 24, "Expected 24 hourly measurements")
            for i in range(24):
                present = hourly is not None and hourly[i] is not None
                v = float(hourly[i]) if present else 0.0
                require(math.isfinite(v), "Nonfinite hourly feature")
                values.append(v / scale if name == "hourlySales" else v)
                mask.append(float(present))
        d = date.fromisoformat(row["date"])
        calendar = [
            math.sin(2 * math.pi * d.weekday() / 7),
            math.cos(2 * math.pi * d.weekday() / 7),
            math.sin(2 * math.pi * d.timetuple().tm_yday / 366),
            math.cos(2 * math.pi * d.timetuple().tm_yday / 366),
        ]
        result.append(values + mask + calendar)
    return torch.tensor(result, dtype=torch.float32), scale


class DemandEncoder(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.gru = nn.GRU(INPUT_DIM, hidden, num_layers=2, batch_first=True)
        self.condition = nn.Linear(hidden + 1, 256)
        self.family = Head(256, 128, 3)
        self.params = nn.ModuleDict({f: Head(256, 128, 3) for f in demand.FAMILIES})
        initial = {
            "truncated_normal": [-3.0, math.log(math.expm1(1)), -1.0],
            "lognormal": [-3.0, 0.0, -1.0],
            "weibull": [-3.0, math.log(math.expm1(1.3)), math.log(math.expm1(0.95))],
        }
        with torch.no_grad():
            for family, head in self.params.items():
                head.layers[-1].weight.mul_(0.1)
                head.layers[-1].bias.copy_(torch.tensor(initial[family]))

    def forward(self, sequences, horizons, context=None):
        device = next(self.parameters()).device
        require(
            len(sequences) == len(horizons) > 0 and all(h > 0 for h in horizons),
            "Missing forecast horizons",
        )
        packed = pack_padded_sequence(
            pad_sequence(sequences, batch_first=True).to(device),
            [len(s) for s in sequences],
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        days = torch.tensor(horizons, dtype=torch.float32, device=device).log1p().unsqueeze(-1)
        state = self.condition(torch.cat([hidden[-1], days], -1)).tanh()
        if context is not None:
            require(context.shape == state.shape, "Linked language/demand context mismatch")
            state = state + context
        return {
            "state": state,
            "scores": self.family(state),
            "raw": {f: h(state) for f, h in self.params.items()},
        }


def samples(rows, labels, min_history=28):
    result = []
    for row in rows:
        if row["component"] != "retail":
            continue
        past = row["input"]["observations"]
        target = labels[row["id"]]
        final = [
            {"date": d, "sales": s, "stockoutHours": 0 if complete else 1}
            for d, s, complete in zip(
                target["dates"], target["answer"], target["complete"], strict=True
            )
        ]
        require(past[-1]["date"] < final[0]["date"], "Future observation in history")
        # Future values are outcomes only; never concatenate them into model features.
        for end in range(min_history, len(past) + 1):
            history = past[:end]
            sequence, scale = features(history)
            for horizon in (1, 7):
                future = (past[end:] + final)[:horizon]
                if len(future) != horizon:
                    continue
                dates = [f["date"] for f in future]
                require(history[-1]["date"] < min(dates), "Cutoff leakage")
                require(
                    dates
                    == [
                        (date.fromisoformat(history[-1]["date"]) + timedelta(days=i)).isoformat()
                        for i in range(1, horizon + 1)
                    ],
                    "Forecast period contains calendar gaps",
                )
                result.append(
                    {
                        "id": row["id"],
                        "family": row["family"],
                        "split": row["split"],
                        "sequence": sequence,
                        "horizon": horizon,
                        "scale": scale * horizon,
                        "y": sum(f["sales"] for f in future) / (scale * horizon),
                        "censored": any(f["stockoutHours"] > 0 for f in future),
                        "cutoff": history[-1]["date"],
                        "dates": dates,
                        "historyHash": digest(history),
                    }
                )
    return result


def losses(output, rows):
    device = output["scores"].device
    observed = torch.tensor([r["y"] for r in rows], device=device)
    censored = torch.tensor([r["censored"] for r in rows], dtype=torch.bool, device=device)
    return torch.stack(
        [demand.nll(f, output["raw"][f], observed, censored) for f in demand.FAMILIES], -1
    )


def fit_params(model, rows, config, seed):
    require(rows, "Empty demand training fold")
    optimizer = torch.optim.AdamW(
        [p for name, p in model.named_parameters() if not name.startswith("family.")],
        lr=config["lr"],
    )
    rand = torch.Generator().manual_seed(seed)
    model.train()
    history = []
    for _ in range(config["epochs"]):
        epoch = []
        for ids in torch.randperm(len(rows), generator=rand).split(config.get("batchSize", 64)):
            batch = [rows[i] for i in ids.tolist()]
            output = model([r["sequence"] for r in batch], [r["horizon"] for r in batch])
            values = losses(output, batch)
            # Both horizons and all families receive likelihood gradients.
            weights = torch.tensor(
                [config.get("dailyWeight", 0.3) if r["horizon"] == 1 else 1 for r in batch],
                device=values.device,
            )
            objective = (values.mean(-1) * weights).sum() / weights.sum()
            require(torch.isfinite(objective).item(), "Nonfinite sequence likelihood")
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            epoch.append(float(objective.detach()))
        history.append(float(np.mean(epoch)))
    return history


@torch.inference_mode()
def measure(model, rows, batch_size=64):
    model.eval()
    result = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        output = model([r["sequence"] for r in batch], [r["horizon"] for r in batch])
        result.append(losses(output, batch).cpu())
    return torch.cat(result)


def cross_fit(model, training, development, config, seed, progress=None):
    require(all(r["split"] == "train" for r in training), "Cross-fit requires Train only")
    require(all(r["split"] == "dev" for r in development), "Checkpoint selection requires Dev")
    groups = sorted({r["family"] for r in training}, key=lambda f: digest([seed, f]))
    count = config.get("folds", 3)
    require(2 <= count <= len(groups), "Insufficient source groups for cross-fit")
    folds = {g: i % count for i, g in enumerate(groups)}
    initial = copy.deepcopy(model.state_dict())
    target = torch.zeros(len(training), 3)
    records, reports = [], []
    for fold in range(count):
        if progress:
            progress.update("demand", fold=fold + 1, folds=count)
        fit = [r for r in training if folds[r["family"]] != fold]
        ids = [i for i, r in enumerate(training) if folds[r["family"]] == fold]
        held = [training[i] for i in ids]
        model.load_state_dict(initial)
        history = fit_params(model, fit, config, seed + fold)
        measured = measure(model, held)
        target[ids] = measured
        reports.append(
            {
                "fold": fold,
                "fitFamilies": sorted({r["family"] for r in fit}),
                "heldFamilies": sorted({r["family"] for r in held}),
                "losses": history,
            }
        )
        for row, values in zip(held, measured.tolist(), strict=True):
            records.append(
                {
                    k: row[k]
                    for k in (
                        "id",
                        "family",
                        "cutoff",
                        "dates",
                        "horizon",
                        "historyHash",
                        "censored",
                        "y",
                        "scale",
                    )
                }
                | {"fold": fold, "familyNLL": values}
            )
    model.load_state_dict(initial)
    if progress:
        progress.update("demand", activity="full_parameter_fit")
    full = fit_params(model, training, config, seed + count)
    # Fit only the loss-score selector; parameter models remain the Train-only final fits.
    optimizer = torch.optim.AdamW(model.family.parameters(), lr=config["lr"])
    rand = torch.Generator().manual_seed(seed)
    best, saved, selected = math.inf, copy.deepcopy(model.state_dict()), 0
    selector_losses = []
    for epoch in range(config.get("selectorEpochs", config["epochs"])):
        current = []
        for ids in torch.randperm(len(training), generator=rand).split(config.get("batchSize", 64)):
            batch = [training[i] for i in ids.tolist()]
            with torch.no_grad():
                state = model([r["sequence"] for r in batch], [r["horizon"] for r in batch])[
                    "state"
                ]
            objective = fn.smooth_l1_loss(model.family(state), target[ids].to(state.device))
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            optimizer.step()
            current.append(float(objective.detach()))
        selector_losses.append(float(np.mean(current)))
        if development:
            with torch.no_grad():
                values = []
                for start in range(0, len(development), 64):
                    batch = development[start : start + 64]
                    out = model([r["sequence"] for r in batch], [r["horizon"] for r in batch])
                    chosen = losses(out, batch).gather(1, out["scores"].argmin(-1, keepdim=True))
                    values.extend(chosen[[r["horizon"] == 7 for r in batch]].flatten().tolist())
                loss = float(np.mean(values))
            if loss < best:
                best, selected, saved = loss, epoch + 1, copy.deepcopy(model.state_dict())
    if development:
        model.load_state_dict(saved)
    model.eval()
    return {
        "folds": reports,
        "parameterLosses": full,
        "selectorLosses": selector_losses,
        "selectedEpoch": selected,
        "devSelectedNLL": best if development else None,
        "objective": "Train source-held-out family NLL regression; minimum predicted loss",
        "targetHash": digest(records),
    }, records


@torch.inference_mode()
def predict(model, value, horizon=7, context=None):
    model.eval()
    seq, daily_scale = features(value["observations"])
    output = model([seq], [horizon], context=context)
    scores = output["scores"][0].float().cpu()
    family = demand.FAMILIES[int(scores.argmin())]
    raw = output["raw"][family][0].float().cpu()
    first, second = demand.parameters(family, raw)
    distribution = {
        "family": family,
        "familyScores": dict(zip(demand.FAMILIES, scores.tolist(), strict=True)),
        "parameters": {
            "zeroProbability": min(max(float(raw[0].sigmoid()), 1e-8), 1 - 1e-8),
            demand.PARAMETERS[family][0]: float(first),
            demand.PARAMETERS[family][1]: float(second),
        },
        "normalizationScale": daily_scale * horizon,
    }
    F = demand.pmf(distribution)
    start = date.fromisoformat(value["observations"][-1]["date"]) + timedelta(days=1)
    return {
        "action": "answer",
        "F": F,
        "distribution": distribution,
        "period": {
            "start": start.isoformat(),
            "end": (start + timedelta(days=horizon - 1)).isoformat(),
            "days": horizon,
            "unit": "globally-normalized-sales",
        },
        "orders": [
            {"underage": u, "overage": 1, **demand.order(F, underage=u)} for u in demand.COST_RATIOS
        ],
    }
