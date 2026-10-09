"""Numerical counterexample for collapsing daily censoring into one total lower bound."""

import math
from pathlib import Path

import torch

from newsvendor.io import digest, require, write

torch.set_num_threads(2)
torch.set_default_dtype(torch.float64)


def total_loss(theta, sales, censored):
    days = sales.shape[1]
    total = sales.sum(-1)
    ratio = total / theta
    density = (days - 1) * total.log() - ratio - days * theta.log() - math.lgamma(days)
    powers = torch.arange(days)
    log_tail = -ratio + torch.logsumexp(ratio.log()[:, None] * powers - torch.lgamma(powers + 1), -1)
    return -torch.where(censored.any(-1), log_tail, density).mean()


def daily_loss(theta, sales, censored):
    return (sales / theta + (~censored) * theta.log()).mean()


def fit(sales, censored, daily_weight):
    parameter = torch.tensor(0., requires_grad=True)
    optimizer = torch.optim.LBFGS([parameter], max_iter=100, tolerance_grad=1e-10, tolerance_change=1e-12, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        theta = parameter.exp()
        loss = total_loss(theta, sales, censored) + daily_weight * daily_loss(theta, sales, censored)
        loss.backward()
        return loss

    optimizer.step(closure)
    require(torch.isfinite(parameter).item(), "Nonfinite numerical fit")
    return float(parameter.detach().exp())


records = []
for seed in (42, 43, 44):
    random = torch.Generator().manual_seed(seed)
    latent = -torch.rand((50000, 7), generator=random).log()
    for days in (1, 2, 7):
        demand = latent[:, :days]
        censored = demand >= 1
        sales = demand.clamp(max=1)
        observed_mle = float(sales.sum() / (~censored).sum())
        aggregate_mle = fit(sales, censored, 0)
        composite_mle = fit(sales, censored, 1)
        if days == 1:
            require(abs(aggregate_mle - observed_mle) < 1e-5, "Daily likelihoods must agree")
        records.append({"seed": seed, "days": days, "periods": len(sales),
                        "trueDailyMean": 1., "dailyObservedLikelihoodMean": observed_mle,
                        "aggregateLowerBoundMean": aggregate_mle, "aggregatePlusDailyMean": composite_mle,
                        "censoredFraction": float(censored.any(-1).double().mean())})
report = {"scope": "Mathematical/optimizer smoke only; generated data do not demonstrate real demand effectiveness",
          "model": "Independent exponential daily demand, known fixed cap 1; sum has exact Gamma(shape=days, scale=dailyMean)",
          "correctLikelihood": "For observed daily values and censor indicators: sum(sales)/theta + count(exact)*log(theta)",
          "aggregateSurrogate": "Gamma density on complete totals, Gamma survival at summed sales when any day stocks out",
          "dailyAuxiliary": "Mean correct daily NLL added with weight 1, analogous to horizon auxiliary training",
          "records": records, "scriptHash": digest(Path(__file__).read_bytes()),
          "testUsed": False,
          "limitation": "This counterexample diagnoses an assumption, not the size of this effect on FreshRetail or a replacement model's effectiveness."}
path = Path("results/research-checks/aggregate-censoring-smoke.json")
require(not path.exists(), "Preserve prior numerical audit")
write(path, report)
print(records, flush=True)
