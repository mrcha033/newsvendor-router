"""Daily observations of a zero-inflated total with uniform positive-day allocation.

The total retains the existing three positive families and zero probability p0.
Conditional on a positive total, active days follow independent Bernoulli draws
conditioned on a nonempty set, using the model's one-day zero probability. The
positive total is allocated uniformly on that simplex.
Exact positive days have a joint density, exact zero days have probability mass,
and censored days are integrated separately. This is a proper likelihood for
this explicit model under noninformative daily right censoring; it does not
identify the real intraday stockout mechanism from daily lower bounds alone.
"""

import math
from functools import lru_cache

import numpy as np
import torch
from torch.nn import functional as fn

from . import demand
from .io import require


@lru_cache(maxsize=8)
def quadrature(size):
    nodes, weights = np.polynomial.legendre.leggauss(size)
    nodes, weights = (nodes + 1) / 2, weights / 2
    # Resolve both the observed bound and the remote tail without subtracting
    # a CDF rounded to one. The transform is v=t^4/(t^4+(1-t)^4).
    left, right = np.log(nodes), np.log1p(-nodes)
    denominator = np.logaddexp(4 * left, 4 * right)
    log_fraction = 4 * right - denominator
    weights = weights * np.exp(math.log(4) + 3 * (left + right) - 2 * denominator)
    return log_fraction, weights / weights.sum()


def normal_isf(log_probability):
    """Normal inverse survival from log probabilities, including extreme tails."""
    reflect = log_probability > -math.log(2)
    small = torch.where(reflect, (-log_probability.expm1()).log(), log_probability)
    z = (-2 * small).sqrt()
    z = z - (z.log() + math.log(2 * math.pi) / 2) / z
    for _ in range(6):
        log_tail = torch.special.log_ndtr(-z)
        slope = -(z.square().mul(-0.5) - math.log(2 * math.pi) / 2 - log_tail).exp()
        z = z - (log_tail - small) / slope
    return torch.where(reflect, -z, z)


def conditional_nodes(family, first, second, lower, size):
    """Log conditional positive-total quantiles above a lower bound, with weights."""
    log_fraction, weights = quadrature(size)
    log_fraction = torch.as_tensor(log_fraction, dtype=lower.dtype, device=lower.device)
    weights = torch.as_tensor(weights, dtype=lower.dtype, device=lower.device)
    _, log_tail = demand.positive_logs(family, first, second, lower)
    if family == "weibull":
        log_value = (
            second[:, None].log() + (-(log_tail[:, None] + log_fraction)).log() / first[:, None]
        )
    elif family == "lognormal":
        z = normal_isf(log_tail[:, None] + log_fraction)
        log_value = first[:, None] + second[:, None] * z
    else:
        require(family == "truncated_normal", "Unknown allocation family")
        tail = torch.special.log_ndtr((first - lower) / second)
        z = normal_isf(tail[:, None] + log_fraction)
        value = first[:, None] + second[:, None] * z
        log_value = value.clamp_min(torch.finfo(value.dtype).tiny).log()
    return log_value, log_tail, weights.log()


def log_integral(log_nodes, log_tail, log_weights, lower, exact, missing):
    """Integrate the simplex density after exact days and active censored days.

    For k exact positive and m active censored days, the integrand multiplier is
    Gamma(k+m)/Gamma(m) * (D-sum(sales))**(m-1) / D**(k+m-1).
    The exact values enter through their sum; individual censor indicators and
    structural zeros remain separate. m must be positive for the returned formula.
    """
    log_lower = lower.clamp_min(torch.finfo(lower.dtype).tiny).log()
    log_ratio = (log_lower[:, None] - log_nodes).clamp(max=-1e-14)
    log_overshoot_fraction = (-log_ratio.expm1()).log()
    kernel = (missing[..., None] - 1) * log_overshoot_fraction[:, None, :] - exact[
        :, None, None
    ] * log_nodes[:, None, :]
    integral = (
        log_tail[:, None]
        + torch.logsumexp(kernel + log_weights, dim=-1)
        + torch.lgamma(exact[:, None] + missing)
        - torch.lgamma(missing)
    )
    # No positive observed value supplies no bound and no density constraint.
    return torch.where(lower[:, None] == 0, torch.zeros_like(integral), integral)


def allocation_nll(family, raw, sales, censored, present=None, size=64, daily_zero_logit=None):
    """Joint daily observation NLL for batches with padding outside their horizon."""
    require(sales.ndim == 2 and censored.shape == sales.shape, "Daily observation shape")
    require(raw.shape == (len(sales), 3), "Distribution parameter shape")
    if present is None:
        present = torch.ones_like(censored, dtype=torch.bool)
    require(present.shape == sales.shape, "Observation mask shape")
    require(bool((sales >= 0).all()) and bool(present.any(-1).all()), "Invalid observations")
    # The integral includes tail quantiles and cancelling normal locations. Keep
    # it in float64; gradients return to the unchanged float32 GRU and heads.
    raw, sales = raw.double(), sales.double()
    positive = (sales > 0) & present
    exact = (positive & ~censored).sum(-1).double()
    forced = (positive & censored).sum(-1).double()
    optional = (present & censored & ~positive).sum(-1).double()
    horizon = present.sum(-1).double()
    lower = (sales * present).sum(-1)
    first, second = demand.parameters(family, raw)
    density, _ = demand.positive_logs(family, first, second, lower)
    log_nodes, log_tail, log_weights = conditional_nodes(family, first, second, lower, size)

    added = torch.arange(sales.shape[1] + 1, dtype=raw.dtype, device=raw.device)[None, :]
    missing = forced[:, None] + added
    active = exact[:, None] + missing
    valid = added <= optional[:, None]
    daily_zero_logit = raw[:, 0] if daily_zero_logit is None else daily_zero_logit.double()
    log_zero = fn.logsigmoid(daily_zero_logit)
    log_active = (-log_zero.expm1()).log()
    log_nonempty = (-(horizon * log_zero).expm1()).log()
    choices = (
        torch.lgamma(optional[:, None] + 1)
        - torch.lgamma(added + 1)
        - torch.lgamma((optional[:, None] - added + 1).clamp_min(1))
    )
    pattern = (
        choices
        + active * log_active[:, None]
        + (horizon[:, None] - active) * log_zero[:, None]
        + (fn.logsigmoid(-raw[:, 0]) - log_nonempty)[:, None]
    )
    pattern = torch.where(active == 0, fn.logsigmoid(raw[:, 0])[:, None], pattern)
    complete_density = (
        density
        + torch.lgamma(exact.clamp_min(1))
        - (exact - 1) * lower.clamp_min(torch.finfo(raw.dtype).tiny).log()
    )
    integral = log_integral(log_nodes, log_tail, log_weights, lower, exact, missing.clamp_min(1))
    observed = torch.where(missing == 0, complete_density[:, None], integral)
    observed = torch.where(active == 0, torch.zeros_like(observed), observed)
    terms = (pattern + observed).masked_fill(~valid, -torch.inf)
    result = -torch.logsumexp(terms, dim=-1)
    # All-zero lower bounds on entirely censored days contain no information.
    return torch.where((lower == 0) & (censored | ~present).all(-1), result * 0, result)
