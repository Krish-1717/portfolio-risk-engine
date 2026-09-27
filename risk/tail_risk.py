"""
Tail Risk Measures â CVaR, EVT, and Extreme Loss Estimation
Day 16 â portfolio-risk-engine/risk/tail_risk.py

Implements:
  - Historical and parametric VaR / CVaR (Expected Shortfall)
  - Extreme Value Theory: GEV (block maxima) and GPD (POT method)
  - Tail dependence analysis
  - Cornish-Fisher expansion for non-Gaussian tails
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Normal distribution helpers
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


def _norm_ppf(p: float) -> float:
    """Approximation to the normal quantile (Beasley-Springer-Moro)."""
    if p <= 0:
        return -math.inf
    if p >= 1:
        return math.inf
    if p < 0.5:
        sign = -1
        q = p
    else:
        sign = 1
        q = 1 - p

    t = math.sqrt(-2 * math.log(q))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    x = t - (c0 + c1 * t + c2 * t ** 2) / (1 + d1 * t + d2 * t ** 2 + d3 * t ** 3)
    return sign * x


# ---------------------------------------------------------------------------
# Historical VaR and CVaR
# ---------------------------------------------------------------------------

def historical_var(returns: List[float], confidence: float = 0.95) -> float:
    """Historical VaR at given confidence level (as a positive loss)."""
    sorted_r = sorted(returns)
    idx = int((1 - confidence) * len(sorted_r))
    return -sorted_r[idx]


def historical_cvar(returns: List[float], confidence: float = 0.95) -> float:
    """
    Historical CVaR (Expected Shortfall) â average of losses beyond VaR.
    """
    sorted_r = sorted(returns)
    idx = int((1 - confidence) * len(sorted_r))
    tail = sorted_r[:max(idx, 1)]
    return -sum(tail) / len(tail) if tail else 0.0


# ---------------------------------------------------------------------------
# Parametric VaR / CVaR (Gaussian)
# ---------------------------------------------------------------------------

def parametric_var(mu: float, sigma: float,
                   confidence: float = 0.95,
                   horizon: int = 1) -> float:
    """Gaussian parametric VaR (annualised to horizon days)."""
    z = _norm_ppf(confidence)
    return -(mu * horizon + sigma * math.sqrt(horizon) * (-z))


def parametric_cvar(mu: float, sigma: float,
                    confidence: float = 0.95,
                    horizon: int = 1) -> float:
    """Gaussian parametric CVaR."""
    z = _norm_ppf(confidence)
    phi_z = math.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi)
    es = -(mu * horizon - sigma * math.sqrt(horizon) * phi_z / (1 - confidence))
    return es


# ---------------------------------------------------------------------------
# Cornish-Fisher expansion (non-Gaussian)
# ---------------------------------------------------------------------------

def cornish_fisher_var(mu: float, sigma: float, skew: float, kurt: float,
                       confidence: float = 0.95) -> float:
    """
    Modified VaR using Cornish-Fisher expansion for skewed / fat-tailed returns.
    kurt here is excess kurtosis.
    """
    z = _norm_ppf(confidence)
    # Adjusted quantile
    z_cf = (z
            + (z ** 2 - 1) * skew / 6
            + (z ** 3 - 3 * z) * kurt / 24
            - (2 * z ** 3 - 5 * z) * skew ** 2 / 36)
    return -(mu + sigma * z_cf)


# ---------------------------------------------------------------------------
# Moments from returns
# ---------------------------------------------------------------------------

def _moments(returns: List[float]) -> Tuple[float, float, float, float]:
    """Compute mean, std, skewness, excess kurtosis."""
    n = len(returns)
    mu = sum(returns) / n
    diffs = [r - mu for r in returns]
    var = sum(d ** 2 for d in diffs) / n
    sigma = math.sqrt(var) if var > 0 else 1e-10
    skew = sum(d ** 3 for d in diffs) / (n * sigma ** 3) if sigma > 0 else 0.0
    kurt = sum(d ** 4 for d in diffs) / (n * sigma ** 4) - 3.0
    return mu, sigma, skew, kurt


# ---------------------------------------------------------------------------
# Extreme Value Theory â Generalised Pareto Distribution (POT)
# ---------------------------------------------------------------------------

@dataclass
class GPDParams:
    xi: float      # shape (tail index)
    beta: float    # scale
    threshold: float
    n_total: int
    n_exceed: int


def fit_gpd_mom(exceedances: List[float]) -> Tuple[float, float]:
    """
    Method of Moments estimator for GPD (Î¾, Î²).
    Exceedances are (x - threshold) for x > threshold.
    """
    if len(exceedances) < 2:
        return 0.0, 1.0
    n = len(exceedances)
    mu_e = sum(exceedances) / n
    var_e = sum((x - mu_e) ** 2 for x in exceedances) / n
    if var_e < 1e-12:
        return 0.0, mu_e
    xi = 0.5 * (mu_e ** 2 / var_e - 1)
    beta = 0.5 * mu_e * (mu_e ** 2 / var_e + 1)
    return xi, beta


def fit_gpd_pot(returns: List[float],
                threshold_quantile: float = 0.05) -> GPDParams:
    """
    Peaks-over-threshold GPD fit to the left tail of returns.
    threshold_quantile = fraction of data in the tail.
    """
    sorted_r = sorted(returns)
    idx = int(threshold_quantile * len(sorted_r))
    threshold = sorted_r[max(idx, 1)]   # negative value

    # Exceedances above threshold (losses beyond threshold)
    exceedances = [threshold - r for r in sorted_r[:max(idx, 1)]]
    exceedances = [e for e in exceedances if e > 0]

    xi, beta = fit_gpd_mom(exceedances)
    return GPDParams(xi=xi, beta=beta, threshold=threshold,
                     n_total=len(returns), n_exceed=len(exceedances))


def gpd_var(params: GPDParams, confidence: float = 0.99) -> float:
    """
    GPD-based VaR (as positive loss):
    VaR_p = threshold_loss + (beta/xi) * ((n/N*(1-p))^{-xi} - 1)
    """
    u = -params.threshold
    p_tail = params.n_exceed / params.n_total
    q = 1 - confidence

    if params.xi == 0:
        return u - params.beta * math.log(q / p_tail)

    ratio = q / p_tail
    if ratio <= 0:
        return u
    return u + (params.beta / params.xi) * (ratio ** (-params.xi) - 1)


def gpd_cvar(params: GPDParams, confidence: float = 0.99) -> float:
    """GPD-based CVaR (Expected Shortfall)."""
    var = gpd_var(params, confidence)
    u = -params.threshold

    if params.xi >= 1:
        return math.inf  # mean doesn't exist

    if params.xi == 0:
        return var + params.beta
    return (var + params.beta - params.xi * u) / (1 - params.xi)


# ---------------------------------------------------------------------------
# GEV block maxima
# ---------------------------------------------------------------------------

@dataclass
class GEVParams:
    xi: float     # shape
    mu: float     # location
    sigma: float  # scale


def fit_gev_pwm(block_maxima: List[float]) -> GEVParams:
    """
    Probability-Weighted Moments estimator for GEV on block maxima.
    Simple L-moments approach.
    """
    n = len(block_maxima)
    xs = sorted(block_maxima)

    b0 = sum(xs) / n
    b1 = sum(xs[j] * j / (n * (n - 1)) for j in range(n))
    b2 = sum(xs[j] * j * (j - 1) / (n * (n - 1) * (n - 2))
             for j in range(2, n))

    l1 = b0
    l2 = 2 * b1 - b0
    l3 = 6 * b2 - 6 * b1 + b0

    tau3 = l3 / l2 if abs(l2) > 1e-12 else 0.0

    # GEV shape parameter via numerical approximation
    c = 2 / (3 + tau3) - math.log(2) / math.log(3)
    xi = 7.859 * c + 2.9554 * c ** 2  # approximation

    k = xi
    if abs(k) < 1e-6:
        sigma = l2 / math.log(2)
        mu = l1 - sigma * (math.log(math.log(2)) + 0.5772)
    else:
        gk = math.gamma(1 + k) if abs(k) < 170 else 1.0
        sigma = l2 * k / ((1 - 2 ** (-k)) * gk)
        mu = l1 - sigma * (gk - 1) / k

    return GEVParams(xi=xi, mu=mu, sigma=sigma)


def gev_return_level(params: GEVParams, period: float) -> float:
    """
    GEV return level: value exceeded on average once per 'period' blocks.
    """
    p = 1 - 1 / period
    y = -math.log(p)

    if abs(params.xi) < 1e-6:
        return params.mu - params.sigma * math.log(y)
    return params.mu + params.sigma * (y ** (-params.xi) - 1) / params.xi


# ---------------------------------------------------------------------------
# Tail dependence coefficient
# ---------------------------------------------------------------------------

def tail_dependence(returns_a: List[float], returns_b: List[float],
                    quantile: float = 0.05) -> Tuple[float, float]:
    """
    Lower and upper tail dependence coefficients.
    Î»_L â fraction of joint extremes / fraction in tail.
    """
    n = len(returns_a)
    k = max(int(quantile * n), 1)
    sorted_a = sorted(returns_a)
    sorted_b = sorted(returns_b)
    lower_a = set(i for i, r in enumerate(returns_a) if r <= sorted_a[k - 1])
    lower_b = set(i for i, r in enumerate(returns_b) if r <= sorted_b[k - 1])
    upper_a = set(i for i, r in enumerate(returns_a) if r >= sorted_a[n - k])
    upper_b = set(i for i, r in enumerate(returns_b) if r >= sorted_b[n - k])

    lower_td = len(lower_a & lower_b) / k
    upper_td = len(upper_a & upper_b) / k
    return lower_td, upper_td


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

@dataclass
class TailRiskReport:
    n_obs: int
    mean_return: float
    vol: float
    skewness: float
    excess_kurtosis: float
    hist_var_95: float
    hist_cvar_95: float
    hist_var_99: float
    hist_cvar_99: float
    normal_var_95: float
    normal_cvar_95: float
    cf_var_99: float
    gpd_var_99: float
    gpd_cvar_99: float
    gev_100yr_loss: Optional[float] = None


def tail_risk_report(returns: List[float],
                     ann_factor: float = 252) -> TailRiskReport:
    """Full tail risk report from daily return series."""
    mu, sigma, skew, kurt = _moments(returns)
    mu_ann = mu * ann_factor
    sigma_ann = sigma * math.sqrt(ann_factor)

    h_var95 = historical_var(returns, 0.95)
    h_cvar95 = historical_cvar(returns, 0.95)
    h_var99 = historical_var(returns, 0.99)
    h_cvar99 = historical_cvar(returns, 0.99)

    n_var95 = parametric_var(mu, sigma, 0.95)
    n_cvar95 = parametric_cvar(mu, sigma, 0.95)
    cf_var99 = cornish_fisher_var(mu, sigma, skew, kurt, 0.99)

    gpd = fit_gpd_pot(returns, threshold_quantile=0.05)
    g_var99 = gpd_var(gpd, 0.99)
    g_cvar99 = gpd_cvar(gpd, 0.99)

    # Block maxima (monthly blocks)
    block_size = 21
    blocks = [-min(returns[i: i + block_size])
               for i in range(0, len(returns) - block_size, block_size)]
    gev_100yr = None
    if len(blocks) >= 5:
        gev = fit_gev_pwm(blocks)
        monthly_periods = 100 * 12
        gev_100yr = gev_return_level(gev, monthly_periods)

    return TailRiskReport(
        n_obs=len(returns),
        mean_return=mu_ann, vol=sigma_ann,
        skewness=skew, excess_kurtosis=kurt,
        hist_var_95=h_var95, hist_cvar_95=h_cvar95,
        hist_var_99=h_var99, hist_cvar_99=h_cvar99,
        normal_var_95=n_var95, normal_cvar_95=n_cvar95,
        cf_var_99=cf_var99,
        gpd_var_99=g_var99, gpd_cvar_99=g_cvar99,
        gev_100yr_loss=gev_100yr,
    )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random
    rng = random.Random(42)

    # Simulate fat-tailed returns (t-distribution, 5 dof)
    def rand_t(df: int) -> float:
        # Box-Muller + chi-squared via sum of squares
        z = rng.gauss(0, 1)
        chi2 = sum(rng.gauss(0, 1) ** 2 for _ in range(df))
        return z / math.sqrt(chi2 / df)

    returns = [0.0005 + 0.015 * rand_t(5) for _ in range(2520)]  # 10 years

    report = tail_risk_report(returns)

    print(f"Return series: {report.n_obs} obs, Î¼={report.mean_return:.3%}, Ï={report.vol:.3%}")
    print(f"Skewness: {report.skewness:.3f}, Excess Kurtosis: {report.excess_kurtosis:.3f}")
    print()
    print(f"{'Measure':<30} {'95%':>10} {'99%':>10}")
    print("-" * 52)
    print(f"{'Historical VaR':<30} {report.hist_var_95:>10.4%} {report.hist_var_99:>10.4%}")
    print(f"{'Historical CVaR':<30} {report.hist_cvar_95:>10.4%} {report.hist_cvar_99:>10.4%}")
    print(f"{'Normal VaR':<30} {report.normal_var_95:>10.4%} {'--':>10}")
    print(f"{'Normal CVaR':<30} {report.normal_cvar_95:>10.4%} {'--':>10}")
    print(f"{'Cornish-Fisher VaR':<30} {'--':>10} {report.cf_var_99:>10.4%}")
    print(f"{'GPD VaR (POT)':<30} {'--':>10} {report.gpd_var_99:>10.4%}")
    print(f"{'GPD CVaR (POT)':<30} {'--':>10} {report.gpd_cvar_99:>10.4%}")
    if report.gev_100yr_loss:
        print(f"\n100-year daily loss estimate (GEV): {report.gev_100yr_loss:.4%}")
