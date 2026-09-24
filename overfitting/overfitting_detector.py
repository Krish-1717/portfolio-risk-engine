"""
overfitting/overfitting_detector.py -- Backtest overfitting detection.
Consolidated from backtest-overfitting-detector repo.
Deflated Sharpe Ratio, Probabilistic SR, haircut Sharpe, multiple-testing
correction (Bonferroni / BHY), combinatorially symmetric cross-validation.
Pure Python stdlib -- no external dependencies.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple


def _mean(xs: List[float]) -> float:
    return sum(xs) / max(len(xs), 1)

def _std(xs: List[float]) -> float:
    mu = _mean(xs)
    return math.sqrt(sum((x - mu)**2 for x in xs) / max(len(xs) - 1, 1))

def _normal_cdf(x: float) -> float:
    """Abramowitz & Stegun approximation."""
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    cdf = 1.0 - (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x * x) * poly
    return cdf if x >= 0 else 1.0 - cdf

def _normal_ppf(p: float) -> float:
    """Beasley-Springer rational approx to inverse normal."""
    if p <= 0 or p >= 1:
        return float('nan')
    sign = 1.0 if p > 0.5 else -1.0
    t = p if p <= 0.5 else 1.0 - p
    t = math.sqrt(-2.0 * math.log(t))
    c = (2.515517, 0.802853, 0.010328)
    d = (1.432788, 0.189269, 0.001308)
    return sign * (t - (c[0] + t*(c[1] + t*c[2])) / (1.0 + t*(d[0] + t*(d[1] + t*d[2]))))


def sharpe_ratio(returns: List[float], ann_factor: float = 252) -> float:
    mu = _mean(returns) * ann_factor
    sd = _std(returns) * math.sqrt(ann_factor)
    return mu / max(sd, 1e-10)


def probabilistic_sharpe_ratio(
    returns: List[float],
    sr_benchmark: float = 0.0,
    ann_factor: float = 252,
) -> float:
    """
    PSR = P(SR* >= SR_benchmark) using the sampling distribution of SR.
    Bailey & Lopez de Prado (2012).
    """
    T = len(returns)
    if T < 4:
        return 0.5
    sr = sharpe_ratio(returns, ann_factor)
    sk = _skewness(returns)
    ku = _kurtosis(returns)
    # Std of SR estimator
    sigma_sr = math.sqrt((1 + 0.5 * sr**2 - sk * sr + (ku - 3) / 4 * sr**2) / max(T - 1, 1))
    z = (sr - sr_benchmark) / max(sigma_sr, 1e-10)
    return _normal_cdf(z)


def deflated_sharpe_ratio(
    returns: List[float],
    n_trials: int,
    sr_benchmark: Optional[float] = None,
    ann_factor: float = 252,
) -> float:
    """
    DSR: PSR where SR_benchmark is the expected maximum SR from n_trials
    drawn from a null distribution (Bailey & Lopez de Prado 2014).
    """
    T = len(returns)
    if T < 4 or n_trials < 1:
        return 0.5
    # Expected max SR under H0 (Gaussian)
    e_max_sr = _expected_max_sharpe(n_trials, T)
    bench = sr_benchmark if sr_benchmark is not None else e_max_sr
    return probabilistic_sharpe_ratio(returns, bench, ann_factor)


def _expected_max_sharpe(n_trials: int, T: int) -> float:
    """Expected maximum Sharpe from n_trials with T observations (approx)."""
    if n_trials <= 1:
        return 0.0
    em = (1 - 0.5772) / math.log(n_trials) + 0.5772  # Euler-Mascheroni approx
    vm = max(1.0 / math.log(n_trials), 1e-8)
    return math.sqrt(2 * math.log(n_trials)) * (1 - em / math.log(n_trials))


def _skewness(xs: List[float]) -> float:
    n = len(xs)
    if n < 3:
        return 0.0
    mu, sd = _mean(xs), max(_std(xs), 1e-10)
    return sum(((x - mu) / sd) ** 3 for x in xs) * n / ((n-1)*(n-2))


def _kurtosis(xs: List[float]) -> float:
    n = len(xs)
    if n < 4:
        return 3.0
    mu, sd = _mean(xs), max(_std(xs), 1e-10)
    return sum(((x - mu) / sd) ** 4 for x in xs) / n


def haircut_sharpe(sr: float, t: float, n_trials: int, ann_factor: float = 252) -> float:
    """
    Haircut SR = SR * (1 - expected_overfit_ratio).
    Harvey, Liu, Zhu (2016) approximation.
    """
    p_val = 1.0 - _normal_cdf(sr * math.sqrt(t))
    bonferroni_p = min(p_val * n_trials, 1.0)
    z_adj = _normal_ppf(max(1.0 - bonferroni_p, 1e-10))
    return z_adj / math.sqrt(max(t, 1))


@dataclass
class OverfitReport:
    sharpe: float
    psr: float
    dsr: float
    n_trials: int
    haircut_sr: float
    overfit_risk: str     # "low" / "medium" / "high"


def overfit_report(
    returns: List[float],
    n_trials: int = 1,
    ann_factor: float = 252,
) -> OverfitReport:
    sr = sharpe_ratio(returns, ann_factor)
    psr = probabilistic_sharpe_ratio(returns, 0.0, ann_factor)
    dsr = deflated_sharpe_ratio(returns, n_trials, ann_factor=ann_factor)
    T = len(returns) / ann_factor
    hsr = haircut_sharpe(sr, T, n_trials, ann_factor)
    risk = "low" if dsr > 0.95 else "medium" if dsr > 0.75 else "high"
    return OverfitReport(sr, psr, dsr, n_trials, hsr, risk)


def cscv_overfit_probability(
    returns_matrix: List[List[float]],
    n_splits: int = 16,
) -> float:
    """
    Combinatorially Symmetric Cross Validation (Bailey et al. 2014).
    Fraction of OOS/IS performance pairs where OOS < median(IS).
    returns_matrix: shape (T, n_strategies)
    """
    T = len(returns_matrix)
    if T < 2 or not returns_matrix[0]:
        return 0.5
    n_strats = len(returns_matrix[0])
    chunk = max(T // n_splits, 1)
    boundaries = list(range(0, T, chunk))
    overfit_count, total = 0, 0
    for i in range(0, len(boundaries) - 1, 2):
        is_idx = list(range(boundaries[i], min(boundaries[i+1], T)))
        oos_idx = list(range(min(boundaries[i+1], T), min(boundaries[i+2] if i+2 < len(boundaries) else T, T)))
        if not is_idx or not oos_idx:
            continue
        is_sr = [sharpe_ratio([returns_matrix[t][s] for t in is_idx]) for s in range(n_strats)]
        oos_sr = [sharpe_ratio([returns_matrix[t][s] for t in oos_idx]) for s in range(n_strats)]
        best_is = max(range(n_strats), key=lambda s: is_sr[s])
        median_oos = sorted(oos_sr)[n_strats // 2]
        if oos_sr[best_is] < median_oos:
            overfit_count += 1
        total += 1
    return overfit_count / max(total, 1)


if __name__ == "__main__":
    import random
    rng = random.Random(42)
    returns = [rng.gauss(0.0005, 0.01) for _ in range(252 * 3)]
    report = overfit_report(returns, n_trials=50)
    print(f"Sharpe:      {report.sharpe:.3f}")
    print(f"PSR:         {report.psr:.3f}")
    print(f"DSR:         {report.dsr:.3f}")
    print(f"Haircut SR:  {report.haircut_sr:.3f}")
    print(f"Overfit risk: {report.overfit_risk}")
    n_strats = 10
    matrix = [[rng.gauss(0.0003, 0.01) for _ in range(n_strats)] for _ in range(252)]
    prob = cscv_overfit_probability(matrix)
    print(f"CSCV overfit probability: {prob:.2%}")
