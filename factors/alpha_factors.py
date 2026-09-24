"""
factors/alpha_factors.py -- Alpha factor research library.
Consolidated from alpha-factor-research repo.
Momentum, value, quality, low-vol, and mean-reversion factors.
Pure Python stdlib -- no external dependencies.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


def _mean(xs: List[float]) -> float:
    return sum(xs) / max(len(xs), 1)

def _std(xs: List[float]) -> float:
    mu = _mean(xs)
    return math.sqrt(sum((x - mu)**2 for x in xs) / max(len(xs) - 1, 1))

def _rank(xs: List[float]) -> List[float]:
    """Cross-sectional rank, normalized to [0,1]."""
    n = len(xs)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: xs[i])
    ranks = [0.0] * n
    for r, i in enumerate(order):
        ranks[i] = r / max(n - 1, 1)
    return ranks

def _zscore(xs: List[float]) -> List[float]:
    mu, sd = _mean(xs), max(_std(xs), 1e-8)
    return [(x - mu) / sd for x in xs]


@dataclass
class FactorScores:
    asset_names: List[str]
    scores: Dict[str, List[float]]   # factor_name -> cross-sectional scores
    composite: List[float]           # equal-weight composite


# ---------------------------------------------------------------------------
# Individual factors
# ---------------------------------------------------------------------------

def momentum_factor(returns: List[List[float]], lookback: int = 252, skip: int = 21) -> List[float]:
    """12-1 month price momentum (Jegadeesh-Titman)."""
    T, n = len(returns), len(returns[0]) if returns else 0
    if T < lookback:
        return [0.0] * n
    window = returns[-(lookback):-skip] if skip > 0 else returns[-lookback:]
    cum = [sum(window[t][i] for t in range(len(window))) for i in range(n)]
    return _zscore(cum)


def short_term_reversal(returns: List[List[float]], lookback: int = 5) -> List[float]:
    """Short-term mean reversion (1-week return, sign-flipped)."""
    T, n = len(returns), len(returns[0]) if returns else 0
    if T < lookback:
        return [0.0] * n
    window = returns[-lookback:]
    cum = [sum(window[t][i] for t in range(len(window))) for i in range(n)]
    return _zscore([-c for c in cum])  # flip sign


def low_volatility_factor(returns: List[List[float]], lookback: int = 63) -> List[float]:
    """Low-volatility anomaly: lower vol -> higher score."""
    T, n = len(returns), len(returns[0]) if returns else 0
    if T < lookback or n == 0:
        return [0.0] * n
    window = returns[-lookback:]
    vols = []
    for i in range(n):
        col = [window[t][i] for t in range(len(window))]
        vols.append(_std(col))
    return _zscore([-v for v in vols])  # flip: low vol = high score


def quality_factor(
    roe: List[float],
    debt_to_equity: List[float],
    earnings_stability: List[float],
) -> List[float]:
    """
    Quality composite: high ROE + low D/E + stable earnings.
    All inputs are cross-sectional vectors.
    """
    n = len(roe)
    z_roe = _zscore(roe)
    z_de  = _zscore([-d for d in debt_to_equity])
    z_es  = _zscore(earnings_stability)
    return [(z_roe[i] + z_de[i] + z_es[i]) / 3.0 for i in range(n)]


def value_factor(
    book_to_price: List[float],
    earnings_yield: List[float],
) -> List[float]:
    """Simple value composite: book/price + earnings yield."""
    z_bp = _zscore(book_to_price)
    z_ey = _zscore(earnings_yield)
    return [(a + b) / 2 for a, b in zip(z_bp, z_ey)]


def size_factor(market_caps: List[float]) -> List[float]:
    """Small-minus-big: lower market cap -> higher score."""
    return _zscore([-mc for mc in market_caps])


# ---------------------------------------------------------------------------
# Composite factor builder
# ---------------------------------------------------------------------------

def build_composite_scores(
    asset_names: List[str],
    factor_scores: Dict[str, List[float]],
    weights: Optional[Dict[str, float]] = None,
) -> FactorScores:
    """
    Combine multiple factor score vectors into a weighted composite.

    Args:
        asset_names:   list of asset identifiers
        factor_scores: dict of factor_name -> z-score vector
        weights:       optional dict of factor_name -> weight (equal-weight if None)
    """
    n = len(asset_names)
    factors = list(factor_scores.keys())
    if weights is None:
        weights = {f: 1.0 / len(factors) for f in factors}
    total_w = sum(weights.get(f, 0) for f in factors)

    composite = [0.0] * n
    for f in factors:
        w = weights.get(f, 0) / max(total_w, 1e-10)
        scores = factor_scores[f]
        for i in range(n):
            composite[i] += w * scores[i]

    return FactorScores(
        asset_names=asset_names,
        scores=factor_scores,
        composite=composite,
    )


def factor_ic(scores: List[float], forward_returns: List[float]) -> float:
    """Information coefficient: Spearman rank correlation between scores and fwd returns."""
    n = len(scores)
    rs = _rank(scores)
    rr = _rank(forward_returns)
    cov = sum((rs[i] - 0.5) * (rr[i] - 0.5) for i in range(n)) / max(n, 1)
    var_s = sum((r - 0.5)**2 for r in rs) / max(n, 1)
    var_r = sum((r - 0.5)**2 for r in rr) / max(n, 1)
    return cov / max(math.sqrt(var_s * var_r), 1e-10)


def top_n_portfolio(composite: List[float], asset_names: List[str], n: int = 10) -> Dict[str, float]:
    """Equal-weight long portfolio from top-n factor scores."""
    ranked = sorted(range(len(composite)), key=lambda i: composite[i], reverse=True)
    top = ranked[:n]
    w = 1.0 / len(top)
    return {asset_names[i]: w for i in top}


if __name__ == "__main__":
    import random
    rng = random.Random(99)
    names = [f"STK_{i:02d}" for i in range(20)]
    T, n = 300, 20
    rets = [[rng.gauss(0.0003, 0.015) for _ in range(n)] for _ in range(T)]
    mom = momentum_factor(rets, lookback=252, skip=21)
    lvol = low_volatility_factor(rets, lookback=63)
    rev = short_term_reversal(rets, lookback=5)
    composite = build_composite_scores(names, {"momentum": mom, "low_vol": lvol, "reversal": rev})
    port = top_n_portfolio(composite.composite, names, n=5)
    print("Top-5 composite portfolio:")
    for sym, w in port.items():
        print(f"  {sym}: {w:.1%}")
    fwd = [rng.gauss(0.001, 0.02) for _ in range(n)]
    ic = factor_ic(composite.composite, fwd)
    print(f"IC (composite vs 1-period fwd): {ic:.4f}")
