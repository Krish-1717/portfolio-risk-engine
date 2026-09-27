"""
Kelly Criterion & Fractional Kelly Position Sizing
Day 16 â portfolio-risk-engine/portfolio/kelly_criterion.py

Implements full-Kelly and fractional-Kelly position sizing for
single assets and multi-asset portfolios, with turnover constraints.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Single-asset Kelly
# ---------------------------------------------------------------------------

def kelly_fraction(mu: float, sigma2: float) -> float:
    """
    Continuous-time Kelly fraction: f* = Î¼ / ÏÂ².
    Assumes log-normal returns with drift Î¼ and variance ÏÂ².
    """
    if sigma2 <= 0:
        return 0.0
    return mu / sigma2


def kelly_win_loss(p_win: float, win_return: float,
                   loss_return: float) -> float:
    """
    Discrete Kelly for binary bets:
    f* = p/|b| - q/a  where b = win_return, a = |loss_return|, q = 1-p.
    loss_return should be negative (e.g., -0.3).
    """
    if win_return <= 0 or loss_return >= 0:
        return 0.0
    p_loss = 1.0 - p_win
    f = p_win / win_return - p_loss / abs(loss_return)
    return max(f, 0.0)


def fractional_kelly(full_kelly: float, fraction: float = 0.5) -> float:
    """
    Fractional Kelly (common: half-Kelly).
    Reduces volatility at the cost of lower long-run growth.
    """
    return full_kelly * fraction


def kelly_growth_rate(f: float, mu: float, sigma2: float) -> float:
    """
    Expected log-growth rate under continuous Kelly:
    g(f) = f*Î¼ - 0.5*fÂ²*ÏÂ²
    """
    return f * mu - 0.5 * f ** 2 * sigma2


def optimal_leverage_with_cost(mu: float, sigma2: float,
                                margin_cost: float = 0.0,
                                max_leverage: float = 5.0) -> float:
    """Kelly with borrowing cost: f* = (Î¼ - c) / ÏÂ², capped at max_leverage."""
    if sigma2 <= 0:
        return 0.0
    f = (mu - margin_cost) / sigma2
    return max(0.0, min(f, max_leverage))


# ---------------------------------------------------------------------------
# Multi-asset Kelly (portfolio)
# ---------------------------------------------------------------------------

def _mat_vec(M: List[List[float]], v: List[float]) -> List[float]:
    return [sum(M[i][j] * v[j] for j in range(len(v))) for i in range(len(M))]


def _invert(A: List[List[float]]) -> List[List[float]]:
    n = len(A)
    aug = [A[i][:] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        aug[col], aug[pivot] = aug[pivot], aug[col]
        p = aug[col][col]
        if abs(p) < 1e-14:
            continue
        aug[col] = [x / p for x in aug[col]]
        for row in range(n):
            if row != col:
                f = aug[row][col]
                aug[row] = [aug[row][k] - f * aug[col][k] for k in range(2 * n)]
    return [row[n:] for row in aug]


def portfolio_kelly(mu: List[float],
                    sigma: List[List[float]],
                    fraction: float = 1.0) -> List[float]:
    """
    Multi-asset Kelly: f* = Î£^{-1} Î¼, scaled by fraction.
    Unconstrained (can be short/levered).
    """
    inv_sigma = _invert(sigma)
    f_star = _mat_vec(inv_sigma, mu)
    return [fraction * f for f in f_star]


def portfolio_kelly_constrained(mu: List[float],
                                 sigma: List[List[float]],
                                 fraction: float = 0.5,
                                 max_gross: float = 1.0,
                                 min_weight: float = 0.0) -> List[float]:
    """
    Constrained Kelly: long-only (min_weight=0) or long-short with gross limit.
    Uses iterative projection onto constraints.
    """
    raw = portfolio_kelly(mu, sigma, fraction)

    # Apply long-only floor
    weights = [max(w, min_weight) for w in raw]

    # Scale to meet gross leverage target
    gross = sum(abs(w) for w in weights)
    if gross > max_gross and gross > 1e-10:
        weights = [w * max_gross / gross for w in weights]

    return weights


# ---------------------------------------------------------------------------
# Growth optimal analysis
# ---------------------------------------------------------------------------

@dataclass
class KellyAnalysis:
    full_kelly: float
    half_kelly: float
    quarter_kelly: float
    growth_full: float
    growth_half: float
    growth_quarter: float
    ruin_probability_full: float   # approximate
    max_drawdown_estimate: float


def analyse_kelly(mu: float, sigma2: float,
                  horizon: float = 1.0) -> KellyAnalysis:
    """
    Full Kelly analysis for a single-asset strategy.
    """
    fk = kelly_fraction(mu, sigma2)
    hk = fractional_kelly(fk
    , 0.5)
    qk = fractional_kelly(fk, 0.25)

    gf = kelly_growth_rate(fk, mu, sigma2)
    gh = kelly_growth_rate(hk, mu, sigma2)
    gq = kelly_growth_rate(qk, mu, sigma2)

    # Approximate ruin probability (log-normal, bankrupt if wealth falls to 0)
    # For log-normal wealth with drift g and vol f*Ï:
    # P(ruin) â exp(-2g/(fÂ²ÏÂ²)) for the continuous case with absorbing barrier
    vol_fk = fk * math.sqrt(sigma2)
    if vol_fk > 1e-10 and gf > 0:
        ruin_prob = math.exp(-2 * gf / (vol_fk ** 2 + 1e-10))
    else:
        ruin_prob = 0.5

    # Expected maximum drawdown estimate: E[DD] â Ï_strategy / (2 * Sharpe)
    strategy_vol = abs(fk) * math.sqrt(sigma2 * horizon)
    sharpe = gf / (strategy_vol + 1e-10)
    max_dd = 0.5 * strategy_vol / (sharpe + 1e-10) if sharpe > 0 else 1.0

    return KellyAnalysis(
        full_kelly=fk,
        half_kelly=hk,
        quarter_kelly=qk,
        growth_full=gf,
        growth_half=gh,
        growth_quarter=gq,
        ruin_probability_full=ruin_prob,
        max_drawdown_estimate=min(max_dd, 1.0),
    )


# ---------------------------------------------------------------------------
# Kelly with estimation error (shrinkage)
# ---------------------------------------------------------------------------

def kelly_with_shrinkage(mu: List[float],
                          sigma: List[List[float]],
                          n_obs: int,
                          shrink_factor: Optional[float] = None) -> List[float]:
    """
    Adjust Kelly for parameter estimation uncertainty.
    Shrinks alpha towards zero by factor (1 - (n+1)/(n*T)).
    """
    n = len(mu)
    if shrink_factor is None:
        # James-Stein style: shrink more with fewer observations
        shrink_factor = max(0.0, 1.0 - (n + 2) / max(n_obs - n - 1, 1))

    mu_shrunk = [m * shrink_factor for m in mu]
    return portfolio_kelly(mu_shrunk, sigma, fraction=1.0)


# ---------------------------------------------------------------------------
# Betting schedule: Kelly for mean-reverting signal
# ---------------------------------------------------------------------------

def dynamic_kelly(signal: float, signal_vol: float,
                  alpha: float, sigma: float,
                  fraction: float = 0.5) -> float:
    """
    Signal-scaled Kelly: size proportional to signal strength.
    f = fraction * (alpha * signal) / ÏÂ²
    """
    mu_estimate = alpha * signal
    return fractional_kelly(kelly_fraction(mu_estimate, sigma ** 2), fraction)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Single asset: strategy with 15% expected return, 20% volatility
    mu, sigma2 = 0.15, 0.04

    analysis = analyse_kelly(mu, sigma2)
    print("=== Single-Asset Kelly Analysis ===")
    print(f"Full Kelly:   {analysis.full_kelly:.4f} ({analysis.full_kelly*100:.1f}% of capital)")
    print(f"Half Kelly:   {analysis.half_kelly:.4f}")
    print(f"Quarter Kelly: {analysis.quarter_kelly:.4f}")
    print(f"")
    print(f"Growth rate (full):    {analysis.growth_full:.4f}")
    print(f"Growth rate (half):    {analysis.growth_half:.4f}")
    print(f"Growth rate (quarter): {analysis.growth_quarter:.4f}")
    print(f"")
    print(f"Ruin probability (full Kelly): {analysis.ruin_probability_full:.4f}")
    print(f"Max drawdown estimate:         {analysis.max_drawdown_estimate:.4f}")

    # Multi-asset portfolio
    assets = ["US_EQ", "BONDS", "COMMOD"]
    mu_vec = [0.08, 0.03, 0.05]
    sigma_mat = [
        [0.04, 0.002, 0.005],
        [0.002, 0.01,  0.001],
        [0.005, 0.001, 0.09],
    ]

    print("\n=== Multi-Asset Kelly ===")
    weights_full = portfolio_kelly(mu_vec, sigma_mat, fraction=1.0)
    weights_half = portfolio_kelly(mu_vec, sigma_mat, fraction=0.5)
    weights_con  = portfolio_kelly_constrained(mu_vec, sigma_mat,
                                               fraction=0.5, max_gross=1.0,
                                               min_weight=0.0)

    print(f"{'Asset':<10} {'Full Kelly':>12} {'Half Kelly':>12} {'Constrained':>12}")
    for a, wf, wh, wc in zip(assets, weights_full, weights_half, weights_con):
        print(f"{a:<10} {wf:>12.4f} {wh:>12.4f} {wc:>12.4f}")

    # With estimation uncertainty
    weights_shrunk = kelly_with_shrinkage(mu_vec, sigma_mat, n_obs=120)
    print(f"\nKelly with shrinkage (120 obs): {[round(w, 4) for w in weights_shrunk]}")
