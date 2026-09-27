"""
Variance Swap Pricing and Replication
Day 16 â portfolio-risk-engine/derivatives/variance_swap.py

Implements:
  - Model-free variance swap strike (Demeterfi et al. 1999)
  - Log-contract replication via options strip
  - Realised variance calculation
  - P&L calculation for long/short variance positions
  - VIX-style volatility index construction
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Black-Scholes helpers
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


def _bs_call(F: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(F - K, 0.0) * math.exp(-r * T)
    d1 = (math.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return math.exp(-r * T) * (F * _norm_cdf(d1) - K * _norm_cdf(d2))


def _bs_put(F: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(K - F, 0.0) * math.exp(-r * T)
    d1 = (math.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return math.exp(-r * T) * (K * _norm_cdf(-d2) - F * _norm_cdf(-d1))


# ---------------------------------------------------------------------------
# Variance swap data structures
# ---------------------------------------------------------------------------

@dataclass
class VarianceSwap:
    """A variance swap contract."""
    notional: float          # variance notional ($/vol pointÂ²)
    strike: float            # K_var = variance strike (annualised, e.g., 0.04 = 20% vol)
    T: float                 # time to expiry (years)
    vega_notional: float = 0.0  # vega notional (derived)

    def __post_init__(self):
        # Convert variance notional to vega notional
        # Vega notional = notional / (2 * K_vol) where K_vol = sqrt(K_var)
        K_vol = math.sqrt(max(self.strike, 1e-8))
        self.vega_notional = self.notional / (2 * K_vol)


@dataclass
class VarianceSwapResult:
    """Variance swap pricing output."""
    strike_variance: float       # K_var (annualised variance)
    strike_vol: float            # K_vol = sqrt(K_var)
    fair_value: float            # approximate fair value
    replication_integral: float  # numerical replication value
    vix_style_vol: float         # VIX-style annualised vol


# ---------------------------------------------------------------------------
# Model-free variance swap strike â strip of options
# ---------------------------------------------------------------------------

def variance_swap_strike(F: float, T: float, r: float,
                          calls: List[Tuple[float, float]],   # (K, price) K > F
                          puts: List[Tuple[float, float]],    # (K, price) K < F
                          K0: Optional[float] = None) -> VarianceSwapResult:
    """
    Model-free variance swap strike via Demeterfi et al. (1999) replication.

    ÏÂ²_K = (2/T) * [Î£ ÎK/KÂ² * e^{rT} * P(K)  for K â¤ K0
                   + Î£ ÎK/KÂ² * e^{rT} * C(K)  for K > K0]

    F: forward price
    T: time to maturity
    r: risk-free rate
    calls: list of (strike, call_price) for K > F
    puts:  list of (strike, put_price)  for K < F
    K0: ATM-ish pivot strike (default: F)
    """
    K0 = K0 or F
    discount = math.exp(r * T)
    integral = 0.0

    # Put contribution (K < K0)
    if puts:
        sorted_puts = sorted(puts, key=lambda x: x[0])
        for i, (K, P) in enumerate(sorted_puts):
            # Trapezoid width
            if i == 0:
                dK = sorted_puts[1][0] - K if len(sorted_puts) > 1 else K * 0.01
            elif i == len(sorted_puts) - 1:
                dK = K - sorted_puts[i - 1][0]
            else:
                dK = (sorted_puts[i + 1][0] - sorted_puts[i - 1][0]) / 2
            integral += (dK / K ** 2) * discount * P

    # Call contribution (K > K0)
    if calls:
        sorted_calls = sorted(calls, key=lambda x: x[0])
        for i, (K, C) in enumerate(sorted_calls):
            if i == 0:
                dK = sorted_calls[1][0] - K if len(sorted_calls) > 1 else K * 0.01
            elif i == len(sorted_calls) - 1:
                dK = K - sorted_calls[i - 1][0]
            else:
                dK = (sorted_calls[i + 1][0] - sorted_calls[i - 1][0]) / 2
            integral += (dK / K ** 2) * discount * C

    # Subtract the forward term
    fwd_term = (F / K0 - 1) - math.log(F / K0)
    variance_strike = (2 / T) * (integral - fwd_term)
    variance_strike = max(variance_strike, 1e-8)

    # VIX-style: ÏÂ² â (2/T) * Î£ ...
    vix_vol = math.sqrt(variance_strike)

    fair_value = 0.0  # at inception

    return VarianceSwapResult(
        strike_variance=variance_strike,
        strike_vol=vix_vol,
        fair_value=fair_value,
        replication_integral=integral,
        vix_style_vol=vix_vol,
    )


def variance_swap_strike_from_vol_surface(
        F: float, T: float, r: float,
        strikes: List[float],
        implied_vols: List[float],
        n_put_strikes: int = 20) -> VarianceSwapResult:
    """
    Compute variance swap strike from an implied vol surface via numerical integration.
    Generates BS prices at each strike and integrates.
    """
    K0 = F  # ATM pivot

    puts = [(K, _bs_put(F, K, T, r, iv))
            for K, iv in zip(strikes, implied_vols) if K <= K0]
    calls = [(K, _bs_call(F, K, T, r, iv))
             for K, iv in zip(strikes, implied_vols) if K > K0]

    return variance_swap_strike(F, T, r, calls, puts, K0)


# ---------------------------------------------------------------------------
# Realised variance
# ---------------------------------------------------------------------------

def realised_variance(prices: List[float], ann_factor: float = 252) -> float:
    """
    Compute annualised realised variance from a price series.
    Uses log-returns: R_i = ln(S_i / S_{i-1}).
    """
    if len(prices) < 2:
        return 0.0
    log_rets = [math.log(prices[i] / prices[i - 1])
                for i in range(1, len(prices))]
    n = len(log_rets)
    mu = sum(log_rets) / n
    var = sum((r - mu) ** 2 for r in log_rets) / n
    return var * ann_factor


def realised_vol(prices: List[float], ann_factor: float = 252) -> float:
    return math.sqrt(realised_variance(prices, ann_factor))


# ---------------------------------------------------------------------------
# Variance swap P&L
# ---------------------------------------------------------------------------

def variance_swap_pnl(swap: VarianceSwap,
                       realised_var: float,
                       long: bool = True) -> float:
    """
    Payoff of a variance swap:
    Payoff = Notional Ã (ÏÂ²_realised â K_var)  [long]
    """
    payoff = swap.notional * (realised_var - swap.strike)
    return payoff if long else -payoff


def vega_pnl(swap: VarianceSwap,
              realised_vol_val: float,
              long: bool = True) -> float:
    """
    Approximate P&L in vega terms:
    â Vega Notional Ã (Ï_realised â K_vol)
    """
    K_vol = math.sqrt(max(swap.strike, 1e-10))
    payoff = swap.vega_notional * (realised_vol_val - K_vol)
    return payoff if long else -payoff


# ---------------------------------------------------------------------------
# VIX-style index construction
# ---------------------------------------------------------------------------

def vix_index(near_calls: List[Tuple[float, float]],
               near_puts: List[Tuple[float, float]],
               next_calls: List[Tuple[float, float]],
               next_puts: List[Tuple[float, float]],
               F_near: float, F_next: float,
               T1: float, T2: float,
               r1: float, r2: float) -> float:
    """
    CBOE VIX-style calculation:
    Blend two maturities T1, T2 to target 30-day constant maturity.
    VIX = 100 Ã sqrt([(T1 Ï1Â² w1 + T2 Ï2Â² w2) Ã (N30/N365)])
    """
    res1 = variance_swap_strike(F_near, T1, r1, near_calls, near_puts)
    res2 = variance_swap_strike(F_next, T2, r2, next_calls, next_puts)

    sigma2_1 = res1.strike_variance
    sigma2_2 = res2.strike_variance

    N30 = 30 / 365
    NT1 = T1
    NT2 = T2

    # Interpolation weight
    if abs(NT2 - NT1) < 1e-10:
        w1, w2 = 0.5, 0.5
    else:
        w1 = (NT2 - N30) / (NT2 - NT1)
        w2 = (N30 - NT1) / (NT2 - NT1)
        w1 = max(0.0, min(1.0, w1))
        w2 = 1.0 - w1

    vix2 = (NT1 * sigma2_1 * w1 + NT2 * sigma2_2 * w2) * (365 / 30)
    return 100.0 * math.sqrt(max(vix2, 0.0))


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random
    rng = random.Random(42)

    F, T, r = 100.0, 1.0 / 12, 0.05  # 1-month

    # Simulate vol smile
    def smile_iv(K: float) -> float:
        moneyness = math.log(K / F)
        return 0.20 + 0.05 * moneyness ** 2 - 0.02 * moneyness

    strikes = list(range(80, 125, 5))
    ivols = [smile_iv(K) for K in strikes]

    result = variance_swap_strike_from_vol_surface(F, T, r, strikes, ivols)

    print(f"Variance swap strike: {result.strike_variance:.6f}")
    print(f"Implied vol strike:   {result.strike_vol:.4%}")
    print(f"VIX-style vol:        {result.vix_style_vol:.4%}")

    # P&L scenario
    swap = VarianceSwap(notional=1_000_000, strike=result.strike_variance, T=T)
    print(f"\nVariance notional: ${swap.notional:,.0f}")
    print(f"Vega notional:     ${swap.vega_notional:,.0f}")

    # Simulate realised path
    prices = [F]
    for _ in range(21):
        prices.append(prices[-1] * math.exp(rng.gauss(-0.0001, 0.015)))

    rv = realised_variance(prices)
    rvol = math.sqrt(rv)

    pnl = variance_swap_pnl(swap, rv, long=True)
    vega_pnl_val = vega_pnl(swap, rvol, long=True)

    print(f"\nRealised variance: {rv:.6f}  (vol: {rvol:.4%})")
    print(f"Variance P&L (long): ${pnl:,.0f}")
    print(f"Vega-approx P&L:     ${vega_pnl_val:,.0f}")
