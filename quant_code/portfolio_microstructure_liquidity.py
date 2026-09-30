"""
portfolio_microstructure_liquidity.py
Microstructure: Liquidity Metrics for portfolio-risk-engine.
Amihud illiquidity ratio, Roll implicit spread, Corwin-Schultz H/L estimator,
Hasbrouck lambda, L-VaR (liquidity-adjusted VaR).
Pure Python stdlib only.
"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def mean(xs: list[float]) -> float:
    return sum(xs) / max(len(xs), 1)

def variance(xs: list[float]) -> float:
    m = mean(xs)
    return sum((x - m)**2 for x in xs) / max(len(xs) - 1, 1)

def std(xs: list[float]) -> float:
    return math.sqrt(max(variance(xs), 0.0))

# ---------------------------------------------------------------------------
# 1. Amihud Illiquidity Ratio (Amihud 2002)
# ---------------------------------------------------------------------------
def amihud_illiquidity(returns: list[float], dollar_volumes: list[float]) -> float:
    """
    ILLIQ = (1/T) * sum(|r_t| / DollarVolume_t)
    Units: percent return per dollar traded (x1e6 for millions).
    Higher ILLIQ → less liquid.
    """
    T = min(len(returns), len(dollar_volumes))
    ratios = [abs(returns[t]) / max(dollar_volumes[t], 1.0)
              for t in range(T) if dollar_volumes[t] > 0]
    return mean(ratios) if ratios else float('nan')

def amihud_normalized(returns: list[float], dollar_volumes: list[float],
                       scale: float = 1e6) -> float:
    """Scale ILLIQ by 1e6 to report in pct per $1M traded."""
    return amihud_illiquidity(returns, dollar_volumes) * scale

def rolling_amihud(returns: list[float], dollar_volumes: list[float],
                    window: int = 21) -> list[float]:
    """Rolling Amihud ratio."""
    T = min(len(returns), len(dollar_volumes))
    out = []
    for i in range(window - 1, T):
        r_w = returns[i - window + 1:i + 1]
        v_w = dollar_volumes[i - window + 1:i + 1]
        out.append(amihud_illiquidity(r_w, v_w))
    return out

# ---------------------------------------------------------------------------
# 2. Roll Implicit Spread (Roll 1984)
# ---------------------------------------------------------------------------
def roll_spread(prices: list[float]) -> float:
    """
    Roll (1984): s = 2 * sqrt(-Cov(Δp_t, Δp_{t-1}))
    If serial covariance is positive (unusual), returns 0.
    """
    diffs = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    if len(diffs) < 2:
        return 0.0
    m = mean(diffs)
    cov = sum((diffs[i] - m) * (diffs[i-1] - m) for i in range(1, len(diffs))) / max(len(diffs) - 2, 1)
    if cov >= 0:
        return 0.0
    return 2 * math.sqrt(-cov)

def roll_effective_spread(prices: list[float]) -> dict:
    """Returns Roll spread and percent spread relative to mid-price."""
    spread = roll_spread(prices)
    avg_price = mean(prices)
    pct_spread = spread / avg_price if avg_price > 0 else 0.0
    return {
        'roll_spread_dollars': spread,
        'roll_spread_pct': pct_spread,
        'avg_price': avg_price,
    }

# ---------------------------------------------------------------------------
# 3. Corwin-Schultz High-Low Spread Estimator (2012)
# ---------------------------------------------------------------------------
def corwin_schultz_spread(highs: list[float], lows: list[float]) -> list[float]:
    """
    CS2012: daily spread estimate from consecutive H/L pairs.
    Returns per-period spread estimates (positive values).
    """
    T = min(len(highs), len(lows))
    spreads = []

    for t in range(1, T):
        H1, L1 = highs[t], lows[t]
        H0, L0 = highs[t-1], lows[t-1]

        if H1 <= 0 or L1 <= 0 or H0 <= 0 or L0 <= 0:
            spreads.append(float('nan'))
            continue

        # Beta = sum over two days of (log H/L)^2
        beta = (math.log(H1/L1))**2 + (math.log(H0/L0))**2

        # Gamma = (log(max(H1,H0) / min(L1,L0)))^2
        gamma = (math.log(max(H1, H0) / min(L1, L0)))**2

        # Alpha = (sqrt(2*beta) - sqrt(beta)) / (3 - 2*sqrt(2)) - sqrt(gamma/(3 - 2*sqrt(2)))
        root2 = math.sqrt(2)
        denom = 3 - 2 * root2
        alpha = (math.sqrt(2 * beta) - math.sqrt(beta)) / denom - math.sqrt(gamma / denom)

        # Spread = 2*(exp(alpha) - 1) / (1 + exp(alpha))
        if alpha > 10 or alpha < -10:
            spreads.append(0.0)
            continue
        exp_alpha = math.exp(alpha)
        spread = 2 * (exp_alpha - 1) / (1 + exp_alpha)
        spreads.append(max(spread, 0.0))  # non-negative constraint

    return spreads

def cs_mean_spread(highs: list[float], lows: list[float]) -> float:
    """Mean Corwin-Schultz spread (ignoring NaN)."""
    spreads = [s for s in corwin_schultz_spread(highs, lows) if not math.isnan(s)]
    return mean(spreads) if spreads else 0.0

# ---------------------------------------------------------------------------
# 4. Hasbrouck Lambda (price impact)
# ---------------------------------------------------------------------------
def hasbrouck_lambda(returns: list[float], signed_volumes: list[float]) -> float:
    """
    Price impact coefficient from Kyle model:
    r_t = lambda * signed_vol_t + epsilon_t
    OLS: lambda = cov(r, sv) / var(sv)
    Higher lambda → greater price impact per unit volume.
    """
    n = min(len(returns), len(signed_volumes))
    r, sv = returns[:n], signed_volumes[:n]
    mr, msv = mean(r), mean(sv)
    cov = sum((r[i] - mr) * (sv[i] - msv) for i in range(n)) / max(n - 1, 1)
    var_sv = variance(sv)
    return cov / max(var_sv, 1e-10)

# ---------------------------------------------------------------------------
# 5. Liquidity-Adjusted VaR (L-VaR)
# ---------------------------------------------------------------------------
def _norm_ppf(p: float) -> float:
    """Approximate inverse normal CDF via bisection."""
    p = max(1e-9, min(1 - 1e-9, p))
    norm_cdf = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
    lo, hi = -10.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2

def liquidity_adjusted_var(portfolio_value: float,
                            sigma_return: float,
                            bid_ask_spread: float,
                            confidence: float = 0.99,
                            horizon_days: int = 1,
                            liquidation_days: int = 5,
                            n_shares: float = 1.0) -> dict:
    """
    L-VaR = Market VaR + Liquidity Cost
    Bangia et al. (1999) approach:
    LiqCost = 0.5 * spread * portfolio_value
    Extended to multi-day horizon via sqrt-of-time scaling.
    """
    z = _norm_ppf(confidence)

    # Market risk VaR (normal approximation)
    market_var = portfolio_value * sigma_return * math.sqrt(horizon_days) * z

    # Liquidity spread cost: half-spread applied at liquidation
    # Amortized over liquidation horizon
    spread_cost_per_trade = 0.5 * bid_ask_spread * portfolio_value

    # Exogenous spread component (Bangia): 0.5 * (mean spread + z * std_spread)
    # We use just the mean spread here (std_spread = 0 simplification)
    exogenous_liquidity_cost = 0.5 * bid_ask_spread * portfolio_value

    # Endogenous (market impact): assumed square-root impact over liquidation horizon
    # Simple approximation: additional 0.1 * spread per extra day of unwinding
    endogenous_extra = 0.1 * bid_ask_spread * portfolio_value * max(liquidation_days - 1, 0)

    total_liquidity_cost = exogenous_liquidity_cost + endogenous_extra
    lvar = market_var + total_liquidity_cost

    return {
        'portfolio_value': portfolio_value,
        'confidence': confidence,
        'market_var': market_var,
        'liquidity_cost': total_liquidity_cost,
        'l_var': lvar,
        'liquidity_premium_pct': total_liquidity_cost / max(market_var, 1e-10),
        'bid_ask_spread': bid_ask_spread,
    }

def portfolio_lvар(assets: list[dict], weights: list[float],
                    corr_matrix: list[list[float]],
                    portfolio_value: float,
                    confidence: float = 0.99) -> dict:
    """
    Portfolio L-VaR accounting for correlation across assets.
    Each asset: {'sigma': float, 'spread': float, 'name': str}
    """
    n = len(assets)

    # Portfolio sigma (correlated)
    sigmas = [a['sigma'] for a in assets]
    port_var = sum(
        weights[i] * weights[j] * sigmas[i] * sigmas[j] * corr_matrix[i][j]
        for i in range(n) for j in range(n)
    )
    port_sigma = math.sqrt(max(port_var, 0.0))

    z = _norm_ppf(confidence)
    market_var = portfolio_value * port_sigma * z

    # Weighted average spread cost
    avg_spread = sum(weights[i] * assets[i]['spread'] for i in range(n))
    liquidity_cost = 0.5 * avg_spread * portfolio_value

    return {
        'portfolio_sigma': port_sigma,
        'market_var': market_var,
        'liquidity_cost': liquidity_cost,
        'l_var': market_var + liquidity_cost,
        'weighted_spread': avg_spread,
    }

# ---------------------------------------------------------------------------
# 6. Liquidity score (composite)
# ---------------------------------------------------------------------------
def composite_liquidity_score(amihud: float, roll_pct: float, cs_spread: float,
                               has_lambda: float,
                               w: Optional[list[float]] = None) -> float:
    """
    Composite liquidity score ∈ [0, 1] where 1 = most liquid.
    Uses user-provided weights (or equal weights by default).
    Each metric is inverted and normalized.
    """
    if w is None:
        w = [0.25, 0.25, 0.25, 0.25]

    # Normalize each metric (lower value = more liquid = higher score)
    # We use simple ratio: score_i = 1 / (1 + metric_i * scale)
    scores = [
        1 / (1 + amihud * 1e4),     # Amihud (per $1M)
        1 / (1 + roll_pct * 100),    # Roll pct spread
        1 / (1 + cs_spread * 100),   # CS pct spread
        1 / (1 + has_lambda * 1e5),  # Hasbrouck lambda
    ]
    return sum(w[i] * scores[i] for i in range(4))

# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 65)
    print("MICROSTRUCTURE: Liquidity Metrics")
    print("=" * 65)

    rng = random.Random(42)
    T = 252

    def randn():
        u = max(rng.random(), 1e-15)
        return math.sqrt(-2*math.log(u)) * math.cos(2*math.pi*rng.random())

    # Simulate price series and volumes for a liquid vs illiquid stock
    def simulate_stock(mu=0.0002, sigma=0.01, spread_pct=0.001, avg_vol=1e6):
        prices = [100.0]
        returns, highs, lows, volumes, signed_vols = [], [], [], [], []
        for _ in range(T):
            r = mu + sigma * randn()
            returns.append(r)
            p = prices[-1] * (1 + r)
            prices.append(p)
            # H/L as price ± intraday range
            intraday_range = abs(sigma * randn()) * p
            highs.append(p + intraday_range * 0.5)
            lows.append(max(p - intraday_range * 0.5, 1e-3))
            # Volume proportional to volatility (empirical stylized fact)
            v = avg_vol * (1 + abs(randn()) * 0.5)
            volumes.append(v)
            dollar_vol = v * p
            signed_vols.append(v * (1 if r > 0 else -1))
        return prices[1:], returns, highs, lows, volumes, signed_vols

    print("\n[Liquid Stock: AAPL-like]")
    p1, r1, h1, l1, vol1, sv1 = simulate_stock(spread_pct=0.0005, avg_vol=5e6)
    dv1 = [v * p for v, p in zip(vol1, p1)]

    amihud1 = amihud_normalized(r1, dv1)
    roll1 = roll_effective_spread(p1)
    cs1 = cs_mean_spread(h1, l1)
    has1 = hasbrouck_lambda(r1, sv1)
    liq_score1 = composite_liquidity_score(amihud1/1e6, roll1['roll_spread_pct'], cs1, has1)

    print(f"  Amihud ILLIQ (per $1M): {amihud1:.4f}")
    print(f"  Roll Spread: ${roll1['roll_spread_dollars']:.4f} ({roll1['roll_spread_pct']:.4%})")
    print(f"  Corwin-Schultz Spread: {cs1:.4%}")
    print(f"  Hasbrouck Lambda: {has1:.2e}")
    print(f"  Composite Liquidity Score: {liq_score1:.4f}")

    print("\n[Illiquid Stock: small-cap]")
    p2, r2, h2, l2, vol2, sv2 = simulate_stock(sigma=0.02, spread_pct=0.01, avg_vol=1e4)
    dv2 = [v * p for v, p in zip(vol2, p2)]

    amihud2 = amihud_normalized(r2, dv2)
    roll2 = roll_effective_spread(p2)
    cs2 = cs_mean_spread(h2, l2)
    has2 = hasbrouck_lambda(r2, sv2)
    liq_score2 = composite_liquidity_score(amihud2/1e6, roll2['roll_spread_pct'], cs2, has2)

    print(f"  Amihud ILLIQ (per $1M): {amihud2:.4f}")
    print(f"  Roll Spread: ${roll2['roll_spread_dollars']:.4f} ({roll2['roll_spread_pct']:.4%})")
    print(f"  Corwin-Schultz Spread: {cs2:.4%}")
    print(f"  Hasbrouck Lambda: {has2:.2e}")
    print(f"  Composite Liquidity Score: {liq_score2:.4f}")

    print("\n[L-VaR Comparison]")
    for name, sigma, spread in [('Liquid', 0.01, 0.0005), ('Illiquid', 0.02, 0.01)]:
        lvар = liquidity_adjusted_var(
            portfolio_value=1_000_000, sigma_return=sigma,
            bid_ask_spread=spread, confidence=0.99,
            horizon_days=1, liquidation_days=5
        )
        print(f"  {name:10s}: Market VaR=${lvар['market_var']:>10,.0f}  "
              f"Liq Cost=${lvар['liquidity_cost']:>8,.0f}  "
              f"L-VaR=${lvар['l_var']:>10,.0f}  "
              f"Liq Premium={lvар['liquidity_premium_pct']:.2%}")

    print("\n[Done] Microstructure: Liquidity Metrics complete.")
