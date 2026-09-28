"""
Liquidity Risk â L-VaR, Market Impact, and Optimal Liquidation
Day 17 â portfolio-risk-engine/risk/liquidity_risk.py

Implements:
  - Liquidity-adjusted VaR (L-VaR) with bid-ask spread
  - Roll (1984) implicit spread estimator
  - Kyle (1985) lambda â price impact coefficient
  - Almgren-Chriss (2000) linear market impact model
  - Optimal liquidation schedule (TWAP vs optimal)
  - Liquidation cost under various urgency levels
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Normal distribution helper
# ---------------------------------------------------------------------------

def _norm_ppf(p: float) -> float:
    """Beasley-Springer-Moro approximation to normal quantile."""
    if p <= 0:
        return -math.inf
    if p >= 1:
        return math.inf
    if p < 0.5:
        sign, q = -1, p
    else:
        sign, q = 1, 1 - p
    t = math.sqrt(-2 * math.log(q))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    x = t - (c0 + c1 * t + c2 * t ** 2) / (1 + d1 * t + d2 * t ** 2 + d3 * t ** 3)
    return sign * x


# ---------------------------------------------------------------------------
# Bid-ask spread estimation
# ---------------------------------------------------------------------------

def roll_spread(prices: List[float]) -> float:
    """
    Roll (1984) implicit bid-ask spread estimator.
    spread = 2 * sqrt(-cov(dP_t, dP_{t-1})) if cov < 0, else 0.
    Estimates the effective bid-ask spread from transaction prices.
    """
    if len(prices) < 3:
        return 0.0
    diffs = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    n = len(diffs) - 1
    if n < 1:
        return 0.0
    mu = sum(diffs) / len(diffs)
    cov = sum((diffs[i] - mu) * (diffs[i - 1] - mu) for i in range(1, len(diffs))) / n
    if cov >= 0:
        return 0.0
    return 2 * math.sqrt(-cov)


def relative_spread(bid: float, ask: float) -> float:
    """Relative (percentage) bid-ask spread."""
    mid = (bid + ask) / 2
    return (ask - bid) / mid if mid > 0 else 0.0


# ---------------------------------------------------------------------------
# Kyle lambda â price impact
# ---------------------------------------------------------------------------

def kyle_lambda(price_changes: List[float], order_flows: List[float]) -> float:
    """
    Kyle (1985) lambda: price impact per unit of order flow.
    lambda = cov(delta_p, x) / var(x)
    where x = signed order flow (+ = buy, - = sell).
    """
    if len(price_changes) != len(order_flows) or len(price_changes) < 2:
        return 0.0
    n = len(price_changes)
    mu_dp = sum(price_changes) / n
    mu_x = sum(order_flows) / n
    cov = sum((price_changes[i] - mu_dp) * (order_flows[i] - mu_x) for i in range(n)) / n
    var_x = sum((order_flows[i] - mu_x) ** 2 for i in range(n)) / n
    return cov / var_x if var_x > 1e-12 else 0.0


# ---------------------------------------------------------------------------
# Almgren-Chriss market impact model
# ---------------------------------------------------------------------------

@dataclass
class AlmgrenChrissParams:
    """
    Parameters for the Almgren-Chriss (2000) market impact model.
    Temporary impact: g(v) = eta * v   (linear)
    Permanent impact: h(v) = gamma * v (linear)
    where v = trading rate (shares per unit time).
    """
    sigma: float           # daily return volatility
    eta: float = 0.01      # temporary impact coefficient ($ per share per share/day)
    gamma: float = 0.001   # permanent impact coefficient ($ per share per share/day)
    risk_aversion: float = 1e-6   # lambda â trader's risk aversion

    @property
    def kappa_sq(self) -> float:
        """ÎºÂ² = risk_aversion * sigmaÂ² / eta"""
        return self.risk_aversion * self.sigma ** 2 / self.eta

    @property
    def kappa(self) -> float:
        return math.sqrt(max(self.kappa_sq, 1e-12))


@dataclass
class LiquidationSchedule:
    """Optimal liquidation trajectory."""
    times: List[float]
    holdings: List[float]          # shares remaining at each time
    trade_list: List[float]        # shares sold in each interval
    expected_cost: float           # E[total cost]
    variance_cost: float           # Var[total cost]
    impl_shortfall: float          # implementation shortfall (mean)


def optimal_liquidation(
        X: float,           # initial position (shares)
        T: float,           # liquidation horizon (days)
        n_steps: int,
        params: AlmgrenChrissParams,
        price: float = 1.0,
) -> LiquidationSchedule:
    """
    Almgren-Chriss (2000) optimal liquidation schedule.
    Minimises E[cost] + lambda * Var[cost].
    Closed-form solution: x(t) = X * sinh(kappa*(T-t)) / sinh(kappa*T)
    """
    tau = T / n_steps
    kappa = params.kappa
    sinh_kT = math.sinh(kappa * T) if kappa * T < 700 else 1e300

    times = [i * tau for i in range(n_steps + 1)]
    holdings = []
    for t in times:
        if kappa < 1e-8:  # TWAP limit
            x_t = X * (1 - t / T)
        else:
            num = math.sinh(kappa * (T - t))
            holdings_t = X * (num / sinh_kT if abs(sinh_kT) > 1e-10 else 0.0)
            x_t = max(holdings_t, 0.0)
        holdings.append(x_t)

    trade_list = [holdings[i] - holdings[i + 1] for i in range(n_steps)]

    # Expected cost = permanent impact + temporary impact
    perm_cost = params.gamma * X ** 2 / 2  # total permanent impact
    temp_cost = params.eta / tau * sum(n ** 2 for n in trade_list)  # temporary impact
    exp_cost = (perm_cost + temp_cost) * price

    # Variance of cost
    var_cost = params.sigma ** 2 * tau * sum(h ** 2 for h in holdings[:-1]) * price ** 2

    # Implementation shortfall (simplified)
    impl_shortfall = exp_cost / (X * price) if X > 0 else 0.0

    return LiquidationSchedule(
        times=times,
        holdings=holdings,
        trade_list=trade_list,
        expected_cost=exp_cost,
        variance_cost=var_cost,
        impl_shortfall=impl_shortfall,
    )


def twap_liquidation(X: float, T: float, n_steps: int,
                     params: AlmgrenChrissParams,
                     price: float = 1.0) -> LiquidationSchedule:
    """Time-Weighted Average Price (TWAP) equal-slice liquidation."""
    tau = T / n_steps
    trade = X / n_steps
    times = [i * tau for i in range(n_steps + 1)]
    holdings = [X - i * trade for i in range(n_steps + 1)]
    trade_list = [trade] * n_steps

    perm_cost = params.gamma * X ** 2 / 2
    temp_cost = params.eta / tau * n_steps * trade ** 2
    exp_cost = (perm_cost + temp_cost) * price

    var_cost = params.sigma ** 2 * tau * sum(h ** 2 for h in holdings[:-1]) * price ** 2
    impl_shortfall = exp_cost / (X * price) if X > 0 else 0.0

    return LiquidationSchedule(
        times=times, holdings=holdings, trade_list=trade_list,
        expected_cost=exp_cost, variance_cost=var_cost,
        impl_shortfall=impl_shortfall,
    )


# ---------------------------------------------------------------------------
# Liquidity-adjusted VaR
# ---------------------------------------------------------------------------

@dataclass
class LiquidityRiskReport:
    var_market: float            # standard VaR (ignoring liquidity)
    spread_cost: float           # expected liquidation spread cost
    impact_cost: float           # expected market impact cost
    lvar: float                  # Liquidity-adjusted VaR
    lvar_99: float               # L-VaR at 99%
    spread_pct: float            # spread as % of position value
    time_to_liquidate: float     # estimated days to liquidate safely
    liquidation_cost_pct: float  # total liquidation cost as % of position


def liquidity_adjusted_var(
        position_value: float,
        daily_vol: float,
        spread: float,            # bid-ask spread (absolute, per unit of value)
        adv: float,               # average daily volume (in value terms)
        kyle_lam: float = 0.0,    # Kyle lambda ($/$ of value traded)
        confidence: float = 0.95,
        horizon_days: int = 1,
) -> LiquidityRiskReport:
    """
    Compute Liquidity-adjusted VaR (L-VaR).

    L-VaR = VaR_market + spread_cost + impact_cost
    where:
      spread_cost = 0.5 * spread * position_value
      impact_cost = kyle_lambda * (position_value / adv)Â² * adv (if kyle_lam provided)
    """
    z = _norm_ppf(confidence)
    var_market = position_value * daily_vol * math.sqrt(horizon_days) * z

    # Spread cost (one-way: cost to exit)
    spread_cost = 0.5 * spread * position_value

    # Market impact cost
    if adv > 0 and kyle_lam > 0:
        participation = position_value / adv
        impact_cost = kyle_lam * participation ** 2 * adv
    else:
        impact_cost = 0.0

    lvar = var_market + spread_cost + impact_cost
    lvar_99 = position_value * daily_vol * math.sqrt(horizon_days) * _norm_ppf(0.99) + spread_cost + impact_cost

    spread_pct = spread_cost / position_value if position_value > 0 else 0.0

    # Days to liquidate: assume max 20% of ADV per day
    max_daily_trade = 0.20 * adv
    time_to_liquidate = position_value / max_daily_trade if max_daily_trade > 0 else math.inf

    total_liq_cost = spread_cost + impact_cost
    liq_cost_pct = total_liq_cost / position_value if position_value > 0 else 0.0

    return LiquidityRiskReport(
        var_market=var_market,
        spread_cost=spread_cost,
        impact_cost=impact_cost,
        lvar=lvar,
        lvar_99=lvar_99,
        spread_pct=spread_pct,
        time_to_liquidate=time_to_liquidate,
        liquidation_cost_pct=liq_cost_pct,
    )


# ---------------------------------------------------------------------------
# Amihud illiquidity ratio
# ---------------------------------------------------------------------------

def amihud_ratio(returns: List[float], volumes: List[float]) -> float:
    """
    Amihud (2002) illiquidity ratio: E[|R_t| / Volume_t].
    Higher values = less liquid.
    """
    if not returns or len(returns) != len(volumes):
        return 0.0
    ratios = [abs(r) / v for r, v in zip(returns, volumes) if v > 0]
    return sum(ratios) / len(ratios) if ratios else 0.0


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random
    rng = random.Random(42)

    # Simulate daily prices and volumes
    prices = [100.0]
    for _ in range(252):
        prices.append(prices[-1] * math.exp(rng.gauss(0, 0.015)))
    volumes = [1_000_000 * (0.8 + 0.4 * rng.random()) for _ in range(252)]
    returns = [(prices[i] / prices[i - 1] - 1) for i in range(1, len(prices))]

    # Roll spread estimate
    roll = roll_spread(prices)
    print(f"Roll implicit spread: ${roll:.4f} ({roll / prices[-1]:.3%})")

    # Amihud ratio
    amihud = amihud_ratio(returns, volumes)
    print(f"Amihud illiquidity:   {amihud:.2e}")

    # Liquidity-adjusted VaR
    pos_value = 5_000_000   # $5M position
    daily_vol = 0.015
    spread = 0.002           # 20bps spread
    adv = 2_000_000          # $2M ADV

    report = liquidity_adjusted_var(pos_value, daily_vol, spread, adv,
                                     kyle_lam=0.1, confidence=0.95)
    print(f"\nLiquidity Risk Report (95% VaR, 1-day):")
    print(f"  Market VaR:       ${report.var_market:>12,.0f}")
    print(f"  Spread cost:      ${report.spread_cost:>12,.0f}  ({report.spread_pct:.2%})")
    print(f"  Impact cost:      ${report.impact_cost:>12,.0f}")
    print(f"  L-VaR (95%):      ${report.lvar:>12,.0f}")
    print(f"  L-VaR (99%):      ${report.lvar_99:>12,.0f}")
    print(f"  Days to liquidate: {report.time_to_liquidate:.1f}")
    print(f"  Liq. cost (%):    {report.liquidation_cost_pct:.2%}")

    # Optimal liquidation
    ac_params = AlmgrenChrissParams(sigma=daily_vol, eta=0.005, gamma=0.0005,
                                     risk_aversion=1e-6)
    n_shares = 10_000
    price = pos_value / n_shares
    T_liq = 5  # 5-day horizon

    opt = optimal_liquidation(n_shares, T_liq, T_liq, ac_params, price)
    twap = twap_liquidation(n_shares, T_liq, T_liq, ac_params, price)

    print(f"\nLiquidation Comparison (10,000 shares over {T_liq} days):")
    print(f"{'Strategy':<12} {'E[Cost]':>12} {'Std[Cost]':>12} {'IS (bps)':>10}")
    print("-" * 50)
    print(f"{'Optimal':<12} ${opt.expected_cost:>10,.0f} ${math.sqrt(opt.variance_cost):>10,.0f} "
          f"{opt.impl_shortfall * 10000:>9.1f}")
    print(f"{'TWAP':<12} ${twap.expected_cost:>10,.0f} ${math.sqrt(twap.variance_cost):>10,.0f} "
          f"{twap.impl_shortfall * 10000:>9.1f}")
