"""
portfolio_day24_transaction_costs.py
Day 24: Transaction Cost Modeling — market impact (linear + square-root),
bid-ask spread estimation, VWAP participation timing risk,
cost-adjusted Sharpe, optimal rebalancing frequency.
Pure Python stdlib only.
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Market microstructure data
# ---------------------------------------------------------------------------

@dataclass
class SecurityInfo:
    ticker: str
    adv: float            # average daily volume ($)
    price: float          # current price
    spread_bps: float     # quoted bid-ask spread in basis points
    sigma_daily: float    # daily volatility (annualized / sqrt(252))


# ---------------------------------------------------------------------------
# Bid-Ask Spread Estimation (from OHLC)
# ---------------------------------------------------------------------------

def corwin_schultz_spread(
    highs: list[float], lows: list[float],
    annualize: bool = False,
) -> list[float]:
    """
    Corwin-Schultz (2012) spread estimator from daily H/L prices.
    Spread = 2*(e^{alpha} - 1) / (1 + e^{alpha})
    where alpha is derived from 1-day and 2-day H/L ranges.
    """
    n = len(highs)
    spreads = []

    for t in range(1, n):
        beta = (math.log(highs[t] / lows[t]))**2 + (math.log(highs[t-1] / lows[t-1]))**2
        gamma = (math.log(max(highs[t], highs[t-1]) / min(lows[t], lows[t-1])))**2

        # alpha = sqrt(2*beta) - sqrt(beta) / (3 - 2*sqrt(2)) - sqrt(gamma / (3 - 2*sqrt(2)))
        denom = 3 - 2 * math.sqrt(2)
        alpha = (math.sqrt(2 * beta) - math.sqrt(beta)) / denom - math.sqrt(gamma / denom)
        alpha = max(alpha, 0.0)

        spread = 2 * (math.exp(alpha) - 1) / (1 + math.exp(alpha))
        spreads.append(max(spread, 0.0))

    return spreads


def roll_spread(prices: list[float]) -> float:
    """
    Roll (1984) implicit spread from price autocorrelation.
    S = 2 * sqrt(-cov(delta_p_t, delta_p_{t-1}))
    """
    n = len(prices)
    if n < 3:
        return 0.0

    diffs = [prices[t] - prices[t-1] for t in range(1, n)]
    n_d = len(diffs) - 1
    cov = sum(diffs[t] * diffs[t-1] for t in range(n_d)) / max(n_d, 1)

    return 2 * math.sqrt(max(-cov, 0.0))


# ---------------------------------------------------------------------------
# Market Impact Models
# ---------------------------------------------------------------------------

def linear_market_impact(
    trade_size: float,   # in dollars
    adv: float,          # average daily volume in dollars
    sigma_daily: float,  # daily vol (daily, not annualized)
    price: float,
    eta: float = 0.1,    # linear impact coefficient
) -> float:
    """
    Linear market impact (Kyle lambda):
    MI = eta * sigma_daily * (trade_size / adv)
    Returns impact in basis points.
    """
    participation = trade_size / max(adv, 1.0)
    mi_pct = eta * sigma_daily * participation
    return mi_pct * 10_000  # in bps


def sqrt_market_impact(
    trade_size: float,
    adv: float,
    sigma_daily: float,
    price: float = 100.0,
    kappa: float = 0.6,    # square-root coefficient (Almgren-Chriss)
    gamma: float = 0.5,    # exponent (0.5 = square-root law)
) -> float:
    """
    Square-root market impact (Almgren et al. 2005):
    MI = kappa * sigma * sqrt(trade_size / adv)
    Returns impact in basis points.
    """
    participation = trade_size / max(adv, 1.0)
    mi_pct = kappa * sigma_daily * (participation ** gamma)
    return mi_pct * 10_000  # bps


def total_transaction_cost(
    trade_size: float,
    security: SecurityInfo,
    model: str = 'sqrt',
    commissions_bps: float = 1.0,
) -> dict:
    """
    Full transaction cost breakdown: spread + market impact + commissions.
    """
    sigma_daily = security.sigma_daily / math.sqrt(252)
    abs_size = abs(trade_size)  # costs are always positive (direction-independent)

    # Spread cost (half-spread on entry)
    spread_cost_bps = security.spread_bps / 2

    # Market impact
    if model == 'linear':
        impact_bps = linear_market_impact(abs_size, security.adv, sigma_daily, security.price)
    else:
        impact_bps = sqrt_market_impact(abs_size, security.adv, sigma_daily, security.price)

    total_bps = spread_cost_bps + impact_bps + commissions_bps
    total_pct = total_bps / 10_000
    cost_dollars = abs_size * total_pct

    return {
        'spread_bps': spread_cost_bps,
        'impact_bps': impact_bps,
        'commissions_bps': commissions_bps,
        'total_bps': total_bps,
        'total_pct': total_pct,
        'cost_dollars': cost_dollars,
        'trade_size': trade_size,
        'participation_rate': trade_size / security.adv,
    }


# ---------------------------------------------------------------------------
# VWAP Participation Rate and Timing Risk
# ---------------------------------------------------------------------------

def vwap_cost(
    participation_rate: float,  # fraction of ADV
    sigma_daily: float,
    eta: float = 0.142,         # empirical coefficient
) -> float:
    """
    VWAP implementation shortfall vs arrival price.
    Per Almgren (2003): E[IS] ≈ eta * sigma * participation_rate^(3/2)
    Returns in bps.
    """
    daily_sigma = sigma_daily / math.sqrt(252)
    is_bps = eta * daily_sigma * (participation_rate ** 1.5) * 10_000
    return is_bps


def twap_timing_risk(
    trade_duration_days: float,
    sigma_daily: float,
    n_slices: int = 10,
) -> dict:
    """
    TWAP timing risk: variance of execution price around TWAP.
    Execution over uniform slices; each slice has independent price risk.
    """
    daily_variance = (sigma_daily / math.sqrt(252))**2

    # TWAP execution: average of n_slices prices over T days
    # Variance of TWAP = variance_daily * T / n_slices (if independent, approximate)
    slice_duration = trade_duration_days / n_slices
    timing_variance = daily_variance * slice_duration / n_slices
    timing_vol_bps = math.sqrt(timing_variance) * 10_000

    # Market timing risk: variance around the mid-period price
    mid_timing_variance = daily_variance * trade_duration_days / 4

    return {
        'twap_timing_vol_bps': timing_vol_bps,
        'n_slices': n_slices,
        'duration_days': trade_duration_days,
        'mid_timing_vol_bps': math.sqrt(mid_timing_variance) * 10_000,
        'expected_slippage_bps': vwap_cost(
            trade_duration_days, sigma_daily
        ),
    }


# ---------------------------------------------------------------------------
# Cost-Adjusted Sharpe Ratio
# ---------------------------------------------------------------------------

def cost_adjusted_sharpe(
    gross_returns: list[float],          # daily returns before costs
    turnover_series: list[float],        # daily turnover fraction (0-1)
    cost_per_unit_turnover: float = 20,  # cost in bps per unit turnover
    rf_daily: float = 0.05/252,
) -> dict:
    """
    Net Sharpe after deducting daily transaction cost drag.
    """
    n = len(gross_returns)

    # Net returns: subtract cost drag
    net_returns = []
    for i in range(n):
        cost_drag = turnover_series[i] * cost_per_unit_turnover / 10_000
        net_returns.append(gross_returns[i] - cost_drag)

    # Gross stats
    gross_mean = sum(gross_returns) / n * 252
    gross_var = sum((r - gross_mean/252)**2 for r in gross_returns) / (n-1)
    gross_std = math.sqrt(gross_var * 252)
    gross_sharpe = (gross_mean - rf_daily * 252) / max(gross_std, 1e-8)

    # Net stats
    net_mean = sum(net_returns) / n * 252
    net_var = sum((r - net_mean/252)**2 for r in net_returns) / (n-1)
    net_std = math.sqrt(net_var * 252)
    net_sharpe = (net_mean - rf_daily * 252) / max(net_std, 1e-8)

    avg_turnover = sum(turnover_series) / n
    annual_cost_drag = avg_turnover * cost_per_unit_turnover / 10_000 * 252

    return {
        'gross_return': gross_mean,
        'net_return': net_mean,
        'gross_sharpe': gross_sharpe,
        'net_sharpe': net_sharpe,
        'annual_cost_drag': annual_cost_drag,
        'annual_turnover': avg_turnover * 252,
        'sharpe_decay': gross_sharpe - net_sharpe,
    }


# ---------------------------------------------------------------------------
# Optimal Rebalancing Frequency
# ---------------------------------------------------------------------------

def optimal_rebalancing_frequency(
    expected_alpha_bps_per_day: float,    # alpha generation rate
    cost_per_rebalance_bps: float,        # round-trip cost in bps
    alpha_decay_half_life_days: float = 20.0,
    holding_periods: list[int] | None = None,
) -> dict:
    """
    Find optimal holding period T that maximizes net alpha.
    Alpha(T) = alpha_daily * (1 - e^{-T/tau}) * tau  (integrated alpha with decay)
    Net(T) = Alpha(T) - cost
    """
    if holding_periods is None:
        holding_periods = list(range(1, 63))

    tau = alpha_decay_half_life_days / math.log(2)

    results = {}
    best_T, best_net = 1, float('-inf')

    for T in holding_periods:
        # Integrated alpha over T days with exponential decay
        integrated_alpha = expected_alpha_bps_per_day * tau * (1 - math.exp(-T / tau))
        net_alpha = integrated_alpha - cost_per_rebalance_bps
        results[T] = {
            'gross_alpha_bps': integrated_alpha,
            'net_alpha_bps': net_alpha,
            'alpha_ir': integrated_alpha / max(math.sqrt(T) * 5, 1),  # proxy IR
        }
        if net_alpha > best_net:
            best_net = net_alpha
            best_T = T

    return {
        'optimal_holding_days': best_T,
        'best_net_alpha_bps': best_net,
        'optimal_annualized_turnover': 252.0 / best_T,
        'schedule': results,
    }


# ---------------------------------------------------------------------------
# Transaction Cost Attribution
# ---------------------------------------------------------------------------

def attribute_transaction_costs(
    trades: list[dict],  # [{'size': float, 'ticker': str, 'security': SecurityInfo}]
    model: str = 'sqrt',
) -> dict:
    """Break down costs across a set of trades."""
    total_traded = 0.0
    total_spread = 0.0
    total_impact = 0.0
    total_commission = 0.0
    trade_details = []

    for trade in trades:
        tc = total_transaction_cost(
            trade['size'], trade['security'],
            model=model, commissions_bps=1.0
        )
        total_traded += abs(trade['size'])
        total_spread += tc['spread_bps'] * abs(trade['size']) / 10_000
        total_impact += tc['impact_bps'] * abs(trade['size']) / 10_000
        total_commission += tc['commissions_bps'] * abs(trade['size']) / 10_000
        trade_details.append({**trade, **tc})

    total_cost = total_spread + total_impact + total_commission

    return {
        'total_traded': total_traded,
        'total_cost_dollars': total_cost,
        'total_cost_bps': total_cost / total_traded * 10_000 if total_traded > 0 else 0,
        'breakdown': {
            'spread_dollars': total_spread,
            'impact_dollars': total_impact,
            'commission_dollars': total_commission,
        },
        'breakdown_pct': {
            'spread_pct': total_spread / max(total_cost, 1e-10),
            'impact_pct': total_impact / max(total_cost, 1e-10),
            'commission_pct': total_commission / max(total_cost, 1e-10),
        },
        'trade_details': trade_details,
    }


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 65)
    print("DAY 24: Transaction Cost Modeling")
    print("=" * 65)

    rng = random.Random(42)

    # Sample securities
    securities = {
        'AAPL': SecurityInfo('AAPL', adv=5e9,  price=175, spread_bps=1.5,  sigma_daily=0.25),
        'SPY':  SecurityInfo('SPY',  adv=30e9, price=450, spread_bps=0.5,  sigma_daily=0.15),
        'IWM':  SecurityInfo('IWM',  adv=5e9,  price=190, spread_bps=2.0,  sigma_daily=0.22),
        'TSLA': SecurityInfo('TSLA', adv=20e9, price=250, spread_bps=3.0,  sigma_daily=0.55),
        'GLD':  SecurityInfo('GLD',  adv=1e9,  price=185, spread_bps=5.0,  sigma_daily=0.12),
    }

    print("\n1. Transaction Cost by Security and Size")
    print(f"{'Ticker':>6} | {'Size $M':>8} | {'Spread bps':>10} | {'Impact bps':>10} | {'Total bps':>10}")
    print("-" * 60)
    for ticker, sec in securities.items():
        for size_m in [1, 10, 50]:
            tc = total_transaction_cost(size_m * 1e6, sec, model='sqrt')
            print(f"{ticker:>6} | {size_m:>8}M | {tc['spread_bps']:>10.2f} | {tc['impact_bps']:>10.2f} | {tc['total_bps']:>10.2f}")

    # --- Bid-ask spread estimation ---
    print("\n2. Bid-Ask Spread Estimation (Corwin-Schultz)")
    # Simulate OHLC prices
    price = 100.0
    highs, lows = [], []
    for _ in range(30):
        r = rng.gauss(0, 0.01)
        price *= math.exp(r)
        hl_range = price * rng.uniform(0.005, 0.02)
        highs.append(price + hl_range / 2)
        lows.append(price - hl_range / 2)

    cs_spreads = corwin_schultz_spread(highs, lows)
    avg_cs = sum(cs_spreads) / len(cs_spreads)

    prices = [(h + l) / 2 for h, l in zip(highs, lows)]
    roll_s = roll_spread(prices)

    print(f"  Corwin-Schultz spread (avg): {avg_cs:.6f} ({avg_cs*10000:.2f} bps)")
    print(f"  Roll implicit spread       : {roll_s:.6f} ({roll_s*10000:.2f} bps)")

    # --- VWAP timing risk ---
    print("\n3. VWAP / TWAP Timing Risk")
    for participation in [0.05, 0.10, 0.20, 0.30]:
        vwap_bps = vwap_cost(participation, sigma_daily=0.20)
        twap = twap_timing_risk(0.5, sigma_daily=0.20, n_slices=10)
        print(f"  Part={participation:.0%}: VWAP IS={vwap_bps:.2f}bps | TWAP timing vol={twap['twap_timing_vol_bps']:.2f}bps")

    # --- Cost-adjusted Sharpe ---
    print("\n4. Cost-Adjusted Sharpe Ratio")
    gross_rets = [rng.gauss(0.0004, 0.01) for _ in range(252)]
    turnovers = [rng.uniform(0.05, 0.20) for _ in range(252)]  # 5-20% daily turnover

    sharpe_stats = cost_adjusted_sharpe(gross_rets, turnovers, cost_per_unit_turnover=20)
    print(f"  Gross return (ann)  : {sharpe_stats['gross_return']:>8.4%}")
    print(f"  Net return (ann)    : {sharpe_stats['net_return']:>8.4%}")
    print(f"  Annual cost drag    : {sharpe_stats['annual_cost_drag']:>8.4%}")
    print(f"  Annual turnover     : {sharpe_stats['annual_turnover']:>8.1f}x")
    print(f"  Gross Sharpe        : {sharpe_stats['gross_sharpe']:>8.4f}")
    print(f"  Net Sharpe          : {sharpe_stats['net_sharpe']:>8.4f}")
    print(f"  Sharpe decay        : {sharpe_stats['sharpe_decay']:>8.4f}")

    # --- Optimal rebalancing frequency ---
    print("\n5. Optimal Rebalancing Frequency")
    opt = optimal_rebalancing_frequency(
        expected_alpha_bps_per_day=2.0,
        cost_per_rebalance_bps=25.0,
        alpha_decay_half_life_days=15.0,
    )
    print(f"  Optimal holding period: {opt['optimal_holding_days']} days")
    print(f"  Best net alpha       : {opt['best_net_alpha_bps']:.2f} bps")
    print(f"  Optimal annual turns : {opt['optimal_annualized_turnover']:.1f}x")

    print(f"\n  Holding period vs net alpha (top 10):")
    sorted_sched = sorted(opt['schedule'].items(), key=lambda x: x[1]['net_alpha_bps'], reverse=True)
    print(f"  {'Hold Days':>10} | {'Gross bps':>10} | {'Net bps':>10}")
    print("  " + "-" * 38)
    for T, vals in sorted_sched[:10]:
        print(f"  {T:>10} | {vals['gross_alpha_bps']:>10.2f} | {vals['net_alpha_bps']:>10.2f}")

    # --- Attribution ---
    print("\n6. Trade Basket Cost Attribution")
    trades = [
        {'size': 2e6,  'security': securities['AAPL']},
        {'size': -1e6, 'security': securities['SPY']},
        {'size': 5e5,  'security': securities['TSLA']},
        {'size': 1e6,  'security': securities['GLD']},
    ]
    attr = attribute_transaction_costs(trades, model='sqrt')

    print(f"  Total traded      : ${attr['total_traded']/1e6:.2f}M")
    print(f"  Total cost        : ${attr['total_cost_dollars']/1000:.2f}K")
    print(f"  Average cost      : {attr['total_cost_bps']:.2f} bps")
    print(f"  Spread share      : {attr['breakdown_pct']['spread_pct']:.1%}")
    print(f"  Market impact share: {attr['breakdown_pct']['impact_pct']:.1%}")
    print(f"  Commission share  : {attr['breakdown_pct']['commission_pct']:.1%}")

    print("\n[Done] Day 24: Transaction Costs complete.")
