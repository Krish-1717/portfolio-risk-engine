"""
portfolio_microstructure_impact.py
Microstructure: Market Impact for portfolio-risk-engine.
Kyle lambda, square-root market impact (Almgren-Chriss), TWAP/VWAP
slippage estimation, optimal execution horizon.
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
# 1. Kyle Lambda (Kyle 1985) — linear price impact
# ---------------------------------------------------------------------------
@dataclass
class KyleModel:
    """
    Kyle (1985): Price impact is linear in order flow.
    Δp = lambda * Q + noise
    where lambda = sigma / (2 * sqrt(V)) in the one-period model.
    """
    sigma: float    # daily return volatility
    avg_daily_volume: float  # shares/day
    price: float    # current price

    @property
    def lambda_theoretical(self) -> float:
        """
        Theoretical Kyle lambda:
        lambda = sigma_dollar / (2 * sqrt(V))
        where sigma_dollar = sigma * price (dollar volatility).
        Units: $/share per share traded.
        """
        sigma_dollar = self.sigma * self.price
        return sigma_dollar / (2 * math.sqrt(max(self.avg_daily_volume, 1.0)))

    def price_impact(self, trade_size: float) -> float:
        """Expected price change from trading `trade_size` shares."""
        return self.lambda_theoretical * trade_size

    def percent_impact(self, trade_size: float) -> float:
        """Price impact as fraction of current price."""
        return self.price_impact(trade_size) / self.price

def kyle_lambda_regression(price_changes: list[float],
                            order_flows: list[float]) -> dict:
    """
    Estimate Kyle lambda from price change / order flow data via OLS.
    price_changes in $, order_flows in shares (signed).
    """
    n = min(len(price_changes), len(order_flows))
    y = price_changes[:n]
    x = order_flows[:n]
    mx, my = mean(x), mean(y)

    cov = sum((x[i] - mx) * (y[i] - my) for i in range(n)) / max(n - 1, 1)
    var_x = variance(x)

    lambda_hat = cov / max(var_x, 1e-10)
    alpha_hat = my - lambda_hat * mx

    # R-squared
    y_hat = [alpha_hat + lambda_hat * x[i] for i in range(n)]
    ss_res = sum((y[i] - y_hat[i])**2 for i in range(n))
    ss_tot = sum((y[i] - my)**2 for i in range(n))
    r2 = 1 - ss_res / max(ss_tot, 1e-10)

    return {
        'lambda': lambda_hat,
        'alpha': alpha_hat,
        'r_squared': r2,
        'n': n,
    }

# ---------------------------------------------------------------------------
# 2. Square-Root Market Impact (Almgren-Chriss)
# ---------------------------------------------------------------------------
def sqrt_market_impact(trade_size: float, avg_daily_volume: float,
                        sigma: float, price: float,
                        eta: float = 0.1, gamma: float = 0.5) -> dict:
    """
    Almgren-Chriss square-root law:
    Impact (bps) = eta * sigma * (|Q| / ADV)^gamma

    eta: impact coefficient (~0.1 for liquid stocks)
    gamma: exponent (~0.5 for square-root law)
    """
    participation = abs(trade_size) / max(avg_daily_volume, 1.0)
    impact_bps = eta * sigma * (participation ** gamma)
    impact_pct = impact_bps  # sigma already in decimal form
    impact_dollars = abs(trade_size) * price * impact_pct

    return {
        'trade_size': trade_size,
        'participation_rate': participation,
        'impact_pct': impact_pct,
        'impact_bps': impact_bps * 10000,
        'impact_dollars': impact_dollars,
    }

def almgren_chriss_total_cost(trade_size: float, T_days: int,
                               sigma: float, price: float,
                               avg_daily_volume: float,
                               eta: float = 0.1, epsilon: float = 0.001,
                               gamma_temp: float = 0.0) -> dict:
    """
    Almgren-Chriss total cost for liquidating `trade_size` over `T_days`:
    Cost = sum over days of:
      - Temporary impact: eta * sigma * (n_t / ADV)^0.5
      - Permanent impact: gamma * (total_shares / ADV)
      - Risk: sigma * price * sigma_risk * remaining_inventory

    Simple equal-slice TWAP approximation.
    """
    daily_qty = abs(trade_size) / max(T_days, 1)

    total_temp_cost = 0.0
    total_perm_cost = 0.0
    remaining = abs(trade_size)

    perm_impact_total = gamma_temp * (abs(trade_size) / max(avg_daily_volume, 1.0)) * price

    for _ in range(T_days):
        slice_q = min(daily_qty, remaining)
        temp = sqrt_market_impact(slice_q, avg_daily_volume, sigma, price, eta)
        total_temp_cost += temp['impact_dollars']
        remaining -= slice_q

    total_cost = total_temp_cost + perm_impact_total

    return {
        'trade_size': trade_size,
        'horizon_days': T_days,
        'temporary_cost': total_temp_cost,
        'permanent_cost': perm_impact_total,
        'total_cost': total_cost,
        'cost_bps': total_cost / (abs(trade_size) * price) * 10000 if trade_size != 0 else 0,
    }

# ---------------------------------------------------------------------------
# 3. TWAP execution model
# ---------------------------------------------------------------------------
def twap_slippage(trade_size: float, T_periods: int,
                   realized_prices: list[float],
                   decision_price: float) -> dict:
    """
    TWAP implementation shortfall.
    Executes equal-sized slices each period.
    Slippage = VWAP_execution - decision_price (for buys, slippage > 0 is bad).
    """
    if not realized_prices:
        return {'slippage': 0.0, 'implementation_shortfall': 0.0}

    n = min(T_periods, len(realized_prices))
    slice_qty = trade_size / max(n, 1)

    # TWAP execution price = equal-weighted avg of period prices
    twap_price = mean(realized_prices[:n])

    slippage = (twap_price - decision_price) / decision_price  # positive for buys (adverse)
    implementation_shortfall = abs(trade_size) * abs(twap_price - decision_price)

    # Timing risk: std dev of slippage
    period_slippages = [(p - decision_price) / decision_price for p in realized_prices[:n]]
    timing_risk = std(period_slippages)

    return {
        'decision_price': decision_price,
        'twap_execution_price': twap_price,
        'slippage_pct': slippage,
        'implementation_shortfall': implementation_shortfall,
        'timing_risk': timing_risk,
        'n_periods': n,
    }

# ---------------------------------------------------------------------------
# 4. VWAP execution model
# ---------------------------------------------------------------------------
def vwap_slippage(trade_size: float, prices: list[float],
                   volumes: list[float], decision_price: float,
                   participation_rate: float = 0.10) -> dict:
    """
    VWAP participation strategy.
    Participates at constant rate in each period.
    """
    n = min(len(prices), len(volumes))
    total_volume = sum(volumes[:n])

    if total_volume <= 0:
        return {'vwap_price': decision_price, 'slippage_pct': 0.0}

    # Market VWAP
    market_vwap = sum(prices[i] * volumes[i] for i in range(n)) / total_volume

    # Participation: we trade participation_rate * volume_i each period
    our_qty = [participation_rate * volumes[i] for i in range(n)]
    our_total = sum(our_qty)

    if our_total <= 0:
        return {'vwap_price': market_vwap, 'slippage_pct': 0.0}

    our_vwap = sum(prices[i] * our_qty[i] for i in range(n)) / our_total

    slippage = (our_vwap - decision_price) / decision_price
    shortfall = abs(our_total) * abs(our_vwap - decision_price)

    return {
        'market_vwap': market_vwap,
        'our_vwap': our_vwap,
        'participation_rate': participation_rate,
        'total_shares_traded': our_total,
        'slippage_pct': slippage,
        'implementation_shortfall': shortfall,
    }

# ---------------------------------------------------------------------------
# 5. Optimal execution horizon
# ---------------------------------------------------------------------------
def optimal_execution_horizon(trade_size: float, avg_daily_volume: float,
                               sigma: float, price: float,
                               eta: float = 0.1,
                               risk_aversion: float = 1e-5,
                               max_days: int = 20) -> dict:
    """
    Find optimal T* (execution days) minimizing:
    E[Cost] = temporary_impact(T) + risk_cost(T)
    where risk_cost = 0.5 * lambda * sigma^2 * X^2 * T

    lambda: risk aversion coefficient
    """
    best_T = 1
    best_total = float('inf')
    costs = []

    for T in range(1, max_days + 1):
        imp = almgren_chriss_total_cost(trade_size, T, sigma, price, avg_daily_volume, eta)

        # Risk cost: variance of inventory * risk aversion * time
        # Simplified: 0.5 * lambda * sigma^2 * price^2 * |X|^2 * T
        risk_cost = 0.5 * risk_aversion * sigma**2 * price**2 * trade_size**2 * T

        total = imp['total_cost'] + risk_cost
        costs.append({'T': T, 'impact_cost': imp['total_cost'],
                      'risk_cost': risk_cost, 'total': total})

        if total < best_total:
            best_total = total
            best_T = T

    return {
        'optimal_T': best_T,
        'optimal_cost': best_total,
        'all_costs': costs,
    }

# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 65)
    print("MICROSTRUCTURE: Market Impact")
    print("=" * 65)

    rng = random.Random(42)

    def randn():
        u = max(rng.random(), 1e-15)
        return math.sqrt(-2*math.log(u)) * math.cos(2*math.pi*rng.random())

    # Example: executing a large position
    price = 100.0
    sigma = 0.015     # daily vol
    ADV = 1_000_000   # average daily volume (shares)
    trade_size = 50_000  # shares to buy

    print(f"\nTrade: {trade_size:,} shares @ ${price:.2f}")
    print(f"ADV: {ADV:,.0f} shares, Daily vol: {sigma:.1%}")
    print(f"Participation: {trade_size/ADV:.1%} of ADV")

    print("\n1. Kyle Lambda (Theoretical)")
    kyle = KyleModel(sigma=sigma, avg_daily_volume=ADV, price=price)
    print(f"   Lambda: ${kyle.lambda_theoretical:.6f}/share")
    print(f"   Linear impact on {trade_size:,} shares: ${kyle.price_impact(trade_size):.4f}")
    print(f"   As pct of price: {kyle.percent_impact(trade_size):.4%}")

    print("\n2. Square-Root Impact (Almgren-Chriss)")
    for T in [1, 2, 5, 10]:
        result = almgren_chriss_total_cost(trade_size, T, sigma, price, ADV)
        print(f"   T={T:2d} days: cost=${result['total_cost']:>10,.2f} ({result['cost_bps']:.1f} bps)")

    print("\n3. Optimal Execution Horizon")
    opt = optimal_execution_horizon(trade_size, ADV, sigma, price, risk_aversion=1e-6)
    print(f"   Optimal T*: {opt['optimal_T']} days (total cost=${opt['optimal_cost']:,.2f})")
    print(f"   Cost curve:")
    for row in opt['all_costs'][:8]:
        marker = " ← OPTIMAL" if row['T'] == opt['optimal_T'] else ""
        print(f"     T={row['T']:2d}: impact=${row['impact_cost']:>10,.2f}  "
              f"risk=${row['risk_cost']:>10,.2f}  total=${row['total']:>10,.2f}{marker}")

    print("\n4. TWAP Slippage Simulation (T=5 days)")
    # Simulate 5-day price path (trending slightly adverse)
    sim_prices = [price * (1 + 0.001 * t + sigma * randn()) for t in range(5)]
    twap_result = twap_slippage(trade_size, 5, sim_prices, price)
    print(f"   Decision price: ${twap_result['decision_price']:.4f}")
    print(f"   TWAP exec price: ${twap_result['twap_execution_price']:.4f}")
    print(f"   Slippage: {twap_result['slippage_pct']:+.4%}")
    print(f"   Implementation shortfall: ${twap_result['implementation_shortfall']:,.2f}")
    print(f"   Timing risk (vol of slippage): {twap_result['timing_risk']:.4%}")

    print("\n5. VWAP Participation Slippage")
    sim_volumes = [ADV / 5 * (1 + 0.3 * randn()) for _ in range(5)]
    vwap_result = vwap_slippage(trade_size, sim_prices, sim_volumes, price, participation_rate=0.10)
    print(f"   Market VWAP: ${vwap_result['market_vwap']:.4f}")
    print(f"   Our VWAP: ${vwap_result['our_vwap']:.4f}")
    print(f"   Shares traded: {vwap_result['total_shares_traded']:,.0f}")
    print(f"   Slippage: {vwap_result['slippage_pct']:+.4%}")
    print(f"   Implementation shortfall: ${vwap_result['implementation_shortfall']:,.2f}")

    print("\n6. Kyle Lambda from Regression")
    # Simulate order flow data
    true_lambda = 0.00001
    order_flows = [rng.choice([-1, 1]) * abs(ADV * 0.01 * randn()) for _ in range(200)]
    price_changes = [true_lambda * q + sigma * price * randn() for q in order_flows]
    reg = kyle_lambda_regression(price_changes, order_flows)
    print(f"   True lambda: {true_lambda:.2e}")
    print(f"   Estimated:   {reg['lambda']:.2e}")
    print(f"   R-squared:   {reg['r_squared']:.4f}")

    print("\n[Done] Microstructure: Market Impact complete.")
