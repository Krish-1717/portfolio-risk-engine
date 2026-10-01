"""
options_day24_tail_risk_hedging.py
Day 24: Tail Risk Hedging — put spread optimizer, variance swap fair value,
model-free VIX approximation, CVaR hedge ratio, hedging P&L attribution.
Pure Python stdlib only.
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Black-Scholes helpers
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)

def bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(S - K * math.exp(-r * T), 0.0)
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)

def bs_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    return bs_call(S, K, T, r, sigma) - S + K * math.exp(-r * T)

def bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    return S * sqrtT * _norm_pdf(d1)

def implied_vol(price: float, S: float, K: float, T: float, r: float,
                option_type: str = 'call', tol: float = 1e-8, max_iter: int = 100) -> float:
    """Newton-Raphson implied vol."""
    sigma = 0.3
    for _ in range(max_iter):
        if option_type == 'call':
            model = bs_call(S, K, T, r, sigma)
        else:
            model = bs_put(S, K, T, r, sigma)
        vega = bs_vega(S, K, T, r, sigma)
        if abs(vega) < 1e-12:
            break
        sigma -= (model - price) / vega
        sigma = max(sigma, 1e-5)
        if abs(model - price) < tol:
            break
    return sigma


# ---------------------------------------------------------------------------
# Variance Swap
# ---------------------------------------------------------------------------

@dataclass
class VarianceSwap:
    """
    Variance swap: pays realized variance - strike variance at expiry.
    Fair variance strike: K_var = (2/T) * [sum of OTM option prices / strike^2 * dK]
    (Demeterfi et al. approximation via log-contract).
    """
    T: float
    r: float
    # Discrete option strip
    put_strikes: list[float]
    put_ivs: list[float]      # implied vols for puts
    call_strikes: list[float]
    call_ivs: list[float]     # implied vols for calls
    S0: float

    def fair_variance_strike(self) -> float:
        """
        Model-free variance swap strike via discrete approximation:
        K_var = (2/T) * sum_i [C(K_i) or P(K_i)] / K_i^2 * delta_K
        (Demeterfi, Derman, Kamal, Zou 1999)
        """
        F = self.S0 * math.exp(self.r * self.T)  # forward
        total = 0.0
        discount = math.exp(-self.r * self.T)

        # Put side (K < F): use puts
        all_K_put = sorted(self.put_strikes)
        for idx, K in enumerate(all_K_put):
            iv = self.put_ivs[self.put_strikes.index(K)]
            price = bs_put(self.S0, K, self.T, self.r, iv)
            # Delta K
            if idx == 0:
                dK = all_K_put[1] - all_K_put[0] if len(all_K_put) > 1 else 1.0
            elif idx == len(all_K_put) - 1:
                dK = all_K_put[-1] - all_K_put[-2]
            else:
                dK = (all_K_put[idx + 1] - all_K_put[idx - 1]) / 2
            total += price * dK / K**2

        # Call side (K >= F): use calls
        all_K_call = sorted(self.call_strikes)
        for idx, K in enumerate(all_K_call):
            iv = self.call_ivs[self.call_strikes.index(K)]
            price = bs_call(self.S0, K, self.T, self.r, iv)
            if idx == 0:
                dK = all_K_call[1] - all_K_call[0] if len(all_K_call) > 1 else 1.0
            elif idx == len(all_K_call) - 1:
                dK = all_K_call[-1] - all_K_call[-2]
            else:
                dK = (all_K_call[idx + 1] - all_K_call[idx - 1]) / 2
            total += price * dK / K**2

        K_var = (2 / self.T) * total  # annualized variance
        return K_var

    def vol_strike(self) -> float:
        """Approximate vol strike = sqrt(K_var)."""
        return math.sqrt(self.fair_variance_strike())

    def pnl(self, realized_variance: float) -> float:
        """Payoff = realized_var - K_var (per unit notional)."""
        return realized_variance - self.fair_variance_strike()


def realized_variance(returns: list[float], annualize: int = 252) -> float:
    """Annualized realized variance from daily log-returns."""
    n = len(returns)
    if n < 2:
        return 0.0
    mean_r = sum(returns) / n
    var = sum((r - mean_r)**2 for r in returns) / (n - 1)
    return var * annualize


# ---------------------------------------------------------------------------
# Model-free VIX approximation
# ---------------------------------------------------------------------------

def model_free_vix(
    S: float,
    T: float,
    r: float,
    put_strikes: list[float],
    put_prices: list[float],
    call_strikes: list[float],
    call_prices: list[float],
) -> float:
    """
    CBOE VIX methodology (simplified):
    VIX^2 = (2/T) * sum [Q(K_i)/K_i^2 * delta_K] - (1/T)*[(F/K0) - 1]^2
    Q = OTM option price.
    """
    F = S * math.exp(r * T)
    # Find K0: largest strike <= F
    all_strikes = sorted(set(put_strikes + call_strikes))
    K0 = max((K for K in all_strikes if K <= F), default=S)

    total = 0.0

    # Puts
    sorted_puts = sorted(zip(put_strikes, put_prices))
    for idx, (K, P) in enumerate(sorted_puts):
        if idx == 0:
            dK = sorted_puts[1][0] - sorted_puts[0][0] if len(sorted_puts) > 1 else 1.0
        elif idx == len(sorted_puts) - 1:
            dK = sorted_puts[-1][0] - sorted_puts[-2][0]
        else:
            dK = (sorted_puts[idx+1][0] - sorted_puts[idx-1][0]) / 2
        total += dK * P / K**2

    # Calls
    sorted_calls = sorted(zip(call_strikes, call_prices))
    for idx, (K, C) in enumerate(sorted_calls):
        if idx == 0:
            dK = sorted_calls[1][0] - sorted_calls[0][0] if len(sorted_calls) > 1 else 1.0
        elif idx == len(sorted_calls) - 1:
            dK = sorted_calls[-1][0] - sorted_calls[-2][0]
        else:
            dK = (sorted_calls[idx+1][0] - sorted_calls[idx-1][0]) / 2
        total += dK * C / K**2

    vix_sq = (2 / T) * total - (1 / T) * ((F / K0 - 1)**2)
    return math.sqrt(max(vix_sq, 0)) * 100  # in percentage


# ---------------------------------------------------------------------------
# Put Spread Hedge Optimizer
# ---------------------------------------------------------------------------

@dataclass
class PutSpreadHedge:
    """
    Put spread hedge: buy put at K_low, sell put at K_high (K_low < K_high < S).
    Net cost = P(K_high) - P(K_low)  (may be negative if K_low much lower)
    Wait: typically buy the higher strike, sell lower.
    Long put at K_high (OTM), short put at K_low (further OTM) — classic put spread.
    """
    S: float
    T: float
    r: float
    sigma: float

    def spread_cost(self, K_long: float, K_short: float) -> float:
        """Net premium: long put at K_long, short put at K_short (K_short < K_long)."""
        return bs_put(self.S, K_long, self.T, self.r, self.sigma) - \
               bs_put(self.S, K_short, self.T, self.r, self.sigma)

    def spread_payoff(self, S_T: float, K_long: float, K_short: float) -> float:
        """Payoff at expiry."""
        return max(K_long - S_T, 0) - max(K_short - S_T, 0)

    def net_protection(self, S_T: float, K_long: float, K_short: float) -> float:
        """Net P&L including premium paid."""
        cost = self.spread_cost(K_long, K_short)
        payoff = self.spread_payoff(S_T, K_long, K_short)
        return payoff - cost

    def optimize_spread(
        self,
        budget: float,            # max premium to pay
        n_contracts: int = 1,
        protection_levels: list[float] | None = None,
    ) -> dict:
        """
        Find optimal put spread (K_long, K_short) given cost budget.
        Maximizes expected protection under log-normal S_T.
        """
        if protection_levels is None:
            protection_levels = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

        best = {'K_long': None, 'K_short': None, 'cost': None,
                'max_payoff': None, 'score': float('-inf')}

        long_strikes = [self.S * p for p in protection_levels]
        short_strikes = [self.S * p for p in protection_levels]

        # Expected payoff under log-normal
        def expected_payoff(K_long: float, K_short: float) -> float:
            """E[payoff] under risk-neutral log-normal."""
            # Using BS put prices as risk-neutral expected payoffs discounted
            p_long = bs_put(self.S, K_long, self.T, self.r, self.sigma)
            p_short = bs_put(self.S, K_short, self.T, self.r, self.sigma)
            return (p_long - p_short) * math.exp(self.r * self.T)  # undiscounted expected payoff

        for K_long in long_strikes:
            for K_short in short_strikes:
                if K_short >= K_long:
                    continue
                cost = self.spread_cost(K_long, K_short) * n_contracts
                if cost > budget or cost < 0:
                    continue
                max_payout = (K_long - K_short) * n_contracts
                ep = expected_payoff(K_long, K_short) * n_contracts
                score = ep / max(cost, 1e-6)  # protection per dollar
                if score > best['score']:
                    best = {
                        'K_long': K_long, 'K_short': K_short,
                        'cost': cost, 'max_payoff': max_payout,
                        'score': score,
                        'expected_payoff': ep,
                    }

        return best


# ---------------------------------------------------------------------------
# Tail Risk Metrics
# ---------------------------------------------------------------------------

def tail_risk_metrics(returns: list[float], confidence: float = 0.99) -> dict:
    """
    Compute tail risk metrics from historical returns:
    - VaR, CVaR, left tail index, conditional drawdown, max drawdown
    """
    n = len(returns)
    if n < 10:
        return {}

    sorted_r = sorted(returns)
    idx = max(int(n * (1 - confidence)), 1)

    var = -sorted_r[idx]
    cvar = -sum(sorted_r[:idx]) / idx if idx > 0 else var

    # Left tail index (Hill estimator)
    k_hill = max(int(n * 0.10), 5)
    negatives = sorted([-r for r in returns if r < 0], reverse=True)
    if len(negatives) >= k_hill + 1:
        log_ratios = [math.log(negatives[i] / negatives[k_hill])
                      for i in range(k_hill) if negatives[k_hill] > 0]
        hill_index = k_hill / sum(log_ratios) if log_ratios else float('nan')
    else:
        hill_index = float('nan')

    # Maximum drawdown
    wealth = 1.0
    peak = 1.0
    max_dd = 0.0
    current_dd_start = 0
    for r in returns:
        wealth *= math.exp(r)
        if wealth > peak:
            peak = wealth
        dd = 1 - wealth / peak
        max_dd = max(max_dd, dd)

    # Conditional drawdown at risk (CDaR)
    # Simulate wealth path drawdowns
    drawdowns = []
    wealth = 1.0
    peak = 1.0
    for r in returns:
        wealth *= math.exp(r)
        if wealth > peak:
            peak = wealth
        drawdowns.append(1 - wealth / peak)

    dd_sorted = sorted(drawdowns, reverse=True)
    cdar_idx = max(int(len(dd_sorted) * (1 - confidence)), 1)
    cdar = sum(dd_sorted[:cdar_idx]) / cdar_idx if cdar_idx > 0 else 0.0

    return {
        f'VaR_{confidence:.0%}': var,
        f'CVaR_{confidence:.0%}': cvar,
        'hill_tail_index': hill_index,
        'max_drawdown': max_dd,
        f'CDaR_{confidence:.0%}': cdar,
        'skewness': _skewness(returns),
        'kurtosis': _kurtosis(returns),
    }


def _skewness(data: list[float]) -> float:
    n = len(data)
    mean = sum(data) / n
    std = math.sqrt(sum((x - mean)**2 for x in data) / n)
    if std == 0:
        return 0.0
    return sum((x - mean)**3 for x in data) / (n * std**3)


def _kurtosis(data: list[float]) -> float:
    n = len(data)
    mean = sum(data) / n
    std = math.sqrt(sum((x - mean)**2 for x in data) / n)
    if std == 0:
        return 0.0
    return sum((x - mean)**4 for x in data) / (n * std**4) - 3


# ---------------------------------------------------------------------------
# CVaR-optimal hedge ratio
# ---------------------------------------------------------------------------

def cvar_optimal_hedge_ratio(
    portfolio_returns: list[float],
    hedge_returns: list[float],
    confidence: float = 0.99,
    n_points: int = 20,
) -> dict:
    """
    Find hedge ratio h* that minimizes CVaR of (portfolio + h * hedge).
    Searches over h in [-2, 0] (protective put hedge: negative h).
    """
    n = len(portfolio_returns)
    if n != len(hedge_returns):
        raise ValueError("Length mismatch")

    def portfolio_cvar(h: float) -> float:
        combined = [portfolio_returns[i] + h * hedge_returns[i] for i in range(n)]
        metrics = tail_risk_metrics(combined, confidence)
        return metrics.get(f'CVaR_{confidence:.0%}', float('inf'))

    best_h = 0.0
    best_cvar = portfolio_cvar(0.0)

    hedge_ratios = [i / n_points * (-2.0) for i in range(n_points + 1)]

    for h in hedge_ratios:
        cv = portfolio_cvar(h)
        if cv < best_cvar:
            best_cvar = cv
            best_h = h

    # Refine with golden-section search near best_h
    unhedged_cvar = portfolio_cvar(0.0)

    return {
        'optimal_h': best_h,
        'hedged_CVaR': best_cvar,
        'unhedged_CVaR': unhedged_cvar,
        'CVaR_reduction': unhedged_cvar - best_cvar,
        'CVaR_reduction_pct': (unhedged_cvar - best_cvar) / max(unhedged_cvar, 1e-6),
    }


# ---------------------------------------------------------------------------
# Hedging P&L Attribution
# ---------------------------------------------------------------------------

def hedge_pnl_attribution(
    portfolio_returns: list[float],
    put_spread_payoffs: list[float],
    put_spread_cost: float,
    hedge_ratio: float = 1.0,
) -> dict:
    """
    Decompose hedged portfolio P&L:
    - Unhedged P&L
    - Hedge payoff
    - Hedge cost (drag)
    - Net hedged P&L
    """
    n = len(portfolio_returns)
    unhedged_pnl = sum(portfolio_returns)
    hedge_payoff = hedge_ratio * sum(put_spread_payoffs)

    hedged_returns = [portfolio_returns[i] + hedge_ratio * put_spread_payoffs[i] / n
                      for i in range(n)]

    unhedged_metrics = tail_risk_metrics(portfolio_returns)
    hedged_metrics = tail_risk_metrics(hedged_returns)

    # Sharpe ratio comparison
    def sharpe(rets: list[float], annualize: int = 252) -> float:
        mean_r = sum(rets) / len(rets) * annualize
        std_r = math.sqrt(sum(r**2 for r in rets) / len(rets)) * math.sqrt(annualize)
        return mean_r / max(std_r, 1e-8)

    return {
        'unhedged_total_pnl': unhedged_pnl,
        'hedge_payoff': hedge_payoff,
        'hedge_cost': put_spread_cost * hedge_ratio,
        'net_hedge_pnl': hedge_payoff - put_spread_cost * hedge_ratio,
        'net_hedged_total_pnl': unhedged_pnl + hedge_payoff - put_spread_cost * hedge_ratio,
        'unhedged_VaR_99': unhedged_metrics.get('VaR_99%', float('nan')),
        'hedged_VaR_99': hedged_metrics.get('VaR_99%', float('nan')),
        'unhedged_max_dd': unhedged_metrics.get('max_drawdown', float('nan')),
        'hedged_max_dd': hedged_metrics.get('max_drawdown', float('nan')),
        'unhedged_sharpe': sharpe(portfolio_returns),
        'hedged_sharpe': sharpe(hedged_returns),
    }


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 65)
    print("DAY 24: Tail Risk Hedging")
    print("=" * 65)

    S, T, r, sigma = 100.0, 0.5, 0.05, 0.25

    # --- Variance swap ---
    print("\n1. Variance Swap Fair Value")
    put_K = [70, 75, 80, 85, 90, 95]
    call_K = [100, 105, 110, 115, 120, 125]
    put_iv = [sigma + 0.05 * (1 - k/100) for k in put_K]   # skewed smile
    call_iv = [sigma + 0.02 * (k/100 - 1) for k in call_K]

    vs = VarianceSwap(T, r, put_K, put_iv, call_K, call_iv, S)
    K_var = vs.fair_variance_strike()
    vol_strike = vs.vol_strike()
    print(f"  Fair variance strike : {K_var:.6f}  ({K_var*100:.4f}% vol^2 annualized)")
    print(f"  Vol strike           : {vol_strike:.4%}")
    print(f"  ATM implied vol      : {sigma:.4%}")

    # --- Model-free VIX ---
    print("\n2. Model-Free VIX Approximation")
    put_prices = [bs_put(S, K, T, r, iv) for K, iv in zip(put_K, put_iv)]
    call_prices = [bs_call(S, K, T, r, iv) for K, iv in zip(call_K, call_iv)]
    vix = model_free_vix(S, T, r, put_K, put_prices, call_K, call_prices)
    print(f"  Model-free VIX approx: {vix:.2f}")
    print(f"  Input ATM sigma      : {sigma*100:.2f}")

    # --- Put spread optimizer ---
    print("\n3. Put Spread Hedge Optimizer")
    hedge = PutSpreadHedge(S, T, r, sigma)
    budget = 2.0  # $2 max premium per contract

    result = hedge.optimize_spread(budget=budget, n_contracts=1)
    if result['K_long']:
        print(f"  Budget: ${budget:.2f}")
        print(f"  Optimal K_long  : {result['K_long']:.1f} ({result['K_long']/S:.1%} of spot)")
        print(f"  Optimal K_short : {result['K_short']:.1f} ({result['K_short']/S:.1%} of spot)")
        print(f"  Net cost        : ${result['cost']:.4f}")
        print(f"  Max payout      : ${result['max_payoff']:.4f}")
        print(f"  Expected payoff : ${result['expected_payoff']:.4f}")
        print(f"  Prot per dollar : {result['score']:.4f}")

        # P&L at various spot levels
        K_l = result['K_long']
        K_s = result['K_short']
        print(f"\n  Spread P&L profile (long {K_l:.0f}P / short {K_s:.0f}P):")
        print(f"  {'S_T':>6} | {'Payoff':>8} | {'Net PnL':>8}")
        print("  " + "-" * 28)
        for s_t in [70, 75, 80, 85, 90, 95, 100, 105]:
            payoff = hedge.spread_payoff(s_t, K_l, K_s)
            net = hedge.net_protection(s_t, K_l, K_s)
            print(f"  {s_t:>6} | {payoff:>8.4f} | {net:>8.4f}")

    # --- Tail risk metrics on simulated portfolio ---
    print("\n4. Tail Risk Metrics")
    rng = random.Random(42)
    # Simulate fat-tailed returns (mix of Gaussian and jumps)
    port_returns = []
    for _ in range(1000):
        r_norm = rng.gauss(0.0003, 0.015)
        if rng.random() < 0.02:  # 2% jump prob
            r_norm -= rng.random() * 0.05 + 0.02
        port_returns.append(r_norm)

    metrics = tail_risk_metrics(port_returns, confidence=0.99)
    print("  Metric              Value")
    print("  " + "-" * 30)
    for k, v in metrics.items():
        if not math.isnan(v):
            print(f"  {k:<22} {v:.6f}")

    # --- CVaR-optimal hedge ratio ---
    print("\n5. CVaR-Optimal Hedge Ratio")
    put_returns = [max(-port_r - 0.02, 0) - 0.003 for port_r in port_returns]
    hedge_result = cvar_optimal_hedge_ratio(port_returns, put_returns, n_points=20)
    print(f"  Optimal hedge ratio : {hedge_result['optimal_h']:.4f}")
    print(f"  Unhedged CVaR (99%) : {hedge_result['unhedged_CVaR']:.6f}")
    print(f"  Hedged CVaR (99%)   : {hedge_result['hedged_CVaR']:.6f}")
    print(f"  CVaR reduction      : {hedge_result['CVaR_reduction']:.6f} ({hedge_result['CVaR_reduction_pct']:.2%})")

    # --- P&L attribution ---
    print("\n6. Hedge P&L Attribution")
    spread_cost = result['cost'] if result['K_long'] else 1.5
    spread_payoffs = [max(result['K_long'] - S * math.exp(r_i), 0) -
                      max(result['K_short'] - S * math.exp(r_i), 0)
                      if result['K_long'] else 0.0
                      for r_i in port_returns]

    attr = hedge_pnl_attribution(port_returns, spread_payoffs, spread_cost)
    print(f"  Unhedged total P&L  : {attr['unhedged_total_pnl']:>10.4f}")
    print(f"  Hedge payoff        : {attr['hedge_payoff']:>10.4f}")
    print(f"  Hedge cost          : {attr['hedge_cost']:>10.4f}")
    print(f"  Net hedge P&L       : {attr['net_hedge_pnl']:>10.4f}")
    print(f"  Unhedged max DD     : {attr['unhedged_max_dd']:>10.4%}")
    print(f"  Hedged max DD       : {attr['hedged_max_dd']:>10.4%}")
    print(f"  Unhedged Sharpe     : {attr['unhedged_sharpe']:>10.4f}")
    print(f"  Hedged Sharpe       : {attr['hedged_sharpe']:>10.4f}")

    print("\n[Done] Day 24: Tail Risk Hedging complete.")
