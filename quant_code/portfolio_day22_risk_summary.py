"""
portfolio_day22_risk_summary.py
Day 22: Risk Summary Report — Portfolio-level risk metrics, risk budget,
factor contribution, ASCII table formatting, alert thresholds.
Pure Python stdlib only.
"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------
def mean(xs: list[float]) -> float:
    return sum(xs) / max(len(xs), 1)

def variance(xs: list[float]) -> float:
    m = mean(xs)
    return sum((x - m)**2 for x in xs) / max(len(xs) - 1, 1)

def std(xs: list[float]) -> float:
    return math.sqrt(max(variance(xs), 0.0))

def _norm_ppf(p: float) -> float:
    """Approximate inverse normal CDF via bisection."""
    p = max(1e-9, min(1 - 1e-9, p))
    lo, hi = -10.0, 10.0
    norm_cdf = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
    for _ in range(60):
        mid = (lo + hi) / 2
        (lo if norm_cdf(mid) < p else None) or setattr(type('_', (), {})(), '_', hi := mid)
        if norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2

# ---------------------------------------------------------------------------
# Core risk metrics
# ---------------------------------------------------------------------------
def compute_var(returns: list[float], confidence: float = 0.95) -> float:
    """Historical VaR (negative returns → positive VaR)."""
    sorted_r = sorted(returns)
    idx = int((1 - confidence) * len(sorted_r))
    return -sorted_r[max(idx, 0)]

def compute_cvar(returns: list[float], confidence: float = 0.95) -> float:
    """Expected Shortfall / CVaR."""
    sorted_r = sorted(returns)
    idx = int((1 - confidence) * len(sorted_r))
    tail = sorted_r[:max(idx, 1)]
    return -mean(tail)

def compute_max_drawdown(returns: list[float]) -> tuple[float, int, int]:
    """Max drawdown and peak/trough indices."""
    cum = 1.0
    nav = [1.0]
    for r in returns:
        cum *= (1 + r)
        nav.append(cum)

    peak = nav[0]
    max_dd = 0.0
    peak_idx = trough_idx = 0
    cur_peak_idx = 0

    for i, v in enumerate(nav):
        if v > peak:
            peak = v
            cur_peak_idx = i
        dd = (peak - v) / peak
        if dd > max_dd:
            max_dd = dd
            peak_idx = cur_peak_idx
            trough_idx = i

    return max_dd, peak_idx, trough_idx

def compute_sharpe(returns: list[float], rf_annual: float = 0.05,
                   periods_per_year: int = 252) -> float:
    rf_per = rf_annual / periods_per_year
    excess = [r - rf_per for r in returns]
    return mean(excess) / max(std(excess), 1e-10) * math.sqrt(periods_per_year)

def compute_sortino(returns: list[float], rf_annual: float = 0.05,
                    periods_per_year: int = 252) -> float:
    rf_per = rf_annual / periods_per_year
    excess = [r - rf_per for r in returns]
    downside = [min(r, 0.0) for r in excess]
    downside_std = math.sqrt(sum(d**2 for d in downside) / max(len(downside) - 1, 1))
    return mean(excess) / max(downside_std, 1e-10) * math.sqrt(periods_per_year)

def compute_calmar(returns: list[float], periods_per_year: int = 252) -> float:
    ann_return = mean(returns) * periods_per_year
    max_dd, _, _ = compute_max_drawdown(returns)
    return ann_return / max(max_dd, 1e-10)

def compute_beta(portfolio_returns: list[float],
                 benchmark_returns: list[float]) -> tuple[float, float]:
    """Returns (beta, alpha_annualized)."""
    n = min(len(portfolio_returns), len(benchmark_returns))
    p, b = portfolio_returns[:n], benchmark_returns[:n]

    bm = mean(b)
    pm = mean(p)
    cov_pb = sum((p[i] - pm) * (b[i] - bm) for i in range(n)) / max(n - 1, 1)
    var_b = variance(b)

    beta = cov_pb / max(var_b, 1e-10)
    alpha = (pm - beta * bm) * 252  # annualized
    return beta, alpha

def compute_tracking_error(portfolio_returns: list[float],
                            benchmark_returns: list[float],
                            periods_per_year: int = 252) -> float:
    n = min(len(portfolio_returns), len(benchmark_returns))
    active = [portfolio_returns[i] - benchmark_returns[i] for i in range(n)]
    return std(active) * math.sqrt(periods_per_year)

def compute_info_ratio(portfolio_returns: list[float],
                        benchmark_returns: list[float],
                        periods_per_year: int = 252) -> float:
    n = min(len(portfolio_returns), len(benchmark_returns))
    active = [portfolio_returns[i] - benchmark_returns[i] for i in range(n)]
    te = std(active) * math.sqrt(periods_per_year)
    active_return = mean(active) * periods_per_year
    return active_return / max(te, 1e-10)

# ---------------------------------------------------------------------------
# Risk budget / factor contribution
# ---------------------------------------------------------------------------
def marginal_contribution_to_risk(weights: list[float],
                                   cov: list[list[float]]) -> list[float]:
    """MCR_i = (C*w)_i / sqrt(w'*C*w)"""
    n = len(weights)
    Cw = [sum(cov[i][j] * weights[j] for j in range(n)) for i in range(n)]
    port_var = sum(weights[i] * Cw[i] for i in range(n))
    port_std = math.sqrt(max(port_var, 1e-10))
    return [Cw[i] / port_std for i in range(n)]

def risk_contribution(weights: list[float], cov: list[list[float]]) -> list[float]:
    """RC_i = w_i * MCR_i"""
    mcr = marginal_contribution_to_risk(weights, cov)
    return [weights[i] * mcr[i] for i in range(len(weights))]

def percent_risk_contribution(weights: list[float], cov: list[list[float]]) -> list[float]:
    """Percentage risk contribution per asset."""
    rc = risk_contribution(weights, cov)
    total = sum(abs(r) for r in rc)
    return [r / max(total, 1e-10) for r in rc]

def sample_cov_matrix(returns_matrix: list[list[float]]) -> list[list[float]]:
    """Compute sample covariance matrix from TxN returns matrix."""
    T, N = len(returns_matrix), len(returns_matrix[0])
    mu = [sum(returns_matrix[t][i] for t in range(T)) / T for i in range(N)]
    cov = [[0.0] * N for _ in range(N)]
    for t in range(T):
        for i in range(N):
            for j in range(N):
                cov[i][j] += (returns_matrix[t][i] - mu[i]) * (returns_matrix[t][j] - mu[j])
    for i in range(N):
        for j in range(N):
            cov[i][j] /= max(T - 1, 1)
    return cov

# ---------------------------------------------------------------------------
# Alert thresholds
# ---------------------------------------------------------------------------
@dataclass
class RiskThresholds:
    max_var_95: float = 0.02       # 2% daily VaR at 95%
    max_cvar_95: float = 0.03      # 3% daily CVaR at 95%
    max_drawdown: float = 0.15     # 15% max drawdown
    min_sharpe: float = 0.5        # annualized Sharpe
    max_vol: float = 0.20          # 20% annualized vol
    max_concentration: float = 0.40  # no single asset > 40% risk contrib

def check_alerts(metrics: dict, thresholds: RiskThresholds) -> list[dict]:
    alerts = []
    checks = [
        ('VaR 95%', metrics.get('var_95', 0), thresholds.max_var_95, '>'),
        ('CVaR 95%', metrics.get('cvar_95', 0), thresholds.max_cvar_95, '>'),
        ('Max Drawdown', metrics.get('max_drawdown', 0), thresholds.max_drawdown, '>'),
        ('Annualized Sharpe', metrics.get('sharpe', 0), thresholds.min_sharpe, '<'),
        ('Annualized Vol', metrics.get('ann_vol', 0), thresholds.max_vol, '>'),
        ('Max Risk Contrib', metrics.get('max_risk_contrib', 0), thresholds.max_concentration, '>'),
    ]
    for name, value, threshold, direction in checks:
        triggered = (value > threshold) if direction == '>' else (value < threshold)
        alerts.append({
            'metric': name,
            'value': value,
            'threshold': threshold,
            'direction': direction,
            'alert': triggered,
            'severity': 'HIGH' if triggered else 'OK',
        })
    return alerts

# ---------------------------------------------------------------------------
# ASCII table formatter
# ---------------------------------------------------------------------------
def ascii_table(headers: list[str], rows: list[list[str]], col_widths: Optional[list[int]] = None) -> str:
    if col_widths is None:
        col_widths = [max(len(h), max((len(str(r[i])) for r in rows), default=0))
                      for i, h in enumerate(headers)]

    sep = '+' + '+'.join('-' * (w + 2) for w in col_widths) + '+'
    fmt_row = lambda cells: '|' + '|'.join(f' {str(c):<{col_widths[i]}} ' for i, c in enumerate(cells)) + '|'

    lines = [sep, fmt_row(headers), sep]
    for row in rows:
        lines.append(fmt_row(row))
    lines.append(sep)
    return '\n'.join(lines)

# ---------------------------------------------------------------------------
# Full risk report
# ---------------------------------------------------------------------------
def generate_risk_report(portfolio_returns: list[float],
                          benchmark_returns: list[float],
                          weights: list[float],
                          asset_names: list[str],
                          returns_matrix: list[list[float]],
                          thresholds: Optional[RiskThresholds] = None,
                          rf: float = 0.05,
                          periods: int = 252) -> str:
    if thresholds is None:
        thresholds = RiskThresholds()

    ann_vol = std(portfolio_returns) * math.sqrt(periods)
    ann_ret = mean(portfolio_returns) * periods
    var95 = compute_var(portfolio_returns, 0.95)
    var99 = compute_var(portfolio_returns, 0.99)
    cvar95 = compute_cvar(portfolio_returns, 0.95)
    max_dd, pk, tr = compute_max_drawdown(portfolio_returns)
    sharpe = compute_sharpe(portfolio_returns, rf, periods)
    sortino = compute_sortino(portfolio_returns, rf, periods)
    calmar = compute_calmar(portfolio_returns, periods)
    beta, alpha = compute_beta(portfolio_returns, benchmark_returns)
    te = compute_tracking_error(portfolio_returns, benchmark_returns, periods)
    ir = compute_info_ratio(portfolio_returns, benchmark_returns, periods)

    cov = sample_cov_matrix(returns_matrix)
    prc = percent_risk_contribution(weights, cov)
    max_rc = max(abs(p) for p in prc)

    metrics = {
        'ann_vol': ann_vol, 'var_95': var95, 'cvar_95': cvar95,
        'max_drawdown': max_dd, 'sharpe': sharpe,
        'max_risk_contrib': max_rc,
    }
    alerts = check_alerts(metrics, thresholds)

    lines = []
    lines.append("=" * 65)
    lines.append("PORTFOLIO RISK SUMMARY REPORT")
    lines.append("=" * 65)

    # Section 1: Return & Risk
    lines.append("\n[1] RETURN & RISK METRICS")
    rows_rr = [
        ['Ann. Return', f'{ann_ret:+.2%}'],
        ['Ann. Volatility', f'{ann_vol:.2%}'],
        ['VaR 95% (daily)', f'{var95:.4%}'],
        ['VaR 99% (daily)', f'{var99:.4%}'],
        ['CVaR 95% (daily)', f'{cvar95:.4%}'],
        ['Max Drawdown', f'{max_dd:.2%}'],
    ]
    lines.append(ascii_table(['Metric', 'Value'], rows_rr, [20, 12]))

    # Section 2: Ratios
    lines.append("\n[2] RISK-ADJUSTED RATIOS")
    rows_ratio = [
        ['Sharpe Ratio', f'{sharpe:.3f}'],
        ['Sortino Ratio', f'{sortino:.3f}'],
        ['Calmar Ratio', f'{calmar:.3f}'],
        ['Beta', f'{beta:.3f}'],
        ['Alpha (ann)', f'{alpha:+.2%}'],
        ['Tracking Error', f'{te:.2%}'],
        ['Info Ratio', f'{ir:.3f}'],
    ]
    lines.append(ascii_table(['Metric', 'Value'], rows_ratio, [20, 12]))

    # Section 3: Risk Budget
    lines.append("\n[3] RISK BUDGET (% Contribution to Portfolio Vol)")
    rc_rows = [[name, f'{w:.1%}', f'{abs(prc[i]):.1%}',
                '⚠' if abs(prc[i]) > thresholds.max_concentration else '✓']
               for i, (name, w) in enumerate(zip(asset_names, weights))]
    lines.append(ascii_table(['Asset', 'Weight', 'Risk%', 'Status'], rc_rows, [12, 8, 8, 7]))

    # Section 4: Alerts
    lines.append("\n[4] ALERT DASHBOARD")
    alert_rows = [[a['metric'], f"{a['value']:.4f}", f"{a['threshold']:.4f}", a['severity']]
                  for a in alerts]
    lines.append(ascii_table(['Metric', 'Current', 'Threshold', 'Status'], alert_rows,
                              [22, 10, 12, 8]))

    n_alerts = sum(1 for a in alerts if a['alert'])
    lines.append(f"\n  {n_alerts}/{len(alerts)} alerts triggered")

    return '\n'.join(lines)

# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    rng = random.Random(42)
    T = 252  # 1 year of daily returns
    N = 5

    def randn():
        u = max(rng.random(), 1e-15)
        return math.sqrt(-2*math.log(u)) * math.cos(2*math.pi*rng.random())

    # Simulate portfolio: mix of assets with different risk profiles
    vols = [0.15, 0.20, 0.25, 0.12, 0.18]  # per asset
    mus = [0.0005, 0.0003, 0.0006, 0.0002, 0.0004]

    returns_matrix = [[mus[i] + vols[i]/math.sqrt(252) * randn() for i in range(N)]
                      for _ in range(T)]
    weights = [0.30, 0.25, 0.20, 0.15, 0.10]
    asset_names = ['US Equity', 'Intl Eq', 'EM Equity', 'US Bonds', 'Commodit']

    portfolio_returns = [sum(weights[i] * returns_matrix[t][i] for i in range(N))
                         for t in range(T)]
    benchmark_returns = [sum(1/N * returns_matrix[t][i] for i in range(N))
                         for t in range(T)]

    report = generate_risk_report(
        portfolio_returns, benchmark_returns, weights, asset_names,
        returns_matrix, thresholds=RiskThresholds(), rf=0.05
    )
    print(report)
    print("\n[Done] Day 22: Risk Summary Report complete.")
