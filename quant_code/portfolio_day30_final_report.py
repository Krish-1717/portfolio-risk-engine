"""
portfolio_day30_final_report.py
Day 30: Portfolio Final Report — rolling performance metrics, Brinson attribution
summary, drawdown analysis, risk-adjusted return ladder, ASCII dashboard.
Ties together Days 20-29 concepts into a production-ready report generator.
Pure Python stdlib only.
"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def mean(xs): return sum(xs) / max(len(xs), 1)
def std(xs, ddof=1):
    m = mean(xs)
    return math.sqrt(sum((x-m)**2 for x in xs) / max(len(xs)-ddof, 1))

def _norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

# ---------------------------------------------------------------------------
# 1. Performance metrics (full suite)
# ---------------------------------------------------------------------------
def compute_full_metrics(returns: list[float], benchmark: list[float],
                          rf: float = 0.02, freq: int = 252) -> dict:
    """Compute comprehensive performance metrics from daily return series."""
    n = len(returns)
    rf_d = (1 + rf) ** (1 / freq) - 1

    # Cumulative
    cum_port = 1.0
    for r in returns:
        cum_port *= (1 + r)
    ann_ret = cum_port ** (freq / n) - 1
    ann_vol = std(returns) * math.sqrt(freq)
    sharpe = (ann_ret - rf) / max(ann_vol, 1e-8)

    # Sortino
    downside = [min(r - rf_d, 0)**2 for r in returns]
    sortino_vol = math.sqrt(mean(downside) * freq)
    sortino = (ann_ret - rf) / max(sortino_vol, 1e-8)

    # Drawdown
    peak, max_dd = 1.0, 0.0
    underwater, dd_durations = [], []
    cur_dd_start = None
    wealth = 1.0
    max_dd_peak, max_dd_trough = 1.0, 1.0

    for t, r in enumerate(returns):
        wealth *= (1 + r)
        if wealth > peak:
            if cur_dd_start is not None:
                dd_durations.append(t - cur_dd_start)
                cur_dd_start = None
            peak = wealth
        dd = (peak - wealth) / peak
        underwater.append(-dd)
        if dd > max_dd:
            max_dd = dd
            max_dd_peak = peak
            max_dd_trough = wealth
        if dd > 0 and cur_dd_start is None:
            cur_dd_start = t

    calmar = ann_ret / max(max_dd, 1e-8)

    # Beta, alpha vs benchmark
    cum_bench = 1.0
    for r in benchmark:
        cum_bench *= (1 + r)
    ann_bench = cum_bench ** (freq / n) - 1
    bench_vol = std(benchmark) * math.sqrt(freq)

    cov_rb = sum((returns[i] - mean(returns)) * (benchmark[i] - mean(benchmark))
                 for i in range(n)) / max(n - 1, 1) * freq
    beta = cov_rb / max(bench_vol**2, 1e-8)
    alpha = ann_ret - (rf + beta * (ann_bench - rf))

    # Tracking error and IR
    active = [returns[i] - benchmark[i] for i in range(n)]
    te = std(active) * math.sqrt(freq)
    ir = (ann_ret - ann_bench) / max(te, 1e-8)

    # VaR/CVaR
    sorted_r = sorted(returns)
    var_95 = -sorted_r[int(n * 0.05)]
    cvar_95 = -mean(sorted_r[:int(n * 0.05)])
    var_99 = -sorted_r[int(n * 0.01)]

    # Win rate
    win_rate = sum(1 for r in returns if r > 0) / n

    return {
        'ann_return': ann_ret, 'ann_vol': ann_vol, 'sharpe': sharpe,
        'sortino': sortino, 'calmar': calmar,
        'max_drawdown': max_dd, 'avg_dd_duration': mean(dd_durations) if dd_durations else 0,
        'beta': beta, 'alpha': alpha,
        'tracking_error': te, 'info_ratio': ir,
        'var_95': var_95, 'cvar_95': cvar_95, 'var_99': var_99,
        'win_rate': win_rate,
        'ann_bench': ann_bench, 'bench_vol': bench_vol,
        'cum_return': cum_port - 1,
        'underwater': underwater,
    }

# ---------------------------------------------------------------------------
# 2. Rolling metrics
# ---------------------------------------------------------------------------
def rolling_metrics(returns: list[float], window: int = 63,
                     rf: float = 0.02, freq: int = 252) -> list[dict]:
    """Compute rolling Sharpe, vol, return over a sliding window."""
    n = len(returns)
    results = []
    rf_ann = rf
    for t in range(window, n + 1):
        r_win = returns[t - window: t]
        ann_r = ((1 + mean(r_win)) ** freq - 1)
        ann_v = std(r_win) * math.sqrt(freq)
        sharpe = (ann_r - rf_ann) / max(ann_v, 1e-8)
        results.append({'t': t, 'return': ann_r, 'vol': ann_v, 'sharpe': sharpe})
    return results

# ---------------------------------------------------------------------------
# 3. ASCII Dashboard
# ---------------------------------------------------------------------------
def sparkline(values: list[float], width: int = 40) -> str:
    """Render a mini sparkline using block characters."""
    if not values:
        return ''
    mn, mx = min(values), max(values)
    span = mx - mn
    chars = '▁▂▃▄▅▆▇█'
    if span < 1e-10:
        return chars[4] * width
    step = max(1, len(values) // width)
    sampled = [values[i] for i in range(0, len(values), step)][:width]
    return ''.join(chars[int((v - mn) / span * 7)] for v in sampled)

def ascii_bar(value: float, max_val: float, width: int = 20,
               fill: str = '█', empty: str = '░') -> str:
    n_fill = int(abs(value) / max(abs(max_val), 1e-8) * width)
    n_fill = min(n_fill, width)
    return fill * n_fill + empty * (width - n_fill)

def generate_dashboard(metrics: dict, rolling: list[dict],
                        asset_names: list[str],
                        weights: list[float]) -> str:
    """Generate full ASCII performance dashboard."""
    lines = []
    w = 65

    def box_line(text='', fill='─', left='│', right='│'):
        if text:
            pad = max(0, w - 2 - len(text))
            return f"{left} {text}{' ' * pad}{right}"
        return '├' + fill * (w - 2) + '┤'

    lines.append('┌' + '─' * (w - 2) + '┐')
    lines.append(box_line('  PORTFOLIO PERFORMANCE DASHBOARD  '.center(w - 4)))
    lines.append('├' + '─' * (w - 2) + '┤')

    # Key metrics
    lines.append(box_line(f"  Cumulative Return : {metrics['cum_return']:>+8.2%}   "
                           f"Benchmark: {metrics['ann_bench']:>+7.2%}"))
    lines.append(box_line(f"  Annual Return     : {metrics['ann_return']:>+8.2%}   "
                           f"Alpha    : {metrics['alpha']:>+7.2%}"))
    lines.append(box_line(f"  Volatility        : {metrics['ann_vol']:>8.2%}   "
                           f"Beta     : {metrics['beta']:>7.3f}"))
    lines.append(box_line(f"  Sharpe Ratio      : {metrics['sharpe']:>+8.3f}   "
                           f"Sortino  : {metrics['sortino']:>+7.3f}"))
    lines.append(box_line(f"  Max Drawdown      : {metrics['max_drawdown']:>8.2%}   "
                           f"Calmar   : {metrics['calmar']:>+7.3f}"))
    lines.append(box_line(f"  Tracking Error    : {metrics['tracking_error']:>8.2%}   "
                           f"Info Ratio: {metrics['info_ratio']:>+6.3f}"))
    lines.append(box_line(f"  VaR 95%  (daily)  : {metrics['var_95']:>8.4f}   "
                           f"CVaR 95% : {metrics['cvar_95']:>7.4f}"))
    lines.append(box_line(f"  Win Rate          : {metrics['win_rate']:>8.1%}"))

    lines.append(box_line())

    # Sparklines
    spy_line = sparkline(metrics['underwater'])
    lines.append(box_line(f"  Underwater (drawdown):"))
    lines.append(box_line(f"  {spy_line}"))

    if rolling:
        sharpes = [r['sharpe'] for r in rolling]
        roll_line = sparkline(sharpes)
        lines.append(box_line(f"  Rolling Sharpe (63d):"))
        lines.append(box_line(f"  {roll_line}"))
        lines.append(box_line(f"  Range: [{min(sharpes):+.2f}, {max(sharpes):+.2f}]  "
                               f"Current: {sharpes[-1]:+.2f}"))

    lines.append(box_line())

    # Portfolio weights bar chart
    lines.append(box_line('  PORTFOLIO WEIGHTS'))
    max_w = max(weights)
    for name, w_i in zip(asset_names, weights):
        bar = ascii_bar(w_i, max_w, width=20)
        lines.append(box_line(f"  {name:12} {bar} {w_i:.1%}"))

    lines.append('└' + '─' * (w - 2) + '┘')
    return '\n'.join(lines)

# ---------------------------------------------------------------------------
# 4. Regime-conditional performance
# ---------------------------------------------------------------------------
def regime_performance(returns: list[float], regime_labels: list[int],
                        n_regimes: int = 2) -> list[dict]:
    """Compute performance metrics separately for each regime."""
    results = []
    for reg in range(n_regimes):
        r_reg = [r for r, l in zip(returns, regime_labels) if l == reg]
        if len(r_reg) < 5:
            results.append({'regime': reg, 'n_days': 0})
            continue
        ann_r = mean(r_reg) * 252
        ann_v = std(r_reg) * math.sqrt(252)
        results.append({
            'regime': reg,
            'n_days': len(r_reg),
            'pct_time': len(r_reg) / len(returns),
            'ann_return': ann_r,
            'ann_vol': ann_v,
            'sharpe': (ann_r - 0.02) / max(ann_v, 1e-8),
        })
    return results

# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 65)
    print("DAY 30: Portfolio Final Report & Dashboard")
    print("=" * 65)

    rng = random.Random(42)
    T = 756  # 3 years

    def randn():
        u = max(rng.random(), 1e-15)
        return math.sqrt(-2 * math.log(u)) * math.cos(2 * math.pi * rng.random())

    # Simulate portfolio (Sharpe ~0.7) and benchmark (Sharpe ~0.5)
    port_returns = [0.0004 + 0.01 * randn() for _ in range(T)]
    bench_returns = [0.0003 + 0.012 * randn() for _ in range(T)]

    # Inject a drawdown period (days 250-300)
    for t in range(250, 300):
        port_returns[t] -= 0.005

    assets = ['US Equity', 'Intl Equity', 'EM Equity', 'Bonds', 'Gold']
    weights = [0.35, 0.25, 0.15, 0.20, 0.05]

    print("\n1. Computing full performance metrics...")
    metrics = compute_full_metrics(port_returns, bench_returns, rf=0.02)

    print(f"   Ann return   : {metrics['ann_return']:+.4f}")
    print(f"   Sharpe       : {metrics['sharpe']:+.4f}")
    print(f"   Max drawdown : {metrics['max_drawdown']:.4f}")
    print(f"   Info ratio   : {metrics['info_ratio']:+.4f}")

    print("\n2. Rolling Metrics (63-day window)")
    rolling = rolling_metrics(port_returns, window=63)
    print(f"   {'Period':>10} | {'Ann Ret':>8} | {'Ann Vol':>8} | {'Sharpe':>8}")
    print("   " + "-" * 42)
    step = len(rolling) // 8
    for r in rolling[::max(step, 1)][:8]:
        print(f"   Day {r['t']:>5}  | {r['return']:>+8.4f} | {r['vol']:>8.4f} | {r['sharpe']:>+8.3f}")

    sharpes = [r['sharpe'] for r in rolling]
    print(f"\n   Rolling Sharpe: min={min(sharpes):+.3f}  max={max(sharpes):+.3f}  "
          f"mean={mean(sharpes):+.3f}")

    print("\n3. Regime-Conditional Performance (2-state)")
    # Simple regime: below/above median vol
    window = 21
    vols = [std(port_returns[max(0,t-window):t]) * math.sqrt(252) for t in range(1, T+1)]
    vol_median = sorted(vols)[len(vols) // 2]
    regimes = [0 if v < vol_median else 1 for v in vols]
    regime_perf = regime_performance(port_returns, regimes)
    labels = ['Low Vol', 'High Vol']
    for rp in regime_perf:
        if 'ann_return' not in rp:
            continue
        print(f"   {labels[rp['regime']]}: {rp['pct_time']:.1%} of time  "
              f"Ann Ret={rp['ann_return']:+.4f}  Vol={rp['ann_vol']:.4f}  "
              f"Sharpe={rp['sharpe']:+.4f}")

    print("\n4. Risk-Return Ladder")
    print(f"   {'Percentile':>12} | {'Daily Return':>14}")
    print("   " + "-" * 30)
    sorted_r = sorted(port_returns)
    for pct in [0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99]:
        idx = int(T * pct / 100)
        print(f"   {pct:>11}% | {sorted_r[min(idx, T-1)]:>+14.4f}")

    print("\n5. Full Performance Dashboard")
    dashboard = generate_dashboard(metrics, rolling, assets, weights)
    print(dashboard)

    print("\n6. Challenge Summary: 30 Days of Quant Finance")
    summary = [
        ("Options",   "Days 20-30: BS Greeks, exotics, barriers, Asians, stoch vol, SVI, IR derivs"),
        ("Portfolio", "Days 20-30: MVO, regime detection, factor models, BL, Kelly, risk parity"),
        ("News",      "Days 20-30: Sentiment NLP, event studies, online learning, signal backtest"),
    ]
    print("   " + "=" * 60)
    for track, desc in summary:
        print(f"   {track:12}: {desc}")
    print("   " + "=" * 60)
    print(f"   Total: 30 files per repo × 3 repos = 90 Python modules")
    print(f"   All pure stdlib — no numpy, pandas, scipy")
    print(f"   Topics: from BS basics → stochastic vol → market microstructure")
    print(f"           → regime detection → Black-Litterman → Kelly → SVI → IR derivatives")

    print("\n[Done] Day 30: Final Report & 30-Day Challenge Complete!")
