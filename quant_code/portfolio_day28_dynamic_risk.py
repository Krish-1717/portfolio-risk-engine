"""
portfolio_day28_dynamic_risk.py
Day 28: Dynamic Risk Budgeting & Kelly Criterion —
EWMA covariance updating, risk parity rebalancing, Kelly portfolio
(single-asset and multi-asset), fractional Kelly, turnover constraints.
Pure Python stdlib only.
"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Matrix helpers
# ---------------------------------------------------------------------------
def mat_vec(A, v):
    return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]

def vec_dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def mat_inv(A):
    n = len(A)
    aug = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(A)]
    for col in range(n):
        max_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        aug[col], aug[max_row] = aug[max_row], aug[col]
        pivot = aug[col][col]
        if abs(pivot) < 1e-12:
            raise ValueError("Singular matrix")
        aug[col] = [x / pivot for x in aug[col]]
        for row in range(n):
            if row != col:
                f = aug[row][col]
                aug[row] = [aug[row][k] - f * aug[col][k] for k in range(2 * n)]
    return [row[n:] for row in aug]

def cholesky(A):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                L[i][j] = math.sqrt(max(A[i][i] - s, 1e-12))
            else:
                L[i][j] = (A[i][j] - s) / max(L[j][j], 1e-12)
    return L

# ---------------------------------------------------------------------------
# 1. EWMA covariance estimation (dynamic)
# ---------------------------------------------------------------------------
def ewma_covariance(returns_matrix: list[list[float]],
                     lambda_: float = 0.94) -> list[list[float]]:
    """
    RiskMetrics EWMA covariance: Σ_t = λ*Σ_{t-1} + (1-λ)*r_{t-1}*r_{t-1}'.
    returns_matrix[t][i] = return of asset i at time t.
    Returns most recent covariance estimate.
    """
    T = len(returns_matrix)
    n = len(returns_matrix[0])
    # Initialize with outer product of first observation
    r0 = returns_matrix[0]
    sigma = [[r0[i] * r0[j] for j in range(n)] for i in range(n)]

    for t in range(1, T):
        r = returns_matrix[t]
        outer = [[r[i] * r[j] for j in range(n)] for i in range(n)]
        sigma = [
            [lambda_ * sigma[i][j] + (1 - lambda_) * outer[i][j] for j in range(n)]
            for i in range(n)
        ]
    return sigma

def ewma_covariance_series(returns_matrix: list[list[float]],
                            lambda_: float = 0.94) -> list[list[list[float]]]:
    """Return full series of EWMA covariance estimates."""
    T = len(returns_matrix)
    n = len(returns_matrix[0])
    r0 = returns_matrix[0]
    sigma = [[r0[i] * r0[j] for j in range(n)] for i in range(n)]
    series = [sigma]
    for t in range(1, T):
        r = returns_matrix[t]
        outer = [[r[i] * r[j] for j in range(n)] for i in range(n)]
        sigma = [
            [lambda_ * sigma[i][j] + (1 - lambda_) * outer[i][j] for j in range(n)]
            for i in range(n)
        ]
        series.append(sigma)
    return series

# ---------------------------------------------------------------------------
# 2. Risk Parity (Equal Risk Contribution)
# ---------------------------------------------------------------------------
def risk_contribution(weights: list[float], sigma: list[list[float]]) -> list[float]:
    """RC_i = w_i * (Σw)_i / sqrt(w'Σw) — fractional risk contribution."""
    n = len(weights)
    sigma_w = mat_vec(sigma, weights)
    port_var = vec_dot(weights, sigma_w)
    port_vol = math.sqrt(max(port_var, 1e-12))
    return [weights[i] * sigma_w[i] / port_vol for i in range(n)]

def risk_parity_weights(sigma: list[list[float]], risk_budgets: list[float] | None = None,
                         n_iter: int = 300, lr: float = 0.01, tol: float = 1e-7) -> list[float]:
    """
    Cyclical coordinate descent for equal (or budgeted) risk contribution.
    Spinu (2013): each w_i update sets RC_i = budget_i analytically.
    """
    n = len(sigma)
    if risk_budgets is None:
        risk_budgets = [1.0 / n] * n
    b = risk_budgets

    w = [1.0 / n] * n  # initial equal weight
    for _ in range(n_iter):
        for i in range(n):
            # Solve: w_i * (sigma_ii * w_i + sum_{j!=i} sigma_ij * w_j) = b_i * port_var
            sigma_w = mat_vec(sigma, w)
            port_var = vec_dot(w, sigma_w)
            # Quadratic in w_i: sigma_ii * w_i^2 + A_i * w_i - b_i * port_var = 0
            A_i = sum(sigma[i][j] * w[j] for j in range(n) if j != i)
            sii = sigma[i][i]
            # Positive root of: sii*w_i^2 + A_i*w_i - b_i*port_var = 0
            disc = A_i**2 + 4 * sii * b[i] * port_var
            w_new = (-A_i + math.sqrt(max(disc, 0))) / (2 * sii)
            w[i] = max(w_new, 1e-8)
        # Normalize
        total = sum(w)
        w = [x / total for x in w]

    return w

def dynamic_risk_parity(returns_matrix: list[list[float]],
                         lambda_: float = 0.94,
                         rebal_freq: int = 21) -> list[list[float]]:
    """
    Rolling risk parity: rebalance every `rebal_freq` days using EWMA cov.
    Returns list of weight vectors (one per day, updated at rebalance).
    """
    T = len(returns_matrix)
    n = len(returns_matrix[0])
    sigma_series = ewma_covariance_series(returns_matrix, lambda_)

    weights_history = []
    current_w = [1.0 / n] * n

    for t in range(T):
        if t % rebal_freq == 0 and t > 0:
            try:
                current_w = risk_parity_weights(sigma_series[t])
            except Exception:
                pass  # keep previous weights if optimization fails
        weights_history.append(current_w[:])

    return weights_history

# ---------------------------------------------------------------------------
# 3. Kelly Criterion
# ---------------------------------------------------------------------------
def kelly_single_asset(mu: float, sigma: float, rf: float = 0.0) -> float:
    """
    Single-asset Kelly fraction: f* = (mu - rf) / sigma^2.
    Maximizes E[log(1 + f*R)].
    """
    return (mu - rf) / max(sigma**2, 1e-10)

def kelly_multiasset(mu: list[float], sigma: list[list[float]],
                      rf: float = 0.0) -> list[float]:
    """
    Multi-asset Kelly: f* = Sigma^{-1} * (mu - rf).
    Returns unconstrained Kelly fractions (can be > 1).
    """
    n = len(mu)
    excess = [mu[i] - rf for i in range(n)]
    sigma_inv = mat_inv(sigma)
    return mat_vec(sigma_inv, excess)

def fractional_kelly(kelly_weights: list[float], fraction: float = 0.5) -> list[float]:
    """Apply fractional Kelly: scale toward cash (reduces leverage + variance)."""
    return [f * fraction for f in kelly_weights]

def constrained_kelly(kelly_weights: list[float],
                       max_gross_leverage: float = 2.0,
                       min_weight: float = -0.5,
                       max_weight: float = 1.0) -> list[float]:
    """
    Project Kelly weights to box + gross leverage constraint.
    Scales all weights proportionally if gross leverage exceeded.
    """
    w = [max(min_weight, min(max_weight, f)) for f in kelly_weights]
    gross = sum(abs(x) for x in w)
    if gross > max_gross_leverage:
        scale = max_gross_leverage / gross
        w = [x * scale for x in w]
    return w

def kelly_growth_rate(weights: list[float], mu: list[float],
                       sigma: list[list[float]]) -> float:
    """
    Log-optimal portfolio growth rate: g = f'μ - 0.5 * f'Σf.
    Exact for log-normal returns.
    """
    n = len(weights)
    sigma_f = mat_vec(sigma, weights)
    port_var = vec_dot(weights, sigma_f)
    port_ret = sum(weights[i] * mu[i] for i in range(n))
    return port_ret - 0.5 * port_var

# ---------------------------------------------------------------------------
# 4. Turnover-constrained rebalancing
# ---------------------------------------------------------------------------
def turnover_constrained_rebalance(w_current: list[float],
                                    w_target: list[float],
                                    max_turnover: float = 0.10) -> list[float]:
    """
    Move from w_current toward w_target, limited by max_turnover (one-way).
    Greedy: apply largest moves first.
    """
    n = len(w_current)
    diffs = [(abs(w_target[i] - w_current[i]), i) for i in range(n)]
    diffs.sort(reverse=True)

    w_new = w_current[:]
    remaining_to = max_turnover
    for _, i in diffs:
        if remaining_to <= 0:
            break
        delta = w_target[i] - w_current[i]
        capped = max(-remaining_to, min(remaining_to, delta))
        w_new[i] += capped
        remaining_to -= abs(capped)

    # Renormalize
    total = sum(w_new)
    if total > 1e-10:
        w_new = [x / total for x in w_new]
    return w_new

# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 65)
    print("DAY 28: Dynamic Risk Budgeting & Kelly Criterion")
    print("=" * 65)

    rng = random.Random(42)
    N = 5
    T_sim = 252
    assets = ['US Equity', 'Intl Equity', 'EM Equity', 'US Bonds', 'Gold']

    # True parameters (annualized)
    mu_true = [0.10, 0.08, 0.12, 0.04, 0.05]
    vols    = [0.20, 0.23, 0.30, 0.04, 0.14]
    corr    = [
        [1.00, 0.75, 0.65, 0.05, 0.10],
        [0.75, 1.00, 0.70, 0.03, 0.08],
        [0.65, 0.70, 1.00, 0.00, 0.12],
        [0.05, 0.03, 0.00, 1.00, -0.05],
        [0.10, 0.08, 0.12, -0.05, 1.00],
    ]
    sigma_ann = [[corr[i][j] * vols[i] * vols[j] for j in range(N)] for i in range(N)]
    mu_daily = [m / 252 for m in mu_true]
    sigma_daily = [[sigma_ann[i][j] / 252 for j in range(N)] for i in range(N)]

    # Simulate returns
    L = cholesky(sigma_daily)
    def randn():
        u = max(rng.random(), 1e-15)
        return math.sqrt(-2 * math.log(u)) * math.cos(2 * math.pi * rng.random())

    returns_sim = []
    for _ in range(T_sim):
        z = [randn() for _ in range(N)]
        r = [mu_daily[i] + sum(L[i][j] * z[j] for j in range(i + 1)) for i in range(N)]
        returns_sim.append(r)

    print("\n1. EWMA Covariance (λ=0.94) vs True Covariance")
    sigma_ewma = ewma_covariance(returns_sim)
    print(f"   {'':20} | {'EWMA vol':>10} | {'True vol':>10}")
    print("   " + "-" * 45)
    for i, a in enumerate(assets):
        ewma_vol = math.sqrt(sigma_ewma[i][i] * 252)
        true_vol = vols[i]
        print(f"   {a:20} | {ewma_vol:>10.4f} | {true_vol:>10.4f}")

    print("\n2. Risk Parity Weights (Equal Risk Contribution)")
    w_rp = risk_parity_weights(sigma_ann)
    rc = risk_contribution(w_rp, sigma_ann)
    port_vol_rp = math.sqrt(vec_dot(w_rp, mat_vec(sigma_ann, w_rp)))
    print(f"   {'Asset':20} | {'Weight':>8} | {'Risk Contrib':>14} | {'RC%':>6}")
    print("   " + "-" * 56)
    for a, w, r in zip(assets, w_rp, rc):
        print(f"   {a:20} | {w:>8.4f} | {r:>14.4f} | {r/port_vol_rp:>6.1%}")
    print(f"   Portfolio vol (ann): {port_vol_rp:.4f}")

    print("\n3. Dynamic Risk Parity (rebalance every 21 days)")
    w_hist = dynamic_risk_parity(returns_sim, lambda_=0.94, rebal_freq=21)
    # Show weights at key dates
    for t in [0, 63, 126, 189, 251]:
        w_t = w_hist[t]
        print(f"   Day {t:3d}: " + "  ".join(f"{a[:4]}={w:.3f}" for a, w in zip(assets, w_t)))

    # Compute portfolio returns under dynamic RP
    port_returns_rp = [vec_dot(w_hist[t], returns_sim[t]) for t in range(T_sim)]
    cum_ret_rp = 1.0
    for r in port_returns_rp:
        cum_ret_rp *= (1 + r)
    ann_ret_rp = cum_ret_rp ** (252 / T_sim) - 1
    ann_vol_rp = math.sqrt(sum(r**2 for r in port_returns_rp) / T_sim * 252)
    print(f"\n   Dynamic RP performance (1-year sim):")
    print(f"   Annual return: {ann_ret_rp:.4f}  Vol: {ann_vol_rp:.4f}  Sharpe: {ann_ret_rp/ann_vol_rp:.4f}")

    print("\n4. Kelly Criterion")
    rf = 0.02
    print("   Single-asset Kelly fractions:")
    for a, mu, vol in zip(assets, mu_true, vols):
        f = kelly_single_asset(mu, vol, rf)
        print(f"   {a:20}: f*={f:.4f}  (leverage={f:.1f}x)")

    print("\n   Multi-asset Kelly portfolio:")
    kelly_w = kelly_multiasset(mu_true, sigma_ann, rf)
    print(f"   {'Asset':20} | {'Kelly f*':>10} | {'Gross lev sum':>14}")
    gross = sum(abs(x) for x in kelly_w)
    for a, f in zip(assets, kelly_w):
        print(f"   {a:20} | {f:>+10.4f} |")
    print(f"   Total gross leverage: {gross:.2f}x")

    g_full = kelly_growth_rate(kelly_w, mu_true, sigma_ann)
    print(f"   Full Kelly growth rate: {g_full:.4f}")

    print("\n5. Fractional Kelly (leverage management)")
    print(f"   {'Fraction':>10} | {'Gross Lev':>10} | {'Growth Rate':>12} | {'Growth/Full':>12}")
    print("   " + "-" * 50)
    for frac in [0.25, 0.50, 0.75, 1.00, 1.50]:
        fk = fractional_kelly(kelly_w, frac)
        ck = constrained_kelly(fk, max_gross_leverage=3.0)
        g = kelly_growth_rate(ck, mu_true, sigma_ann)
        gl = sum(abs(x) for x in ck)
        print(f"   {frac:>10.2f} | {gl:>10.2f} | {g:>12.4f} | {g/g_full:>12.3f}")

    print("\n6. Turnover-Constrained Rebalancing")
    w_current = [0.30, 0.25, 0.15, 0.20, 0.10]
    w_rp_target = risk_parity_weights(sigma_ann)
    print(f"   Current weights:  {[f'{w:.3f}' for w in w_current]}")
    print(f"   RP target weights:{[f'{w:.3f}' for w in w_rp_target]}")
    for max_to in [0.05, 0.10, 0.20, 1.00]:
        w_reb = turnover_constrained_rebalance(w_current, w_rp_target, max_to)
        actual_to = sum(abs(w_reb[i] - w_current[i]) for i in range(N)) / 2
        print(f"   Max TO={max_to:.2f}: actual TO={actual_to:.4f}  weights={[f'{w:.3f}' for w in w_reb]}")

    print("\n[Done] Day 28: Dynamic Risk Budgeting & Kelly complete.")
