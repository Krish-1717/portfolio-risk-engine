"""
portfolio_day27_black_litterman.py
Day 27: Black-Litterman Model — equilibrium returns via reverse optimization,
investor views, posterior return distribution, BL portfolio weights.
Pure Python stdlib only.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Matrix helpers (pure Python)
# ---------------------------------------------------------------------------
def mat_mul(A, B):
    n, m, p = len(A), len(A[0]), len(B[0])
    return [[sum(A[i][k] * B[k][j] for k in range(m)) for j in range(p)] for i in range(n)]

def mat_T(A):
    return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]

def mat_add(A, B):
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def mat_scale(A, s):
    return [[x * s for x in row] for row in A]

def vec_dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def mat_inv(A):
    """Gauss-Jordan inversion for small matrices."""
    n = len(A)
    aug = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(A)]
    for col in range(n):
        # Pivot
        max_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        aug[col], aug[max_row] = aug[max_row], aug[col]
        pivot = aug[col][col]
        if abs(pivot) < 1e-12:
            raise ValueError("Singular matrix")
        aug[col] = [x / pivot for x in aug[col]]
        for row in range(n):
            if row != col:
                factor = aug[row][col]
                aug[row] = [aug[row][k] - factor * aug[col][k] for k in range(2 * n)]
    return [row[n:] for row in aug]

def mat_vec(A, v):
    """Matrix-vector product."""
    return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]

# ---------------------------------------------------------------------------
# 1. Market equilibrium (reverse optimization)
# ---------------------------------------------------------------------------
def market_equilibrium_returns(weights_mkt: list[float],
                                 sigma: list[list[float]],
                                 risk_aversion: float = 2.5) -> list[float]:
    """
    Pi = delta * Sigma * w_mkt  (CAPM implied excess returns).
    delta = market risk aversion (typically 2-3).
    """
    Sigma_w = mat_vec(sigma, weights_mkt)
    return [risk_aversion * x for x in Sigma_w]

# ---------------------------------------------------------------------------
# 2. Black-Litterman posterior
# ---------------------------------------------------------------------------
@dataclass
class BLView:
    """A single view: P @ mu = Q with confidence Omega_ii."""
    name: str
    P: list[float]      # 1 x n pick vector (relative or absolute view)
    Q: float            # view expected return
    confidence: float   # 1/omega_ii (higher = more confident)

def black_litterman(pi: list[float],
                     sigma: list[list[float]],
                     views: list[BLView],
                     tau: float = 0.05) -> tuple[list[float], list[list[float]]]:
    """
    Black-Litterman posterior:
      mu_BL = [(tau*Sigma)^{-1} + P' Omega^{-1} P]^{-1} [(tau*Sigma)^{-1} Pi + P' Omega^{-1} Q]
      Sigma_BL = [(tau*Sigma)^{-1} + P' Omega^{-1} P]^{-1}

    Returns (mu_BL, Sigma_BL + Sigma).
    """
    n = len(pi)
    k = len(views)

    # tau * Sigma
    tau_sigma = mat_scale(sigma, tau)
    tau_sigma_inv = mat_inv(tau_sigma)

    if k == 0:
        # No views: posterior = prior
        return pi[:], sigma[:]

    # Build P matrix (k x n) and Q vector
    P = [v.P for v in views]
    Q = [v.Q for v in views]
    Omega_inv_diag = [v.confidence for v in views]  # diagonal of Omega^{-1}

    # P' Omega^{-1} P  (n x n)
    Pt = mat_T(P)
    # Omega^{-1} P  (k x n): scale each row of P by confidence
    OmegaInv_P = [[Omega_inv_diag[i] * P[i][j] for j in range(n)] for i in range(k)]
    Pt_OmegaInv_P = mat_mul(Pt, OmegaInv_P)  # n x n

    # [(tau*Sigma)^{-1} + P' Omega^{-1} P]
    M_inv_term = mat_add(tau_sigma_inv, Pt_OmegaInv_P)
    M = mat_inv(M_inv_term)  # Sigma_BL (n x n)

    # P' Omega^{-1} Q  (n,)
    OmegaInv_Q = [Omega_inv_diag[i] * Q[i] for i in range(k)]
    Pt_OmegaInv_Q = mat_vec(Pt, OmegaInv_Q)  # n,

    # (tau*Sigma)^{-1} Pi
    tau_sigma_inv_pi = mat_vec(tau_sigma_inv, pi)

    # RHS: (tau*Sigma)^{-1} Pi + P' Omega^{-1} Q
    rhs = [tau_sigma_inv_pi[i] + Pt_OmegaInv_Q[i] for i in range(n)]

    mu_bl = mat_vec(M, rhs)

    # Posterior covariance: Sigma_BL + Sigma (parameter uncertainty + sampling)
    sigma_post = mat_add(M, sigma)

    return mu_bl, sigma_post

# ---------------------------------------------------------------------------
# 3. MVO on BL returns
# ---------------------------------------------------------------------------
def mvo_weights(mu: list[float], sigma: list[list[float]],
                 risk_aversion: float = 2.5, rf: float = 0.02) -> list[float]:
    """
    Unconstrained MVO: w* = (1/delta) * Sigma^{-1} * (mu - rf)
    Then normalize to sum to 1 (long-only projection).
    """
    n = len(mu)
    sigma_inv = mat_inv(sigma)
    excess = [mu[i] - rf for i in range(n)]
    raw = mat_vec(sigma_inv, excess)
    raw = [x / risk_aversion for x in raw]

    # Long-only projection: clip negative weights, renormalize
    raw = [max(x, 0.0) for x in raw]
    total = sum(raw)
    if total < 1e-10:
        return [1.0 / n] * n
    return [x / total for x in raw]

# ---------------------------------------------------------------------------
# 4. Implied confidence / tau sensitivity
# ---------------------------------------------------------------------------
def bl_tau_sensitivity(pi, sigma, views, tau_values, risk_aversion=2.5, rf=0.02):
    """Show how tau affects BL posterior mu and portfolio weights."""
    results = []
    for tau in tau_values:
        mu_bl, sig_bl = black_litterman(pi, sigma, views, tau)
        w = mvo_weights(mu_bl, sig_bl, risk_aversion, rf)
        results.append((tau, mu_bl[:], w[:]))
    return results

# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 65)
    print("DAY 27: Black-Litterman Portfolio Optimization")
    print("=" * 65)

    # ---- Setup: 5 assets ----
    assets = ['US Equity', 'Intl Equity', 'EM Equity', 'US Bonds', 'Gold']
    n = len(assets)

    # Market cap weights (e.g., world market portfolio approximation)
    w_mkt = [0.40, 0.25, 0.15, 0.15, 0.05]

    # Covariance matrix (annualized, realistic estimates)
    sigma = [
        [0.0400, 0.0320, 0.0360, 0.0020, 0.0040],
        [0.0320, 0.0529, 0.0480, 0.0015, 0.0030],
        [0.0360, 0.0480, 0.0900, 0.0010, 0.0050],
        [0.0020, 0.0015, 0.0010, 0.0016, 0.0000],
        [0.0040, 0.0030, 0.0050, 0.0000, 0.0196],
    ]

    delta = 2.5
    rf = 0.02

    print(f"\nAssets: {assets}")
    print(f"Market weights: {[f'{w:.2f}' for w in w_mkt]}")

    print("\n1. CAPM Equilibrium Returns (Reverse Optimization)")
    pi = market_equilibrium_returns(w_mkt, sigma, delta)
    print(f"   {'Asset':15} | {'Eq. Return':>12} | {'Mkt Weight':>12}")
    print("   " + "-" * 44)
    for a, p, w in zip(assets, pi, w_mkt):
        print(f"   {a:15} | {p:>+12.4f} | {w:>12.4f}")

    mvo_eq = mvo_weights(pi, sigma, delta, rf)
    print(f"\n   MVO on equilibrium returns (should ≈ market weights):")
    for a, w_mv, w_m in zip(assets, mvo_eq, w_mkt):
        print(f"   {a:15}: MVO={w_mv:.4f}  Market={w_m:.4f}")

    print("\n2. Investor Views")
    views = [
        BLView("US Equity outperforms Intl by 3%",
               P=[1, -1, 0, 0, 0], Q=0.03, confidence=50.0),
        BLView("EM Equity returns 12% (absolute)",
               P=[0, 0, 1, 0, 0], Q=0.12, confidence=25.0),
        BLView("Gold safe-haven +5% in risk-off",
               P=[0, 0, 0, 0, 1], Q=0.05, confidence=15.0),
    ]
    for v in views:
        print(f"   View: {v.name}")
        print(f"         Q={v.Q:.2%}  confidence={v.confidence:.1f}")

    print("\n3. Black-Litterman Posterior Returns (tau=0.05)")
    tau = 0.05
    mu_bl, sigma_bl = black_litterman(pi, sigma, views, tau)

    print(f"   {'Asset':15} | {'Prior Pi':>10} | {'BL Posterior':>13} | {'View Effect':>12}")
    print("   " + "-" * 57)
    for a, p, bl in zip(assets, pi, mu_bl):
        print(f"   {a:15} | {p:>+10.4f} | {bl:>+13.4f} | {bl - p:>+12.4f}")

    print("\n4. Portfolio Weights: Market vs BL-MVO")
    w_bl = mvo_weights(mu_bl, sigma_bl, delta, rf)
    print(f"   {'Asset':15} | {'Market':>8} | {'BL-MVO':>8} | {'Diff':>8}")
    print("   " + "-" * 48)
    for a, wm, wb in zip(assets, w_mkt, w_bl):
        print(f"   {a:15} | {wm:>8.4f} | {wb:>8.4f} | {wb - wm:>+8.4f}")

    # Expected portfolio return and vol
    ret_bl = sum(w_bl[i] * mu_bl[i] for i in range(n))
    var_bl = sum(w_bl[i] * w_bl[j] * sigma[i][j] for i in range(n) for j in range(n))
    print(f"\n   BL portfolio expected return : {ret_bl:.4f}")
    print(f"   BL portfolio volatility      : {math.sqrt(var_bl):.4f}")
    print(f"   BL portfolio Sharpe          : {(ret_bl - rf) / math.sqrt(var_bl):.4f}")

    print("\n5. Tau Sensitivity")
    tau_vals = [0.01, 0.05, 0.10, 0.25, 0.50]
    print(f"   {'tau':>6} | {'US Eq return':>14} | {'US Eq weight':>14} | {'Portfolio Sharpe':>17}")
    print("   " + "-" * 58)
    for tau, mu_t, w_t in bl_tau_sensitivity(pi, sigma, views, tau_vals, delta, rf):
        ret_t = sum(w_t[i] * mu_t[i] for i in range(n))
        var_t = sum(w_t[i] * w_t[j] * sigma[i][j] for i in range(n) for j in range(n))
        sharpe_t = (ret_t - rf) / math.sqrt(var_t)
        print(f"   {tau:>6.2f} | {mu_t[0]:>+14.4f} | {w_t[0]:>14.4f} | {sharpe_t:>17.4f}")

    print("\n6. Confidence Sensitivity (vary View 1 confidence)")
    print(f"   {'Confidence':>12} | {'US Eq weight':>14} | {'Intl weight':>12}")
    print("   " + "-" * 45)
    for conf in [5, 10, 25, 50, 100, 200]:
        views_c = [
            BLView("US>Intl +3%", [1, -1, 0, 0, 0], 0.03, conf),
            BLView("EM 12%",      [0,  0, 1, 0, 0], 0.12, 25.0),
            BLView("Gold 5%",     [0,  0, 0, 0, 1], 0.05, 15.0),
        ]
        mu_c, sig_c = black_litterman(pi, sigma, views_c, tau=0.05)
        w_c = mvo_weights(mu_c, sig_c, delta, rf)
        print(f"   {conf:>12} | {w_c[0]:>14.4f} | {w_c[1]:>12.4f}")

    print("\n[Done] Day 27: Black-Litterman complete.")
