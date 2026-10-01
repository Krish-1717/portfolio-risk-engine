"""
options_day21_multi_asset.py
Day 21: Multi-Asset Options — Margrabe spread, Kirk spread, basket options
(geometric/arithmetic/moment-matched), exchange options, rainbow options,
Gaussian copula correlation structure, correlation Greeks.
Pure Python stdlib only.
"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Black-Scholes helpers
# ---------------------------------------------------------------------------
def _N(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def _n(x: float) -> float:
    return math.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)

def bs_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K * math.exp(-r * T), 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _N(d1) - K * math.exp(-r * T) * _N(d2)

def bs_put(S, K, T, r, sigma):
    return bs_call(S, K, T, r, sigma) - S + K * math.exp(-r * T)

# ---------------------------------------------------------------------------
# Margrabe exchange option: max(S1 - S2, 0)
# ---------------------------------------------------------------------------
def margrabe_exchange(S1, S2, T, sigma1, sigma2, rho, q1=0.0, q2=0.0):
    """Price of option to exchange asset 2 for asset 1: max(S1-S2, 0)."""
    sigma = math.sqrt(sigma1**2 + sigma2**2 - 2 * rho * sigma1 * sigma2)
    if sigma < 1e-8 or T <= 0:
        return max(S1 * math.exp(-q1*T) - S2 * math.exp(-q2*T), 0.0)
    F1 = S1 * math.exp(-q1 * T)
    F2 = S2 * math.exp(-q2 * T)
    d1 = (math.log(F1 / F2) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return F1 * _N(d1) - F2 * _N(d2)

# ---------------------------------------------------------------------------
# Kirk spread option approximation
# ---------------------------------------------------------------------------
def kirk_spread(F1, F2, K, T, sigma1, sigma2, rho, r=0.0):
    """
    Kirk (1995) approximation for spread option max(F1 - F2 - K, 0).
    """
    if K <= 0:
        return margrabe_exchange(F1, F2, T, sigma1, sigma2, rho)
    F2_adj = F2 + K
    sigma_adj = math.sqrt(
        sigma1**2 + (sigma2 * F2 / F2_adj)**2 - 2 * rho * sigma1 * sigma2 * F2 / F2_adj
    )
    if sigma_adj < 1e-8:
        return max(math.exp(-r*T) * (F1 - F2 - K), 0.0)
    d1 = (math.log(F1 / F2_adj) + 0.5 * sigma_adj**2 * T) / (sigma_adj * math.sqrt(T))
    d2 = d1 - sigma_adj * math.sqrt(T)
    return math.exp(-r*T) * (F1 * _N(d1) - F2_adj * _N(d2))

# ---------------------------------------------------------------------------
# Geometric basket option (exact closed-form)
# ---------------------------------------------------------------------------
def geometric_basket_call(spots, vols, weights, corr_matrix, K, T, r):
    """
    Geometric average basket call: max(prod(S_i^w_i) - K, 0).
    Closed-form under log-normal assumption.
    """
    n = len(spots)
    # Log-spot means
    log_spots = [math.log(S) for S in spots]
    # Weighted sum for log of geometric average
    log_F = sum(weights[i] * (log_spots[i] + (r - 0.5 * vols[i]**2) * T) for i in range(n))

    # Effective variance of log geometric basket
    var_log = sum(
        weights[i] * weights[j] * vols[i] * vols[j] * corr_matrix[i][j] * T
        for i in range(n) for j in range(n)
    )
    sigma_G = math.sqrt(max(var_log, 0.0)) / math.sqrt(T) if T > 0 else 0.0
    F_G = math.exp(log_F + 0.5 * var_log)

    return bs_call(F_G, K, T, r, sigma_G)

# ---------------------------------------------------------------------------
# Arithmetic basket option via moment matching
# ---------------------------------------------------------------------------
def arithmetic_basket_call_mm(spots, vols, weights, corr_matrix, K, T, r):
    """
    Arithmetic basket call via moment matching (Curran-style approximation).
    Match mean and variance of arithmetic basket to log-normal.
    """
    n = len(spots)
    disc = math.exp(-r * T)

    # Forward prices
    forwards = [S * math.exp(r * T) for S in spots]

    # E[A] = sum w_i * F_i
    E_A = sum(weights[i] * forwards[i] for i in range(n))

    # E[A^2] = sum_i sum_j w_i w_j * F_i F_j * exp(rho_ij * sig_i * sig_j * T)
    E_A2 = sum(
        weights[i] * weights[j] * forwards[i] * forwards[j] *
        math.exp(corr_matrix[i][j] * vols[i] * vols[j] * T)
        for i in range(n) for j in range(n)
    )

    # Effective sigma_A from moment matching
    sigma_A_sq = math.log(E_A2 / E_A**2) / T if E_A > 0 else 0.0
    sigma_A = math.sqrt(max(sigma_A_sq, 0.0))

    if sigma_A < 1e-8:
        return max(disc * (E_A - K), 0.0)

    return disc * bs_call(E_A * disc, K * disc, T, r, sigma_A)  # simplified approximation

def arithmetic_basket_call_mc(spots, vols, weights, corr_matrix, K, T, r,
                               n_paths=20000, seed=42):
    """Monte Carlo arithmetic basket call."""
    rng = random.Random(seed)
    n = len(spots)

    def cholesky_2d(corr):
        """Cholesky of correlation matrix."""
        k = len(corr)
        L = [[0.0]*k for _ in range(k)]
        for i in range(k):
            for j in range(i+1):
                s = sum(L[i][m] * L[j][m] for m in range(j))
                if i == j:
                    L[i][j] = math.sqrt(max(corr[i][i] - s, 1e-10))
                else:
                    L[i][j] = (corr[i][j] - s) / max(L[j][j], 1e-10)
        return L

    L = cholesky_2d(corr_matrix)

    def randn():
        u1 = max(rng.random(), 1e-15)
        return math.sqrt(-2*math.log(u1)) * math.cos(2*math.pi*rng.random())

    drift = [(r - 0.5 * vols[i]**2) * T for i in range(n)]
    sqT = math.sqrt(T)
    disc = math.exp(-r * T)
    total = 0.0

    for _ in range(n_paths):
        z_ind = [randn() for _ in range(n)]
        z_corr = [sum(L[i][j] * z_ind[j] for j in range(i+1)) for i in range(n)]
        log_ST = [math.log(spots[i]) + drift[i] + vols[i] * sqT * z_corr[i] for i in range(n)]
        ST = [math.exp(ls) for ls in log_ST]
        basket = sum(weights[i] * ST[i] for i in range(n))
        total += max(basket - K, 0.0)

    return disc * total / n_paths

# ---------------------------------------------------------------------------
# Rainbow options: best-of and worst-of
# ---------------------------------------------------------------------------
def rainbow_bestof_mc(spots, vols, corr_matrix, K, T, r, n_paths=20000, seed=42):
    """Best-of-n-assets call: max(max(S_i) - K, 0)."""
    rng = random.Random(seed)
    n = len(spots)

    def cholesky_2d(corr):
        k = len(corr)
        L = [[0.0]*k for _ in range(k)]
        for i in range(k):
            for j in range(i+1):
                s = sum(L[i][m] * L[j][m] for m in range(j))
                if i == j:
                    L[i][j] = math.sqrt(max(corr[i][i] - s, 1e-10))
                else:
                    L[i][j] = (corr[i][j] - s) / max(L[j][j], 1e-10)
        return L

    L = cholesky_2d(corr_matrix)
    drift = [(r - 0.5 * vols[i]**2) * T for i in range(n)]
    sqT = math.sqrt(T)
    disc = math.exp(-r * T)
    total_best, total_worst = 0.0, 0.0

    for _ in range(n_paths):
        z_ind = [math.sqrt(-2*math.log(max(rng.random(),1e-15))) * math.cos(2*math.pi*rng.random())
                 for _ in range(n)]
        z_corr = [sum(L[i][j] * z_ind[j] for j in range(i+1)) for i in range(n)]
        ST = [spots[i] * math.exp(drift[i] + vols[i] * sqT * z_corr[i]) for i in range(n)]
        total_best += max(max(ST) - K, 0.0)
        total_worst += max(min(ST) - K, 0.0)

    return disc * total_best / n_paths, disc * total_worst / n_paths

# ---------------------------------------------------------------------------
# Correlation Greeks
# ---------------------------------------------------------------------------
def correlation_delta(price_func, rho, h=0.01, **kwargs):
    """Correlation delta: dV/d_rho."""
    p_up = price_func(rho=rho + h, **kwargs)
    p_dn = price_func(rho=rho - h, **kwargs)
    return (p_up - p_dn) / (2 * h)

def basket_sensitivity(spots, vols, weights, corr_matrix, K, T, r, h_spot=0.01, h_vol=0.01):
    """Compute delta and vega for each asset in arithmetic basket."""
    n = len(spots)
    base = arithmetic_basket_call_mc(spots, vols, weights, corr_matrix, K, T, r, n_paths=10000)

    deltas, vegas = [], []
    for i in range(n):
        s_up = spots[:]; s_up[i] *= (1 + h_spot)
        s_dn = spots[:]; s_dn[i] *= (1 - h_spot)
        delta_i = (arithmetic_basket_call_mc(s_up, vols, weights, corr_matrix, K, T, r, n_paths=10000) -
                   arithmetic_basket_call_mc(s_dn, vols, weights, corr_matrix, K, T, r, n_paths=10000)) / (2 * spots[i] * h_spot)

        v_up = vols[:]; v_up[i] += h_vol
        v_dn = vols[:]; v_dn[i] = max(vols[i] - h_vol, 1e-4)
        vega_i = (arithmetic_basket_call_mc(spots, v_up, weights, corr_matrix, K, T, r, n_paths=10000) -
                  arithmetic_basket_call_mc(spots, v_dn, weights, corr_matrix, K, T, r, n_paths=10000)) / (2 * h_vol)

        deltas.append(delta_i)
        vegas.append(vega_i)

    return deltas, vegas

# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 65)
    print("DAY 21: Multi-Asset Options")
    print("=" * 65)

    r, T = 0.05, 0.5
    S1, S2 = 100.0, 95.0
    sig1, sig2, rho = 0.20, 0.25, 0.60

    print("\n1. Margrabe Exchange Option: max(S1 - S2, 0)")
    margrabe_p = margrabe_exchange(S1, S2, T, sig1, sig2, rho)
    print(f"   S1={S1}, S2={S2}, sig1={sig1}, sig2={sig2}, rho={rho}, T={T}")
    print(f"   Exchange option price: {margrabe_p:.4f}")

    print("\n   Correlation sensitivity:")
    for rho_val in [0.0, 0.3, 0.6, 0.9]:
        p = margrabe_exchange(S1, S2, T, sig1, sig2, rho_val)
        print(f"   rho={rho_val:.1f}: {p:.4f}")

    print("\n2. Kirk Spread Option: max(F1 - F2 - K, 0)")
    F1, F2_k, K_spread = 105.0, 95.0, 5.0
    kirk_p = kirk_spread(F1, F2_k, K_spread, T, sig1, sig2, rho, r)
    print(f"   F1={F1}, F2={F2_k}, K={K_spread}")
    print(f"   Kirk spread price: {kirk_p:.4f}")

    print("\n3. Basket Options (3 assets)")
    spots = [100.0, 105.0, 95.0]
    vols = [0.20, 0.25, 0.22]
    weights = [1/3, 1/3, 1/3]
    corr = [[1.0, 0.5, 0.4], [0.5, 1.0, 0.6], [0.4, 0.6, 1.0]]
    K_basket = 100.0

    geo_p = geometric_basket_call(spots, vols, weights, corr, K_basket, T, r)
    mm_p = arithmetic_basket_call_mm(spots, vols, weights, corr, K_basket, T, r)
    mc_p = arithmetic_basket_call_mc(spots, vols, weights, corr, K_basket, T, r,
                                      n_paths=30000)
    print(f"   Geometric basket call   : {geo_p:.4f}")
    print(f"   Arithmetic (MM approx)  : {mm_p:.4f}")
    print(f"   Arithmetic (MC 30K)     : {mc_p:.4f}")

    print("\n4. Rainbow Options: Best-of and Worst-of")
    best_p, worst_p = rainbow_bestof_mc(spots, vols, corr, K_basket, T, r, n_paths=30000)
    print(f"   Best-of-3 call  : {best_p:.4f}")
    print(f"   Worst-of-3 call : {worst_p:.4f}")

    # Sanity: best-of >= equal weight basket >= worst-of
    print(f"   Ordering check  : best ({best_p:.4f}) >= basket ({mc_p:.4f}) >= worst ({worst_p:.4f}): "
          f"{best_p >= mc_p - 0.01 and mc_p >= worst_p - 0.01}")

    print("\n5. Correlation Sensitivity")
    corr_deltas = []
    for rho_val in [0.0, 0.3, 0.6, 0.9]:
        corr2 = [[1.0 if i==j else rho_val for j in range(3)] for i in range(3)]
        p = arithmetic_basket_call_mc(spots, vols, weights, corr2, K_basket, T, r, n_paths=20000)
        corr_deltas.append((rho_val, p))
        print(f"   rho={rho_val:.1f}: basket call = {p:.4f}")

    print("\n   Rho increase reduces basket vol (diversification) → lower ATM call prices")

    print("\n6. Basket Greeks (per-asset Delta and Vega)")
    deltas, vegas = basket_sensitivity(spots, vols, weights, corr, K_basket, T, r)
    print(f"   {'Asset':>8} | {'Delta':>10} | {'Vega':>10}")
    print("   " + "-" * 35)
    for i in range(3):
        print(f"   {'Asset '+str(i+1):>8} | {deltas[i]:>10.4f} | {vegas[i]:>10.4f}")

    print("\n[Done] Day 21: Multi-Asset Options complete.")
