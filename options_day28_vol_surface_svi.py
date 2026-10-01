"""
options_day28_vol_surface_svi.py
Day 28: SVI (Stochastic Volatility Inspired) parametrization — fit to market
smile, arbitrage-free checks (butterfly + calendar), implied density extraction,
SSVI surface construction.
Pure Python stdlib only.
"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

def bs_call_from_iv(F: float, K: float, T: float, r: float, sigma: float) -> float:
    """Call price in forward measure: C = e^{-rT}[F*N(d1) - K*N(d2)]."""
    if T <= 0 or sigma <= 0:
        return math.exp(-r * T) * max(F - K, 0.0)
    sqT = math.sqrt(T)
    d1 = math.log(F / K) / (sigma * sqT) + 0.5 * sigma * sqT
    d2 = d1 - sigma * sqT
    return math.exp(-r * T) * (F * _norm_cdf(d1) - K * _norm_cdf(d2))

def bs_implied_vol(price: float, F: float, K: float, T: float, r: float,
                   tol: float = 1e-7) -> float:
    """Brent's method implied vol."""
    intrinsic = math.exp(-r * T) * max(F - K, 0.0)
    if price <= intrinsic + 1e-10:
        return 0.001
    lo, hi = 1e-4, 5.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if bs_call_from_iv(F, K, T, r, mid) < price:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2

# ---------------------------------------------------------------------------
# 1. SVI parametrization  (Gatheral 2004)
# ---------------------------------------------------------------------------
@dataclass
class SVIParams:
    """
    Raw SVI: w(k) = a + b*(ρ*(k-m) + sqrt((k-m)^2 + σ^2))
    k = log(K/F): log-moneyness
    w = σ_BS^2 * T: total implied variance
    """
    a: float     # overall variance level (vertical shift)
    b: float     # angle between left and right asymptotes
    rho: float   # correlation, controls skew (-1 < ρ < 1)
    m: float     # translation of smile (ATM tilt)
    sigma: float # curvature / ATM smoothness (> 0)

    def total_variance(self, k: float) -> float:
        """w(k) = a + b*(ρ*(k-m) + sqrt((k-m)^2 + σ^2))"""
        return self.a + self.b * (self.rho * (k - self.m) +
                                   math.sqrt((k - self.m)**2 + self.sigma**2))

    def implied_vol(self, k: float, T: float) -> float:
        """σ_BS(k, T) = sqrt(w(k) / T)"""
        w = self.total_variance(k)
        return math.sqrt(max(w, 0.0) / T)

    def butterfly_arbitrage_free(self) -> bool:
        """
        Necessary conditions for no butterfly arbitrage (Roper 2010):
        b >= 0, |ρ| < 1, σ > 0, a + b*σ*sqrt(1-ρ^2) >= 0.
        """
        return (self.b >= 0 and abs(self.rho) < 1 and self.sigma > 0
                and self.a + self.b * self.sigma * math.sqrt(1 - self.rho**2) >= 0)

    def min_variance(self) -> float:
        """Minimum of w(k) at k* = m - ρ*σ/sqrt(1-ρ^2) (if |ρ| < 1)."""
        if abs(self.rho) >= 1:
            return self.a
        k_star = self.m - self.rho * self.sigma / math.sqrt(1 - self.rho**2)
        return self.total_variance(k_star)

def svi_fit(log_moneyness: list[float], market_total_var: list[float],
             n_iter: int = 2000, lr: float = 0.002, seed: int = 42) -> SVIParams:
    """
    Fit SVI to market total variance w(k) = σ_mkt^2 * T via gradient descent.
    """
    rng = random.Random(seed)
    # Initialize near market midpoint
    w_mid = sum(market_total_var) / len(market_total_var)
    a = w_mid * 0.8
    b = 0.1
    rho = -0.3
    m = 0.0
    sig = 0.1

    def loss(a, b, rho, m, s):
        total = 0.0
        for k, w_mkt in zip(log_moneyness, market_total_var):
            w_svi = a + b * (rho * (k - m) + math.sqrt((k - m)**2 + s**2))
            total += (w_svi - w_mkt)**2
        return total / len(log_moneyness)

    eps = 1e-5
    for it in range(n_iter):
        L = loss(a, b, rho, m, sig)
        ga = (loss(a + eps, b, rho, m, sig) - L) / eps
        gb = (loss(a, b + eps, rho, m, sig) - L) / eps
        gr = (loss(a, b, rho + eps, m, sig) - L) / eps
        gm = (loss(a, b, rho, m + eps, sig) - L) / eps
        gs = (loss(a, b, rho, m, sig + eps) - L) / eps

        a   -= lr * ga
        b    = max(b   - lr * gb, 1e-6)
        rho  = max(-0.999, min(0.999, rho - lr * gr))
        m   -= lr * gm
        sig  = max(sig - lr * gs, 1e-4)

        if it > 100 and it % 100 == 0:
            lr *= 0.97  # decay

    return SVIParams(a, b, rho, m, sig)

# ---------------------------------------------------------------------------
# 2. Calendar spread arbitrage check
# ---------------------------------------------------------------------------
def calendar_arbitrage_free(params_t1: SVIParams, params_t2: SVIParams,
                              T1: float, T2: float,
                              k_range: tuple = (-0.5, 0.5), n_points: int = 50) -> bool:
    """
    Check calendar spread no-arbitrage: w(k, T2) >= w(k, T1) for all k.
    (Total variance must be non-decreasing in T.)
    """
    ks = [k_range[0] + i * (k_range[1] - k_range[0]) / (n_points - 1) for i in range(n_points)]
    for k in ks:
        w1 = params_t1.total_variance(k)
        w2 = params_t2.total_variance(k)
        if w2 < w1 - 1e-8:
            return False
    return True

# ---------------------------------------------------------------------------
# 3. Implied density (Breeden-Litzenberger)
# ---------------------------------------------------------------------------
def implied_density(params: SVIParams, T: float, F: float, r: float,
                    k_range: tuple = (-0.5, 0.5), n_points: int = 100) -> list[tuple[float, float]]:
    """
    Breeden-Litzenberger: q(K) = e^{rT} * d²C/dK²
    Numerically via second-order finite differences on call prices.
    Returns list of (K, density) pairs.
    """
    dk_log = (k_range[1] - k_range[0]) / (n_points - 1)
    result = []
    for i in range(1, n_points - 1):
        k = k_range[0] + i * dk_log
        k_m = k - dk_log
        k_p = k + dk_log

        K   = F * math.exp(k)
        K_m = F * math.exp(k_m)
        K_p = F * math.exp(k_p)

        iv   = params.implied_vol(k, T)
        iv_m = params.implied_vol(k_m, T)
        iv_p = params.implied_vol(k_p, T)

        C   = bs_call_from_iv(F, K,   T, r, iv)
        C_m = bs_call_from_iv(F, K_m, T, r, iv_m)
        C_p = bs_call_from_iv(F, K_p, T, r, iv_p)

        dK = K_p - K_m
        d2C_dK2 = (C_p - 2 * C + C_m) / ((0.5 * dK)**2)
        density = math.exp(r * T) * max(d2C_dK2, 0.0)
        result.append((K, density))

    return result

# ---------------------------------------------------------------------------
# 4. SSVI (Surface SVI, Gatheral & Jacquier 2014)
# ---------------------------------------------------------------------------
def ssvi_total_variance(k: float, theta_t: float, rho: float, phi_func_val: float) -> float:
    """
    SSVI: w(k, θ_t) = θ_t/2 * (1 + ρ*φ*k + sqrt((φ*k + ρ)^2 + 1 - ρ^2))
    θ_t = ATM total variance (σ_ATM^2 * T), φ = φ(θ_t) curvature function.
    Uses power-law: φ(θ) = η / (θ^γ * (1 + θ)^(1-γ)).
    """
    x = phi_func_val * k
    return theta_t / 2 * (1 + rho * x + math.sqrt((x + rho)**2 + (1 - rho**2)))

def ssvi_surface(strikes: list[float], tenors: list[float], F: float,
                  atm_vols: list[float], rho: float = -0.4,
                  eta: float = 1.0, gamma: float = 0.5) -> list[list[float]]:
    """
    Build SSVI implied vol surface.
    atm_vols[i] = σ_ATM for tenor tenors[i].
    Returns vol surface: surface[i][j] = σ(tenors[i], strikes[j]).
    """
    surface = []
    for i, T in enumerate(tenors):
        theta_t = atm_vols[i]**2 * T
        phi = eta / (theta_t**gamma * (1 + theta_t)**(1 - gamma))
        row = []
        for K in strikes:
            k = math.log(K / F)
            w = ssvi_total_variance(k, theta_t, rho, phi)
            vol = math.sqrt(max(w, 0.0) / T)
            row.append(vol)
        surface.append(row)
    return surface

# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 65)
    print("DAY 28: SVI Vol Surface Calibration")
    print("=" * 65)

    S, r, T = 100.0, 0.05, 0.5
    F = S * math.exp(r * T)

    # Simulate market smile from Heston (from Day 26)
    heston_ivs = {
        85: 0.2287, 90: 0.2177, 95: 0.2075,
        100: 0.1983, 105: 0.1889, 110: 0.1792, 115: 0.1706
    }
    strikes_mkt = list(heston_ivs.keys())
    ivs_mkt = list(heston_ivs.values())

    log_m = [math.log(K / F) for K in strikes_mkt]
    total_var_mkt = [iv**2 * T for iv in ivs_mkt]

    print(f"\nForward F = {F:.4f}, T = {T}")

    print("\n1. Raw SVI Calibration to Heston Smile")
    svi = svi_fit(log_m, total_var_mkt)
    print(f"   Fitted SVI: a={svi.a:.5f}, b={svi.b:.5f}, ρ={svi.rho:.4f}, m={svi.m:.4f}, σ={svi.sigma:.4f}")
    print(f"   Butterfly arb-free : {svi.butterfly_arbitrage_free()}")
    print(f"   Min total variance : {svi.min_variance():.5f} ({'OK' if svi.min_variance() >= 0 else 'VIOLATION'})")

    print(f"\n   {'K':>5} | {'Mkt IV':>9} | {'SVI IV':>9} | {'Error bps':>11}")
    print("   " + "-" * 44)
    for K, iv_mkt, k, w_mkt in zip(strikes_mkt, ivs_mkt, log_m, total_var_mkt):
        iv_svi = svi.implied_vol(k, T)
        err_bps = (iv_svi - iv_mkt) * 10000
        print(f"   {K:>5} | {iv_mkt:>9.4f} | {iv_svi:>9.4f} | {err_bps:>+10.1f}")

    print("\n2. Fine-Grid SVI Smile with Dupire g(k) butterfly check")
    fine_strikes = [80 + i * 2 for i in range(21)]
    print(f"   {'K':>5} | {'SVI IV':>10} | {'Total Var':>11} | {'g(k)>=0':>10}")
    print("   " + "-" * 46)
    dk = 0.01
    for K in fine_strikes:
        k = math.log(K / F)
        w   = svi.total_variance(k)
        w_p = svi.total_variance(k + dk)
        w_m = svi.total_variance(k - dk)
        dw  = (w_p - w_m) / (2 * dk)
        d2w = (w_p - 2 * w + w_m) / dk**2
        # Dupire local variance condition: g(k) >= 0
        # g(k) = (1 - k*dw/(2w))^2 - dw^2/4*(1/w + 1/4) + d2w/2
        g = (1 - k * dw / (2 * w))**2 - dw**2 / 4 * (1 / w + 0.25) + d2w / 2
        iv = svi.implied_vol(k, T)
        ok = "✓" if g >= -1e-6 else "✗"
        print(f"   {K:>5} | {iv:>10.4f} | {w:>11.6f} | {g:>+9.5f} {ok}")

    print("\n3. Calendar Spread Arbitrage Check")
    # Fit SVI for two tenors: T=0.5 (already fitted) and T=1.0
    heston_ivs_1y = {85:0.2350, 90:0.2220, 95:0.2110, 100:0.2010, 105:0.1920, 110:0.1840, 115:0.1770}
    log_m_1y = [math.log(K / F) for K in heston_ivs_1y.keys()]
    tv_1y = [iv**2 * 1.0 for iv in heston_ivs_1y.values()]
    svi_1y = svi_fit(log_m_1y, tv_1y)
    arb_free = calendar_arbitrage_free(svi, svi_1y, T, 1.0)
    print(f"   SVI T=0.5 vs T=1.0 calendar arb-free: {arb_free}")
    print(f"   w(ATM, T=0.5) = {svi.total_variance(0.0):.5f}")
    print(f"   w(ATM, T=1.0) = {svi_1y.total_variance(0.0):.5f}")
    print(f"   Monotone in T: {svi_1y.total_variance(0.0) >= svi.total_variance(0.0)}")

    print("\n4. Implied Risk-Neutral Density (Breeden-Litzenberger)")
    density = implied_density(svi, T, F, r, k_range=(-0.4, 0.3))
    # Normalize
    K_vals = [d[0] for d in density]
    q_vals = [d[1] for d in density]
    dK = K_vals[1] - K_vals[0]
    q_sum = sum(q_vals) * dK
    q_norm = [q / q_sum for q in q_vals]

    print(f"   Density integrates to ≈ {q_sum * dK / dK * dK:.3f} (should → 1.0)")
    print(f"   {'K':>5} | {'q(K)':>12} | {'bar':}")
    print("   " + "-" * 40)
    for K, q in zip(K_vals[::5], q_norm[::5]):  # every 5th point
        bar_len = int(q * 200)
        bar = '█' * bar_len
        print(f"   {K:>5.1f} | {q:>12.4f} | {bar}")

    print("\n5. SSVI Surface")
    tenors = [0.25, 0.5, 1.0, 2.0]
    atm_vols_t = [0.19, 0.20, 0.21, 0.22]
    strikes_grid = [85, 90, 95, 100, 105, 110, 115]
    surface = ssvi_surface(strikes_grid, tenors, F, atm_vols_t)

    tk_label = 'T\\K'
    header = f"   {tk_label:>5} | " + " | ".join(f"{K:>5}" for K in strikes_grid)
    print(f"\n   SSVI Implied Vol Surface (ρ=-0.4):")
    print("   " + header)
    print("   " + "-" * len(header))
    for i, T_s in enumerate(tenors):
        row = " | ".join(f"{v:5.4f}" for v in surface[i])
        print(f"   {T_s:>5.2f} | {row}")

    print("\n[Done] Day 28: SVI Vol Surface Calibration complete.")
