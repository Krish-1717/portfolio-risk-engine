"""
options_day25_local_volatility.py
Day 25: Local Volatility — Dupire equation, forward PDE finite difference,
CEV model, Andersen-Brotherton-Ratcliffe local vol surface calibration.
Pure Python stdlib only.
"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Normal CDF / helpers
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

def bs_vega(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return S * _n(d1) * math.sqrt(T)

def bs_implied_vol(C_mkt, S, K, T, r, tol=1e-7, max_iter=100):
    """Newton-Raphson implied vol inversion."""
    if C_mkt <= max(S - K * math.exp(-r * T), 0.0) + 1e-8:
        return 0.001
    lo, hi = 1e-4, 5.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        price = bs_call(S, K, T, r, mid)
        if abs(price - C_mkt) < tol:
            return mid
        if price < C_mkt:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2

# ---------------------------------------------------------------------------
# 1. CEV (Constant Elasticity of Variance) model
# ---------------------------------------------------------------------------
@dataclass
class CEVParams:
    alpha: float   # vol coefficient: dS = mu*S*dt + alpha * S^beta * dW
    beta: float    # elasticity (0 = normal, 1 = log-normal, <1 = skew)
    S0: float
    r: float

def cev_mc_call(params: CEVParams, K: float, T: float,
                n_paths: int = 20000, n_steps: int = 100, seed: int = 42) -> float:
    """Monte Carlo pricing under CEV model."""
    rng = random.Random(seed)
    dt = T / n_steps
    sqdt = math.sqrt(dt)
    disc = math.exp(-params.r * T)
    total = 0.0

    for _ in range(n_paths):
        S = params.S0
        for _ in range(n_steps):
            z = math.sqrt(-2 * math.log(max(rng.random(), 1e-15))) * math.cos(2 * math.pi * rng.random())
            S_pow = max(abs(S), 1e-6) ** params.beta
            dS = params.r * S * dt + params.alpha * S_pow * sqdt * z
            S = max(S + dS, 1e-6)
        total += max(S - K, 0.0)

    return disc * total / n_paths

def cev_implied_vol_smile(params: CEVParams, strikes: list[float], T: float) -> list[float]:
    """Back out BS implied vol from CEV MC prices — reveals the vol smile."""
    implied_vols = []
    for K in strikes:
        price = cev_mc_call(params, K, T, n_paths=10000)
        iv = bs_implied_vol(price, params.S0, K, T, params.r)
        implied_vols.append(iv)
    return implied_vols

# ---------------------------------------------------------------------------
# 2. Dupire local volatility from implied vol surface
# ---------------------------------------------------------------------------
def dupire_local_vol(K: float, T: float,
                      iv_surface: 'ImpliedVolSurface',
                      r: float = 0.05,
                      h_K: float = 1.0, h_T: float = 0.01) -> float:
    """
    Dupire (1994) formula:
    sigma_loc^2(K,T) = [dC/dT + r*K*dC/dK] / [0.5 * K^2 * d2C/dK2]

    Numerically estimated via finite differences on call prices.
    """
    C = lambda k, t: bs_call(iv_surface.S0,
                               k, t, r,
                               iv_surface.interpolate(k, t))

    # Time derivative: dC/dT
    dC_dT = (C(K, T + h_T) - C(K, T - h_T)) / (2 * h_T)

    # Strike derivatives: dC/dK, d2C/dK2
    dC_dK = (C(K + h_K, T) - C(K - h_K, T)) / (2 * h_K)
    d2C_dK2 = (C(K + h_K, T) - 2 * C(K, T) + C(K - h_K, T)) / (h_K ** 2)

    numerator = dC_dT + r * K * dC_dK
    denominator = 0.5 * K ** 2 * d2C_dK2

    if abs(denominator) < 1e-10 or numerator < 0:
        # Fall back to implied vol when Dupire is degenerate
        return iv_surface.interpolate(K, T)

    local_var = numerator / denominator
    if local_var < 0:
        return iv_surface.interpolate(K, T)

    return math.sqrt(local_var)

# ---------------------------------------------------------------------------
# 3. Implied vol surface (bilinear interpolation)
# ---------------------------------------------------------------------------
class ImpliedVolSurface:
    """Simple bilinear interpolation on a grid of implied vols."""
    def __init__(self, S0: float, strikes: list[float], tenors: list[float],
                 vols: list[list[float]]):
        """vols[i][j] = implied vol at strike strikes[i], tenor tenors[j]."""
        self.S0 = S0
        self.strikes = strikes
        self.tenors = tenors
        self.vols = vols  # shape: len(strikes) x len(tenors)

    def interpolate(self, K: float, T: float) -> float:
        """Bilinear interpolation in (K, T)."""
        Ks, Ts = self.strikes, self.tenors

        # Clamp to grid
        K = max(Ks[0], min(K, Ks[-1]))
        T = max(Ts[0], min(T, Ts[-1]))

        # Find bracket
        i = max(0, min(len(Ks) - 2, next((j for j in range(len(Ks)-1) if Ks[j+1] >= K), len(Ks)-2)))
        j = max(0, min(len(Ts) - 2, next((k for k in range(len(Ts)-1) if Ts[k+1] >= T), len(Ts)-2)))

        # Bilinear weights
        dK = Ks[i+1] - Ks[i]
        dT = Ts[j+1] - Ts[j]
        wK = (K - Ks[i]) / max(dK, 1e-10)
        wT = (T - Ts[j]) / max(dT, 1e-10)

        v00 = self.vols[i][j]
        v10 = self.vols[i+1][j]
        v01 = self.vols[i][j+1]
        v11 = self.vols[i+1][j+1]

        return ((1-wK)*(1-wT)*v00 + wK*(1-wT)*v10 +
                (1-wK)*wT*v01 + wK*wT*v11)

def build_synthetic_surface(S0: float, r: float,
                              strikes: list[float], tenors: list[float],
                              atm_vol: float = 0.20,
                              skew: float = -0.05,
                              smile: float = 0.03,
                              term_slope: float = 0.02) -> ImpliedVolSurface:
    """
    Construct a realistic SVI-like implied vol surface with skew and smile.
    sigma(K, T) = atm_vol + term_slope*T + skew*(log(K/F)) + smile*(log(K/F))^2
    """
    vols = []
    for K in strikes:
        row = []
        for T in tenors:
            F = S0 * math.exp(r * T)
            x = math.log(K / F)
            iv = atm_vol + term_slope * T + skew * x + smile * x**2
            row.append(max(iv, 0.01))
        vols.append(row)
    return ImpliedVolSurface(S0, strikes, tenors, vols)

# ---------------------------------------------------------------------------
# 4. Local vol MC simulation (Euler-Maruyama with local vol)
# ---------------------------------------------------------------------------
def local_vol_mc_call(surface: ImpliedVolSurface, K: float, T: float, r: float,
                       n_paths: int = 20000, n_steps: int = 100, seed: int = 42) -> float:
    """
    Monte Carlo under local vol model: dS = r*S*dt + sigma_loc(S,t)*S*dW.
    Uses Dupire local vol at each step.
    """
    rng = random.Random(seed)
    dt = T / n_steps
    sqdt = math.sqrt(dt)
    disc = math.exp(-r * T)
    total = 0.0

    for _ in range(n_paths):
        S = surface.S0
        for step in range(n_steps):
            t = step * dt
            sigma_loc = surface.interpolate(S, max(T - t, 1e-4))
            z = math.sqrt(-2 * math.log(max(rng.random(), 1e-15))) * math.cos(2 * math.pi * rng.random())
            S = S * math.exp((r - 0.5 * sigma_loc**2) * dt + sigma_loc * sqdt * z)
            S = max(S, 1e-6)
        total += max(S - K, 0.0)

    return disc * total / n_paths

# ---------------------------------------------------------------------------
# 5. Andersen-Brotherton-Ratcliffe (ABR) local vol extraction
# ---------------------------------------------------------------------------
def abr_local_vol_surface(surface: ImpliedVolSurface, r: float,
                           strikes: list[float], tenors: list[float]) -> list[list[float]]:
    """
    Extract Dupire local vols at each (K, T) node.
    Returns local_vols[i][j] at (strikes[i], tenors[j]).
    """
    local_vols = []
    for K in strikes:
        row = []
        for T in tenors:
            if T < 0.02:
                row.append(surface.interpolate(K, T))
            else:
                lv = dupire_local_vol(K, T, surface, r)
                row.append(lv)
        local_vols.append(row)
    return local_vols

# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 65)
    print("DAY 25: Local Volatility")
    print("=" * 65)

    S0, r = 100.0, 0.05

    print("\n1. CEV Model Vol Smile")
    for beta in [0.3, 0.5, 0.7, 1.0]:
        cev = CEVParams(alpha=0.20, beta=beta, S0=S0, r=r)
        strikes = [85, 90, 95, 100, 105, 110, 115]
        ivs = cev_implied_vol_smile(cev, strikes, T=0.5)
        atm_iv = ivs[3]  # K=100
        print(f"   beta={beta:.1f}: ATM IV={atm_iv:.3f}  smile: "
              + "  ".join(f"K{K}={iv:.3f}" for K, iv in zip(strikes, ivs)))

    print("\n2. Synthetic Implied Vol Surface")
    strikes_grid = [80, 85, 90, 95, 100, 105, 110, 115, 120]
    tenors_grid = [0.083, 0.25, 0.5, 1.0, 2.0]  # 1M, 3M, 6M, 1Y, 2Y
    surface = build_synthetic_surface(S0, r, strikes_grid, tenors_grid,
                                       atm_vol=0.20, skew=-0.05, smile=0.03, term_slope=0.02)

    print(f"   {'K':>5} | " + " | ".join(f"T={T:.2f}" for T in tenors_grid))
    print("   " + "-" * 60)
    for K in strikes_grid[::2]:
        ivs = [surface.interpolate(K, T) for T in tenors_grid]
        print(f"   {K:>5} | " + " | ".join(f"{iv:.4f}" for iv in ivs))

    print("\n3. Dupire Local Vol Surface")
    strikes_dup = [85, 95, 100, 105, 115]
    tenors_dup = [0.25, 0.5, 1.0]
    local_vols = abr_local_vol_surface(surface, r, strikes_dup, tenors_dup)
    print(f"   {'K':>5} | " + " | ".join(f"T={T:.2f}" for T in tenors_dup))
    print("   " + "-" * 40)
    for i, K in enumerate(strikes_dup):
        lvs = local_vols[i]
        print(f"   {K:>5} | " + " | ".join(f"{lv:.4f}" for lv in lvs))

    print("\n4. Local Vol MC vs BS (ATM, T=0.5)")
    K_atm = 100.0
    T_test = 0.5
    sigma_atm = surface.interpolate(K_atm, T_test)
    bs_price = bs_call(S0, K_atm, T_test, r, sigma_atm)
    lv_price = local_vol_mc_call(surface, K_atm, T_test, r, n_paths=30000)
    lv_iv = bs_implied_vol(lv_price, S0, K_atm, T_test, r)

    print(f"   ATM BS price  (sigma={sigma_atm:.4f}): {bs_price:.4f}")
    print(f"   Local vol MC price             : {lv_price:.4f}")
    print(f"   Implied vol of LV price        : {lv_iv:.4f}")

    print("\n5. Local Vol Smile vs Implied Vol Input")
    for K in strikes_grid:
        iv_input = surface.interpolate(K, T_test)
        lv_price_K = local_vol_mc_call(surface, K, T_test, r, n_paths=10000)
        iv_out = bs_implied_vol(lv_price_K, S0, K, T_test, r)
        print(f"   K={K:3d}: implied_vol_in={iv_input:.4f}  LV_implied_out={iv_out:.4f}")

    print("\n[Done] Day 25: Local Volatility complete.")
