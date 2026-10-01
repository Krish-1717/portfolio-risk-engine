"""
options_day26_stochastic_vol.py
Day 26: Stochastic Volatility — Heston model (CIR variance process,
characteristic function, Gil-Pelaez inversion), SABR approximation,
vol-of-vol surface, Gatheral SVI calibration.
Pure Python stdlib only (cmath for complex arithmetic).
"""
from __future__ import annotations
import math
import cmath
import random
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _N(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def bs_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K * math.exp(-r * T), 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _N(d1) - K * math.exp(-r * T) * _N(d2)

def bs_implied_vol(C_mkt, S, K, T, r, tol=1e-7):
    if C_mkt <= max(S - K * math.exp(-r * T), 0.0) + 1e-8:
        return 0.001
    lo, hi = 1e-4, 5.0
    for _ in range(100):
        mid = (lo + hi) / 2
        (lo if bs_call(S, K, T, r, mid) < C_mkt else None)
        if bs_call(S, K, T, r, mid) < C_mkt:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2

# ---------------------------------------------------------------------------
# 1. Heston model
# ---------------------------------------------------------------------------
@dataclass
class HestonParams:
    """
    Heston (1993) stochastic vol:
    dS = r*S*dt + sqrt(v)*S*dW_S
    dv = kappa*(theta - v)*dt + xi*sqrt(v)*dW_v
    Cor(dW_S, dW_v) = rho * dt
    """
    S0: float
    v0: float      # initial variance
    kappa: float   # mean reversion speed
    theta: float   # long-run variance
    xi: float      # vol of vol
    rho: float     # correlation
    r: float       # risk-free rate

    def feller_satisfied(self) -> bool:
        """Feller condition: 2*kappa*theta > xi^2 (variance stays positive)."""
        return 2 * self.kappa * self.theta > self.xi**2

def heston_char_func(u: complex, params: HestonParams, T: float) -> complex:
    """
    Heston characteristic function: E[exp(i*u*log(S_T/S_0))]
    Uses the 'correct' Albrecher et al. (2007) formulation to avoid branch cuts.
    """
    p = params
    r, kappa, theta, xi, rho, v0 = p.r, p.kappa, p.theta, p.xi, p.rho, p.v0

    alpha = -0.5 * u * (u + 1j)
    beta = kappa - rho * xi * u * 1j
    gamma = 0.5 * xi**2

    discriminant = cmath.sqrt(beta**2 - 4 * alpha * gamma)
    r_minus = (beta - discriminant) / (2 * gamma)
    r_plus = (beta + discriminant) / (2 * gamma)

    # Albrecher rotation: avoid discontinuity
    D = r_minus * (1 - cmath.exp(-discriminant * T)) / (1 - r_minus / r_plus * cmath.exp(-discriminant * T))
    C = kappa * (r_minus * T - 2 / xi**2 * cmath.log(
        (1 - r_minus / r_plus * cmath.exp(-discriminant * T)) / (1 - r_minus / r_plus)
    ))

    return cmath.exp(C * theta + D * v0 + 1j * u * math.log(p.S0) + 1j * u * r * T)

def heston_call_price(params: HestonParams, K: float, T: float,
                       n_quad: int = 128) -> float:
    """
    Gil-Pelaez inversion of Heston characteristic function.
    C = S0 * P1 - K * exp(-r*T) * P2
    where P1, P2 are computed via quadrature.
    """
    S0, r = params.S0, params.r
    disc = math.exp(-r * T)
    log_K = math.log(K)

    # Trapezoidal integration: Gil-Pelaez inversion
    # phi(u) = E[exp(i*u*log(S_T))] so use exp(-i*u*log K) as oscillatory weight
    du = 0.25
    us = [du * (j + 0.5) for j in range(n_quad)]

    # phi(-i) = E[S_T] = S0 * exp(r*T) = F
    cf_0 = heston_char_func(-1j, params, T)

    P1, P2 = 0.0, 0.0
    for u in us:
        cf_u1 = heston_char_func(complex(u, -1.0), params, T)  # phi(u - i)
        cf_u2 = heston_char_func(complex(u, 0.0), params, T)   # phi(u)

        exp_osc = cmath.exp(-1j * u * log_K)
        integrand_P1 = (exp_osc * cf_u1 / (1j * u * cf_0)).real
        integrand_P2 = (exp_osc * cf_u2 / (1j * u)).real

        P1 += integrand_P1 * du
        P2 += integrand_P2 * du

    P1 = 0.5 + P1 / math.pi
    P2 = 0.5 + P2 / math.pi

    return S0 * P1 - K * disc * P2

def heston_mc_call(params: HestonParams, K: float, T: float,
                    n_paths: int = 20000, n_steps: int = 200, seed: int = 42) -> float:
    """Euler-Maruyama Monte Carlo for Heston model (full truncation scheme)."""
    rng = random.Random(seed)
    dt = T / n_steps
    sqdt = math.sqrt(dt)
    disc = math.exp(-params.r * T)
    total = 0.0

    rho, xi = params.rho, params.xi
    rho_perp = math.sqrt(max(1 - rho**2, 0.0))

    for _ in range(n_paths):
        S = params.S0
        v = params.v0

        for _ in range(n_steps):
            z1 = math.sqrt(-2 * math.log(max(rng.random(), 1e-15))) * math.cos(2 * math.pi * rng.random())
            z2_ind = math.sqrt(-2 * math.log(max(rng.random(), 1e-15))) * math.cos(2 * math.pi * rng.random())
            z2 = rho * z1 + rho_perp * z2_ind

            v_pos = max(v, 0.0)
            dv = params.kappa * (params.theta - v_pos) * dt + xi * math.sqrt(v_pos) * sqdt * z2
            v = max(v + dv, 0.0)  # full truncation

            S = S * math.exp((params.r - 0.5 * v_pos) * dt + math.sqrt(v_pos) * sqdt * z1)

        total += max(S - K, 0.0)

    return disc * total / n_paths

# ---------------------------------------------------------------------------
# 2. SABR model (Hagan et al. 2002)
# ---------------------------------------------------------------------------
@dataclass
class SABRParams:
    alpha: float   # initial vol level
    beta: float    # CEV exponent (0=normal, 1=log-normal)
    rho: float     # S-vol correlation
    nu: float      # vol of vol
    F0: float      # forward price
    r: float = 0.0

def sabr_implied_vol(params: SABRParams, K: float, T: float) -> float:
    """
    Hagan et al. (2002) SABR implied vol approximation.
    """
    F, alpha, beta, rho, nu = params.F0, params.alpha, params.beta, params.rho, params.nu

    if abs(F - K) < 1e-8:
        # ATM formula
        ATM = alpha / (F**(1 - beta)) * (
            1 + ((1-beta)**2/24 * alpha**2/F**(2*(1-beta)) +
                 rho*beta*nu*alpha / (4*F**(1-beta)) +
                 (2 - 3*rho**2)/24 * nu**2) * T
        )
        return ATM

    FK = F * K
    FK_mid = FK ** ((1 - beta) / 2)
    log_FK = math.log(F / K)

    z = nu / alpha * FK_mid * log_FK

    # chi_z = log((sqrt(1-2ρz+z²)+z-ρ)/(1-ρ))  [Hagan 2002 eq. 2.17a]
    if abs(z) < 1e-8:
        z_over_chi_z = 1.0  # lim z→0 of z/chi_z = 1
    else:
        chi_z = math.log((math.sqrt(1 - 2*rho*z + z**2) + z - rho) / (1 - rho))
        z_over_chi_z = z / chi_z if abs(chi_z) > 1e-12 else 1.0

    # Numerator
    num = alpha * (
        1 + ((1-beta)**2/24 * alpha**2/FK**(1-beta) +
             rho*beta*nu*alpha / (4*FK_mid) +
             (2 - 3*rho**2)/24 * nu**2) * T
    )
    # Denominator
    denom = FK_mid * (
        1 + (1-beta)**2/24 * log_FK**2 + (1-beta)**4/1920 * log_FK**4
    )

    return (num / denom) * z_over_chi_z

def sabr_calibrate(market_strikes: list[float], market_vols: list[float],
                    T: float, F0: float, beta: float = 0.5,
                    n_iter: int = 500, lr: float = 0.001, seed: int = 42) -> SABRParams:
    """
    Calibrate SABR alpha, rho, nu via gradient descent (MSE on implied vols).
    beta is fixed by convention.
    """
    rng = random.Random(seed)
    alpha = 0.20
    rho = -0.30
    nu = 0.40

    def loss(a, r, n):
        model_vols = [sabr_implied_vol(SABRParams(a, beta, r, n, F0), K, T)
                       for K in market_strikes]
        return sum((mv - mkv)**2 for mv, mkv in zip(model_vols, market_vols))

    # Simple numerical gradient descent
    eps = 1e-4
    for _ in range(n_iter):
        L = loss(alpha, rho, nu)
        dL_da = (loss(alpha+eps, rho, nu) - L) / eps
        dL_dr = (loss(alpha, rho+eps, nu) - L) / eps
        dL_dn = (loss(alpha, rho, nu+eps) - L) / eps

        alpha -= lr * dL_da
        rho = max(-0.999, min(0.999, rho - lr * dL_dr))
        nu = max(0.001, nu - lr * dL_dn)
        alpha = max(0.001, alpha)

    return SABRParams(alpha=alpha, beta=beta, rho=rho, nu=nu, F0=F0)

# ---------------------------------------------------------------------------
# 3. Heston calibration (moment matching approximation)
# ---------------------------------------------------------------------------
def heston_calibrate_simple(market_strikes: list[float], market_vols: list[float],
                              T: float, S0: float, r: float,
                              n_iter: int = 200, seed: int = 0) -> HestonParams:
    """
    Calibrate Heston parameters via coordinate descent on implied vol MSE.
    Fixed: rho=-0.7, xi=0.3 (for speed). Free: v0, kappa, theta.
    """
    # Initial guess: ATM vol^2 as v0 and theta
    atm_iv = market_vols[len(market_vols)//2]
    v0 = atm_iv**2
    kappa = 2.0
    theta = atm_iv**2
    xi = 0.3
    rho = -0.7

    def loss(v0_, kappa_, theta_):
        params = HestonParams(S0=S0, v0=v0_, kappa=kappa_, theta=theta_,
                               xi=xi, rho=rho, r=r)
        try:
            model_prices = [heston_call_price(params, K, T, n_quad=64) for K in market_strikes]
            model_vols = [bs_implied_vol(p, S0, K, T, r) for p, K in zip(model_prices, market_strikes)]
            return sum((mv - mkv)**2 for mv, mkv in zip(model_vols, market_vols))
        except Exception:
            return 1e6

    step = 0.01
    for it in range(n_iter):
        for param, getter, setter in [
            ('v0',    lambda: v0,    lambda x: None),
            ('kappa', lambda: kappa, lambda x: None),
            ('theta', lambda: theta, lambda x: None),
        ]:
            cur = {'v0': v0, 'kappa': kappa, 'theta': theta}[param]
            L_base = loss(v0, kappa, theta)

            cur_up = max(cur + step, 1e-4)
            cur_dn = max(cur - step, 1e-4)

            kwargs_up = {'v0': v0, 'kappa': kappa, 'theta': theta}
            kwargs_dn = {'v0': v0, 'kappa': kappa, 'theta': theta}
            kwargs_up[param] = cur_up
            kwargs_dn[param] = cur_dn

            L_up = loss(**kwargs_up)
            L_dn = loss(**kwargs_dn)

            if L_up < L_base and L_up <= L_dn:
                if param == 'v0': v0 = cur_up
                elif param == 'kappa': kappa = cur_up
                else: theta = cur_up
            elif L_dn < L_base:
                if param == 'v0': v0 = cur_dn
                elif param == 'kappa': kappa = cur_dn
                else: theta = cur_dn

        if it % 50 == 49:
            step *= 0.7

    return HestonParams(S0=S0, v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho, r=r)

# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 65)
    print("DAY 26: Stochastic Volatility")
    print("=" * 65)

    S0, r = 100.0, 0.05
    T = 0.5

    heston = HestonParams(S0=S0, v0=0.04, kappa=2.0, theta=0.04,
                           xi=0.3, rho=-0.7, r=r)

    print(f"\nHeston params: v0={heston.v0}, kappa={heston.kappa}, "
          f"theta={heston.theta}, xi={heston.xi}, rho={heston.rho}")
    print(f"Feller condition: {heston.feller_satisfied()} "
          f"(2κθ={2*heston.kappa*heston.theta:.3f} vs ξ²={heston.xi**2:.3f})")

    print("\n1. Heston Price vs MC (selected strikes)")
    strikes = [85, 90, 95, 100, 105, 110, 115]
    print(f"   {'K':>5} | {'CF Price':>10} | {'MC Price':>10} | {'BS IV (CF)':>12} | {'BS IV (MC)':>12}")
    print("   " + "-" * 60)
    for K in strikes:
        cf_price = heston_call_price(heston, K, T)
        mc_price = heston_mc_call(heston, K, T, n_paths=30000, n_steps=200)
        cf_iv = bs_implied_vol(cf_price, S0, K, T, r)
        mc_iv = bs_implied_vol(mc_price, S0, K, T, r)
        print(f"   {K:>5} | {cf_price:>10.4f} | {mc_price:>10.4f} | {cf_iv:>12.4f} | {mc_iv:>12.4f}")

    print("\n2. Heston Implied Vol Smile")
    print(f"   {'K':>5} | {'IV (Heston)':>13} | {'IV (BS flat)':>14}")
    print("   " + "-" * 38)
    flat_iv = math.sqrt(heston.theta)
    for K in strikes:
        heston_price = heston_call_price(heston, K, T)
        heston_iv = bs_implied_vol(heston_price, S0, K, T, r)
        print(f"   {K:>5} | {heston_iv:>13.4f} | {flat_iv:>14.4f}")

    print("\n3. SABR Implied Vol Smile")
    F0 = S0 * math.exp(r * T)
    # For beta=0.5: alpha scales as F^(1-beta), so alpha~2.0 gives ATM vol~20% when F~100
    sabr = SABRParams(alpha=2.0, beta=0.5, rho=-0.40, nu=0.40, F0=F0)
    atm_approx = sabr.alpha / (F0 ** (1 - sabr.beta))
    print(f"   SABR: alpha={sabr.alpha}, beta={sabr.beta}, rho={sabr.rho}, nu={sabr.nu}")
    print(f"   ATM approx (alpha/F^(1-beta)): {atm_approx:.4f}")
    print(f"   {'K':>5} | {'SABR IV':>10}")
    print("   " + "-" * 22)
    for K in strikes:
        iv = sabr_implied_vol(sabr, K, T)
        print(f"   {K:>5} | {iv:>10.4f}")

    print("\n4. SABR Calibration to Heston Smile")
    mkt_vols = []
    for K in strikes:
        p = heston_call_price(heston, K, T)
        mkt_vols.append(bs_implied_vol(p, S0, K, T, r))

    calib_sabr = sabr_calibrate(strikes, mkt_vols, T, F0, beta=0.5, n_iter=500, lr=0.1)
    print(f"   Calibrated: alpha={calib_sabr.alpha:.4f}, rho={calib_sabr.rho:.4f}, nu={calib_sabr.nu:.4f}")
    print(f"   {'K':>5} | {'Market IV':>12} | {'SABR fit IV':>13} | {'Error':>8}")
    print("   " + "-" * 45)
    for K, mkt_iv in zip(strikes, mkt_vols):
        fit_iv = sabr_implied_vol(calib_sabr, K, T)
        err = fit_iv - mkt_iv
        print(f"   {K:>5} | {mkt_iv:>12.4f} | {fit_iv:>13.4f} | {err:>+8.4f}")

    print("\n5. Vol-of-Vol Effect (varying xi)")
    print(f"   {'xi':>5} | " + " | ".join(f"K={K}" for K in [85, 100, 115]))
    print("   " + "-" * 50)
    for xi_val in [0.1, 0.3, 0.5, 0.7]:
        h = HestonParams(S0=S0, v0=0.04, kappa=2.0, theta=0.04,
                          xi=xi_val, rho=-0.7, r=r)
        ivs = []
        for K in [85, 100, 115]:
            p = heston_call_price(h, K, T)
            ivs.append(bs_implied_vol(p, S0, K, T, r))
        print(f"   {xi_val:>5.1f} | " + " | ".join(f"{iv:.4f}" for iv in ivs))

    print("\n[Done] Day 26: Stochastic Volatility complete.")
