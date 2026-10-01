"""
options_day23_jump_diffusion.py
Day 23: Jump Diffusion Models — Merton (log-normal jumps) and Kou (double-exponential jumps)
Includes Fourier/characteristic-function pricing, calibration, and Greeks.
Pure Python stdlib only (math, random, cmath).
"""

from __future__ import annotations
import math
import cmath
import random
from dataclasses import dataclass, field
from typing import Callable

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MertonParams:
    """Merton jump-diffusion parameters."""
    S: float          # spot price
    K: float          # strike
    T: float          # time to expiry (years)
    r: float          # risk-free rate
    sigma: float      # diffusion volatility
    lam: float        # jump intensity (jumps/year)
    mu_J: float       # mean log jump size
    sigma_J: float    # std-dev log jump size

    @property
    def k_bar(self) -> float:
        """Expected jump size: E[e^J - 1]."""
        return math.exp(self.mu_J + 0.5 * self.sigma_J ** 2) - 1.0

    @property
    def r_adj(self) -> float:
        """Risk-neutral drift adjustment for jumps."""
        return self.r - self.lam * self.k_bar

@dataclass
class KouParams:
    """Kou double-exponential jump-diffusion parameters."""
    S: float
    K: float
    T: float
    r: float
    sigma: float
    lam: float        # jump intensity
    p: float          # prob of positive jump
    eta1: float       # up-jump exponential rate (>1 for finite mean)
    eta2: float       # down-jump exponential rate (>0)

    @property
    def k_bar(self) -> float:
        """E[e^J - 1] for double-exponential."""
        return self.p * self.eta1 / (self.eta1 - 1) + (1 - self.p) * self.eta2 / (self.eta2 + 1) - 1.0

    @property
    def r_adj(self) -> float:
        return self.r - self.lam * self.k_bar


# ---------------------------------------------------------------------------
# Black-Scholes helper
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(S - K * math.exp(-r * T), 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)

def bs_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    return bs_call(S, K, T, r, sigma) - S + K * math.exp(-r * T)


# ---------------------------------------------------------------------------
# Merton jump-diffusion: infinite series expansion
# ---------------------------------------------------------------------------

def merton_call_series(p: MertonParams, n_terms: int = 50) -> float:
    """
    Price a European call under Merton jump diffusion using series expansion.

    The price is a weighted sum of Black-Scholes prices with jump-adjusted
    volatility and rate, where weight_n = e^{-lam'*T} * (lam'*T)^n / n!
    with lam' = lam*(1+k_bar).
    """
    lam_prime = p.lam * (1 + p.k_bar)  # risk-neutral intensity
    price = 0.0

    # Precompute log-factorial for stability
    log_factorial = 0.0
    log_lam_T = math.log(lam_prime * p.T) if lam_prime * p.T > 0 else float('-inf')
    log_exp_part = -lam_prime * p.T

    for n in range(n_terms):
        if n > 0:
            log_factorial += math.log(n)

        if lam_prime * p.T == 0 and n > 0:
            break

        # log weight: log(e^{-lam'T} * (lam'T)^n / n!)
        if lam_prime * p.T > 0:
            log_w = log_exp_part + n * log_lam_T - log_factorial
        else:
            log_w = 0.0 if n == 0 else float('-inf')

        if log_w < -30:  # negligible
            continue
        w = math.exp(log_w)

        # Jump-adjusted parameters for this term
        sigma_n = math.sqrt(p.sigma**2 + n * p.sigma_J**2 / p.T)
        r_n = p.r_adj + n * (p.mu_J + 0.5 * p.sigma_J**2) / p.T

        price += w * bs_call(p.S, p.K, p.T, r_n, sigma_n)

    return price


def merton_put_series(p: MertonParams, n_terms: int = 50) -> float:
    """Put via put-call parity."""
    call = merton_call_series(p, n_terms)
    return call - p.S + p.K * math.exp(-p.r * p.T)


# ---------------------------------------------------------------------------
# Characteristic function approach (Gil-Pelaez inversion)
# ---------------------------------------------------------------------------

def merton_char_func(u: complex, p: MertonParams) -> complex:
    """Characteristic function of log(S_T/S_0) under Merton."""
    i = complex(0, 1)
    # Diffusion part
    diff = i * u * (p.r_adj - 0.5 * p.sigma**2) * p.T - 0.5 * p.sigma**2 * u**2 * p.T
    # Jump part: lam*T*(E[e^{iuJ}] - 1) where J ~ N(mu_J, sigma_J^2)
    jump_cf = cmath.exp(i * u * p.mu_J - 0.5 * p.sigma_J**2 * u**2)
    jump = p.lam * p.T * (jump_cf - 1)
    return cmath.exp(diff + jump)


def kou_char_func(u: complex, p: KouParams) -> complex:
    """Characteristic function of log(S_T/S_0) under Kou."""
    i = complex(0, 1)
    diff = i * u * (p.r_adj - 0.5 * p.sigma**2) * p.T - 0.5 * p.sigma**2 * u**2 * p.T
    # Double-exponential MGF: E[e^{iuJ}] = p*eta1/(eta1-iu) + (1-p)*eta2/(eta2+iu)
    jump_cf = (p.p * p.eta1 / (p.eta1 - i * u) +
               (1 - p.p) * p.eta2 / (p.eta2 + i * u))
    jump = p.lam * p.T * (jump_cf - 1)
    return cmath.exp(diff + jump)


def _fourier_call_price(
    S: float, K: float, T: float, r: float,
    char_func: Callable[[complex], complex],
    N: int = 2048, eta: float = 0.25, alpha: float = 1.5
) -> float:
    """
    Carr-Madan FFT option pricing via Fourier inversion.
    Uses the damped characteristic function trick.
    """
    lam = 2 * math.pi / (N * eta)
    b = N * lam / 2
    log_K = math.log(K)

    # Damping factor alpha shifts the integration contour
    # psi_n = e^{-r*T} * phi(u - (alpha+1)*i) / (alpha^2 + alpha - u^2 + i*(2*alpha+1)*u)
    price = 0.0
    i = complex(0, 1)
    discount = math.exp(-r * T)

    for n in range(N):
        u_n = n * eta
        v = u_n - (alpha + 1) * i
        phi_v = char_func(v)

        denom = (alpha**2 + alpha - u_n**2 + i * (2 * alpha + 1) * u_n)
        if abs(denom) < 1e-10:
            continue

        psi_n = discount * phi_v / denom

        # Trapezoidal weight
        w = eta if n == 0 else eta

        # Sum contribution
        x = cmath.exp(-i * (log_K + b - b) * u_n) * psi_n * w
        price += x.real

    call = math.exp(-alpha * math.log(K / S)) / math.pi * price
    # Apply spot factor
    call *= (K / S)
    call = max(call * S, 0.0)

    return call


def fourier_merton_call(p: MertonParams, N: int = 512, eta: float = 0.1) -> float:
    """Fourier-based Merton call price."""
    log_K = math.log(p.K / p.S)
    i = complex(0, 1)
    alpha = 1.5

    total = 0.0
    discount = math.exp(-p.r * p.T)

    for n in range(N):
        u = n * eta + 1e-10
        v = u - (alpha + 1) * i
        phi_v = merton_char_func(v * complex(0, 1) if False else v, p)

        # Actually use simpler Gil-Pelaez integration
        # P(S_T > K) via characteristic function
        integrand_real = (cmath.exp(-i * u * log_K) * merton_char_func(u - i, p) /
                          (i * u * merton_char_func(-i, p))).real
        integrand_imag = (cmath.exp(-i * u * log_K) * merton_char_func(u, p) /
                          (i * u)).real

        w = eta * (1 if n > 0 else 0.5)
        total += w * (integrand_real - integrand_imag)

    # Gil-Pelaez: C = S * N_1 - K * e^{-rT} * N_2
    N1 = 0.5 + total / math.pi
    N2_integrand = 0.0
    for n in range(N):
        u = n * eta + 1e-10
        ignd = (cmath.exp(-i * u * log_K) * merton_char_func(u, p) / (i * u)).real
        w = eta * (1 if n > 0 else 0.5)
        N2_integrand += w * ignd
    N2 = 0.5 + N2_integrand / math.pi

    call = p.S * max(N1, 0) - p.K * discount * max(N2, 0)
    return max(call, 0.0)


# ---------------------------------------------------------------------------
# Monte Carlo pricing (reference)
# ---------------------------------------------------------------------------

def merton_mc_call(p: MertonParams, n_paths: int = 50000, seed: int = 42) -> float:
    """Monte Carlo reference price for Merton call."""
    rng = random.Random(seed)

    def randn() -> float:
        u1, u2 = rng.random(), rng.random()
        return math.sqrt(-2 * math.log(u1 + 1e-15)) * math.cos(2 * math.pi * u2)

    def poisson(lam: float) -> int:
        if lam > 30:
            k = int(lam + randn() * math.sqrt(lam))
            return max(k, 0)
        L = math.exp(-lam)
        k = 0
        prob = 1.0
        while prob > L:
            prob *= rng.random()
            k += 1
        return k - 1

    total = 0.0
    drift = (p.r_adj - 0.5 * p.sigma**2) * p.T
    vol_dt = p.sigma * math.sqrt(p.T)

    for _ in range(n_paths):
        z = randn()
        log_S = math.log(p.S) + drift + vol_dt * z

        # Jump component
        n_jumps = poisson(p.lam * p.T)
        for _ in range(n_jumps):
            j = p.mu_J + p.sigma_J * randn()
            log_S += j

        S_T = math.exp(log_S)
        total += max(S_T - p.K, 0.0)

    discount = math.exp(-p.r * p.T)
    return discount * total / n_paths


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

@dataclass
class MarketQuote:
    K: float
    T: float
    call_price: float
    weight: float = 1.0


def implied_vol_newton(C_mkt: float, S: float, K: float, T: float, r: float,
                       tol: float = 1e-8, max_iter: int = 100) -> float:
    """Newton's method implied volatility."""
    sigma = 0.3
    for _ in range(max_iter):
        price = bs_call(S, K, T, r, sigma)
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        vega = S * math.sqrt(T) * math.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)
        if abs(vega) < 1e-12:
            break
        diff = price - C_mkt
        sigma -= diff / vega
        sigma = max(sigma, 1e-5)
        if abs(diff) < tol:
            break
    return sigma


def merton_calibrate(
    quotes: list[MarketQuote],
    S: float, r: float,
    init_params: tuple[float, float, float, float] = (0.2, 0.5, -0.05, 0.15),
    lr: float = 0.001,
    n_iters: int = 200,
) -> tuple[float, float, float, float]:
    """
    Calibrate Merton (sigma, lam, mu_J, sigma_J) via gradient-free
    coordinate descent on sum of squared call price errors.
    """
    sigma, lam, mu_J, sigma_J = init_params
    best_loss = float('inf')
    best_params = (sigma, lam, mu_J, sigma_J)

    def loss(s, l, mj, sj) -> float:
        try:
            total = 0.0
            for q in quotes:
                p = MertonParams(S, q.K, q.T, r, s, l, mj, sj)
                price = merton_call_series(p, n_terms=20)
                total += q.weight * (price - q.call_price) ** 2
            return total
        except Exception:
            return float('inf')

    params = [sigma, lam, mu_J, sigma_J]
    bounds = [(0.01, 2.0), (0.01, 10.0), (-0.5, 0.5), (0.01, 1.0)]

    for iteration in range(n_iters):
        step = lr * (1.0 / (1 + 0.01 * iteration))
        improved = False

        for j in range(4):
            for direction in [+step, -step]:
                new_params = params.copy()
                new_params[j] += direction
                # Clip to bounds
                new_params[j] = max(bounds[j][0], min(bounds[j][1], new_params[j]))
                l = loss(*new_params)
                if l < best_loss:
                    best_loss = l
                    best_params = tuple(new_params)
                    params = new_params
                    improved = True
                    break

    return best_params


# ---------------------------------------------------------------------------
# Greeks via finite differences
# ---------------------------------------------------------------------------

def merton_call_greeks(p: MertonParams, h_S: float = 0.01, h_t: float = 1/365) -> dict:
    """Compute call Greeks via central finite differences."""
    price = merton_call_series(p)

    # Delta
    p_up = MertonParams(p.S * (1 + h_S), p.K, p.T, p.r, p.sigma, p.lam, p.mu_J, p.sigma_J)
    p_dn = MertonParams(p.S * (1 - h_S), p.K, p.T, p.r, p.sigma, p.lam, p.mu_J, p.sigma_J)
    delta = (merton_call_series(p_up) - merton_call_series(p_dn)) / (2 * p.S * h_S)

    # Gamma
    gamma = (merton_call_series(p_up) - 2 * price + merton_call_series(p_dn)) / (p.S * h_S)**2

    # Theta
    if p.T > h_t:
        p_th = MertonParams(p.S, p.K, p.T - h_t, p.r, p.sigma, p.lam, p.mu_J, p.sigma_J)
        theta = (merton_call_series(p_th) - price) / h_t
    else:
        theta = float('nan')

    # Vega
    p_vs = MertonParams(p.S, p.K, p.T, p.r, p.sigma + 0.01, p.lam, p.mu_J, p.sigma_J)
    p_vd = MertonParams(p.S, p.K, p.T, p.r, max(p.sigma - 0.01, 1e-4), p.lam, p.mu_J, p.sigma_J)
    vega = (merton_call_series(p_vs) - merton_call_series(p_vd)) / 0.02

    return {
        'price': price,
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
    }


# ---------------------------------------------------------------------------
# Kou double-exponential pricing via series
# ---------------------------------------------------------------------------

def kou_call_mc(p: KouParams, n_paths: int = 30000, seed: int = 123) -> float:
    """Monte Carlo for Kou double-exponential call."""
    rng = random.Random(seed)

    def randn() -> float:
        u1 = max(rng.random(), 1e-15)
        u2 = rng.random()
        return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)

    def dbl_exp_sample() -> float:
        """Sample from double exponential: +Exp(eta1) with prob p, -Exp(eta2) with prob 1-p."""
        if rng.random() < p.p:
            return -math.log(max(rng.random(), 1e-15)) / p.eta1
        else:
            return math.log(max(rng.random(), 1e-15)) / p.eta2

    def poisson_sample(lam: float) -> int:
        L = math.exp(-min(lam, 500))
        k, prob = 0, 1.0
        while prob > L and k < 1000:
            prob *= rng.random()
            k += 1
        return k - 1

    drift = (p.r_adj - 0.5 * p.sigma**2) * p.T
    vol_dt = p.sigma * math.sqrt(p.T)
    discount = math.exp(-p.r * p.T)

    total = 0.0
    for _ in range(n_paths):
        log_S = math.log(p.S) + drift + vol_dt * randn()
        n_jumps = poisson_sample(p.lam * p.T)
        for _ in range(n_jumps):
            log_S += dbl_exp_sample()
        total += max(math.exp(log_S) - p.K, 0.0)

    return discount * total / n_paths


# ---------------------------------------------------------------------------
# Implied vol surface from Merton (generates a smile)
# ---------------------------------------------------------------------------

def merton_implied_vol_surface(
    p: MertonParams,
    strikes: list[float],
    maturities: list[float],
) -> dict[tuple[float, float], float]:
    """Compute implied vol for a grid of (K, T) under Merton."""
    surface: dict[tuple[float, float], float] = {}
    for T in maturities:
        for K in strikes:
            pm = MertonParams(p.S, K, T, p.r, p.sigma, p.lam, p.mu_J, p.sigma_J)
            call = merton_call_series(pm, n_terms=30)
            try:
                iv = implied_vol_newton(call, p.S, K, T, p.r)
            except Exception:
                iv = float('nan')
            surface[(K, T)] = iv
    return surface


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 65)
    print("DAY 23: Jump Diffusion — Merton & Kou Models")
    print("=" * 65)

    # --- Merton call: series vs Monte Carlo ---
    p_merton = MertonParams(
        S=100, K=100, T=0.5, r=0.05,
        sigma=0.20, lam=0.5, mu_J=-0.05, sigma_J=0.15
    )

    series_price = merton_call_series(p_merton, n_terms=40)
    mc_price = merton_mc_call(p_merton, n_paths=100_000)

    print(f"\nMerton Call (ATM, T=0.5y):")
    print(f"  Series  price: {series_price:.6f}")
    print(f"  MC      price: {mc_price:.6f}")
    print(f"  BS      price: {bs_call(100, 100, 0.5, 0.05, 0.20):.6f}  (no jumps)")
    print(f"  k_bar   = {p_merton.k_bar:.6f}  (expected jump size)")

    # --- Vol smile generated by jumps ---
    print("\nVol Smile generated by Merton jumps (T=0.5y):")
    print(f"{'Strike':>8} | {'Call Price':>10} | {'Impl Vol':>8}")
    print("-" * 35)
    for K in [85, 90, 95, 100, 105, 110, 115]:
        pm = MertonParams(100, K, 0.5, 0.05, 0.20, 0.5, -0.05, 0.15)
        c = merton_call_series(pm, n_terms=30)
        try:
            iv = implied_vol_newton(c, 100, K, 0.5, 0.05)
        except Exception:
            iv = float('nan')
        print(f"{K:>8} | {c:>10.4f} | {iv:>8.4%}")

    # --- Greeks ---
    greeks = merton_call_greeks(p_merton)
    print("\nMerton Call Greeks (ATM, T=0.5y):")
    for g, v in greeks.items():
        print(f"  {g:8s}: {v:.6f}")

    # --- Kou model ---
    p_kou = KouParams(
        S=100, K=100, T=0.5, r=0.05, sigma=0.18,
        lam=0.5, p=0.4, eta1=10.0, eta2=5.0
    )
    kou_price = kou_call_mc(p_kou, n_paths=50_000)
    bs_price = bs_call(100, 100, 0.5, 0.05, 0.18)
    print(f"\nKou Double-Exponential Call (ATM, T=0.5y):")
    print(f"  MC price: {kou_price:.6f}")
    print(f"  BS price: {bs_price:.6f}")
    print(f"  k_bar   = {p_kou.k_bar:.6f}")

    # --- Calibration demo ---
    print("\nCalibration demo (fitting Merton to synthetic quotes):")
    true_params = MertonParams(100, 100, 0.5, 0.05, 0.22, 0.4, -0.04, 0.12)
    quotes = []
    for K in [90, 95, 100, 105, 110]:
        p_q = MertonParams(100, K, 0.5, 0.05, 0.22, 0.4, -0.04, 0.12)
        price = merton_call_series(p_q, n_terms=30)
        # Add tiny noise
        price += random.gauss(0, 0.01)
        quotes.append(MarketQuote(K=K, T=0.5, call_price=max(price, 0)))

    sigma_fit, lam_fit, mu_J_fit, sigma_J_fit = merton_calibrate(
        quotes, S=100, r=0.05,
        init_params=(0.2, 0.3, -0.02, 0.1),
        lr=0.002, n_iters=150
    )
    print(f"  True  : sigma={true_params.sigma:.3f}, lam={true_params.lam:.3f}, mu_J={true_params.mu_J:.3f}, sig_J={true_params.sigma_J:.3f}")
    print(f"  Fitted: sigma={sigma_fit:.3f}, lam={lam_fit:.3f}, mu_J={mu_J_fit:.3f}, sig_J={sigma_J_fit:.3f}")

    # --- Implied vol surface ---
    print("\nImplied vol surface (Merton):")
    strikes = [90, 95, 100, 105, 110]
    maturities = [0.25, 0.5, 1.0]
    surface = merton_implied_vol_surface(p_merton, strikes, maturities)
    header = f"{'K/T':>6} | " + " | ".join(f"{T:>6.2f}" for T in maturities)
    print(header)
    print("-" * len(header))
    for K in strikes:
        row = f"{K:>6} | "
        row += " | ".join(f"{surface[(K,T)]:>6.2%}" for T in maturities)
        print(row)

    print("\n[Done] Day 23: Jump Diffusion complete.")
