"""
options_day27_exotic_options.py
Day 27: Exotic Options — barrier options (analytical + MC), Asian options
(geometric closed-form + arithmetic MC), lookback options, forward-starting options.
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

def bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(S - K * math.exp(-r * T), 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)

def bs_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    return bs_call(S, K, T, r, sigma) - S + K * math.exp(-r * T)

# ---------------------------------------------------------------------------
# 1. Analytical Barrier Options (Merton 1973 / Reiner-Rubinstein 1991)
# ---------------------------------------------------------------------------
def _barrier_phi(S: float, T: float, lambda_: float, x: float,
                 r: float, b: float, sigma: float) -> float:
    """Helper: φ function for barrier option pricing (Reiner-Rubinstein)."""
    a = (math.log(x / S) + (b + (lambda_ - 0.5) * sigma**2) * T) / (sigma * math.sqrt(T))
    return _norm_cdf(a)

def _d1_d2(S, K, T, r, b, sigma):
    d1 = (math.log(S / K) + (b + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return d1, d1 - sigma * math.sqrt(T)

def down_and_out_call(S: float, K: float, H: float, T: float,
                       r: float, sigma: float, rebate: float = 0.0) -> float:
    """
    Down-and-out call: knocked out if S hits barrier H from above (H < S, H < K).
    Uses Reiner-Rubinstein (1991) closed form.
    """
    if H >= S:
        return 0.0  # already knocked out
    b = r  # cost-of-carry = r for non-dividend stock
    mu = (b - 0.5 * sigma**2) / sigma**2
    lam = math.sqrt(mu**2 + 2 * r / sigma**2)

    x1 = math.log(S / K) / (sigma * math.sqrt(T)) + (1 + mu) * sigma * math.sqrt(T)
    x2 = math.log(S / H) / (sigma * math.sqrt(T)) + (1 + mu) * sigma * math.sqrt(T)
    y1 = math.log(H**2 / (S * K)) / (sigma * math.sqrt(T)) + (1 + mu) * sigma * math.sqrt(T)
    y2 = math.log(H / S) / (sigma * math.sqrt(T)) + (1 + mu) * sigma * math.sqrt(T)
    z  = math.log(H / S) / (sigma * math.sqrt(T)) + lam * sigma * math.sqrt(T)

    A = S * math.exp((b - r) * T) * _norm_cdf(x1) - K * math.exp(-r * T) * _norm_cdf(x1 - sigma * math.sqrt(T))
    B = S * math.exp((b - r) * T) * _norm_cdf(x2) - K * math.exp(-r * T) * _norm_cdf(x2 - sigma * math.sqrt(T))
    C = S * math.exp((b - r) * T) * (H / S)**(2 * (mu + 1)) * _norm_cdf(y1) \
        - K * math.exp(-r * T) * (H / S)**(2 * mu) * _norm_cdf(y1 - sigma * math.sqrt(T))
    D = S * math.exp((b - r) * T) * (H / S)**(2 * (mu + 1)) * _norm_cdf(y2) \
        - K * math.exp(-r * T) * (H / S)**(2 * mu) * _norm_cdf(y2 - sigma * math.sqrt(T))
    E = rebate * math.exp(-r * T) * (_norm_cdf(x2 - sigma * math.sqrt(T)) - (H / S)**(2 * mu) * _norm_cdf(y2 - sigma * math.sqrt(T)))
    F = rebate * ((H / S)**(mu + lam) * _norm_cdf(z) + (H / S)**(mu - lam) * _norm_cdf(z - 2 * lam * sigma * math.sqrt(T)))

    if K >= H:
        return A - C + F
    else:
        return B - D + F

def down_and_in_call(S: float, K: float, H: float, T: float,
                      r: float, sigma: float, rebate: float = 0.0) -> float:
    """Down-and-in call: activated when S hits H from above."""
    vanilla = bs_call(S, K, T, r, sigma)
    doc = down_and_out_call(S, K, H, T, r, sigma, rebate)
    return vanilla - doc

def up_and_out_put(S: float, K: float, H: float, T: float,
                    r: float, sigma: float, rebate: float = 0.0) -> float:
    """
    Up-and-out put: knocked out if S rises to H (H > S, H > K).
    Via put-call symmetry of barrier options.
    """
    if H <= S:
        return 0.0
    # Use reflection: UOP(S,K,H) = DOC(1/S, 1/K, 1/H) with sign flip
    # Numerically: price via MC or use Haug's formula
    b = r
    mu = (b - 0.5 * sigma**2) / sigma**2

    x1 = math.log(S / K) / (sigma * math.sqrt(T)) + (1 + mu) * sigma * math.sqrt(T)
    x2 = math.log(S / H) / (sigma * math.sqrt(T)) + (1 + mu) * sigma * math.sqrt(T)
    y1 = math.log(H**2 / (S * K)) / (sigma * math.sqrt(T)) + (1 + mu) * sigma * math.sqrt(T)
    y2 = math.log(H / S) / (sigma * math.sqrt(T)) + (1 + mu) * sigma * math.sqrt(T)

    lam = math.sqrt(mu**2 + 2 * r / sigma**2)
    z   = math.log(H / S) / (sigma * math.sqrt(T)) + lam * sigma * math.sqrt(T)

    A = -S * math.exp((b - r) * T) * _norm_cdf(-x1) + K * math.exp(-r * T) * _norm_cdf(-x1 + sigma * math.sqrt(T))
    B = -S * math.exp((b - r) * T) * _norm_cdf(-x2) + K * math.exp(-r * T) * _norm_cdf(-x2 + sigma * math.sqrt(T))
    C = -S * math.exp((b - r) * T) * (H / S)**(2 * (mu + 1)) * _norm_cdf(-y1) \
        + K * math.exp(-r * T) * (H / S)**(2 * mu) * _norm_cdf(-y1 + sigma * math.sqrt(T))
    D = -S * math.exp((b - r) * T) * (H / S)**(2 * (mu + 1)) * _norm_cdf(-y2) \
        + K * math.exp(-r * T) * (H / S)**(2 * mu) * _norm_cdf(-y2 + sigma * math.sqrt(T))
    F = rebate * ((H / S)**(mu + lam) * _norm_cdf(-z) + (H / S)**(mu - lam) * _norm_cdf(-z + 2 * lam * sigma * math.sqrt(T)))

    if K <= H:
        return A - B + C - D + F
    else:
        return F

def up_and_in_put(S, K, H, T, r, sigma, rebate=0.0):
    vanilla = bs_put(S, K, T, r, sigma)
    uop = up_and_out_put(S, K, H, T, r, sigma, rebate)
    return vanilla - uop

# ---------------------------------------------------------------------------
# 2. Barrier Option MC (with continuous monitoring approximation)
# ---------------------------------------------------------------------------
def barrier_call_mc(S: float, K: float, H: float, T: float, r: float,
                    sigma: float, barrier_type: str = 'down_out',
                    n_paths: int = 50000, n_steps: int = 252, seed: int = 42) -> float:
    """
    MC barrier call: barrier_type in {'down_out', 'down_in', 'up_out', 'up_in'}.
    Uses Brownian bridge correction for barrier crossing between steps.
    """
    rng = random.Random(seed)
    dt = T / n_steps
    disc = math.exp(-r * T)
    drift = (r - 0.5 * sigma**2) * dt
    vol_dt = sigma * math.sqrt(dt)

    payoffs = []
    is_out = barrier_type in ('down_out', 'up_out')
    is_down = barrier_type in ('down_out', 'down_in')

    for _ in range(n_paths):
        S_t = S
        knocked = False
        for _ in range(n_steps):
            z = math.sqrt(-2 * math.log(max(rng.random(), 1e-15))) * math.cos(2 * math.pi * rng.random())
            S_prev = S_t
            S_t = S_t * math.exp(drift + vol_dt * z)
            if not knocked:
                if is_down and S_prev > H:
                    S_lo = min(S_prev, S_t)
                    if S_lo <= H:
                        knocked = True
                    else:
                        p_cross = math.exp(-2 * math.log(S_prev / H) * math.log(S_t / H) / (sigma**2 * dt))
                        if rng.random() < p_cross:
                            knocked = True
                elif not is_down and S_prev < H:
                    if S_t >= H:
                        knocked = True
                    else:
                        p_cross = math.exp(-2 * math.log(H / S_prev) * math.log(H / S_t) / (sigma**2 * dt))
                        if rng.random() < p_cross:
                            knocked = True
                # For knock-out: stop simulating once knocked (payoff = 0)
                if is_out and knocked:
                    break

        call_payoff = max(S_t - K, 0.0)
        if is_out:
            payoffs.append(0.0 if knocked else call_payoff)
        else:  # knock-in: need terminal S_t, so we don't break early
            payoffs.append(call_payoff if knocked else 0.0)

    return disc * sum(payoffs) / len(payoffs)

# ---------------------------------------------------------------------------
# 3. Asian Options
# ---------------------------------------------------------------------------
def asian_geometric_call(S: float, K: float, T: float, r: float,
                           sigma: float, n: int = 252) -> float:
    """
    Geometric Asian call (closed form).
    Average of geometric means of daily prices → use adjusted parameters.
    """
    # Adjusted vol and drift for geometric average
    sigma_g = sigma * math.sqrt((2 * n + 1) / (6 * (n + 1)))
    r_g = 0.5 * (r - 0.5 * sigma**2 + sigma_g**2)
    return bs_call(S, K, T, r_g + 0.5 * sigma_g**2, sigma_g) * math.exp((r_g - r) * T)

def asian_arithmetic_call_mc(S: float, K: float, T: float, r: float,
                               sigma: float, n_steps: int = 252,
                               n_paths: int = 50000, seed: int = 42) -> tuple[float, float]:
    """
    Arithmetic Asian call (MC with geometric control variate).
    Returns (price, std_error).
    """
    rng = random.Random(seed)
    dt = T / n_steps
    drift = (r - 0.5 * sigma**2) * dt
    vol_dt = sigma * math.sqrt(dt)
    disc = math.exp(-r * T)

    # Control variate: geometric Asian (known price)
    cv_price = asian_geometric_call(S, K, T, r, sigma, n_steps)

    arith_payoffs = []
    geom_payoffs = []

    for _ in range(n_paths):
        path = [S]
        for _ in range(n_steps):
            z = math.sqrt(-2 * math.log(max(rng.random(), 1e-15))) * math.cos(2 * math.pi * rng.random())
            path.append(path[-1] * math.exp(drift + vol_dt * z))

        path = path[1:]  # exclude S0
        arith_avg = sum(path) / n_steps
        geom_avg = math.exp(sum(math.log(p) for p in path) / n_steps)

        arith_payoffs.append(max(arith_avg - K, 0.0))
        geom_payoffs.append(max(geom_avg - K, 0.0))

    n_p = len(arith_payoffs)
    mean_a = sum(arith_payoffs) / n_p
    mean_g = sum(geom_payoffs) / n_p

    # Control variate adjustment
    cov_ag = sum((a - mean_a) * (g - mean_g) for a, g in zip(arith_payoffs, geom_payoffs)) / n_p
    var_g = sum((g - mean_g)**2 for g in geom_payoffs) / n_p
    beta_cv = cov_ag / max(var_g, 1e-12)

    cv_adjusted = [a - beta_cv * (g - cv_price / disc) for a, g in zip(arith_payoffs, geom_payoffs)]
    price = disc * sum(cv_adjusted) / n_p
    variance = sum((x - sum(cv_adjusted) / n_p)**2 for x in cv_adjusted) / (n_p - 1)
    std_err = disc * math.sqrt(variance / n_p)

    return price, std_err

# ---------------------------------------------------------------------------
# 4. Lookback Options
# ---------------------------------------------------------------------------
def lookback_floating_call_mc(S: float, T: float, r: float, sigma: float,
                               n_steps: int = 252, n_paths: int = 50000,
                               seed: int = 42) -> float:
    """
    Floating strike lookback call: payoff = S_T - min(S_t).
    Closed form exists but MC demonstrates the path-dependency clearly.
    """
    rng = random.Random(seed)
    dt = T / n_steps
    drift = (r - 0.5 * sigma**2) * dt
    vol_dt = sigma * math.sqrt(dt)
    disc = math.exp(-r * T)

    payoffs = []
    for _ in range(n_paths):
        S_t = S
        S_min = S
        for _ in range(n_steps):
            z = math.sqrt(-2 * math.log(max(rng.random(), 1e-15))) * math.cos(2 * math.pi * rng.random())
            S_t *= math.exp(drift + vol_dt * z)
            S_min = min(S_min, S_t)
        payoffs.append(max(S_t - S_min, 0.0))

    return disc * sum(payoffs) / len(payoffs)

def lookback_fixed_call_mc(S: float, K: float, T: float, r: float, sigma: float,
                            n_steps: int = 252, n_paths: int = 50000,
                            seed: int = 42) -> float:
    """Fixed strike lookback call: payoff = max(S_max - K, 0)."""
    rng = random.Random(seed)
    dt = T / n_steps
    drift = (r - 0.5 * sigma**2) * dt
    vol_dt = sigma * math.sqrt(dt)
    disc = math.exp(-r * T)

    payoffs = []
    for _ in range(n_paths):
        S_t = S
        S_max = S
        for _ in range(n_steps):
            z = math.sqrt(-2 * math.log(max(rng.random(), 1e-15))) * math.cos(2 * math.pi * rng.random())
            S_t *= math.exp(drift + vol_dt * z)
            S_max = max(S_max, S_t)
        payoffs.append(max(S_max - K, 0.0))

    return disc * sum(payoffs) / len(payoffs)

# ---------------------------------------------------------------------------
# 5. Forward-Starting Options
# ---------------------------------------------------------------------------
def forward_starting_call(S: float, alpha: float, T_start: float, T_end: float,
                           r: float, sigma: float) -> float:
    """
    Forward-starting call: strike set as K = alpha * S_{T_start} at future date T_start.
    Price at t=0. For dividend-free stock: = S * e^{-r*T_start} * BS(1, alpha, T_end-T_start, r, sigma).
    """
    tau = T_end - T_start
    return S * math.exp(-r * T_start) * bs_call(1.0, alpha, tau, r, sigma)

# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 65)
    print("DAY 27: Exotic Options")
    print("=" * 65)

    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    H_down, H_up = 90.0, 110.0

    print(f"\nParams: S={S}, K={K}, T={T}, r={r}, sigma={sigma}")
    print(f"Vanilla call (BS): {bs_call(S, K, T, r, sigma):.4f}")
    print(f"Vanilla put  (BS): {bs_put(S, K, T, r, sigma):.4f}")

    print("\n1. Barrier Options (Analytical vs MC)")
    print(f"   Barrier H_down={H_down}, H_up={H_up}")
    print(f"   {'Type':25} | {'Analytical':>12} | {'MC (50K)':>12} | {'Error':>8}")
    print("   " + "-" * 65)

    barriers = [
        ("Down-and-out call",   down_and_out_call(S, K, H_down, T, r, sigma),
         barrier_call_mc(S, K, H_down, T, r, sigma, 'down_out')),
        ("Down-and-in call",    down_and_in_call(S, K, H_down, T, r, sigma),
         barrier_call_mc(S, K, H_down, T, r, sigma, 'down_in')),
        ("Up-and-out put",      up_and_out_put(S, K, H_up, T, r, sigma),
         None),
        ("Up-and-in put",       up_and_in_put(S, K, H_up, T, r, sigma),
         None),
    ]
    # Parity check: DOC + DIC = vanilla call
    doc = down_and_out_call(S, K, H_down, T, r, sigma)
    dic = down_and_in_call(S, K, H_down, T, r, sigma)

    for name, anal, mc in barriers:
        mc_str = f"{mc:12.4f}" if mc is not None else f"{'—':>12}"
        err_str = f"{mc - anal:+8.4f}" if mc is not None else f"{'—':>8}"
        print(f"   {name:25} | {anal:12.4f} | {mc_str} | {err_str}")

    print(f"\n   Parity: DOC + DIC = {doc:.4f} + {dic:.4f} = {doc+dic:.4f} "
          f"vs vanilla = {bs_call(S, K, T, r, sigma):.4f}")

    print("\n2. Asian Options")
    geo_price = asian_geometric_call(S, K, T, r, sigma)
    arith_price, arith_se = asian_arithmetic_call_mc(S, K, T, r, sigma)
    vanilla = bs_call(S, K, T, r, sigma)
    print(f"   Vanilla call         : {vanilla:.4f}")
    print(f"   Geometric Asian call : {geo_price:.4f}  (analytical)")
    print(f"   Arithmetic Asian call: {arith_price:.4f} ± {arith_se:.4f}  (MC + CV)")
    print(f"   Asian discount       : {(vanilla - arith_price) / vanilla * 100:.1f}% of vanilla")

    print("\n   Strike sensitivity (arithmetic Asian MC):")
    print(f"   {'K':>5} | {'Vanilla':>10} | {'Asian':>10} | {'Asian/Vanilla':>14}")
    print("   " + "-" * 45)
    for Ks in [90, 95, 100, 105, 110]:
        v = bs_call(S, Ks, T, r, sigma)
        a, _ = asian_arithmetic_call_mc(S, Ks, T, r, sigma, n_paths=20000, seed=Ks)
        ratio = a / v if v > 0.001 else 0
        print(f"   {Ks:>5} | {v:>10.4f} | {a:>10.4f} | {ratio:>14.3f}")

    print("\n3. Lookback Options")
    lb_float = lookback_floating_call_mc(S, T, r, sigma, n_paths=30000)
    lb_fixed = lookback_fixed_call_mc(S, K, T, r, sigma, n_paths=30000)
    print(f"   Floating strike lookback call (S_T - min S): {lb_float:.4f}")
    print(f"   Fixed strike lookback call   (max S - K)  : {lb_fixed:.4f}")
    print(f"   Vanilla call                               : {bs_call(S, K, T, r, sigma):.4f}")
    print(f"   Lookback premium vs vanilla: "
          f"{(lb_fixed - bs_call(S, K, T, r, sigma)) / bs_call(S, K, T, r, sigma) * 100:.1f}%")

    print("\n4. Forward-Starting Options")
    print(f"   {'T_start':>8} | {'T_end':>6} | {'alpha':>6} | {'Price':>8} | {'vs Vanilla':>10}")
    print("   " + "-" * 50)
    for T_s, alpha in [(0.25, 1.0), (0.50, 1.0), (0.25, 1.05), (0.50, 1.05)]:
        fs = forward_starting_call(S, alpha, T_s, T_s + T, r, sigma)
        v_equiv = bs_call(S, alpha * S, T, r, sigma) * math.exp(-r * T_s)  # approx comparison
        print(f"   {T_s:>8.2f} | {T_s + T:>6.2f} | {alpha:>6.2f} | {fs:>8.4f} | {fs/v_equiv - 1:>+9.2%}")

    print("\n[Done] Day 27: Exotic Options complete.")
