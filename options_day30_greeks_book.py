"""
options_day30_greeks_book.py
Day 30: Options Book Management — delta hedging simulation, higher-order Greeks
(vanna, volga, charm, speed), Greeks aggregation across positions, P&L
attribution, hedge effectiveness analysis.
Pure Python stdlib only.
"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

# ---------------------------------------------------------------------------
# 1. Full Greeks suite
# ---------------------------------------------------------------------------
@dataclass
class FullGreeks:
    price: float
    delta: float
    gamma: float
    vega: float      # per 1% move in vol
    theta: float     # per calendar day
    rho: float       # per 1% move in rate
    vanna: float     # dDelta/dVol = dVega/dS
    volga: float     # d²Price/dVol² (convexity in vol)
    charm: float     # dDelta/dT (delta decay per day)
    speed: float     # dGamma/dS (gamma convexity)
    zomma: float     # dGamma/dVol

def bs_full_greeks(S: float, K: float, T: float, r: float, sigma: float,
                    option_type: str = 'call', q: float = 0.0) -> FullGreeks:
    """
    Compute full set of Greeks analytically (including vanna, volga, charm, speed, zomma).
    q = continuous dividend yield.
    """
    if T <= 1e-8 or sigma <= 1e-8:
        payoff = max(S - K, 0) if option_type == 'call' else max(K - S, 0)
        empty = FullGreeks(payoff, 1.0 if (option_type=='call' and S>K) else 0.0,
                            0,0,0,0,0,0,0,0,0)
        return empty

    sqT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqT)
    d2 = d1 - sigma * sqT

    nd1 = _norm_cdf(d1)
    nd2 = _norm_cdf(d2)
    npd1 = _norm_pdf(d1)

    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)

    if option_type == 'call':
        price = S * disc_q * nd1 - K * disc_r * nd2
        delta = disc_q * nd1
        rho   = K * T * disc_r * nd2 / 100
    else:
        price = K * disc_r * _norm_cdf(-d2) - S * disc_q * _norm_cdf(-d1)
        delta = -disc_q * _norm_cdf(-d1)
        rho   = -K * T * disc_r * _norm_cdf(-d2) / 100

    gamma  = disc_q * npd1 / (S * sigma * sqT)
    vega   = S * disc_q * npd1 * sqT / 100  # per 1%

    # Theta (per calendar day)
    theta_call = (-S * disc_q * npd1 * sigma / (2 * sqT)
                  - r * K * disc_r * nd2 + q * S * disc_q * nd1) / 365
    theta = theta_call if option_type == 'call' else (theta_call + r * K * disc_r / 365
                                                        - q * S * disc_q / 365)

    # Higher-order Greeks
    vanna  = -disc_q * npd1 * d2 / sigma          # dDelta/dVol
    volga  = S * disc_q * npd1 * sqT * d1 * d2 / sigma  # d²Price/dVol² (per 1%)
    charm  = (-disc_q * npd1 * (2 * (r - q) * T - d2 * sigma * sqT)
               / (2 * T * sigma * sqT)) / 365
    speed  = -gamma / S * (d1 / (sigma * sqT) + 1)
    zomma  = gamma * (d1 * d2 - 1) / sigma

    return FullGreeks(price, delta, gamma, vega, theta, rho,
                      vanna, volga, charm, speed, zomma)

# ---------------------------------------------------------------------------
# 2. Options Book
# ---------------------------------------------------------------------------
@dataclass
class BookPosition:
    name: str
    S: float
    K: float
    T: float       # years to expiry
    r: float
    sigma: float
    option_type: str
    quantity: int  # number of contracts (+ long, - short)
    multiplier: float = 100.0  # shares per contract

    @property
    def greeks(self) -> FullGreeks:
        return bs_full_greeks(self.S, self.K, self.T, self.r, self.sigma, self.option_type)

    def scaled(self, attr: str) -> float:
        return getattr(self.greeks, attr) * self.quantity * self.multiplier

def aggregate_book(positions: list[BookPosition]) -> dict:
    """Aggregate Greeks across all book positions."""
    attrs = ['price', 'delta', 'gamma', 'vega', 'theta', 'rho',
             'vanna', 'volga', 'charm', 'speed', 'zomma']
    total = {a: sum(p.scaled(a) for p in positions) for a in attrs}
    total['book_value'] = sum(p.greeks.price * p.quantity * p.multiplier for p in positions)
    return total

# ---------------------------------------------------------------------------
# 3. Delta Hedging Simulation
# ---------------------------------------------------------------------------
def delta_hedge_simulation(S0: float, K: float, T: float, r: float,
                             sigma: float, option_type: str = 'call',
                             n_steps: int = 63, n_paths: int = 100,
                             sigma_realized: float | None = None,
                             seed: int = 42) -> dict:
    """
    Simulate delta hedging of a short call position over T years.
    Track: daily hedge P&L, cumulative hedging error, gamma/theta balance.
    sigma_realized: actual vol process (can differ from pricing vol sigma).
    """
    if sigma_realized is None:
        sigma_realized = sigma

    rng = random.Random(seed)
    dt = T / n_steps
    disc = math.exp(-r * T)

    terminal_pnls = []
    hedge_errors_all = []
    gamma_pnl_all = []
    theta_pnl_all = []

    for _ in range(n_paths):
        S_t = S0
        T_t = T
        # Short 1 call, value received = call price
        init_price = bs_full_greeks(S0, K, T, r, sigma, option_type).price
        cash = init_price  # cash received from short call
        shares_held = 0.0
        cum_hedge_err = 0.0
        gamma_pnl = 0.0
        theta_pnl = 0.0

        for step in range(n_steps):
            g = bs_full_greeks(S_t, K, T_t, r, sigma, option_type)
            target_delta = g.delta  # short call: hedge by holding +delta shares
            trade = target_delta - shares_held
            cash -= trade * S_t  # buy/sell shares
            shares_held = target_delta

            # Simulate price move
            eps = math.sqrt(-2 * math.log(max(rng.random(), 1e-15))) * \
                  math.cos(2 * math.pi * rng.random())
            dS = S_t * (r * dt + sigma_realized * math.sqrt(dt) * eps)
            S_t += dS
            T_t = max(T_t - dt, 1e-8)

            # Theoretical gamma P&L = 0.5 * gamma * dS^2
            gamma_pnl += 0.5 * g.gamma * dS**2
            # Theoretical theta P&L = theta * dt * 365 (negative for long, positive for short)
            theta_pnl += -g.theta * dt * 365  # short call: benefit from theta decay

        # At expiry: unwind shares, deliver option payoff
        payoff = max(S_t - K, 0) if option_type == 'call' else max(K - S_t, 0)
        cash += shares_held * S_t  # sell shares
        cash -= payoff              # pay option payoff (short position)
        cash *= math.exp(r * T)    # compound cash (simplification)

        terminal_pnls.append(cash)
        hedge_errors_all.append(cash)
        gamma_pnl_all.append(gamma_pnl)
        theta_pnl_all.append(theta_pnl)

    n = len(terminal_pnls)
    mean_pnl = sum(terminal_pnls) / n
    std_pnl = math.sqrt(sum((p - mean_pnl)**2 for p in terminal_pnls) / max(n - 1, 1))

    return {
        'mean_terminal_pnl': mean_pnl,
        'std_terminal_pnl': std_pnl,
        'pnl_dist': sorted(terminal_pnls),
        'mean_gamma_pnl': sum(gamma_pnl_all) / n,
        'mean_theta_pnl': sum(theta_pnl_all) / n,
        'hedge_effectiveness': 1 - std_pnl / init_price,
    }

# ---------------------------------------------------------------------------
# 4. P&L Attribution (Taylor expansion)
# ---------------------------------------------------------------------------
def pnl_attribution(position: BookPosition, dS: float, dvol: float,
                     dT_days: float, dr: float) -> dict:
    """
    Decompose P&L into Greeks contributions:
    ΔP ≈ delta*dS + 0.5*gamma*dS² + vega*dvol*100 + theta*dT + rho*dr*100
          + vanna*dS*dvol*100 + 0.5*volga*(dvol*100)²
    """
    g = position.greeks
    q = position.quantity * position.multiplier

    delta_pnl  = g.delta * dS * q
    gamma_pnl  = 0.5 * g.gamma * dS**2 * q
    vega_pnl   = g.vega * (dvol * 100) * q
    theta_pnl  = g.theta * dT_days * q
    rho_pnl    = g.rho * (dr * 100) * q
    vanna_pnl  = g.vanna * dS * (dvol * 100) * q / 100  # cross-term scaling
    volga_pnl  = 0.5 * g.volga * (dvol * 100)**2 * q / 100

    total_est  = delta_pnl + gamma_pnl + vega_pnl + theta_pnl + rho_pnl + vanna_pnl + volga_pnl

    return {
        'delta': delta_pnl, 'gamma': gamma_pnl, 'vega': vega_pnl,
        'theta': theta_pnl, 'rho': rho_pnl, 'vanna': vanna_pnl, 'volga': volga_pnl,
        'total_estimated': total_est,
    }

# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 65)
    print("DAY 30: Options Book Management")
    print("=" * 65)

    S, r = 100.0, 0.05

    print("\n1. Full Greeks Suite (ATM Call, T=0.5, σ=20%)")
    g = bs_full_greeks(S, 100, 0.5, r, 0.20, 'call')
    print(f"   Price : {g.price:.4f}")
    print(f"   Delta : {g.delta:+.4f}  (dP/dS)")
    print(f"   Gamma : {g.gamma:+.6f}  (d²P/dS²)")
    print(f"   Vega  : {g.vega:+.4f}  (dP per 1% σ move)")
    print(f"   Theta : {g.theta:+.4f}  (dP per calendar day)")
    print(f"   Rho   : {g.rho:+.4f}  (dP per 1% rate move)")
    print(f"   Vanna : {g.vanna:+.4f}  (dΔ/dσ = dVega/dS)")
    print(f"   Volga : {g.volga:+.4f}  (d²P/dσ²  vol convexity)")
    print(f"   Charm : {g.charm:+.6f}  (dΔ/dt per day)")
    print(f"   Speed : {g.speed:+.6f}  (dΓ/dS)")
    print(f"   Zomma : {g.zomma:+.6f}  (dΓ/dσ)")

    print("\n2. Greeks vs Strike (T=0.5, σ=20%)")
    print(f"   {'K':>5} | {'Delta':>8} | {'Gamma':>9} | {'Vega':>8} | {'Vanna':>8} | {'Volga':>8}")
    print("   " + "-" * 60)
    for K in [85, 90, 95, 100, 105, 110, 115]:
        gg = bs_full_greeks(S, K, 0.5, r, 0.20, 'call')
        print(f"   {K:>5} | {gg.delta:>+8.4f} | {gg.gamma:>9.5f} | "
              f"{gg.vega:>8.4f} | {gg.vanna:>+8.4f} | {gg.volga:>8.4f}")

    print("\n3. Options Book Aggregation")
    book = [
        BookPosition('Long ATM call',   S, 100, 0.5, r, 0.20, 'call',  +10),
        BookPosition('Short OTM call',  S, 110, 0.5, r, 0.22, 'call',  -20),
        BookPosition('Long ATM put',    S, 100, 0.5, r, 0.20, 'put',   +10),
        BookPosition('Short ITM put',   S,  90, 0.5, r, 0.18, 'put',   -5),
        BookPosition('Long 3M call',    S, 100, 0.25, r, 0.21, 'call', +15),
    ]
    agg = aggregate_book(book)
    print(f"   {'Position':25} | {'Qty':>5} | {'Delta':>8} | {'Gamma':>9} | {'Vega':>8} | {'Theta/day':>10}")
    print("   " + "-" * 72)
    for p in book:
        g_p = p.greeks
        scale = p.quantity * p.multiplier
        print(f"   {p.name:25} | {p.quantity:>+5} | {g_p.delta*scale:>+8.2f} | "
              f"{g_p.gamma*scale:>9.4f} | {g_p.vega*scale:>+8.2f} | {g_p.theta*scale:>+10.2f}")
    print("   " + "-" * 72)
    print(f"   {'NET BOOK':25} | {'':>5} | {agg['delta']:>+8.2f} | "
          f"{agg['gamma']:>9.4f} | {agg['vega']:>+8.2f} | {agg['theta']:>+10.2f}")
    print(f"\n   Book value : ${agg['book_value']:,.2f}")
    print(f"   Net vanna  : {agg['vanna']:+.4f}   Net volga: {agg['volga']:+.4f}")

    print("\n4. P&L Attribution (dS=+2, dvol=+2%, dT=1 day)")
    pos = book[0]  # Long ATM call x10
    attr = pnl_attribution(pos, dS=2.0, dvol=0.02, dT_days=1, dr=0.0)
    print(f"   Position: {pos.name} (qty={pos.quantity})")
    total_check = (bs_full_greeks(S + 2, 100, 0.5 - 1/365, r, 0.22, 'call').price -
                   bs_full_greeks(S, 100, 0.5, r, 0.20, 'call').price) * pos.quantity * pos.multiplier
    print(f"   {'Component':12} | {'P&L':>12}")
    print("   " + "-" * 28)
    for k, v in attr.items():
        if k != 'total_estimated':
            print(f"   {k:12} | {v:>+12.4f}")
    print("   " + "-" * 28)
    print(f"   {'Estimated':12} | {attr['total_estimated']:>+12.4f}")
    print(f"   {'Actual':12} | {total_check:>+12.4f}")
    print(f"   {'Error':12} | {attr['total_estimated'] - total_check:>+12.4f}")

    print("\n5. Delta Hedging Simulation (Short Call, 63 daily hedges)")
    print("   Case A: Hedge vol = Realized vol = 20%")
    dh_a = delta_hedge_simulation(S, 100, 0.5, r, sigma=0.20, sigma_realized=0.20,
                                   n_steps=63, n_paths=200, seed=42)
    print(f"   Mean terminal P&L : {dh_a['mean_terminal_pnl']:+.4f}")
    print(f"   Std  terminal P&L : {dh_a['std_terminal_pnl']:+.4f}")
    print(f"   Hedge effectiveness: {dh_a['hedge_effectiveness']:.2%}")
    print(f"   Mean gamma P&L    : {dh_a['mean_gamma_pnl']:+.4f}")
    print(f"   Mean theta P&L    : {dh_a['mean_theta_pnl']:+.4f}")

    print("\n   Case B: Hedge σ=20% but Realized σ=30% (vol mismatch)")
    dh_b = delta_hedge_simulation(S, 100, 0.5, r, sigma=0.20, sigma_realized=0.30,
                                   n_steps=63, n_paths=200, seed=42)
    print(f"   Mean terminal P&L : {dh_b['mean_terminal_pnl']:+.4f}")
    print(f"   Std  terminal P&L : {dh_b['std_terminal_pnl']:+.4f}")
    print(f"   (Short call loses when realized vol > implied vol)")

    print("\n6. Gamma/Theta Tradeoff by Moneyness")
    T_short = 0.083  # ~1 month
    print(f"   T = {T_short:.2f}Y (1 month)")
    print(f"   {'K':>5} | {'Gamma':>9} | {'Theta/day':>10} | {'G/T ratio':>10}")
    print("   " + "-" * 43)
    for K in [90, 95, 100, 105, 110]:
        gg = bs_full_greeks(S, K, T_short, r, 0.20, 'call')
        gt_ratio = gg.gamma / max(abs(gg.theta), 1e-8)
        print(f"   {K:>5} | {gg.gamma:>9.5f} | {gg.theta:>+10.4f} | {gt_ratio:>+10.4f}")

    print("\n[Done] Day 30: Options Book Management complete.")
