"""
options_day22_scenario_analysis.py
Day 22: Scenario Analysis — Greeks P&L attribution, crisis scenarios,
vol surface stress testing, tail-risk ladder.
Pure Python stdlib only.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Normal CDF / PDF helpers
# ---------------------------------------------------------------------------
def _N(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def _n(x: float) -> float:
    return math.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)

# ---------------------------------------------------------------------------
# Black-Scholes pricer + Greeks
# ---------------------------------------------------------------------------
def bs_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K * math.exp(-r * T), 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _N(d1) - K * math.exp(-r * T) * _N(d2)

def bs_put(S, K, T, r, sigma):
    return bs_call(S, K, T, r, sigma) - S + K * math.exp(-r * T)

@dataclass
class BSGreeks:
    delta: float
    gamma: float
    vega: float     # per 1 vol point (1%)
    theta: float    # per calendar day
    rho: float      # per 1% rate move
    price: float

def bs_greeks(S, K, T, r, sigma, option_type='call') -> BSGreeks:
    """Compute Black-Scholes price and all first/second order Greeks."""
    if T <= 0:
        price = max(S - K, 0.0) if option_type == 'call' else max(K - S, 0.0)
        return BSGreeks(0.0, 0.0, 0.0, 0.0, 0.0, price)

    sqT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqT)
    d2 = d1 - sigma * sqT
    disc = math.exp(-r * T)

    delta_c = _N(d1)
    delta_p = delta_c - 1.0
    gamma = _n(d1) / (S * sigma * sqT)
    vega_raw = S * _n(d1) * sqT  # per unit vol
    vega = vega_raw / 100.0      # per 1 vol point

    if option_type == 'call':
        price = S * delta_c - K * disc * _N(d2)
        delta = delta_c
        theta_raw = (-S * _n(d1) * sigma / (2 * sqT) - r * K * disc * _N(d2))
        rho_val = K * T * disc * _N(d2) / 100.0
    else:
        price = K * disc * _N(-d2) - S * _N(-d1)
        delta = delta_p
        theta_raw = (-S * _n(d1) * sigma / (2 * sqT) + r * K * disc * _N(-d2))
        rho_val = -K * T * disc * _N(-d2) / 100.0

    theta = theta_raw / 365.0  # per calendar day

    return BSGreeks(delta=delta, gamma=gamma, vega=vega,
                    theta=theta, rho=rho_val, price=price)

# ---------------------------------------------------------------------------
# Portfolio of options
# ---------------------------------------------------------------------------
@dataclass
class OptionPosition:
    S: float
    K: float
    T: float
    r: float
    sigma: float
    option_type: str  # 'call' or 'put'
    quantity: float   # positive = long, negative = short
    name: str = ''

    def greeks(self) -> BSGreeks:
        g = bs_greeks(self.S, self.K, self.T, self.r, self.sigma, self.option_type)
        return BSGreeks(
            delta=g.delta * self.quantity,
            gamma=g.gamma * self.quantity,
            vega=g.vega * self.quantity,
            theta=g.theta * self.quantity,
            rho=g.rho * self.quantity,
            price=g.price * self.quantity,
        )

# ---------------------------------------------------------------------------
# Greeks P&L attribution (Taylor expansion)
# ---------------------------------------------------------------------------
def greeks_pnl_attribution(positions: list[OptionPosition],
                            dS: float, dvol: float, dT_days: float = 1.0,
                            dr: float = 0.0) -> dict:
    """
    Decompose P&L using Greeks:
      dV ≈ Delta*dS + 0.5*Gamma*dS^2 + Vega*dvol + Theta*dT + Rho*dr
    """
    total_delta = total_gamma = total_vega = total_theta = total_rho = 0.0
    for pos in positions:
        g = pos.greeks()
        total_delta += g.delta
        total_gamma += g.gamma
        total_vega += g.vega
        total_theta += g.theta
        total_rho += g.rho

    pnl_delta = total_delta * dS
    pnl_gamma = 0.5 * total_gamma * dS**2
    pnl_vega = total_vega * dvol          # dvol in vol points
    pnl_theta = total_theta * dT_days
    pnl_rho = total_rho * dr              # dr in %
    pnl_total = pnl_delta + pnl_gamma + pnl_vega + pnl_theta + pnl_rho

    return {
        'delta_pnl': pnl_delta,
        'gamma_pnl': pnl_gamma,
        'vega_pnl': pnl_vega,
        'theta_pnl': pnl_theta,
        'rho_pnl': pnl_rho,
        'total_approx_pnl': pnl_total,
        'net_delta': total_delta,
        'net_gamma': total_gamma,
        'net_vega': total_vega,
        'net_theta': total_theta,
        'net_rho': total_rho,
    }

# ---------------------------------------------------------------------------
# Crisis scenario definitions
# ---------------------------------------------------------------------------
CRISIS_SCENARIOS = {
    'COVID_Crash_2020': {'dS_pct': -34.0, 'dvol': 40.0, 'dr_bps': -150},
    'GFC_2008':         {'dS_pct': -57.0, 'dvol': 60.0, 'dr_bps': -400},
    'Black_Monday_1987':{'dS_pct': -22.0, 'dvol': 80.0, 'dr_bps': 50},
    'Dot_Com_2000':     {'dS_pct': -49.0, 'dvol': 35.0, 'dr_bps': -500},
    'Flash_Crash_2010': {'dS_pct': -9.0,  'dvol': 25.0, 'dr_bps': -20},
    'Taper_Tantrum_2013':{'dS_pct': -6.0, 'dvol': 12.0, 'dr_bps': 100},
    'Mild_Correction':  {'dS_pct': -10.0, 'dvol': 10.0, 'dr_bps': 0},
    'Slow_Bleed':       {'dS_pct': -20.0, 'dvol': 20.0, 'dr_bps': -50},
}

def run_crisis_scenarios(positions: list[OptionPosition]) -> list[dict]:
    """Re-price portfolio under historical crisis shocks."""
    results = []
    for name, shock in CRISIS_SCENARIOS.items():
        shocked_pnl = 0.0
        for pos in positions:
            S_new = pos.S * (1 + shock['dS_pct'] / 100.0)
            sigma_new = max(pos.sigma + shock['dvol'] / 100.0, 1e-4)
            r_new = pos.r + shock['dr_bps'] / 10000.0

            T_new = max(pos.T, 0.01)
            old_price = bs_greeks(pos.S, pos.K, pos.T, pos.r, pos.sigma, pos.option_type).price
            new_price = bs_greeks(S_new, pos.K, T_new, r_new, sigma_new, pos.option_type).price
            shocked_pnl += (new_price - old_price) * pos.quantity

        results.append({
            'scenario': name,
            'dS_pct': shock['dS_pct'],
            'dvol': shock['dvol'],
            'dr_bps': shock['dr_bps'],
            'pnl': shocked_pnl,
        })
    return results

# ---------------------------------------------------------------------------
# Vol surface stress testing
# ---------------------------------------------------------------------------
def vol_surface_stress(positions: list[OptionPosition],
                       parallel_shifts: list[float],
                       skew_multipliers: list[float],
                       term_structure_pivots: list[float]) -> list[dict]:
    """
    Stress vol surface with parallel shifts and skew/term-structure tweaks.
    parallel_shifts: list of additive vol shifts (e.g. [-0.10, 0, +0.10])
    skew_multipliers: multiplier on (ATM vol - current) for OTM options
    term_structure_pivots: additive shift to short-dated vol only
    """
    base_value = sum(
        bs_greeks(p.S, p.K, p.T, p.r, p.sigma, p.option_type).price * p.quantity
        for p in positions
    )

    results = []
    for shift in parallel_shifts:
        pnl = 0.0
        for pos in positions:
            sigma_new = max(pos.sigma + shift, 0.01)
            new_p = bs_greeks(pos.S, pos.K, pos.T, pos.r, sigma_new, pos.option_type).price
            old_p = bs_greeks(pos.S, pos.K, pos.T, pos.r, pos.sigma, pos.option_type).price
            pnl += (new_p - old_p) * pos.quantity
        results.append({'type': 'parallel_shift', 'shift': shift, 'pnl': pnl})

    for mult in skew_multipliers:
        pnl = 0.0
        for pos in positions:
            atm_vol = pos.sigma  # approximate ATM as current
            moneyness = math.log(pos.S / pos.K)  # + = ITM call
            skew_adj = -mult * 0.05 * moneyness   # synthetic skew: OTM puts more expensive
            sigma_new = max(pos.sigma + skew_adj, 0.01)
            new_p = bs_greeks(pos.S, pos.K, pos.T, pos.r, sigma_new, pos.option_type).price
            old_p = bs_greeks(pos.S, pos.K, pos.T, pos.r, pos.sigma, pos.option_type).price
            pnl += (new_p - old_p) * pos.quantity
        results.append({'type': 'skew_twist', 'multiplier': mult, 'pnl': pnl})

    for pivot in term_structure_pivots:
        pnl = 0.0
        for pos in positions:
            # Short-dated (<3M) gets the pivot shift, long-dated gets less
            if pos.T < 0.25:
                sigma_new = max(pos.sigma + pivot, 0.01)
            elif pos.T < 0.5:
                sigma_new = max(pos.sigma + pivot * 0.5, 0.01)
            else:
                sigma_new = pos.sigma  # long end unchanged
            new_p = bs_greeks(pos.S, pos.K, pos.T, pos.r, sigma_new, pos.option_type).price
            old_p = bs_greeks(pos.S, pos.K, pos.T, pos.r, pos.sigma, pos.option_type).price
            pnl += (new_p - old_p) * pos.quantity
        results.append({'type': 'term_structure_pivot', 'pivot': pivot, 'pnl': pnl})

    return results

# ---------------------------------------------------------------------------
# Tail-risk ladder
# ---------------------------------------------------------------------------
def tail_risk_ladder(positions: list[OptionPosition],
                     spot_moves: list[float],
                     vol_beta: float = 1.5) -> list[dict]:
    """
    Compute full reprice PnL for a grid of spot moves.
    Vol is assumed to move by -vol_beta * spot_move (inverse leverage effect).
    """
    rows = []
    for dS_pct in spot_moves:
        total_pnl = 0.0
        for pos in positions:
            S_new = pos.S * (1 + dS_pct / 100.0)
            # Inverse leverage: down moves spike vol, up moves compress vol
            dvol = -vol_beta * (dS_pct / 100.0) * pos.sigma
            sigma_new = max(pos.sigma + dvol, 0.01)
            old_p = bs_greeks(pos.S, pos.K, pos.T, pos.r, pos.sigma, pos.option_type).price
            new_p = bs_greeks(S_new, pos.K, pos.T, pos.r, sigma_new, pos.option_type).price
            total_pnl += (new_p - old_p) * pos.quantity

        rows.append({
            'spot_move_pct': dS_pct,
            'pnl': total_pnl,
            'scenario_label': f"S {'↑' if dS_pct > 0 else '↓'} {abs(dS_pct):.0f}%",
        })
    return rows

# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 65)
    print("DAY 22: Scenario Analysis & Stress Testing")
    print("=" * 65)

    # Example portfolio: long straddle + short OTM calls
    S = 100.0
    r = 0.05

    positions = [
        OptionPosition(S=S, K=100.0, T=0.25, r=r, sigma=0.20, option_type='call',
                       quantity=100, name='ATM Call'),
        OptionPosition(S=S, K=100.0, T=0.25, r=r, sigma=0.20, option_type='put',
                       quantity=100, name='ATM Put'),
        OptionPosition(S=S, K=110.0, T=0.25, r=r, sigma=0.22, option_type='call',
                       quantity=-100, name='OTM Call Short'),
        OptionPosition(S=S, K=90.0,  T=0.25, r=r, sigma=0.23, option_type='put',
                       quantity=-100, name='OTM Put Short'),
    ]

    print("\nPortfolio:")
    total_val = 0.0
    for pos in positions:
        g = pos.greeks()
        total_val += g.price
        print(f"  {pos.name:20s} qty={pos.quantity:+5d}  price={g.price / pos.quantity:7.4f}  "
              f"delta={g.delta:+8.4f}  gamma={g.gamma:+8.6f}")
    print(f"  {'Total':20s}  value={total_val:10.4f}")

    print("\n1. Greeks P&L Attribution (S+2%, vol-3pts, 1 day decay)")
    attr = greeks_pnl_attribution(positions, dS=2.0, dvol=-3.0, dT_days=1.0)
    print(f"   Delta P&L  : {attr['delta_pnl']:+10.4f}")
    print(f"   Gamma P&L  : {attr['gamma_pnl']:+10.4f}")
    print(f"   Vega P&L   : {attr['vega_pnl']:+10.4f}")
    print(f"   Theta P&L  : {attr['theta_pnl']:+10.4f}")
    print(f"   Total (approx): {attr['total_approx_pnl']:+10.4f}")

    print("\n2. Crisis Scenario Analysis")
    crisis = run_crisis_scenarios(positions)
    print(f"  {'Scenario':25s} | {'dS%':>6} | {'dVol':>6} | {'P&L':>12}")
    print("  " + "-" * 60)
    for row in crisis:
        print(f"  {row['scenario']:25s} | {row['dS_pct']:>6.1f} | {row['dvol']:>6.1f} | {row['pnl']:>12.2f}")

    print("\n3. Vol Surface Stress")
    stress = vol_surface_stress(
        positions,
        parallel_shifts=[-0.10, -0.05, 0.0, 0.05, 0.10],
        skew_multipliers=[0.5, 1.0, 2.0],
        term_structure_pivots=[-0.05, 0.05]
    )
    for row in stress:
        if row['type'] == 'parallel_shift':
            print(f"  Parallel shift {row['shift']:+.0%}: P&L = {row['pnl']:+.4f}")
    for row in stress:
        if row['type'] == 'skew_twist':
            print(f"  Skew twist {row['multiplier']:>4.1f}x: P&L = {row['pnl']:+.4f}")
    for row in stress:
        if row['type'] == 'term_structure_pivot':
            print(f"  Term pivot {row['pivot']:+.0%}: P&L = {row['pnl']:+.4f}")

    print("\n4. Tail-Risk Ladder")
    ladder = tail_risk_ladder(
        positions,
        spot_moves=[-30, -20, -15, -10, -5, 0, 5, 10, 15, 20, 30],
        vol_beta=1.5
    )
    print(f"  {'Scenario':12s} | {'P&L':>12}")
    print("  " + "-" * 28)
    for row in ladder:
        bar = '█' * max(int(row['pnl'] / 2), 0) if row['pnl'] > 0 else '▒' * max(int(-row['pnl'] / 2), 0)
        print(f"  {row['scenario_label']:12s} | {row['pnl']:+12.2f}  {bar}")

    print("\n[Done] Day 22: Scenario Analysis complete.")
