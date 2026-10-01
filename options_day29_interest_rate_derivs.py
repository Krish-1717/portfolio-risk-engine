"""
options_day29_interest_rate_derivs.py
Day 29: Interest Rate Derivatives — yield curve bootstrapping, Black's model
for caplets/floorlets/swaptions, cap/floor pricing, duration/DV01/convexity,
interest rate sensitivity.
Pure Python stdlib only.
"""
from __future__ import annotations
import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

# ---------------------------------------------------------------------------
# 1. Yield Curve Bootstrapping
# ---------------------------------------------------------------------------
@dataclass
class YieldCurve:
    """Piecewise-linear zero-rate curve bootstrapped from par swap rates."""
    tenors: list[float]      # years
    zero_rates: list[float]  # continuously compounded annual rates

    def discount(self, T: float) -> float:
        """Discount factor P(0, T) = exp(-r(T) * T)."""
        r = self.zero_rate(T)
        return math.exp(-r * T)

    def zero_rate(self, T: float) -> float:
        """Linear interpolation of zero rates."""
        if T <= self.tenors[0]:
            return self.zero_rates[0]
        if T >= self.tenors[-1]:
            return self.zero_rates[-1]
        for i in range(len(self.tenors) - 1):
            t0, t1 = self.tenors[i], self.tenors[i + 1]
            if t0 <= T <= t1:
                alpha = (T - t0) / (t1 - t0)
                return self.zero_rates[i] * (1 - alpha) + self.zero_rates[i + 1] * alpha
        return self.zero_rates[-1]

    def forward_rate(self, T1: float, T2: float) -> float:
        """Simple forward rate: F(T1, T2) = [P(0,T1)/P(0,T2) - 1] / (T2-T1)."""
        if T2 <= T1:
            return self.zero_rate(T1)
        p1 = self.discount(T1)
        p2 = self.discount(T2)
        return (p1 / p2 - 1.0) / (T2 - T1)

def bootstrap_yield_curve(par_swap_rates: list[tuple[float, float]]) -> YieldCurve:
    """
    Bootstrap zero curve from par swap rates [(tenor, rate)].
    Assumes annual coupon payments.
    Simple iterative bootstrapping.
    """
    tenors = [t for t, _ in par_swap_rates]
    zero_rates = []
    discount_factors = {}

    for i, (T, c) in enumerate(par_swap_rates):
        # Sum of discounted coupons up to T-1
        coupon_pv = sum(c * discount_factors.get(t, math.exp(-zero_rates[j] * t))
                        for j, (t, _) in enumerate(par_swap_rates[:i]))
        # Solve: coupon_pv + (1 + c) * P(0,T) = 1
        p_T = (1 - coupon_pv) / (1 + c)
        p_T = max(p_T, 1e-10)
        z_T = -math.log(p_T) / T
        zero_rates.append(z_T)
        discount_factors[T] = p_T

    return YieldCurve(tenors, zero_rates)

# ---------------------------------------------------------------------------
# 2. Black's Model for Caplets / Floorlets
# ---------------------------------------------------------------------------
def black_caplet(F: float, K: float, T_fix: float, T_pay: float,
                  sigma: float, discount: float, notional: float = 1.0) -> float:
    """
    Black's model caplet: pays max(F - K, 0) * tau at T_pay.
    F = forward rate for [T_fix, T_pay], tau = T_pay - T_fix.
    """
    tau = T_pay - T_fix
    if tau <= 0 or sigma <= 0:
        return notional * tau * discount * max(F - K, 0.0)
    sqT = math.sqrt(T_fix)
    d1 = (math.log(F / K) + 0.5 * sigma**2 * T_fix) / (sigma * sqT)
    d2 = d1 - sigma * sqT
    return notional * tau * discount * (F * _norm_cdf(d1) - K * _norm_cdf(d2))

def black_floorlet(F: float, K: float, T_fix: float, T_pay: float,
                    sigma: float, discount: float, notional: float = 1.0) -> float:
    """Black's model floorlet."""
    tau = T_pay - T_fix
    if tau <= 0 or sigma <= 0:
        return notional * tau * discount * max(K - F, 0.0)
    sqT = math.sqrt(T_fix)
    d1 = (math.log(F / K) + 0.5 * sigma**2 * T_fix) / (sigma * sqT)
    d2 = d1 - sigma * sqT
    return notional * tau * discount * (K * _norm_cdf(-d2) - F * _norm_cdf(-d1))

def price_cap(curve: YieldCurve, K: float, tenor: float, sigma: float,
               notional: float = 1e6, freq: float = 0.25) -> dict:
    """
    Cap = sum of caplets at quarterly reset dates.
    First caplet starts at t=freq (spot rate already set).
    """
    reset_dates = [i * freq for i in range(1, int(tenor / freq) + 1)]
    total_cap = 0.0
    total_floor = 0.0
    caplets = []

    for i, T_pay in enumerate(reset_dates):
        T_fix = T_pay - freq
        if T_fix < 0:
            continue
        F = curve.forward_rate(T_fix, T_pay)
        disc = curve.discount(T_pay)
        cl = black_caplet(F, K, max(T_fix, 1e-6), T_pay, sigma, disc, notional)
        fl = black_floorlet(F, K, max(T_fix, 1e-6), T_pay, sigma, disc, notional)
        total_cap += cl
        total_floor += fl
        caplets.append({'T_fix': T_fix, 'T_pay': T_pay, 'F': F, 'caplet': cl})

    return {'cap': total_cap, 'floor': total_floor, 'caplets': caplets}

# ---------------------------------------------------------------------------
# 3. Black's Model for Swaptions (European)
# ---------------------------------------------------------------------------
def annuity(curve: YieldCurve, T_start: float, T_end: float,
             freq: float = 0.5) -> float:
    """PV01 (annuity): A = sum P(0, T_i) * delta for swap payment dates."""
    pay_dates = []
    t = T_start + freq
    while t <= T_end + 1e-8:
        pay_dates.append(t)
        t += freq
    return sum(curve.discount(t) * freq for t in pay_dates)

def black_payer_swaption(curve: YieldCurve, K: float,
                          T_expiry: float, T_maturity: float,
                          sigma_swap: float, notional: float = 1e6,
                          freq: float = 0.5) -> dict:
    """
    Payer swaption (right to enter fixed-pay swap).
    S = par swap rate for [T_expiry, T_maturity].
    A = annuity (PV01).
    Value = N * A * (S*N(d1) - K*N(d2)).
    """
    A = annuity(curve, T_expiry, T_maturity, freq)
    if A < 1e-10:
        return {'payer': 0.0, 'receiver': 0.0, 'swap_rate': 0.0, 'annuity': 0.0}

    # Par swap rate: S = [P(T_expiry) - P(T_maturity)] / A  (in forward measure)
    p_start = curve.discount(T_expiry)
    p_end   = curve.discount(T_maturity)
    S = (p_start - p_end) / A

    sqT = math.sqrt(T_expiry)
    if sqT < 1e-8 or sigma_swap <= 0:
        payer = notional * A * max(S - K, 0.0)
        receiver = notional * A * max(K - S, 0.0)
        return {'payer': payer, 'receiver': receiver, 'swap_rate': S, 'annuity': A}

    d1 = (math.log(S / K) + 0.5 * sigma_swap**2 * T_expiry) / (sigma_swap * sqT)
    d2 = d1 - sigma_swap * sqT

    payer    = notional * A * (S * _norm_cdf(d1) - K * _norm_cdf(d2))
    receiver = notional * A * (K * _norm_cdf(-d2) - S * _norm_cdf(-d1))

    return {'payer': payer, 'receiver': receiver, 'swap_rate': S, 'annuity': A,
            'd1': d1, 'd2': d2}

# ---------------------------------------------------------------------------
# 4. Fixed income: duration, DV01, convexity
# ---------------------------------------------------------------------------
@dataclass
class Bond:
    face: float
    coupon_rate: float   # annual coupon rate
    maturity: float      # years
    freq: int = 2        # coupons per year

    def cash_flows(self) -> list[tuple[float, float]]:
        """Returns [(time, cash_flow)]."""
        n = int(self.maturity * self.freq)
        c = self.face * self.coupon_rate / self.freq
        cfs = [(i / self.freq, c) for i in range(1, n + 1)]
        cfs[-1] = (cfs[-1][0], cfs[-1][1] + self.face)
        return cfs

    def price(self, yield_: float) -> float:
        """Price given yield (semi-annual convention)."""
        y = yield_ / self.freq
        return sum(cf / (1 + y)**int(t * self.freq) for t, cf in self.cash_flows())

    def duration(self, yield_: float) -> float:
        """Macaulay duration (in years)."""
        p = self.price(yield_)
        y = yield_ / self.freq
        wtd = sum(t * cf / (1 + y)**int(t * self.freq) for t, cf in self.cash_flows())
        return wtd / p

    def modified_duration(self, yield_: float) -> float:
        return self.duration(yield_) / (1 + yield_ / self.freq)

    def dv01(self, yield_: float) -> float:
        """Dollar value of 1 bp (DV01 = -dP/dy * 0.0001)."""
        return (self.price(yield_ + 0.0001) - self.price(yield_ - 0.0001)) / 2

    def convexity(self, yield_: float) -> float:
        """Convexity = (d²P/dy²) / P."""
        p = self.price(yield_)
        y = yield_ / self.freq
        n = self.freq
        conv = sum(t * (t + 1 / n) * cf / (1 + y)**int(t * n)
                   for t, cf in self.cash_flows())
        return conv / (p * (1 + y)**2)

# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 65)
    print("DAY 29: Interest Rate Derivatives")
    print("=" * 65)

    # --- 1. Yield Curve Bootstrap ---
    print("\n1. Yield Curve Bootstrapping from Par Swap Rates")
    par_rates = [(1, 0.0430), (2, 0.0460), (3, 0.0475),
                 (5, 0.0490), (7, 0.0500), (10, 0.0510)]
    curve = bootstrap_yield_curve(par_rates)
    print(f"   {'Tenor':>6} | {'Par Rate':>10} | {'Zero Rate':>10} | {'Disc Factor':>12}")
    print("   " + "-" * 44)
    for T, par in par_rates:
        z = curve.zero_rate(T)
        d = curve.discount(T)
        print(f"   {T:>6} | {par:>10.4f} | {z:>10.4f} | {d:>12.6f}")

    print("\n   Forward Rates:")
    fwd_pairs = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 7), (7, 10)]
    print(f"   {'Period':>12} | {'Forward Rate':>14}")
    print("   " + "-" * 30)
    for T1, T2 in fwd_pairs:
        f = curve.forward_rate(T1, T2)
        print(f"   {T1:.0f}Y-{T2:.0f}Y       | {f:>14.4f}")

    # --- 2. Cap / Floor Pricing ---
    print("\n2. Interest Rate Cap Pricing (Black's Model)")
    K_cap = 0.048
    sigma_cap = 0.25  # 25% lognormal vol (normal market)
    cap_result = price_cap(curve, K_cap, tenor=5.0, sigma=sigma_cap, notional=1e6)
    print(f"   Strike = {K_cap:.3%}  Vol = {sigma_cap:.1%}  Notional = $1M  Tenor = 5Y")
    print(f"   {'Reset':>6} | {'Pay':>6} | {'Fwd Rate':>10} | {'Caplet $':>12}")
    print("   " + "-" * 42)
    for cl in cap_result['caplets']:
        print(f"   {cl['T_fix']:.2f}Y→{cl['T_pay']:.2f}Y | {cl['F']:>10.4f} | {cl['caplet']:>12.2f}")
    print(f"\n   Total Cap Value : ${cap_result['cap']:>12,.2f}")
    print(f"   Total Floor Value: ${cap_result['floor']:>12,.2f}")
    print(f"   Cap-Floor Parity: Cap - Floor = {cap_result['cap'] - cap_result['floor']:,.2f}")
    # Theoretical: = PV(floating) - PV(fixed at K)
    float_pv = 1e6 * (curve.discount(0.25) - curve.discount(5.0))
    fixed_pv = 1e6 * K_cap * sum(curve.discount(i * 0.25) * 0.25 for i in range(1, 21))
    print(f"   Swap value check: {float_pv - fixed_pv:,.2f}")

    # Strike sensitivity
    print(f"\n   Strike Sensitivity (5Y Cap, σ=25%):")
    print(f"   {'Strike':>8} | {'Cap ($)':>12} | {'Floor ($)':>12} | {'Net Swap':>12}")
    print("   " + "-" * 50)
    for K in [0.040, 0.044, 0.048, 0.052, 0.056]:
        r = price_cap(curve, K, 5.0, sigma_cap, 1e6)
        swap_val = r['cap'] - r['floor']
        print(f"   {K:>8.3%} | {r['cap']:>12,.2f} | {r['floor']:>12,.2f} | {swap_val:>12,.2f}")

    # --- 3. Swaption Pricing ---
    print("\n3. Swaption Pricing (Black's Model)")
    sigma_swap = 0.18  # 18% swaption vol
    swaptions = [
        (1, 5),   # 1Y into 5Y
        (2, 5),   # 2Y into 5Y
        (1, 10),  # 1Y into 10Y
        (5, 5),   # 5Y into 5Y
    ]
    print(f"   σ_swap = {sigma_swap:.1%}  Notional = $10M")
    print(f"   {'Expiry':>8} | {'Tenor':>6} | {'Swap Rate':>10} | {'Strike':>8} | {'Payer ($)':>12} | {'Receiver ($)':>13}")
    print("   " + "-" * 67)
    for T_exp, sw_tenor in swaptions:
        T_mat = T_exp + sw_tenor
        K_sw = curve.forward_rate(T_exp, T_mat) * 0.5 + curve.zero_rate(T_mat) * 0.5
        K_sw = round(K_sw, 3)
        res = black_payer_swaption(curve, K_sw, T_exp, T_mat, sigma_swap, 10e6)
        print(f"   {T_exp}Y exp  | {sw_tenor}Y sw | {res['swap_rate']:>10.4f} | "
              f"{K_sw:>8.4f} | {res['payer']:>12,.2f} | {res['receiver']:>13,.2f}")

    # --- 4. Bond Analytics ---
    print("\n4. Bond Duration, DV01, Convexity")
    bonds = [
        Bond(face=1000, coupon_rate=0.04, maturity=2, freq=2),
        Bond(face=1000, coupon_rate=0.05, maturity=5, freq=2),
        Bond(face=1000, coupon_rate=0.05, maturity=10, freq=2),
        Bond(face=1000, coupon_rate=0.00, maturity=10, freq=2),  # zero coupon
    ]
    bond_names = ['2Y 4% Bond', '5Y 5% Bond', '10Y 5% Bond', '10Y Zero']
    yield_ = 0.05

    print(f"   Yield = {yield_:.2%}")
    print(f"   {'Bond':15} | {'Price':>8} | {'Dur (yr)':>9} | {'Mod Dur':>8} | {'DV01':>8} | {'Convex':>8}")
    print("   " + "-" * 67)
    for name, bond in zip(bond_names, bonds):
        p  = bond.price(yield_)
        d  = bond.duration(yield_)
        md = bond.modified_duration(yield_)
        dv01 = bond.dv01(yield_)
        conv = bond.convexity(yield_)
        print(f"   {name:15} | {p:>8.3f} | {d:>9.4f} | {md:>8.4f} | {dv01:>8.4f} | {conv:>8.4f}")

    print("\n   Price Change Estimation (10Y 5% Bond, yield +100bp):")
    bond10 = bonds[2]
    p0 = bond10.price(yield_)
    dy = 0.01
    dur_est = -bond10.modified_duration(yield_) * dy * p0
    conv_adj = 0.5 * bond10.convexity(yield_) * dy**2 * p0
    actual = bond10.price(yield_ + dy) - p0
    print(f"   Duration estimate    : {dur_est:+.4f}")
    print(f"   Convexity adjustment : {conv_adj:+.4f}")
    print(f"   Total estimate       : {dur_est + conv_adj:+.4f}")
    print(f"   Actual price change  : {actual:+.4f}")

    print("\n[Done] Day 29: Interest Rate Derivatives complete.")
