"""
fixed_income/bond_analytics.py -- Fixed-income analytics library.
Consolidated from fixed-income-analytics repo.
Bond pricing, duration, convexity, DV01, yield-to-maturity, OAS.
Pure Python stdlib -- no external dependencies.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class Bond:
    face: float          # par value
    coupon_rate: float   # annual coupon rate (decimal)
    maturity: float      # years to maturity
    freq: int = 2        # coupon payments per year (2 = semi-annual)
    price: Optional[float] = None


def _coupon(bond: Bond) -> float:
    return bond.face * bond.coupon_rate / bond.freq


def _cash_flows(bond: Bond) -> List[Tuple[float, float]]:
    """Return list of (time_in_years, cash_flow)."""
    n = int(round(bond.maturity * bond.freq))
    c = _coupon(bond)
    flows = [(i / bond.freq, c) for i in range(1, n + 1)]
    flows[-1] = (flows[-1][0], flows[-1][1] + bond.face)
    return flows


def bond_price(bond: Bond, ytm: float) -> float:
    """Compute dirty price given yield-to-maturity."""
    r = ytm / bond.freq
    return sum(cf / (1 + r) ** (t * bond.freq) for t, cf in _cash_flows(bond))


def yield_to_maturity(bond: Bond, price: float, tol: float = 1e-8, max_iter: int = 100) -> float:
    """Newton-Raphson YTM solver."""
    ytm = bond.coupon_rate  # initial guess
    for _ in range(max_iter):
        p = bond_price(bond, ytm)
        dp = modified_duration(bond, ytm) * p / 100  # approx deriv
        err = p - price
        if abs(err) < tol:
            break
        ytm += err / max(dp * 100, 1e-10)
    return ytm


def macaulay_duration(bond: Bond, ytm: float) -> float:
    """Macaulay duration in years."""
    r = ytm / bond.freq
    price = bond_price(bond, ytm)
    return sum(t * cf / (1 + r) ** (t * bond.freq) for t, cf in _cash_flows(bond)) / max(price, 1e-10)


def modified_duration(bond: Bond, ytm: float) -> float:
    """Modified duration = MacD / (1 + ytm/freq)."""
    macd = macaulay_duration(bond, ytm)
    return macd / (1 + ytm / bond.freq)


def convexity(bond: Bond, ytm: float) -> float:
    """Dollar convexity (second derivative of price w.r.t. yield, scaled)."""
    r = ytm / bond.freq
    price = bond_price(bond, ytm)
    n = bond.freq
    conv = sum(
        t * (t + 1 / n) * cf / (1 + r) ** (t * n + 2)
        for t, cf in _cash_flows(bond)
    )
    return conv / max(price, 1e-10)


def dv01(bond: Bond, ytm: float) -> float:
    """Dollar value of 1 basis point (DV01 = ModDur * Price / 10000)."""
    return modified_duration(bond, ytm) * bond_price(bond, ytm) / 10_000


def price_change_approx(bond: Bond, ytm: float, dy: float) -> float:
    """First+second order Taylor approx of price change for yield shift dy."""
    md = modified_duration(bond, ytm)
    cx = convexity(bond, ytm)
    p = bond_price(bond, ytm)
    return p * (-md * dy + 0.5 * cx * dy ** 2)


def spread_to_treasury(bond_ytm: float, treasury_ytm: float) -> float:
    """Simple spread: bond YTM - treasury YTM (in basis points)."""
    return (bond_ytm - treasury_ytm) * 10_000


@dataclass
class BondPortfolioSummary:
    portfolio_dv01: float
    portfolio_duration: float
    portfolio_convexity: float
    total_market_value: float
    weighted_ytm: float


def portfolio_risk(bonds: List[Bond], ytms: List[float], notionals: List[float]) -> BondPortfolioSummary:
    """Aggregate risk measures for a bond portfolio."""
    total_mv = sum(bond_price(b, y) * n / b.face for b, y, n in zip(bonds, ytms, notionals))
    port_dv01 = sum(dv01(b, y) * n / b.face for b, y, n in zip(bonds, ytms, notionals))
    port_dur = sum(modified_duration(b, y) * bond_price(b, y) * n / b.face
                   for b, y, n in zip(bonds, ytms, notionals)) / max(total_mv, 1e-10)
    port_conv = sum(convexity(b, y) * bond_price(b, y) * n / b.face
                    for b, y, n in zip(bonds, ytms, notionals)) / max(total_mv, 1e-10)
    wmv = [bond_price(b, y) * n / b.face for b, y, n in zip(bonds, ytms, notionals)]
    w_ytm = sum(y * mv for y, mv in zip(ytms, wmv)) / max(total_mv, 1e-10)
    return BondPortfolioSummary(port_dv01, port_dur, port_conv, total_mv, w_ytm)


if __name__ == "__main__":
    b = Bond(face=1000, coupon_rate=0.05, maturity=10, freq=2)
    ytm = 0.06
    p = bond_price(b, ytm)
    print(f"Price:      {p:.4f}")
    print(f"MacD:       {macaulay_duration(b, ytm):.4f} yrs")
    print(f"ModD:       {modified_duration(b, ytm):.4f}")
    print(f"Convexity:  {convexity(b, ytm):.4f}")
    print(f"DV01:       {dv01(b, ytm):.4f}")
    ytm_back = yield_to_maturity(b, p)
    print(f"YTM check:  {ytm_back:.6f} (should be {ytm})")
    dp = price_change_approx(b, ytm, 0.01)
    print(f"Price chg (+100bp): {dp:.4f}")
