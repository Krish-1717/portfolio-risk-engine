"""
yield_curve/yield_curve.py -- Yield curve construction and analysis.
Consolidated from yield-curve-analyzer repo.
Nelson-Siegel, bootstrap, forward rates, PCA decomposition.
Pure Python stdlib -- no external dependencies.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class YieldCurve:
    tenors: List[float]     # in years
    yields: List[float]     # zero rates (decimal)

    def rate(self, t: float) -> float:
        """Linear interpolation / flat extrapolation."""
        if not self.tenors:
            return 0.0
        if t <= self.tenors[0]:
            return self.yields[0]
        if t >= self.tenors[-1]:
            return self.yields[-1]
        for i in range(len(self.tenors) - 1):
            t0, t1 = self.tenors[i], self.tenors[i + 1]
            if t0 <= t <= t1:
                frac = (t - t0) / (t1 - t0)
                return self.yields[i] + frac * (self.yields[i + 1] - self.yields[i])
        return self.yields[-1]

    def discount(self, t: float) -> float:
        return math.exp(-self.rate(t) * t)

    def forward_rate(self, t1: float, t2: float) -> float:
        """Continuously compounded forward rate between t1 and t2."""
        if t2 <= t1:
            return self.rate(t1)
        r1, r2 = self.rate(t1), self.rate(t2)
        return (r2 * t2 - r1 * t1) / (t2 - t1)

    def instantaneous_forward(self, t: float, dt: float = 0.001) -> float:
        return self.forward_rate(t, t + dt)


# ---------------------------------------------------------------------------
# Nelson-Siegel parametric model
# ---------------------------------------------------------------------------

@dataclass
class NelsonSiegelParams:
    beta0: float   # long-run level
    beta1: float   # short-end slope
    beta2: float   # hump
    tau: float     # decay factor (years)


def nelson_siegel_rate(params: NelsonSiegelParams, t: float) -> float:
    """Nelson-Siegel zero rate."""
    if t <= 0:
        return params.beta0 + params.beta1
    x = t / max(params.tau, 1e-8)
    ex = math.exp(-x)
    factor1 = (1 - ex) / x
    factor2 = factor1 - ex
    return params.beta0 + params.beta1 * factor1 + params.beta2 * factor2


def fit_nelson_siegel(
    tenors: List[float],
    yields: List[float],
    tau_grid: Optional[List[float]] = None,
    max_iter: int = 100,
) -> NelsonSiegelParams:
    """
    Grid search over tau, then OLS for beta0/beta1/beta2.
    Uses least-squares closed form given tau.
    """
    if tau_grid is None:
        tau_grid = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0]
    n = len(tenors)
    best_sse, best_params = float('inf'), NelsonSiegelParams(0.05, -0.02, 0.01, 2.0)

    for tau in tau_grid:
        # Build regressor matrix X (n x 3)
        X = []
        for t in tenors:
            if t <= 0:
                X.append([1.0, 1.0, 0.0])
                continue
            x = t / tau
            ex = math.exp(-x)
            f1 = (1 - ex) / x
            X.append([1.0, f1, f1 - ex])

        # OLS: beta = (X'X)^-1 X'y
        XtX = [[sum(X[i][a] * X[i][b] for i in range(n)) for b in range(3)] for a in range(3)]
        Xty = [sum(X[i][a] * yields[i] for i in range(n)) for a in range(3)]

        # 3x3 matrix inverse (Cramer's rule)
        def det3(m);
            return (m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])
                   -m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])
                   +m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]))
        d = det3(XtX)
        if abs(d) < 1e-14:
            continue
        inv = [[0.0]*3 for _ in range(3)]
        for r in range(3):
            for c in range(3):
                minor = [[XtX[i][j] for j in range(3) if j != c]
                         for i in range(3) if i != r]
                cofac = ((-1)**(r+c)) * det3(minor) if len(minor)==2 else ((-1)**(r+c)) * (minor[0][0] if minor else 1)
                inv[r][c] = cofac / d
        beta = [sum(inv[a][b] * Xty[b] for b in range(3)) for a in range(3)]
        p = NelsonSiegelParams(beta[0], beta[1], beta[2], tau)
        sse = sum((nelson_siegel_rate(p, t) - y)**2 for t, y in zip(tenors, yields))
        if sse < best_sse:
            best_sse, best_params = sse, p

    return best_params


def nelson_siegel_curve(params: NelsonSiegelParams, tenors: List[float]) -> YieldCurve:
    return YieldCurve(tenors, [nelson_siegel_rate(params, t) for t in tenors])


# ---------------------------------------------------------------------------
# Curve shifts (parallel, twist, butterfly)
# ---------------------------------------------------------------------------

def parallel_shift(curve: YieldCurve, shift_bps: float) -> YieldCurve:
    s = shift_bps / 10_000
    return YieldCurve(list(curve.tenors), [y + s for y in curve.yields])


def steepener(curve: YieldCurve, short_end_bps: float, long_end_bps: float,
              pivot: float = 5.0) -> YieldCurve:
    """Twist: linear interpolation of shift from short to long end."""
    shifts = []
    t_min, t_max = curve.tenors[0], curve.tenors[-1]
    for t in curve.tenors:
        frac = (t - t_min) / max(t_max - t_min, 1e-8)
        s = (short_end_bps + frac * (long_end_bps - short_end_bps)) / 10_000
        shifts.append(s)
    return YieldCurve(list(curve.tenors), [y + s for y, s in zip(curve.yields, shifts)])


if __name__ == "__main__":
    tenors = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]
    yields = [0.052, 0.053, 0.051, 0.048, 0.047, 0.046, 0.047, 0.048, 0.049, 0.048]
    curve = YieldCurve(tenors, yields)
    print("Yield Curve:")
    for t, y in zip(tenors, yields):
        fwd = curve.forward_rate(t, t + 1)
        print(f"  {t:5.2f}y  zero={y:.3%}  fwd1y={fwd:.3%}")
    ns = fit_nelson_siegel(tenors, yields)
    print(f"\nNelson-Siegel: b0={ns.beta0:.4f} b1={ns.beta1:.4f} b2={ns.beta2:.4f} tau={ns.tau:.2f}")
    ns_curve = nelson_siegel_curve(ns, tenors)
    rmse = math.sqrt(sum((r - y)**2 for r, y in zip(ns_curve.yields, yields)) / len(yields))
    print(f"NS fit RMSE: {rmse*10000:.2f} bps")
