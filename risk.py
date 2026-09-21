"""
src/risk.py -- Portfolio VaR, CVaR, Monte Carlo and factor risk engine
portfolio-risk-engine Day 1 Commit 1
"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Position:
    ticker: str
    quantity: float
    price: float

    @property
    def market_value(self) -> float:
        return self.quantity * self.price


@dataclass
class RiskResult:
    portfolio_value: float
    var_95: float       # 1-day VaR at 95% confidence (loss amount)
    var_99: float
    cvar_95: float      # Expected Shortfall at 95%
    cvar_99: float
    volatility_daily: float
    sharpe_annualised: float
    max_drawdown: float
    positions: List[str] = field(default_factory=list)

    def summary(self) -> str:
        return (f"Portfolio ${self.portfolio_value:,.0f} | "
                f"VaR(95%)=${self.var_95:,.0f} | "
                f"CVaR(95%)=${self.cvar_95:,.0f} | "
                f"Vol(ann.)={self.volatility_daily * math.sqrt(252):.1%}")


class PortfolioRiskEngine:
    """Historical simulation + parametric + Monte Carlo VaR/CVaR."""

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    def _norm_inv(self, p: float) -> float:
        """Rational approximation for the standard normal quantile (Beasley-Springer-Moro)."""
        a = [0, -3.969683028665376e+01, 2.209460984245205e+02,
             -2.759285104469687e+02, 1.383577518672690e+02,
             -3.066479806614716e+01, 2.506628277459239e+00]
        b = [0, -5.447609879822406e+01, 1.615858368580409e+02,
             -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01]
        c = [0, -7.784894002430293e-03, -3.223964580411365e-01,
             -2.400758277161838e+00, -2.549732539343734e+00,
             4.374664141464968e+00, 2.938163982698783e+00]
        d = [0, 7.784695709041462e-03, 3.224671290700398e-01,
             2.445134137142996e+00, 3.754408661907416e+00]
        plow, phigh = 0.02425, 1 - 0.02425
        if p < plow:
            q = math.sqrt(-2 * math.log(p))
            return (((((c[1]*q+c[2])*q+c[3])*q+c[4])*q+c[5])*q+c[6]) / ((((d[1]*q+d[2])*q+d[3])*q+d[4])*q+1)
        elif p <= phigh:
            q = p - 0.5; r = q*q
            return (((((a[1]*r+a[2])*r+a[3])*r+a[4])*r+a[5])*r+a[6])*q / (((((b[1]*r+b[2])*r+b[3])*r+b[4])*r+b[5])*r+1)
        else:
            q = math.sqrt(-2 * math.log(1-p))
            return -(((((c[1]*q+c[2])*q+c[3])*q+c[4])*q+c[5])*q+c[6]) / ((((d[1]*q+d[2])*q+d[3])*q+d[4])*q+1)

    def monte_carlo_var(self, positions: List[Position],
                        daily_vols: Dict[str, float],
                        correlations: Optional[Dict[Tuple[str,str], float]] = None,
                        n_sims: int = 10000, horizon_days: int = 1) -> RiskResult:
        """Parametric Monte Carlo VaR/CVaR via Cholesky decomposition (diagonal if no corr)."""
        tickers = [p.ticker for p in positions]
        weights = [p.market_value for p in positions]
        port_val = sum(weights)
        vols = [daily_vols.get(t, 0.02) * math.sqrt(horizon_days) for t in tickers]
        n = len(positions)

        # Build simple correlation matrix (identity if not provided)
        corr = [[1.0 if i == j else (correlations or {}).get((tickers[i], tickers[j]), 0.0)
                 for j in range(n)] for i in range(n)]

        # Cholesky (simple diagonal fallback for robustness)
        L = [[0.0]*n for _ in range(n)]
        for i in range(n):
            s = sum(L[i][k]**2 for k in range(i))
            L[i][i] = math.sqrt(max(corr[i][i] - s, 1e-12))
            for j in range(i+1, n):
                s2 = sum(L[i][k]*L[j][k] for k in range(i))
                L[j][i] = (corr[j][i] - s2) / L[i][i] if L[i][i] > 1e-12 else 0.0

        # Simulate PnL
        pnl = []
        for _ in range(n_sims):
            z = [self._rng.gauss(0, 1) for _ in range(n)]
            corr_z = [sum(L[i][k] * z[k] for k in range(i+1)) for i in range(n)]
            pnl_sim = sum(weights[i] * corr_z[i] * vols[i] for i in range(n))
            pnl.append(pnl_sim)

        pnl_sorted = sorted(pnl)
        idx_95 = int(0.05 * n_sims)
        idx_99 = int(0.01 * n_sims)
        var_95 = -pnl_sorted[idx_95]
        var_99 = -pnl_sorted[idx_99]
        cvar_95 = -sum(pnl_sorted[:idx_95]) / max(idx_95, 1)
        cvar_99 = -sum(pnl_sorted[:idx_99]) / max(idx_99, 1)

        # Portfolio daily vol
        port_vol = math.sqrt(sum(pnl_sim**2 for pnl_sim in pnl) / n_sims) / port_val
        # Simple Sharpe (assume 0 mean for 1-day)
        sharpe = 0.0

        # Max drawdown from simulated cumulative path
        cum = [0.0]
        for p in pnl:
            cum.append(cum[-1] + p)
        peak = cum[0]
        max_dd = 0.0
        for v in cum:
            if v > peak:
                peak = v
            dd = (peak - v) / port_val if port_val > 0 else 0
            if dd > max_dd:
                max_dd = dd

        return RiskResult(
            portfolio_value=port_val, var_95=var_95, var_99=var_99,
            cvar_95=cvar_95, cvar_99=cvar_99, volatility_daily=port_vol,
            sharpe_annualised=sharpe, max_drawdown=max_dd,
            positions=tickers,
        )
