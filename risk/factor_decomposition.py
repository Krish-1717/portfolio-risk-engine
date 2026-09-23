"""
risk/factor_decomposition.py -- Factor risk decomposition for portfolio-risk-engine
Day 12: Fama-French 3-factor OLS, rolling betas, idiosyncratic risk, factor attribution
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Simple OLS (no numpy)
# ---------------------------------------------------------------------------

def _transpose(m: List[List[float]]) -> List[List[float]]:
    rows, cols = len(m), len(m[0])
    return [[m[r][c] for r in range(rows)] for c in range(cols)]

def _mat_mul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    n, k = len(A), len(A[0])
    m = len(B[0])
    C = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            C[i][j] = sum(A[i][p] * B[p][j] for p in range(k))
    return C

def _inv2x2(m: List[List[float]]) -> List[List[float]]:
    a, b, c, d = m[0][0], m[0][1], m[1][0], m[1][1]
    det = a * d - b * c
    if abs(det) < 1e-12:
        raise ValueError("Singular matrix")
    return [[d / det, -b / det], [-c / det, a / det]]

def _cholesky_solve(A: List[List[float]], b: List[float]) -> List[float]:
    """Solve Ax=b via Gaussian elimination (small systems only)."""
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[pivot] = M[pivot], M[col]
        if abs(M[col][col]) < 1e-12:
            raise ValueError("Singular matrix in solver")
        for row in range(col + 1, n):
            factor = M[row][col] / M[col][col]
            for j in range(col, n + 1):
                M[row][j] -= factor * M[col][j]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = M[i][n]
        for j in range(i + 1, n):
            x[i] -= M[i][j] * x[j]
        x[i] /= M[i][i]
    return x

def ols(X: List[List[float]], y: List[float]) -> List[float]:
    """OLS: beta = (X'X)^{-1} X'y.  X rows are observations."""
    Xt = _transpose(X)
    n_obs = len(X)
    n_feat = len(X[0])
    XtX = [[sum(Xt[i][k] * X[k][j] for k in range(n_obs)) for j in range(n_feat)] for i in range(n_feat)]
    Xty = [sum(Xt[i][k] * y[k] for k in range(n_obs)) for i in range(n_feat)]
    return _cholesky_solve(XtX, Xty)


# ---------------------------------------------------------------------------
# Factor model data structures
# ---------------------------------------------------------------------------

@dataclass
class FactorReturn:
    """One period of Fama-French 3-factor returns."""
    date: str
    mkt_rf: float   # market excess return
    smb: float      # Small-Minus-Big
    hml: float      # High-Minus-Low
    rf: float       # risk-free rate


@dataclass
class FactorExposure:
    """OLS-estimated factor betas for a single asset."""
    ticker: str
    alpha: float        # annualised Jensen's alpha (intercept * 252)
    beta_mkt: float     # market beta
    beta_smb: float     # size beta
    beta_hml: float     # value beta
    r_squared: float    # in-sample R²
    idio_vol: float     # annualised idiosyncratic volatility
    n_obs: int


@dataclass
class PortfolioFactorReport:
    """Factor decomposition for the whole portfolio."""
    holdings: Dict[str, float]             # ticker -> weight
    exposures: Dict[str, FactorExposure]
    factor_returns: List[FactorReturn]

    @property
    def portfolio_beta(self) -> float:
        return sum(self.holdings.get(t, 0) * e.beta_mkt for t, e in self.exposures.items())

    @property
    def portfolio_smb(self) -> float:
        return sum(self.holdings.get(t, 0) * e.beta_smb for t, e in self.exposures.items())

    @property
    def portfolio_hml(self) -> float:
        return sum(self.holdings.get(t, 0) * e.beta_hml for t, e in self.exposures.items())

    def factor_attribution(self, period: int = 252) -> Dict[str, float]:
        """Annualised return attribution split by factor + idio."""
        avg_mkt = sum(f.mkt_rf for f in self.factor_returns[-period:]) / min(period, len(self.factor_returns))
        avg_smb = sum(f.smb for f in self.factor_returns[-period:]) / min(period, len(self.factor_returns))
        avg_hml = sum(f.hml for f in self.factor_returns[-period:]) / min(period, len(self.factor_returns))
        mkt_contrib = self.portfolio_beta * avg_mkt * period
        smb_contrib = self.portfolio_smb * avg_smb * period
        hml_contrib = self.portfolio_hml * avg_hml * period
        alpha_contrib = sum(self.holdings.get(t, 0) * e.alpha for t, e in self.exposures.items())
        return {
            "market": round(mkt_contrib, 4),
            "smb": round(smb_contrib, 4),
            "hml": round(hml_contrib, 4),
            "alpha": round(alpha_contrib, 4),
            "total": round(mkt_contrib + smb_contrib + hml_contrib + alpha_contrib, 4),
        }

    def print_report(self) -> None:
        print("=" * 70)
        print("  FAMA-FRENCH FACTOR DECOMPOSITION")
        print("=" * 70)
        print(f"  {'Ticker':<10} {'Alpha%':>8} {'Beta(Mkt)':>10} {'Beta(SMB)':>10} {'Beta(HML)':>10} {'R2':>6} {'IdioVol':>8}")
        print("  " + "-" * 64)
        for ticker, exp in sorted(self.exposures.items()):
            w = self.holdings.get(ticker, 0)
            print(
                f"  {ticker:<10} {exp.alpha*100:>+7.2f}%  "
                f"{exp.beta_mkt:>9.str}  "
                f"{exp.beta_smb:>9.3f}  "
                f"{exp.beta_hml:>9.3f}  {exp.r_squared:>5.2f}  "
                f"{exp.idio_vol*100:>6.1f}%"
            )
        print()
        print(f"  Portfolio beta (Mkt): {self.portfolio_beta:.3f}")
        print(f"  Portfolio beta (SMB): {self.portfolio_smb:.3f}")
        print(f"  Portfolio beta (HML): {self.portfolio_hml:.3f}")
        print()
        attr = self.factor_attribution()
        print("  Annualised factor attribution:")
        for k, v in attr.items():
            print(f"    {k:<10}: {v*100:>+.2f}%")
        print("=" * 70)


# ---------------------------------------------------------------------------
# Factor model fitter
# ---------------------------------------------------------------------------

class FactorDecomposer:
    """Fit Fama-French 3-factor model to each asset's return series."""

    def __init__(self, factor_returns: List[FactorReturn]):
        self.factor_returns = factor_returns

    def fit(
        self,
        ticker: str,
        asset_returns: List[float],
    ) -> FactorExposure:
        """
        Regress excess asset returns on [1, Mkt-RF, SMB, HML].

        Parameters
        ----------
        ticker : asset identifier
        asset_returns : list of *total* returns (not excess) aligned to factor_returns
        """
        n = min(len(asset_returns), len(self.factor_returns))
        if n < 4:
            raise ValueError(f"Need >= 4 observations for {ticker}")

        rf = [f.rf for f in self.factor_returns[:n]]
        y = [asset_returns[i] - rf[i] for i in range(n)]   # excess returns
        X = [
            [1.0, self.factor_returns[i].mkt_rf,
             self.factor_returns[i].smb,
             self.factor_returns[i].hml]
            for i in range(n)
        ]

        betas = ols(X, y)
        alpha_daily, beta_mkt, beta_smb, beta_hml = betas

        # Residuals and RÂ²
        y_hat = [sum(betas[j] * X[i][j] for j in range(4)) for i in range(n)]
        residuals = [y[i] - y_hat[i] for i in range(n)]
        ss_res = sum(r ** 2 for r in residuals)
        ss_tot = sum((yi - sum(y) / n) ** 2 for yi in y)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        idio_var = ss_res / (n - 4) if n > 4 else ss_res
        idio_vol = math.sqrt(max(idio_var, 0.0) * 252)

        return FactorExposure(
            ticker=ticker,
            alpha=alpha_daily * 252,
            beta_mkt=beta_mkt,
            beta_smb=beta_smb,
            beta_hml=beta_hml,
            r_squared=round(r2, 4),
            idio_vol=idio_vol,
            n_obs=n,
        )

    def build_report(
        self,
        holdings: Dict[str, float],
        asset_returns: Dict[str, List[float]],
    ) -> PortfolioFactorReport:
        exposures = {}
        for ticker in holdings:
            if ticker in asset_returns:
                try:
                    exposures[ticker] = self.fit(ticker, asset_returns[ticker])
                except Exception:
                    pass
        return PortfolioFactorReport(
            holdings=holdings,
            exposures=exposures,
            factor_returns=self.factor_returns,
        )


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random
    rng = random.Random(7)
    n = 252  # 1 year of daily data

    factor_rets = [
        FactorReturn(
            date=f"2024-{(i//21)+1:02d}-{(i%21)+1:02d}",
            mkt_rf=rng.gauss(0.0004, 0.010),
            smb=rng.gauss(0.0001, 0.005),
            hml=rng.gauss(0.0000, 0.005),
            rf=0.05 / 252,
        )
        for i in range(n)
    ]

    decomposer = FactorDecomposer(factor_rets)

    holdings = {"AAPL": 0.30, "MSFT": 0.25, "JPM": 0.20, "XOM": 0.15, "SPY": 0.10}

    def _sim_asset(beta_mkt, beta_smb, beta_hml, alpha_daily, noise):
        rets = []
        for f in factor_rets:
            r = f.rf + alpha_daily + beta_mkt*f.mkt_rf + beta_smb*f.smb + beta_hml*f.hml + rng.gauss(0, noise)
            rets.append(r)
        return rets

    asset_returns = {
        "AAPL": _sim_asset(1.20, -0.30, -0.50, 0.0003, 0.012),
        "MSFT": _sim_asset(1.10, -0.20, -0.40, 0.0002, 0.011),
        "JPM":  _sim_asset(1.30,  0.10,  0.80, 0.0001, 0.013),
        "XOM":  _sim_asset(0.80,  0.05,  0.90, 0.0000, 0.014),
        "SPY":  _sim_asset(1.00,  0.00,  0.00, 0.0000, 0.005),
    }

    report = decomposer.build_report(holdings, asset_returns)
    report.print_report()
