"""
Robust Mean-Variance Optimization
Day 18 â portfolio-risk-engine/optimization/robust_optimization.py

Implements:
  - Standard Markowitz MVO (baseline)
  - Box uncertainty set: mu_i in [mu_i - delta_i, mu_i + delta_i]
  - Ellipsoidal uncertainty set (Goldfarb-Iyengar 2003)
  - Robust efficient frontier construction
  - Parameter uncertainty propagation via resampling
  - Portfolio diagnostics (diversification ratio, concentration)
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict


# ---------------------------------------------------------------------------
# Linear algebra helpers (pure Python, no numpy)
# ---------------------------------------------------------------------------

def _mat_vec(A: List[List[float]], v: List[float]) -> List[float]:
    """Matrix-vector product A @ v."""
    return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]


def _vec_dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _vec_add(a: List[float], b: List[float]) -> List[float]:
    return [x + y for x, y in zip(a, b)]


def _vec_sub(a: List[float], b: List[float]) -> List[float]:
    return [x - y for x, y in zip(a, b)]


def _vec_scale(v: List[float], s: float) -> List[float]:
    return [x * s for x in v]


def _portfolio_variance(w: List[float], Sigma: List[List[float]]) -> float:
    """w' Sigma w"""
    Sw = _mat_vec(Sigma, w)
    return _vec_dot(w, Sw)


def _portfolio_return(w: List[float], mu: List[float]) -> float:
    return _vec_dot(w, mu)


def _cholesky(A: List[List[float]]) -> List[List[float]]:
    """Cholesky decomposition L such that A = L L'. Returns lower triangular L."""
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                val = A[i][i] - s
                L[i][j] = math.sqrt(max(val, 1e-14))
            else:
                L[i][j] = (A[i][j] - s) / L[j][j] if L[j][j] > 1e-10 else 0.0
    return L


def _solve_lower(L: List[List[float]], b: List[float]) -> List[float]:
    """Forward substitution: solve L x = b."""
    n = len(b)
    x = [0.0] * n
    for i in range(n):
        x[i] = (b[i] - sum(L[i][j] * x[j] for j in range(i))) / (L[i][i] if L[i][i] != 0 else 1e-14)
    return x


def _solve_upper(L: List[List[float]], b: List[float]) -> List[float]:
    """Back substitution: solve L' x = b."""
    n = len(b)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - sum(L[j][i] * x[j] for j in range(i + 1, n))) / (L[i][i] if L[i][i] != 0 else 1e-14)
    return x


def _solve_spd(A: List[List[float]], b: List[float]) -> List[float]:
    """Solve A x = b for symmetric positive definite A via Cholesky."""
    L = _cholesky(A)
    y = _solve_lower(L, b)
    return _solve_upper(L, y)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AssetUniverse:
    n: int
    mu: List[float]          # expected returns
    Sigma: List[List[float]] # covariance matrix


@dataclass
class PortfolioResult:
    weights: List[float]
    expected_return: float
    volatility: float
    sharpe: float
    worst_case_return: Optional[float] = None
    method: str = "standard"


# ---------------------------------------------------------------------------
# Markowitz MVO (baseline)
# ---------------------------------------------------------------------------

def _critical_line_mvo(
        mu: List[float],
        Sigma: List[List[float]],
        target_return: float,
        allow_short: bool = False,
) -> List[float]:
    """
    Solve Markowitz MVO for a target return using Lagrangian KKT conditions.
    If allow_short=False, uses projected gradient to enforce w >= 0.

    Lagrangian: min w'Sw - lambda*(w'mu - r*) - gamma*(w'1 - 1)
    KKT: 2Sw = lambda*mu + gamma*1
         w'mu = r*, w'1 = 1
    """
    n = len(mu)
    ones = [1.0] * n

    if allow_short:
        # Analytical solution via KKT
        # Build system: [2S  -mu  -1 ] [w     ]   [0 ]
        #               [-mu' 0    0  ] [lambda] = [r*]
        #               [-1'  0    0  ] [gamma ]   [1 ]
        # Reduce: solve S^{-1} mu and S^{-1} 1, then use 2x2 system
        Sinv_mu = _solve_spd(Sigma, mu)
        Sinv_1 = _solve_spd(Sigma, ones)
        A_val = _vec_dot(mu, Sinv_mu)
        B_val = _vec_dot(mu, Sinv_1)
        C_val = _vec_dot(ones, Sinv_1)
        denom = A_val * C_val - B_val ** 2
        if abs(denom) < 1e-12:
            return [1.0 / n] * n
        lam = (C_val * target_return - B_val) / denom
        gamma = (A_val - B_val * target_return) / denom
        w = _vec_add(_vec_scale(Sinv_mu, lam), _vec_scale(Sinv_1, gamma))
        return w

    # Long-only: projected gradient descent
    w = [1.0 / n] * n
    lr = 0.1
    for _ in range(2000):
        # Gradient of variance: 2 Sigma w
        grad_var = _vec_scale(_mat_vec(Sigma, w), 2.0)
        # Gradient step + projection
        w_new = []
        ret = _vec_dot(w, mu)
        for i in range(n):
            wi = w[i] - lr * grad_var[i]
            w_new.append(max(wi, 0.0))
        # Normalise to sum=1
        total = sum(w_new)
        if total < 1e-10:
            w_new = [1.0 / n] * n
        else:
            w_new = [x / total for x in w_new]
        # Check return constraint
        ret_new = _vec_dot(w_new, mu)
        # If return is below target, shift toward max-return asset
        if ret_new < target_return - 1e-6:
            max_idx = max(range(n), key=lambda i: mu[i])
            w_new = list(w_new)
            w_new[max_idx] = min(1.0, w_new[max_idx] + lr * 0.5)
            tot = sum(w_new)
            w_new = [x / tot for x in w_new]
        if max(abs(w_new[i] - w[i]) for i in range(n)) < 1e-8:
            w = w_new
            break
        w = w_new
    return w


def markowitz_portfolio(
        universe: AssetUniverse,
        target_return: float,
        risk_free_rate: float = 0.0,
        allow_short: bool = False,
) -> PortfolioResult:
    """Standard Markowitz minimum-variance portfolio for a target return."""
    w = _critical_line_mvo(universe.mu, universe.Sigma, target_return, allow_short)
    ret = _portfolio_return(w, universe.mu)
    vol = math.sqrt(max(_portfolio_variance(w, universe.Sigma), 0.0))
    sharpe = (ret - risk_free_rate) / vol if vol > 1e-10 else 0.0
    return PortfolioResult(weights=w, expected_return=ret, volatility=vol,
                           sharpe=sharpe, method="markowitz")


# ---------------------------------------------------------------------------
# Box uncertainty set robust MVO
# ---------------------------------------------------------------------------

def box_robust_portfolio(
        universe: AssetUniverse,
        delta: List[float],         # uncertainty half-widths per asset
        target_return: float,
        risk_free_rate: float = 0.0,
        allow_short: bool = False,
) -> PortfolioResult:
    """
    Robust MVO with box uncertainty set.
    Worst-case return: mu_w(w) = w'mu - |w|' * delta
    Minimise variance subject to worst-case return >= target.

    The worst-case return of portfolio w under box uncertainty:
      min_{mu in U_box} w'mu = w'mu - sum_i |w_i| * delta_i
    We conservatively replace mu with mu_robust = mu - delta (for long-only)
    and solve standard MVO.
    """
    n = universe.n
    if not allow_short:
        # For long-only portfolios: worst-case = mu - delta
        mu_robust = [universe.mu[i] - delta[i] for i in range(n)]
        # Ensure target_return is achievable under robust mu
        max_robust = max(mu_robust)
        target_eff = min(target_return, max_robust * 0.999)
        w = _critical_line_mvo(mu_robust, universe.Sigma, target_eff, False)
        ret = _portfolio_return(w, universe.mu)   # actual expected return
        # Worst-case return
        wc_ret = _portfolio_return(w, mu_robust)
    else:
        # General: worst-case = w'mu - delta'|w|
        mu_robust = [universe.mu[i] - delta[i] for i in range(n)]
        target_eff = min(target_return, max(mu_robust) * 0.999)
        w = _critical_line_mvo(mu_robust, universe.Sigma, target_eff, True)
        ret = _portfolio_return(w, universe.mu)
        wc_ret = sum(w[i] * universe.mu[i] - abs(w[i]) * delta[i] for i in range(n))

    vol = math.sqrt(max(_portfolio_variance(w, universe.Sigma), 0.0))
    sharpe = (ret - risk_free_rate) / vol if vol > 1e-10 else 0.0
    return PortfolioResult(weights=w, expected_return=ret, volatility=vol,
                           sharpe=sharpe, worst_case_return=wc_ret, method="box_robust")


# ---------------------------------------------------------------------------
# Ellipsoidal uncertainty set robust MVO
# ---------------------------------------------------------------------------

def ellipsoidal_robust_portfolio(
        universe: AssetUniverse,
        kappa: float,               # uncertainty radius (e.g., 1.96 for 95% CI)
        Omega: Optional[List[List[float]]] = None,  # uncertainty covariance; default = diagonal sigma^2/T
        n_obs: int = 252,           # number of observations used to estimate mu
        target_return: float = 0.0,
        risk_free_rate: float = 0.0,
        allow_short: bool = False,
) -> PortfolioResult:
    """
    Robust MVO with ellipsoidal uncertainty set (Goldfarb-Iyengar 2003).
    Uncertainty set: U = { mu : (mu - mu_hat)' Omega^{-1} (mu - mu_hat) <= kappa^2 }

    Worst-case return: w'mu_hat - kappa * sqrt(w' Omega w)

    For large kappa (high uncertainty), the optimizer shrinks toward lower-risk assets.
    """
    n = universe.n

    # Default Omega: standard estimation error covariance = Sigma / T
    if Omega is None:
        Omega = [[universe.Sigma[i][j] / n_obs for j in range(n)] for i in range(n)]

    def worst_case_return(w: List[float]) -> float:
        """w'mu - kappa * sqrt(w' Omega w)"""
        Ow = _mat_vec(Omega, w)
        wOw = max(_vec_dot(w, Ow), 0.0)
        return _vec_dot(w, universe.mu) - kappa * math.sqrt(wOw)

    # Solve via outer approximation: iteratively linearise the penalty term
    w = [1.0 / n] * n
    lr = 0.05

    for iteration in range(3000):
        # Gradient of variance: 2 Sigma w
        grad_var = _vec_scale(_mat_vec(universe.Sigma, w), 2.0)

        # Gradient of ellipsoidal penalty: kappa * Omega w / sqrt(w'Omega w)
        Ow = _mat_vec(Omega, w)
        wOw = max(_vec_dot(w, Ow), 1e-14)
        grad_pen = _vec_scale(Ow, kappa / math.sqrt(wOw))

        # Combined gradient: minimise variance + penalty (treat as regularisation)
        grad = _vec_add(grad_var, grad_pen)

        # Gradient step
        w_new = [w[i] - lr * grad[i] for i in range(n)]

        # Project: long-only
        if not allow_short:
            w_new = [max(wi, 0.0) for wi in w_new]

        # Project: sum-to-one
        total = sum(w_new)
        if total < 1e-10:
            w_new = [1.0 / n] * n
        else:
            w_new = [x / total for x in w_new]

        # Return constraint enforcement: if worst-case return < target, shift
        wc = worst_case_return(w_new)
        if wc < target_return - 1e-4 and target_return > 0:
            max_idx = max(range(n), key=lambda i: universe.mu[i])
            w_new[max_idx] = min(1.0, w_new[max_idx] + lr)
            tot = sum(w_new)
            w_new = [x / tot for x in w_new]

        if max(abs(w_new[i] - w[i]) for i in range(n)) < 1e-9:
            w = w_new
            break
        w = w_new
        lr *= 0.9995

    ret = _portfolio_return(w, universe.mu)
    vol = math.sqrt(max(_portfolio_variance(w, universe.Sigma), 0.0))
    sharpe = (ret - risk_free_rate) / vol if vol > 1e-10 else 0.0
    wc_ret = worst_case_return(w)

    return PortfolioResult(weights=w, expected_return=ret, volatility=vol,
                           sharpe=sharpe, worst_case_return=wc_ret,
                           method="ellipsoidal_robust")


# ---------------------------------------------------------------------------
# Robust efficient frontier
# ---------------------------------------------------------------------------

def robust_efficient_frontier(
        universe: AssetUniverse,
        n_points: int = 20,
        method: str = "box",         # "markowitz", "box", "ellipsoidal"
        delta: Optional[List[float]] = None,
        kappa: float = 1.96,
        n_obs: int = 252,
        risk_free_rate: float = 0.0,
) -> List[PortfolioResult]:
    """
    Generate efficient frontier portfolios.
    Returns n_points PortfolioResult objects along the frontier.
    """
    n = universe.n
    mu_min = min(universe.mu)
    mu_max = max(universe.mu)
    targets = [mu_min + (mu_max - mu_min) * i / (n_points - 1) for i in range(n_points)]

    if delta is None:
        vol_est = [math.sqrt(universe.Sigma[i][i]) for i in range(n)]
        delta = [v * 0.1 for v in vol_est]   # 10% of vol as default uncertainty

    results = []
    for t in targets:
        if method == "box":
            r = box_robust_portfolio(universe, delta, t, risk_free_rate)
        elif method == "ellipsoidal":
            r = ellipsoidal_robust_portfolio(universe, kappa, None, n_obs, t, risk_free_rate)
        else:
            r = markowitz_portfolio(universe, t, risk_free_rate)
        results.append(r)
    return results


# ---------------------------------------------------------------------------
# Resampling-based parameter uncertainty
# ---------------------------------------------------------------------------

def resampled_frontier(
        universe: AssetUniverse,
        n_simulations: int = 500,
        n_points: int = 10,
        risk_free_rate: float = 0.0,
        seed: int = 42,
) -> List[PortfolioResult]:
    """
    Michaud (1998) resampled efficient frontier.
    Averages portfolios across simulated return scenarios to reduce estimation error.
    """
    rng = random.Random(seed)
    n = universe.n

    # Cholesky for sampling from N(mu, Sigma)
    L = _cholesky(universe.Sigma)

    mu_min = min(universe.mu)
    mu_max = max(universe.mu)
    targets = [mu_min + (mu_max - mu_min) * i / (n_points - 1) for i in range(n_points)]

    # Accumulate weights across simulations
    w_sum = [[0.0] * n for _ in range(n_points)]

    for sim in range(n_simulations):
        # Sample mu from N(mu, Sigma / n) as estimation uncertainty
        mu_sim = []
        for i in range(n):
            z = rng.gauss(0, 1)
            noise = sum(L[i][j] * (rng.gauss(0, 1) if j < i else (z if j == i else 0.0))
                        for j in range(i + 1)) / math.sqrt(252)
            mu_sim.append(universe.mu[i] + noise)

        for k, t in enumerate(targets):
            t_sim = min(t, max(mu_sim) * 0.999)
            try:
                w = _critical_line_mvo(mu_sim, universe.Sigma, t_sim, False)
                for i in range(n):
                    w_sum[k][i] += w[i]
            except Exception:
                pass

    # Average and build results
    results = []
    for k, t in enumerate(targets):
        w = [w_sum[k][i] / n_simulations for i in range(n)]
        tot = sum(w)
        if tot > 1e-10:
            w = [x / tot for x in w]
        ret = _portfolio_return(w, universe.mu)
        vol = math.sqrt(max(_portfolio_variance(w, universe.Sigma), 0.0))
        sharpe = (ret - risk_free_rate) / vol if vol > 1e-10 else 0.0
        results.append(PortfolioResult(weights=w, expected_return=ret, volatility=vol,
                                       sharpe=sharpe, method="resampled"))
    return results


# ---------------------------------------------------------------------------
# Portfolio diagnostics
# ---------------------------------------------------------------------------

def diversification_ratio(w: List[float], Sigma: List[List[float]]) -> float:
    """
    Diversification ratio = (w' sigma) / sqrt(w' Sigma w)
    where sigma is vector of asset volatilities.
    DR >= 1; DR = 1 for a single asset; higher = more diversified.
    """
    asset_vols = [math.sqrt(max(Sigma[i][i], 0.0)) for i in range(len(w))]
    weighted_vol_sum = _vec_dot(w, asset_vols)
    portfolio_vol = math.sqrt(max(_portfolio_variance(w, Sigma), 0.0))
    return weighted_vol_sum / portfolio_vol if portfolio_vol > 1e-10 else 1.0


def herfindahl_index(w: List[float]) -> float:
    """Herfindahl-Hirschman concentration index. HHI = sum(w_i^2). 1/n = max diversification."""
    return sum(wi ** 2 for wi in w)


def effective_n(w: List[float]) -> float:
    """Effective N: 1 / HHI. Equal-weight portfolio has effective N = n."""
    hhi = herfindahl_index(w)
    return 1.0 / hhi if hhi > 1e-10 else len(w)


def portfolio_diagnostics(result: PortfolioResult, Sigma: List[List[float]],
                           asset_names: Optional[List[str]] = None) -> Dict:
    w = result.weights
    n = len(w)
    names = asset_names or [f"Asset_{i}" for i in range(n)]
    sorted_w = sorted(zip(names, w), key=lambda x: -x[1])
    return {
        "return": result.expected_return,
        "volatility": result.volatility,
        "sharpe": result.sharpe,
        "worst_case_return": result.worst_case_return,
        "method": result.method,
        "diversification_ratio": diversification_ratio(w, Sigma),
        "hhi": herfindahl_index(w),
        "effective_n": effective_n(w),
        "top_holdings": [(name, round(wi * 100, 2)) for name, wi in sorted_w[:5]],
        "n_nonzero": sum(1 for wi in w if wi > 1e-4),
    }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random as _random

    rng = _random.Random(0)
    n = 6
    names = ["AAPL", "MSFT", "JPM", "XOM", "GLD", "TLT"]

    # Realistic expected returns and covariance
    mu = [0.15, 0.13, 0.10, 0.08, 0.05, 0.03]

    # Build a realistic covariance matrix (from correlations and vols)
    vols = [0.28, 0.25, 0.22, 0.24, 0.15, 0.08]
    corr = [
        [1.00, 0.75, 0.60, 0.40, 0.05, -0.30],
        [0.75, 1.00, 0.55, 0.35, 0.00, -0.25],
        [0.60, 0.55, 1.00, 0.45, 0.10, -0.20],
        [0.40, 0.35, 0.45, 1.00, 0.20, -0.15],
        [0.05, 0.00, 0.10, 0.20, 1.00,  0.10],
        [-0.30, -0.25, -0.20, -0.15, 0.10, 1.00],
    ]
    Sigma = [[corr[i][j] * vols[i] * vols[j] for j in range(n)] for i in range(n)]

    universe = AssetUniverse(n=n, mu=mu, Sigma=Sigma)
    delta = [0.03] * n  # 3% uncertainty in each return estimate

    target_return = 0.10

    print("=" * 60)
    print("Robust Portfolio Optimization Comparison")
    print("=" * 60)

    # 1. Standard Markowitz
    std = markowitz_portfolio(universe, target_return)
    print(f"\n1. Markowitz (standard MVO)")
    d = portfolio_diagnostics(std, Sigma, names)
    print(f"   Return: {d['return']:.2%}  Volatility: {d['volatility']:.2%}  Sharpe: {d['sharpe']:.3f}")
    print(f"   DR: {d['diversification_ratio']:.3f}  Eff-N: {d['effective_n']:.1f}  Non-zero: {d['n_nonzero']}")
    print(f"   Top holdings: {[(n, f'{w:.1f}%') for n, w in d['top_holdings'][:3]]}")

    # 2. Box robust
    box = box_robust_portfolio(universe, delta, target_return)
    print(f"\n2. Box Robust (delta={delta[0]:.1%})")
    d = portfolio_diagnostics(box, Sigma, names)
    print(f"   Return: {d['return']:.2%}  Vol: {d['volatility']:.2%}  Sharpe: {d['sharpe']:.3f}")
    print(f"   Worst-case return: {d['worst_case_return']:.2%}")
    print(f"   DR: {d['diversification_ratio']:.3f}  Eff-N: {d['effective_n']:.1f}")

    # 3. Ellipsoidal robust
    ell = ellipsoidal_robust_portfolio(universe, kappa=1.96, target_return=target_return)
    print(f"\n3. Ellipsoidal Robust (kappa=1.96)")
    d = portfolio_diagnostics(ell, Sigma, names)
    print(f"   Return: {d['return']:.2%}  Vol: {d['volatility']:.2%}  Sharpe: {d['sharpe']:.3f}")
    print(f"   Worst-case return: {d['worst_case_return']:.2%}")
    print(f"   DR: {d['diversification_ratio']:.3f}  Eff-N: {d['effective_n']:.1f}")

    # 4. Efficient frontiers
    print(f"\n4. Efficient Frontier Comparison (target return = 8% - 14%)")
    print(f"{'Target':>8} {'MVO Vol':>9} {'Box Vol':>9} {'Ell Vol':>9} {'MVO WC':>9} {'Box WC':>9}")
    print("-" * 60)
    for t in [0.08, 0.09, 0.10, 0.11, 0.12, 0.13]:
        m = markowitz_portfolio(universe, t)
        b = box_robust_portfolio(universe, delta, t)
        e = ellipsoidal_robust_portfolio(universe, 1.96, target_return=t)
        wc_b = b.worst_case_return
        wc_e = e.worst_case_return
        print(f"{t:>8.1%} {m.volatility:>9.2%} {b.volatility:>9.2%} {e.volatility:>9.2%} "
              f"{wc_b:>9.2%} {wc_e:>9.2%}")

    # 5. Resampled frontier
    print(f"\n5. Resampled Frontier (500 simulations)")
    resampled = resampled_frontier(universe, n_simulations=200, n_points=5, seed=42)
    for r in resampled:
        print(f"   Return: {r.expected_return:.2%}  Vol: {r.volatility:.2%}  "
              f"Sharpe: {r.sharpe:.3f}  Eff-N: {effective_n(r.weights):.1f}")
