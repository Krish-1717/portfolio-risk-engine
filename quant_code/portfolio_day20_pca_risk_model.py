"""
PCA Risk Factor Model

Implements:
- Full covariance matrix estimation with shrinkage (Ledoit-Wolf)
- PCA factor decomposition using power iteration (no numpy)
- Factor attribution: systematic vs idiosyncratic risk
- Factor-model portfolio VaR and CVaR
- Rolling PCA for regime detection
- Explained variance and factor loading analysis

Reference: Barra/FactSet risk model methodology; Ledoit & Wolf (2004)
"""

import math
import random
from typing import List, Tuple, Optional, NamedTuple, Dict


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class PCAResult(NamedTuple):
    eigenvalues: List[float]          # Descending sorted
    eigenvectors: List[List[float]]   # Columns are eigenvectors
    explained_variance_ratio: List[float]
    cumulative_explained: List[float]


class FactorModel(NamedTuple):
    n_factors: int
    factor_returns: List[List[float]]  # T x K matrix
    loadings: List[List[float]]        # N x K matrix (B)
    systematic_var: List[float]        # N systematic variance
    idiosyncratic_var: List[float]     # N residual variance
    eigenvalues: List[float]
    explained_ratio: List[float]


class PortfolioRiskDecomp(NamedTuple):
    total_var: float
    systematic_var: float
    idiosyncratic_var: float
    factor_contributions: List[float]   # Per-factor variance contribution
    factor_names: List[str]
    marginal_risk: List[float]          # Per-asset marginal risk contribution
    portfolio_vol: float


# ---------------------------------------------------------------------------
# Matrix utilities (pure Python)
# ---------------------------------------------------------------------------

def mat_mul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Matrix multiply A (m×k) × B (k×n) → m×n."""
    m, k = len(A), len(A[0])
    n = len(B[0])
    C = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            s = 0.0
            for l in range(k):
                s += A[i][l] * B[l][j]
            C[i][j] = s
    return C


def mat_transpose(A: List[List[float]]) -> List[List[float]]:
    m, n = len(A), len(A[0])
    return [[A[i][j] for i in range(m)] for j in range(n)]


def mat_vec_mul(A: List[List[float]], v: List[float]) -> List[float]:
    return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]


def vec_dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def vec_norm(v: List[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def vec_scale(v: List[float], s: float) -> List[float]:
    return [x * s for x in v]


def vec_sub(a: List[float], b: List[float]) -> List[float]:
    return [x - y for x, y in zip(a, b)]


def vec_add(a: List[float], b: List[float]) -> List[float]:
    return [x + y for x, y in zip(a, b)]


def outer_product(a: List[float], b: List[float]) -> List[List[float]]:
    return [[ai * bi for bi in b] for ai in a]


def identity(n: int) -> List[List[float]]:
    M = [[0.0] * n for _ in range(n)]
    for i in range(n):
        M[i][i] = 1.0
    return M


def mat_scale(A: List[List[float]], s: float) -> List[List[float]]:
    return [[A[i][j] * s for j in range(len(A[0]))] for i in range(len(A))]


def mat_add(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def mat_sub(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


# ---------------------------------------------------------------------------
# Covariance estimation
# ---------------------------------------------------------------------------

def sample_covariance(returns: List[List[float]]) -> List[List[float]]:
    """
    Compute sample covariance matrix from returns.

    Args:
        returns: T×N matrix of asset returns

    Returns:
        N×N sample covariance matrix
    """
    T = len(returns)
    N = len(returns[0])

    # Demean
    means = [sum(returns[t][i] for t in range(T)) / T for i in range(N)]
    demeaned = [[returns[t][i] - means[i] for i in range(N)] for t in range(T)]

    # Cov = X'X / (T-1)
    cov = [[0.0] * N for _ in range(N)]
    for t in range(T):
        for i in range(N):
            for j in range(N):
                cov[i][j] += demeaned[t][i] * demeaned[t][j]

    scale = 1.0 / (T - 1)
    for i in range(N):
        for j in range(N):
            cov[i][j] *= scale

    return cov


def ledoit_wolf_shrinkage(returns: List[List[float]]) -> List[List[float]]:
    """
    Ledoit-Wolf analytical shrinkage toward scaled identity.
    Shrinks sample covariance toward mu*I where mu = trace(S)/N.

    Returns shrunk covariance with optimal alpha.
    """
    T = len(returns)
    N = len(returns[0])

    S = sample_covariance(returns)

    # Target: scaled identity
    trace_S = sum(S[i][i] for i in range(N))
    mu = trace_S / N
    target = [[mu if i == j else 0.0 for j in range(N)] for i in range(N)]

    # Oracle shrinkage intensity (simplified Ledoit-Wolf)
    # alpha = arg min || (1-alpha)*S + alpha*T - Sigma ||_F^2
    # Analytical: alpha* = min(1, ((N+2)/T) / (||S-mu*I||^2_F / mu^2))
    norm_sq = sum((S[i][j] - target[i][j]) ** 2 for i in range(N) for j in range(N))

    if norm_sq < 1e-12:
        return S

    alpha_raw = ((N + 2) / T) * (mu * mu) / norm_sq
    alpha = min(1.0, max(0.0, alpha_raw))

    # Shrunk covariance
    shrunk = [[
        (1 - alpha) * S[i][j] + alpha * target[i][j]
        for j in range(N)
    ] for i in range(N)]

    return shrunk


# ---------------------------------------------------------------------------
# Power iteration PCA
# ---------------------------------------------------------------------------

def _deflate(A: List[List[float]], eigvec: List[float], eigval: float) -> List[List[float]]:
    """Hotelling deflation: A ← A - λ * v * v'."""
    N = len(A)
    A_new = [row[:] for row in A]
    for i in range(N):
        for j in range(N):
            A_new[i][j] -= eigval * eigvec[i] * eigvec[j]
    return A_new


def power_iteration(A: List[List[float]], n_iter: int = 200,
                    seed: int = 0) -> Tuple[float, List[float]]:
    """
    Dominant eigenpair via power iteration with normalization.

    Returns:
        (eigenvalue, eigenvector)
    """
    N = len(A)
    rng = random.Random(seed)
    v = [rng.gauss(0, 1) for _ in range(N)]
    nrm = vec_norm(v)
    v = vec_scale(v, 1.0 / nrm)

    for _ in range(n_iter):
        v_new = mat_vec_mul(A, v)
        nrm = vec_norm(v_new)
        if nrm < 1e-14:
            break
        v_new = vec_scale(v_new, 1.0 / nrm)
        # Check for sign flip
        if vec_dot(v_new, v) < 0:
            v_new = vec_scale(v_new, -1.0)
        v = v_new

    # Rayleigh quotient
    av = mat_vec_mul(A, v)
    eigval = vec_dot(v, av)

    return eigval, v


def pca_decompose(cov: List[List[float]], n_components: int) -> PCAResult:
    """
    Extract top n_components principal components via deflation.

    Args:
        cov: N×N covariance matrix
        n_components: Number of components to extract

    Returns:
        PCAResult with eigenvalues, eigenvectors, explained variance
    """
    N = len(cov)
    n_components = min(n_components, N)

    A = [row[:] for row in cov]   # Work on copy
    total_var = sum(A[i][i] for i in range(N))

    eigenvalues = []
    eigenvectors = []

    for k in range(n_components):
        lam, v = power_iteration(A, seed=k)
        if lam <= 1e-10:
            lam = 1e-10
        eigenvalues.append(lam)
        eigenvectors.append(v)
        A = _deflate(A, v, lam)

    # Explained variance ratio
    ev_ratio = [lam / total_var for lam in eigenvalues]
    cumulative = []
    running = 0.0
    for r in ev_ratio:
        running += r
        cumulative.append(running)

    # Eigenvectors as columns of N×K matrix
    N_vecs = len(eigenvectors)
    evec_matrix = [[eigenvectors[k][i] for k in range(N_vecs)] for i in range(N)]

    return PCAResult(
        eigenvalues=eigenvalues,
        eigenvectors=evec_matrix,
        explained_variance_ratio=ev_ratio,
        cumulative_explained=cumulative,
    )


def n_components_for_threshold(pca: PCAResult, threshold: float = 0.90) -> int:
    """Return minimum k such that cumulative explained variance >= threshold."""
    for k, cum in enumerate(pca.cumulative_explained):
        if cum >= threshold:
            return k + 1
    return len(pca.eigenvalues)


# ---------------------------------------------------------------------------
# Factor model estimation
# ---------------------------------------------------------------------------

def build_factor_model(returns: List[List[float]],
                        n_factors: Optional[int] = None,
                        variance_threshold: float = 0.80,
                        use_shrinkage: bool = True) -> FactorModel:
    """
    Build PCA factor model from asset returns.

    Args:
        returns: T×N asset return matrix
        n_factors: Number of factors (None = auto from variance_threshold)
        variance_threshold: If n_factors None, choose k explaining this much variance
        use_shrinkage: Apply Ledoit-Wolf shrinkage

    Returns:
        FactorModel with loadings, factor returns, variance decomposition
    """
    T = len(returns)
    N = len(returns[0])

    cov = ledoit_wolf_shrinkage(returns) if use_shrinkage else sample_covariance(returns)

    # Demean returns
    means = [sum(returns[t][i] for t in range(T)) / T for i in range(N)]
    X = [[returns[t][i] - means[i] for i in range(N)] for t in range(T)]

    # PCA on covariance
    max_k = min(N, T - 1)
    pca = pca_decompose(cov, max_k)

    if n_factors is None:
        n_factors = n_components_for_threshold(pca, variance_threshold)
    n_factors = min(n_factors, max_k)

    # Loadings B: N×K, columns = eigenvectors scaled by sqrt(eigenvalue)
    loadings = []
    for i in range(N):
        row = []
        for k in range(n_factors):
            b = pca.eigenvectors[i][k] * math.sqrt(pca.eigenvalues[k])
            row.append(b)
        loadings.append(row)

    # Factor returns: F_t = X_t @ B / eigenvalue_k  (T×K)
    # More precisely: F_t = X_t @ eigvec_k / sqrt(eigval_k)
    eigvec_matrix = pca.eigenvectors   # N×K
    factor_returns = []
    for t in range(T):
        ft = []
        for k in range(n_factors):
            evk = [eigvec_matrix[i][k] for i in range(N)]
            fval = vec_dot(X[t], evk) / math.sqrt(max(pca.eigenvalues[k], 1e-12))
            ft.append(fval)
        factor_returns.append(ft)

    # Systematic variance per asset: sum_k B_ik^2
    systematic_var = [sum(loadings[i][k] ** 2 for k in range(n_factors))
                      for i in range(N)]

    # Idiosyncratic variance: diag(Cov) - systematic
    idiosyncratic_var = [max(cov[i][i] - systematic_var[i], 0.0)
                         for i in range(N)]

    return FactorModel(
        n_factors=n_factors,
        factor_returns=factor_returns,
        loadings=loadings,
        systematic_var=systematic_var,
        idiosyncratic_var=idiosyncratic_var,
        eigenvalues=pca.eigenvalues[:n_factors],
        explained_ratio=pca.explained_variance_ratio[:n_factors],
    )


# ---------------------------------------------------------------------------
# Portfolio risk decomposition
# ---------------------------------------------------------------------------

def portfolio_risk_decomposition(weights: List[float],
                                   model: FactorModel,
                                   factor_names: Optional[List[str]] = None) -> PortfolioRiskDecomp:
    """
    Decompose portfolio variance into factor and idiosyncratic contributions.

    Total var = w'*B*F_cov*B'*w + w'*D*w
    where D = diag(idiosyncratic_var), F_cov = I (orthogonal factors).

    Args:
        weights: Portfolio weights (N,)
        model: Calibrated FactorModel

    Returns:
        PortfolioRiskDecomp with factor/idiosyncratic breakdown
    """
    N = len(weights)
    K = model.n_factors
    B = model.loadings  # N×K

    if factor_names is None:
        factor_names = [f"PC{k + 1}" for k in range(K)]

    # Portfolio factor exposure: h = B'*w (K,)
    h = []
    for k in range(K):
        hk = sum(weights[i] * B[i][k] for i in range(N))
        h.append(hk)

    # Factor var contributions: h_k^2 (since F_cov = I, eigenvalues absorbed in B)
    factor_contribs = [hk * hk for hk in h]
    systematic_var = sum(factor_contribs)

    # Idiosyncratic variance: sum_i w_i^2 * D_i
    idio_var = sum(weights[i] ** 2 * model.idiosyncratic_var[i] for i in range(N))

    total_var = systematic_var + idio_var

    # Marginal risk contribution per asset
    # dVol/dw_i = (systematic_marginal_i + idio_marginal_i) / portfolio_vol
    vol = math.sqrt(max(total_var, 1e-12))
    marginal_risk = []
    for i in range(N):
        # Systematic: sum_k h_k * B_ik
        sys_i = sum(h[k] * B[i][k] for k in range(K))
        # Idio: w_i * D_i
        idio_i = weights[i] * model.idiosyncratic_var[i]
        marginal_risk.append((sys_i + idio_i) / vol)

    return PortfolioRiskDecomp(
        total_var=total_var,
        systematic_var=systematic_var,
        idiosyncratic_var=idio_var,
        factor_contributions=factor_contribs,
        factor_names=factor_names,
        marginal_risk=marginal_risk,
        portfolio_vol=vol,
    )


# ---------------------------------------------------------------------------
# Portfolio VaR using factor model
# ---------------------------------------------------------------------------

def portfolio_var_cvar(weights: List[float],
                        model: FactorModel,
                        confidence: float = 0.99,
                        horizon_days: int = 1,
                        n_sims: int = 10000,
                        seed: int = 42) -> Tuple[float, float]:
    """
    Monte Carlo VaR/CVaR using the factor model.

    Simulate: r_port = sum_k h_k * f_k + sum_i w_i * eps_i
    where f_k ~ N(0, 1), eps_i ~ N(0, D_i).

    Args:
        weights: Portfolio weights
        model: Calibrated FactorModel
        confidence: VaR confidence (0.99 = 99%)
        horizon_days: Holding period in trading days
        n_sims: Monte Carlo draws

    Returns:
        (VaR, CVaR) in portfolio return units
    """
    N = len(weights)
    K = model.n_factors
    B = model.loadings

    # Portfolio factor exposures
    h = [sum(weights[i] * B[i][k] for i in range(N)) for k in range(K)]
    # Scale to horizon
    scale = math.sqrt(horizon_days)

    rng = random.Random(seed)
    port_returns = []

    for _ in range(n_sims):
        # Factor shocks
        f_shock = sum(h[k] * rng.gauss(0, 1) * scale for k in range(K))

        # Idiosyncratic shocks
        eps_shock = sum(
            weights[i] * rng.gauss(0, math.sqrt(model.idiosyncratic_var[i])) * scale
            for i in range(N)
        )
        port_returns.append(f_shock + eps_shock)

    # Sort ascending (losses are negative)
    port_returns.sort()

    cutoff_idx = int((1 - confidence) * n_sims)
    var = -port_returns[cutoff_idx]

    tail = port_returns[:cutoff_idx + 1]
    cvar = -sum(tail) / len(tail) if tail else var

    return var, cvar


# ---------------------------------------------------------------------------
# Rolling PCA for regime detection
# ---------------------------------------------------------------------------

def rolling_pca_explained_variance(returns: List[List[float]],
                                    window: int = 60,
                                    n_factors: int = 3) -> List[float]:
    """
    Rolling window PCA: track explained variance ratio of top K factors.

    A spike in explained variance → factor concentration → crisis regime.

    Args:
        returns: T×N return matrix
        window: Rolling window length
        n_factors: Number of factors to track

    Returns:
        List of cumulative explained variance (length T - window + 1)
    """
    T = len(returns)
    result = []

    for t in range(window - 1, T):
        window_returns = returns[t - window + 1: t + 1]
        cov = sample_covariance(window_returns)
        pca = pca_decompose(cov, n_factors)
        cum_explained = pca.cumulative_explained[-1]
        result.append(cum_explained)

    return result


# ---------------------------------------------------------------------------
# Factor loading analysis
# ---------------------------------------------------------------------------

def factor_betas(weights: List[float], model: FactorModel) -> List[float]:
    """Portfolio beta to each PCA factor."""
    B = model.loadings
    K = model.n_factors
    N = len(weights)
    return [sum(weights[i] * B[i][k] for i in range(N)) for k in range(K)]


def asset_communality(model: FactorModel) -> List[float]:
    """
    Communality: fraction of each asset's variance explained by factors.
    R^2_i = systematic_var_i / total_var_i
    """
    result = []
    for i in range(len(model.systematic_var)):
        total = model.systematic_var[i] + model.idiosyncratic_var[i]
        if total < 1e-12:
            result.append(0.0)
        else:
            result.append(model.systematic_var[i] / total)
    return result


def factor_correlation_matrix(model: FactorModel, asset_names: Optional[List[str]] = None) -> Dict:
    """
    Compute correlations between each asset and each factor.

    Returns:
        Dict with keys 'factor_k' → list of per-asset correlations
    """
    N = len(model.loadings)
    K = model.n_factors
    B = model.loadings

    result = {}
    for k in range(K):
        corrs = []
        for i in range(N):
            total_std = math.sqrt(model.systematic_var[i] + model.idiosyncratic_var[i])
            factor_std = math.sqrt(model.eigenvalues[k])
            if total_std < 1e-12 or factor_std < 1e-12:
                corrs.append(0.0)
            else:
                # corr(r_i, f_k) = B_ik / sigma_i  (since f_k has unit variance)
                corrs.append(model.loadings[i][k] / (total_std * factor_std))
        result[f"PC{k + 1}"] = corrs

    return result


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== PCA Risk Factor Model Demo ===\n")

    rng = random.Random(1234)
    T, N = 252, 8   # 1 year daily returns, 8 assets

    # Generate correlated returns: 3 latent factors
    K_true = 3
    true_loadings = [[rng.gauss(0, 0.3) for _ in range(K_true)] for _ in range(N)]
    returns = []
    for t in range(T):
        factors = [rng.gauss(0, 0.01) for _ in range(K_true)]
        r = []
        for i in range(N):
            systematic = sum(true_loadings[i][k] * factors[k] for k in range(K_true))
            idio = rng.gauss(0, 0.008)
            r.append(systematic + idio)
        returns.append(r)

    # Fit factor model
    model = build_factor_model(returns, n_factors=3)

    print(f"Factors: {model.n_factors}")
    print("Explained variance per factor:")
    for k in range(model.n_factors):
        print(f"  PC{k + 1}: {model.explained_ratio[k] * 100:.1f}%")

    # Equal-weight portfolio
    w = [1.0 / N] * N

    # Risk decomposition
    decomp = portfolio_risk_decomposition(w, model)
    print(f"\nPortfolio Vol: {decomp.portfolio_vol * math.sqrt(252) * 100:.2f}% annualized")
    print(f"Systematic var: {decomp.systematic_var / decomp.total_var * 100:.1f}%")
    print(f"Idiosyncratic var: {decomp.idiosyncratic_var / decomp.total_var * 100:.1f}%")
    print("Factor contributions:")
    for name, contrib in zip(decomp.factor_names, decomp.factor_contributions):
        print(f"  {name}: {contrib / decomp.total_var * 100:.1f}%")

    # VaR
    var_1d, cvar_1d = portfolio_var_cvar(w, model, confidence=0.99, horizon_days=1)
    print(f"\n1-day 99% VaR:  {var_1d * 100:.3f}%")
    print(f"1-day 99% CVaR: {cvar_1d * 100:.3f}%")

    var_10d, cvar_10d = portfolio_var_cvar(w, model, confidence=0.99, horizon_days=10)
    print(f"10-day 99% VaR: {var_10d * 100:.3f}%")

    # Asset communalities
    comm = asset_communality(model)
    print("\nAsset communalities (% variance from factors):")
    for i, c in enumerate(comm):
        print(f"  Asset {i + 1}: {c * 100:.1f}%")

    # Rolling PCA regime detection
    roll_ev = rolling_pca_explained_variance(returns, window=63, n_factors=3)
    min_ev = min(roll_ev)
    max_ev = max(roll_ev)
    avg_ev = sum(roll_ev) / len(roll_ev)
    print(f"\nRolling PCA (63-day, 3 factors):")
    print(f"  Avg explained var: {avg_ev * 100:.1f}%")
    print(f"  Min: {min_ev * 100:.1f}%  Max: {max_ev * 100:.1f}%")
    if max_ev > 0.9:
        print("  ⚠ High factor concentration detected → possible crisis regime")

    # Factor betas
    betas = factor_betas(w, model)
    print(f"\nPortfolio factor betas: {[round(b, 4) for b in betas]}")
