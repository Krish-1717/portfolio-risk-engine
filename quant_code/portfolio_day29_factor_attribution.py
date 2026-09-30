"""
portfolio_day29_factor_attribution.py
Day 29: Factor Risk Attribution — Barra-style decomposition,
factor exposure matrix, specific risk, marginal contribution to factor risk,
active risk vs benchmark, ex-ante tracking error.
Pure Python stdlib only.
"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Matrix helpers
# ---------------------------------------------------------------------------
def mat_mul(A, B):
    n, m, p = len(A), len(A[0]), len(B[0])
    return [[sum(A[i][k] * B[k][j] for k in range(m)) for j in range(p)] for i in range(n)]

def mat_T(A):
    return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]

def mat_add(A, B):
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def mat_scale(A, s):
    return [[x * s for x in row] for row in A]

def mat_vec(A, v):
    return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]

def vec_dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def outer(a, b):
    return [[a[i] * b[j] for j in range(len(b))] for i in range(len(a))]

def mat_inv(A):
    n = len(A)
    aug = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(A)]
    for col in range(n):
        max_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        aug[col], aug[max_row] = aug[max_row], aug[col]
        pivot = aug[col][col]
        if abs(pivot) < 1e-12:
            raise ValueError("Singular matrix")
        aug[col] = [x / pivot for x in aug[col]]
        for row in range(n):
            if row != col:
                f = aug[row][col]
                aug[row] = [aug[row][k] - f * aug[col][k] for k in range(2 * n)]
    return [row[n:] for row in aug]

def mean(xs): return sum(xs) / max(len(xs), 1)
def std(xs): m = mean(xs); return math.sqrt(sum((x-m)**2 for x in xs) / max(len(xs)-1, 1))

# ---------------------------------------------------------------------------
# 1. Factor model: R_i = alpha_i + sum_k beta_ik * F_k + epsilon_i
# ---------------------------------------------------------------------------
@dataclass
class FactorModel:
    """
    Barra-style linear factor model.
    B: asset exposures matrix (n_assets x n_factors)
    F_cov: factor covariance matrix (n_factors x n_factors)  [annualized]
    D: specific variance diagonal (n_assets,)                [annualized]
    """
    asset_names: list[str]
    factor_names: list[str]
    B: list[list[float]]       # n_assets x n_factors
    F_cov: list[list[float]]   # n_factors x n_factors
    D: list[float]             # n_assets (specific variances)

    @property
    def n_assets(self): return len(self.asset_names)
    @property
    def n_factors(self): return len(self.factor_names)

    def total_covariance(self) -> list[list[float]]:
        """Σ = B * F_cov * B' + diag(D)"""
        BF = mat_mul(self.B, self.F_cov)
        BFBt = mat_mul(BF, mat_T(self.B))
        n = self.n_assets
        return [[BFBt[i][j] + (self.D[i] if i == j else 0.0) for j in range(n)]
                for i in range(n)]

    def factor_risk(self, weights: list[float]) -> float:
        """Annualized factor risk = sqrt(w' B F B' w)."""
        Bw = mat_vec(mat_T(self.B), weights)
        return math.sqrt(max(vec_dot(Bw, mat_vec(self.F_cov, Bw)), 0.0))

    def specific_risk(self, weights: list[float]) -> float:
        """Annualized specific risk = sqrt(w' D w)."""
        return math.sqrt(max(sum(self.D[i] * weights[i]**2 for i in range(self.n_assets)), 0.0))

    def total_risk(self, weights: list[float]) -> float:
        """Total portfolio risk = sqrt(factor_risk² + specific_risk²)."""
        fr = self.factor_risk(weights)
        sr = self.specific_risk(weights)
        return math.sqrt(fr**2 + sr**2)

    def factor_exposures(self, weights: list[float]) -> list[float]:
        """Portfolio factor exposures: b_port = B' * w."""
        return mat_vec(mat_T(self.B), weights)

    def marginal_factor_contribution(self, weights: list[float]) -> list[list[float]]:
        """
        Marginal contribution of each asset to each factor risk component.
        MCFR_i,k = w_i * (B F)_ik / factor_risk
        """
        fr = self.factor_risk(weights)
        if fr < 1e-10:
            return [[0.0] * self.n_factors for _ in range(self.n_assets)]
        BF = mat_mul(self.B, self.F_cov)
        BFBt_w = mat_vec(mat_mul(BF, mat_T(self.B)), weights)
        return [[weights[i] * BF[i][k] / fr for k in range(self.n_factors)]
                for i in range(self.n_assets)]

# ---------------------------------------------------------------------------
# 2. Active risk vs benchmark
# ---------------------------------------------------------------------------
def active_risk_decomposition(model: FactorModel,
                               portfolio_w: list[float],
                               benchmark_w: list[float]) -> dict:
    """
    Decompose active risk (tracking error) into:
    - Active factor risk: from active factor bets (b_port - b_bench)
    - Active specific risk: from stock-selection weights
    """
    n = model.n_assets
    active_w = [portfolio_w[i] - benchmark_w[i] for i in range(n)]

    Sigma = model.total_covariance()
    te_var = vec_dot(active_w, mat_vec(Sigma, active_w))
    te = math.sqrt(max(te_var, 0.0))

    active_factor_exp = model.factor_exposures(active_w)
    factor_risk = model.factor_risk(active_w)
    specific_risk = model.specific_risk(active_w)

    # Factor risk contribution by factor
    BFBt_aw = mat_vec(mat_mul(mat_mul(model.B, model.F_cov), mat_T(model.B)), active_w)
    factor_var = sum(active_w[i] * BFBt_aw[i] for i in range(n))

    factor_risk_pct = factor_var / max(te_var, 1e-12)
    specific_risk_pct = (te_var - factor_var) / max(te_var, 1e-12)

    return {
        'tracking_error': te,
        'active_factor_risk': factor_risk,
        'active_specific_risk': specific_risk,
        'factor_risk_pct': factor_risk_pct,
        'specific_risk_pct': max(specific_risk_pct, 0.0),
        'active_factor_exposures': active_factor_exp,
        'active_weights': active_w,
    }

# ---------------------------------------------------------------------------
# 3. Brinson-Hood-Beebower attribution
# ---------------------------------------------------------------------------
def brinson_attribution(portfolio_w: list[float], benchmark_w: list[float],
                          portfolio_ret: list[float], benchmark_ret: list[float],
                          sector_labels: list[str]) -> dict:
    """
    Brinson-Hood-Beebower (1986) decomposition:
    - Allocation effect: (w_p - w_b) * (r_b - R_b)
    - Selection effect: w_b * (r_p - r_b)
    - Interaction effect: (w_p - w_b) * (r_p - r_b)
    """
    n = len(portfolio_w)
    R_b = vec_dot(benchmark_w, benchmark_ret)  # benchmark total return

    allocation = [(portfolio_w[i] - benchmark_w[i]) * (benchmark_ret[i] - R_b)
                  for i in range(n)]
    selection  = [benchmark_w[i] * (portfolio_ret[i] - benchmark_ret[i])
                  for i in range(n)]
    interaction = [(portfolio_w[i] - benchmark_w[i]) * (portfolio_ret[i] - benchmark_ret[i])
                   for i in range(n)]

    R_p = vec_dot(portfolio_w, portfolio_ret)
    total_active = R_p - R_b

    return {
        'portfolio_return': R_p,
        'benchmark_return': R_b,
        'active_return': total_active,
        'allocation': allocation,
        'selection': selection,
        'interaction': interaction,
        'total_allocation': sum(allocation),
        'total_selection': sum(selection),
        'total_interaction': sum(interaction),
        'sector_labels': sector_labels,
    }

# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 65)
    print("DAY 29: Factor Risk Attribution")
    print("=" * 65)

    rng = random.Random(42)

    # ---- 5 stocks, 3 factors: Market, Size, Value ----
    assets = ['AAPL', 'MSFT', 'JPM', 'XOM', 'JNJ']
    factors = ['Market', 'Size', 'Value']
    n_a, n_f = len(assets), len(factors)

    # Factor exposure matrix (beta to Market, Size, Value)
    B = [
        [1.20, -0.50, -0.80],  # AAPL: high beta, large-growth
        [1.10, -0.40, -0.70],  # MSFT: large-growth
        [1.05, -0.10,  0.50],  # JPM:  value financials
        [0.85,  0.10,  0.80],  # XOM:  value energy
        [0.70, -0.20,  0.30],  # JNJ:  defensive value
    ]

    # Annualized factor covariance (Market vol=16%, Size vol=8%, Value vol=10%)
    F_cov = [
        [0.0256,  0.0040, -0.0080],
        [0.0040,  0.0064,  0.0024],
        [-0.0080, 0.0024,  0.0100],
    ]

    # Specific variances (annualized) — idiosyncratic risk
    D = [0.0100, 0.0090, 0.0120, 0.0130, 0.0070]

    model = FactorModel(assets, factors, B, F_cov, D)

    portfolio_w = [0.35, 0.30, 0.15, 0.10, 0.10]
    benchmark_w = [0.25, 0.25, 0.20, 0.15, 0.15]

    print(f"\nPortfolio: {dict(zip(assets, portfolio_w))}")
    print(f"Benchmark: {dict(zip(assets, benchmark_w))}")

    print("\n1. Portfolio Factor Exposures")
    port_exp = model.factor_exposures(portfolio_w)
    bench_exp = model.factor_exposures(benchmark_w)
    print(f"   {'Factor':10} | {'Port Exp':>10} | {'Bench Exp':>10} | {'Active Bet':>11}")
    print("   " + "-" * 47)
    for f, pe, be in zip(factors, port_exp, bench_exp):
        print(f"   {f:10} | {pe:>+10.4f} | {be:>+10.4f} | {pe-be:>+11.4f}")

    print("\n2. Risk Decomposition")
    port_fr = model.factor_risk(portfolio_w)
    port_sr = model.specific_risk(portfolio_w)
    port_tr = model.total_risk(portfolio_w)
    bench_fr = model.factor_risk(benchmark_w)
    bench_sr = model.specific_risk(benchmark_w)
    bench_tr = model.total_risk(benchmark_w)
    print(f"   {'':15} | {'Factor Risk':>12} | {'Specific Risk':>14} | {'Total Risk':>11}")
    print("   " + "-" * 57)
    print(f"   {'Portfolio':15} | {port_fr:>12.4f} | {port_sr:>14.4f} | {port_tr:>11.4f}")
    print(f"   {'Benchmark':15} | {bench_fr:>12.4f} | {bench_sr:>14.4f} | {bench_tr:>11.4f}")

    print("\n3. Per-Asset Factor Risk Contribution (Portfolio)")
    mcfr = model.marginal_factor_contribution(portfolio_w)
    print(f"   {'Asset':6} | {'Weight':>8} | {'Mkt RC':>8} | {'Size RC':>8} | {'Val RC':>8} | {'Total':>8}")
    print("   " + "-" * 58)
    for i, (a, w) in enumerate(zip(assets, portfolio_w)):
        row_sum = sum(mcfr[i])
        print(f"   {a:6} | {w:>8.4f} | {mcfr[i][0]:>+8.4f} | {mcfr[i][1]:>+8.4f} | "
              f"{mcfr[i][2]:>+8.4f} | {row_sum:>+8.4f}")

    print("\n4. Active Risk Decomposition (Tracking Error)")
    ar = active_risk_decomposition(model, portfolio_w, benchmark_w)
    print(f"   Tracking error (ex-ante) : {ar['tracking_error']:.4f}  ({ar['tracking_error']*100:.2f}% ann)")
    print(f"   Active factor risk       : {ar['active_factor_risk']:.4f}")
    print(f"   Active specific risk     : {ar['active_specific_risk']:.4f}")
    print(f"   Factor risk % of TE²     : {ar['factor_risk_pct']:.1%}")
    print(f"   Specific risk % of TE²   : {ar['specific_risk_pct']:.1%}")
    print(f"\n   Active factor bets:")
    for f, ae in zip(factors, ar['active_factor_exposures']):
        print(f"   {f:10}: {ae:+.4f}")

    print("\n5. Brinson-Hood-Beebower Attribution (Quarterly Returns)")
    sector_names = assets
    # Simulated quarterly returns
    port_rets_q  = [0.08, 0.12, -0.05, 0.03, 0.04]
    bench_rets_q = [0.07, 0.09, -0.06, 0.05, 0.03]

    brison = brinson_attribution(portfolio_w, benchmark_w, port_rets_q, bench_rets_q, sector_names)
    print(f"   Portfolio return : {brison['portfolio_return']:+.4f}")
    print(f"   Benchmark return : {brison['benchmark_return']:+.4f}")
    print(f"   Active return    : {brison['active_return']:+.4f}")
    print(f"\n   {'Asset':6} | {'Alloc':>8} | {'Select':>8} | {'Interact':>9}")
    print("   " + "-" * 38)
    for i, a in enumerate(sector_names):
        print(f"   {a:6} | {brison['allocation'][i]:>+8.5f} | "
              f"{brison['selection'][i]:>+8.5f} | {brison['interaction'][i]:>+9.5f}")
    print("   " + "-" * 38)
    print(f"   {'TOTAL':6} | {brison['total_allocation']:>+8.5f} | "
          f"{brison['total_selection']:>+8.5f} | {brison['total_interaction']:>+9.5f}")
    verify = brison['total_allocation'] + brison['total_selection'] + brison['total_interaction']
    print(f"\n   Sum check: {verify:+.6f} vs active return {brison['active_return']:+.6f}")

    print("\n6. Ex-Ante Information Ratio")
    # Assume alpha forecasts for each stock
    alphas = [0.02, 0.015, -0.005, 0.003, 0.008]
    port_alpha = vec_dot(portfolio_w, alphas)
    bench_alpha = vec_dot(benchmark_w, alphas)
    active_alpha = port_alpha - bench_alpha
    te = ar['tracking_error']
    ir = active_alpha / max(te, 1e-8)
    print(f"   Portfolio alpha forecast : {port_alpha:+.4f}")
    print(f"   Benchmark alpha forecast : {bench_alpha:+.4f}")
    print(f"   Active alpha (ex-ante)   : {active_alpha:+.4f}")
    print(f"   Tracking error           : {te:.4f}")
    print(f"   Information ratio (IR)   : {ir:+.4f}")

    print("\n[Done] Day 29: Factor Risk Attribution complete.")
