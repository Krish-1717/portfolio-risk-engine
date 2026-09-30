"""
portfolio_day26_stress_testing.py
Day 26: Stress Testing Framework — Monte Carlo with fat-tail (Student-t),
reverse stress testing (find scenario that breaks the portfolio),
butterfly scenario generation, copula-based joint stress, drawdown scenario.
Pure Python stdlib only.
"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def mean(xs: list[float]) -> float:
    return sum(xs) / max(len(xs), 1)

def std(xs: list[float]) -> float:
    m = mean(xs)
    return math.sqrt(sum((x - m)**2 for x in xs) / max(len(xs) - 1, 1))

def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def _norm_ppf(p: float) -> float:
    p = max(1e-9, min(1 - 1e-9, p))
    lo, hi = -10.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2

# ---------------------------------------------------------------------------
# 1. Student-t random variate (fat tails)
# ---------------------------------------------------------------------------
def student_t_variate(nu: float, rng: random.Random) -> float:
    """
    Generate a Student-t(nu) variate using ratio of normals / chi.
    Z/sqrt(chi2/nu) where Z~N(0,1), chi2~chi2(nu).
    """
    def randn():
        u = max(rng.random(), 1e-15)
        return math.sqrt(-2 * math.log(u)) * math.cos(2 * math.pi * rng.random())

    Z = randn()
    # Sum of nu squared normals ~ chi2(nu)
    chi2 = sum(randn()**2 for _ in range(max(int(nu), 2)))
    return Z / math.sqrt(chi2 / nu)

def student_t_var(sigma: float, nu: float, confidence: float) -> float:
    """Approximate Student-t VaR via normal scaling + excess kurtosis correction."""
    # t-dist quantile approximation (Cornish-Fisher)
    z = _norm_ppf(1 - confidence)  # negative for VaR
    excess_kurt = 6.0 / max(nu - 4, 0.1) if nu > 4 else 1.5
    # Cornish-Fisher: z_cf = z + (excess_kurt/24) * (z^3 - 3z)
    z_cf = z + (excess_kurt / 24) * (z**3 - 3 * z)
    return -sigma * z_cf  # positive VaR

# ---------------------------------------------------------------------------
# 2. Fat-tail Monte Carlo portfolio stress
# ---------------------------------------------------------------------------
def fat_tail_mc(weights: list[float], mu: list[float], sigma: list[float],
                 corr: list[list[float]], T_days: int = 1, n_paths: int = 50000,
                 nu: float = 5.0, seed: int = 42) -> dict:
    """
    Monte Carlo with Student-t marginals and Gaussian copula.
    Returns distribution stats for portfolio P&L.
    """
    rng = random.Random(seed)
    n = len(weights)

    # Cholesky of correlation matrix with Tikhonov regularization for stability
    # (stressed scenarios can produce near-singular matrices)
    reg = 1e-4
    corr_reg = [[corr[i][j] + (reg if i == j else 0.0) for j in range(n)] for i in range(n)]
    L = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                diag_val = corr_reg[i][i] - s
                L[i][j] = math.sqrt(max(diag_val, 1e-8))
            else:
                L[i][j] = (corr_reg[i][j] - s) / max(L[j][j], 1e-8)

    sqT = math.sqrt(T_days)
    pnl_dist = []

    for _ in range(n_paths):
        # Correlated standard normals
        z_ind = [student_t_variate(nu, rng) for _ in range(n)]
        z_corr = [sum(L[i][j] * z_ind[j] for j in range(i+1)) for i in range(n)]

        port_pnl = sum(
            weights[i] * (mu[i] * T_days + sigma[i] * sqT * z_corr[i])
            for i in range(n)
        )
        pnl_dist.append(port_pnl)

    pnl_dist.sort()
    var_95 = -pnl_dist[int(0.05 * n_paths)]
    var_99 = -pnl_dist[int(0.01 * n_paths)]
    cvar_95 = -mean(pnl_dist[:int(0.05 * n_paths)])
    cvar_99 = -mean(pnl_dist[:int(0.01 * n_paths)])

    return {
        'var_95': var_95,
        'var_99': var_99,
        'cvar_95': cvar_95,
        'cvar_99': cvar_99,
        'mean_pnl': mean(pnl_dist),
        'std_pnl': std(pnl_dist),
        'min_pnl': pnl_dist[0],
        'pnl_dist': pnl_dist,
    }

# ---------------------------------------------------------------------------
# 3. Normal vs fat-tail comparison
# ---------------------------------------------------------------------------
def normal_mc(weights: list[float], mu: list[float], sigma: list[float],
               corr: list[list[float]], T_days: int = 1, n_paths: int = 50000,
               seed: int = 42) -> dict:
    """Same as fat_tail_mc but with Gaussian innovations."""
    rng = random.Random(seed)
    n = len(weights)
    L = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                L[i][j] = math.sqrt(max(corr[i][i] - s, 1e-10))
            else:
                L[i][j] = (corr[i][j] - s) / max(L[j][j], 1e-10)

    def randn():
        u = max(rng.random(), 1e-15)
        return math.sqrt(-2 * math.log(u)) * math.cos(2 * math.pi * rng.random())

    sqT = math.sqrt(T_days)
    pnl_dist = []
    for _ in range(n_paths):
        z_ind = [randn() for _ in range(n)]
        z_corr = [sum(L[i][j] * z_ind[j] for j in range(i+1)) for i in range(n)]
        port_pnl = sum(weights[i] * (mu[i]*T_days + sigma[i]*sqT*z_corr[i]) for i in range(n))
        pnl_dist.append(port_pnl)
    pnl_dist.sort()
    var_99 = -pnl_dist[int(0.01 * n_paths)]
    cvar_99 = -mean(pnl_dist[:int(0.01 * n_paths)])
    return {'var_99': var_99, 'cvar_99': cvar_99, 'std_pnl': std(pnl_dist)}

# ---------------------------------------------------------------------------
# 4. Reverse stress test
# ---------------------------------------------------------------------------
def reverse_stress_test(weights: list[float], sigma: list[float],
                          corr: list[list[float]],
                          loss_threshold: float,
                          n_factors: int = 2,
                          n_scenarios: int = 10000,
                          seed: int = 42) -> list[dict]:
    """
    Find scenarios that produce losses exceeding the threshold.
    Returns the worst-case contributing scenarios.
    Generates random market shocks and filters for extreme losses.
    """
    rng = random.Random(seed)
    n = len(weights)

    L = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                L[i][j] = math.sqrt(max(corr[i][i] - s, 1e-10))
            else:
                L[i][j] = (corr[i][j] - s) / max(L[j][j], 1e-10)

    breaking_scenarios = []
    for _ in range(n_scenarios):
        z_ind = [math.sqrt(-2*math.log(max(rng.random(),1e-15))) * math.cos(2*math.pi*rng.random())
                 for _ in range(n)]
        z_corr = [sum(L[i][j] * z_ind[j] for j in range(i+1)) for i in range(n)]
        shocks = [sigma[i] * z_corr[i] for i in range(n)]
        loss = -sum(weights[i] * shocks[i] for i in range(n))
        if loss >= loss_threshold:
            breaking_scenarios.append({'shocks': shocks, 'loss': loss})

    breaking_scenarios.sort(key=lambda s: -s['loss'])
    return breaking_scenarios[:20]

# ---------------------------------------------------------------------------
# 5. Butterfly scenario (base + bull + bear)
# ---------------------------------------------------------------------------
@dataclass
class ButterflyScenario:
    name: str
    asset_shocks: list[float]       # % shocks per asset
    vol_shocks: list[float]         # additive vol shocks
    correlation_shift: float = 0.0  # shift all correlations by this amount

def generate_butterfly_scenarios(base_vols: list[float],
                                   n_assets: int) -> list[ButterflyScenario]:
    """
    Generate 3 butterfly scenarios: base, bull, bear.
    """
    scenarios = [
        ButterflyScenario(
            name='Base',
            asset_shocks=[0.0] * n_assets,
            vol_shocks=[0.0] * n_assets,
        ),
        ButterflyScenario(
            name='Bull (Risk-On)',
            asset_shocks=[+0.05] * n_assets,
            vol_shocks=[-0.03] * n_assets,
            correlation_shift=-0.10,  # correlations compress in bull markets
        ),
        ButterflyScenario(
            name='Bear (Risk-Off)',
            asset_shocks=[-0.15] * n_assets,
            vol_shocks=[+0.10] * n_assets,
            correlation_shift=+0.20,  # correlations spike in crash
        ),
        ButterflyScenario(
            name='Stagflation',
            asset_shocks=[-0.10, -0.08, -0.05, +0.02, +0.05][:n_assets]
                         + [-0.08] * max(n_assets - 5, 0),
            vol_shocks=[+0.05] * n_assets,
            correlation_shift=+0.10,
        ),
        ButterflyScenario(
            name='Flash Crash',
            asset_shocks=[-0.08] * n_assets,
            vol_shocks=[+0.20] * n_assets,
            correlation_shift=+0.15,  # +0.30 can make matrix non-PSD with high base corr
        ),
    ]
    return scenarios

def apply_butterfly_scenario(weights: list[float], mu: list[float],
                               sigma: list[float], corr: list[list[float]],
                               scenario: ButterflyScenario,
                               n_paths: int = 20000, seed: int = 42) -> dict:
    """Compute portfolio loss distribution under a butterfly scenario."""
    n = len(weights)
    sigma_stressed = [max(sigma[i] + scenario.vol_shocks[i], 0.001) for i in range(n)]
    corr_stressed = [[
        max(-0.99, min(0.99, corr[i][j] + (scenario.correlation_shift if i != j else 0.0)))
        for j in range(n)] for i in range(n)]
    # Force diagonal = 1
    for i in range(n):
        corr_stressed[i][i] = 1.0

    result = fat_tail_mc(weights, mu, sigma_stressed, corr_stressed, n_paths=n_paths, seed=seed)

    # Add deterministic shock from asset_shocks
    det_pnl = sum(weights[i] * scenario.asset_shocks[i] for i in range(n))

    return {
        'scenario': scenario.name,
        'deterministic_pnl': det_pnl,
        'stochastic_var_99': result['var_99'],
        'total_var_99': result['var_99'] - det_pnl,
        'cvar_99': result['cvar_99'],
    }

# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 65)
    print("DAY 26: Stress Testing Framework")
    print("=" * 65)

    N = 5
    weights = [0.30, 0.25, 0.20, 0.15, 0.10]
    mu = [0.0005, 0.0004, 0.0006, 0.0002, 0.0003]
    sigma = [0.015, 0.018, 0.022, 0.008, 0.020]
    corr = [
        [1.00, 0.70, 0.65, 0.20, 0.55],
        [0.70, 1.00, 0.75, 0.15, 0.60],
        [0.65, 0.75, 1.00, 0.10, 0.50],
        [0.20, 0.15, 0.10, 1.00, 0.05],
        [0.55, 0.60, 0.50, 0.05, 1.00],
    ]

    print("\n1. Normal vs Fat-Tail (t=5) 1-Day VaR/CVaR")
    normal_result = normal_mc(weights, mu, sigma, corr, n_paths=50000)
    fat_result = fat_tail_mc(weights, mu, sigma, corr, nu=5.0, n_paths=50000)
    print(f"   {'Metric':20s} | {'Normal':>10} | {'Student-t5':>12} | {'Ratio':>8}")
    print("   " + "-" * 58)
    for metric in ['var_99', 'cvar_99']:
        n_val = normal_result[metric]
        f_val = fat_result[metric]
        print(f"   {metric:20s} | {n_val:>10.5f} | {f_val:>12.5f} | {f_val/max(n_val,1e-10):>8.2f}x")

    print(f"\n   Fat-tail amplifies VaR by {fat_result['var_99']/max(normal_result['var_99'],1e-10):.2f}x")
    print(f"   (Expected: t(5) has ~40% heavier tails than Gaussian)")

    print("\n2. 10-Day VaR Scaling")
    for T_days in [1, 5, 10, 20]:
        r = fat_tail_mc(weights, mu, sigma, corr, T_days=T_days, n_paths=20000, nu=5.0)
        sqrt_T_var = fat_result['var_99'] * math.sqrt(T_days)
        print(f"   T={T_days:2d}: MC VaR99={r['var_99']:.5f}  sqrt(T) rule={sqrt_T_var:.5f}  "
              f"ratio={r['var_99']/max(sqrt_T_var,1e-10):.3f}")

    print("\n3. Butterfly Scenario Analysis")
    scenarios = generate_butterfly_scenarios(sigma, N)
    print(f"   {'Scenario':20s} | {'Det PnL':>10} | {'VaR99':>10} | {'CVaR99':>10}")
    print("   " + "-" * 58)
    for sc in scenarios:
        result = apply_butterfly_scenario(weights, mu, sigma, corr, sc, n_paths=20000)
        print(f"   {result['scenario']:20s} | {result['deterministic_pnl']:>+10.4f} | "
              f"{result['stochastic_var_99']:>10.5f} | {result['cvar_99']:>10.5f}")

    print("\n4. Reverse Stress Test (loss threshold = 5%)")
    breaking = reverse_stress_test(weights, sigma, corr, loss_threshold=0.05, n_scenarios=50000)
    print(f"   Found {len(breaking)} breaking scenarios")
    if breaking:
        worst = breaking[0]
        print(f"   Worst case loss: {worst['loss']:.4%}")
        print(f"   Asset shocks (%) in worst case:")
        for i, shock in enumerate(worst['shocks']):
            print(f"     Asset {i+1}: {shock:+.2%}")

    print("\n5. Tail Dependence (correlation under stress)")
    # Simulate normal vs stress-period correlation
    rng = random.Random(0)
    n_obs = 1000
    normal_returns = []
    for _ in range(n_obs):
        z = math.sqrt(-2*math.log(max(rng.random(),1e-15))) * math.cos(2*math.pi*rng.random())
        r1 = sigma[0] * z
        z2 = math.sqrt(-2*math.log(max(rng.random(),1e-15))) * math.cos(2*math.pi*rng.random())
        r2 = sigma[1] * (corr[0][1] * z + math.sqrt(1-corr[0][1]**2) * z2)
        normal_returns.append((r1, r2))

    # Tail: filter to bottom 10% of r1
    sorted_by_r1 = sorted(normal_returns, key=lambda x: x[0])
    tail_obs = sorted_by_r1[:100]
    tail_r1 = [x[0] for x in tail_obs]
    tail_r2 = [x[1] for x in tail_obs]
    tail_mean_r2 = mean(tail_r2)
    unconditional_r2 = mean([x[1] for x in normal_returns])
    print(f"   Unconditional r2 mean: {unconditional_r2:+.5f}")
    print(f"   Conditional on r1 in bottom 10%: {tail_mean_r2:+.5f}")
    print(f"   (Larger negative → fat tail co-movement in crashes)")

    print("\n[Done] Day 26: Stress Testing complete.")
