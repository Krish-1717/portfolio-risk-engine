"""
portfolio_day25_factor_models.py
Day 25: Equity Factor Models — Fama-French 3-factor exposure estimation,
factor timing (momentum/mean-reversion), factor crowding (correlation-based),
factor-adjusted alpha, smart-beta construction.
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

def variance(xs: list[float]) -> float:
    m = mean(xs)
    return sum((x - m)**2 for x in xs) / max(len(xs) - 1, 1)

def std(xs: list[float]) -> float:
    return math.sqrt(max(variance(xs), 0.0))

def cov(xs: list[float], ys: list[float]) -> float:
    n = min(len(xs), len(ys))
    mx, my = mean(xs[:n]), mean(ys[:n])
    return sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / max(n - 1, 1)

def ols(y: list[float], X: list[list[float]]) -> list[float]:
    """
    OLS regression y = X @ beta via Gaussian elimination.
    X: n x k design matrix (includes intercept as first column if desired).
    Returns beta (k-vector).
    """
    n, k = len(y), len(X[0])
    # X'X
    XtX = [[sum(X[i][a] * X[i][b] for i in range(n)) for b in range(k)] for a in range(k)]
    # X'y
    Xty = [sum(X[i][a] * y[i] for i in range(n)) for a in range(k)]

    # Gaussian elimination with partial pivoting
    aug = [XtX[i][:] + [Xty[i]] for i in range(k)]
    for col in range(k):
        pivot = max(range(col, k), key=lambda r: abs(aug[r][col]))
        aug[col], aug[pivot] = aug[pivot], aug[col]
        if abs(aug[col][col]) < 1e-12:
            continue
        for row in range(k):
            if row != col:
                factor = aug[row][col] / aug[col][col]
                aug[row] = [aug[row][j] - factor * aug[col][j] for j in range(k + 1)]
    beta = [aug[i][k] / aug[i][i] if abs(aug[i][i]) > 1e-12 else 0.0 for i in range(k)]
    return beta

# ---------------------------------------------------------------------------
# 1. Fama-French 3-factor model
# ---------------------------------------------------------------------------
@dataclass
class FFFactors:
    """Fama-French factors: Market (Mkt-Rf), Size (SMB), Value (HML)."""
    mkt_rf: list[float]   # excess market return
    smb: list[float]      # small-minus-big
    hml: list[float]      # high-minus-low (value)
    rf: list[float]       # risk-free rate

@dataclass
class FFExposure:
    alpha: float       # annualized Jensen alpha
    beta_mkt: float    # market beta
    beta_smb: float    # size beta
    beta_hml: float    # value beta
    r_squared: float
    t_stats: list[float]  # [alpha, beta_mkt, beta_smb, beta_hml]

def ff3_regression(stock_returns: list[float], factors: FFFactors) -> FFExposure:
    """Estimate FF3 factor loadings via OLS."""
    T = min(len(stock_returns), len(factors.mkt_rf), len(factors.smb),
            len(factors.hml), len(factors.rf))

    # Excess returns
    y = [stock_returns[t] - factors.rf[t] for t in range(T)]

    # Design matrix [1, Mkt-Rf, SMB, HML]
    X = [[1.0, factors.mkt_rf[t], factors.smb[t], factors.hml[t]] for t in range(T)]

    beta = ols(y, X)
    alpha_per, b_mkt, b_smb, b_hml = beta

    # Residuals and R²
    y_hat = [alpha_per + b_mkt * factors.mkt_rf[t] + b_smb * factors.smb[t] + b_hml * factors.hml[t]
             for t in range(T)]
    residuals = [y[t] - y_hat[t] for t in range(T)]
    ss_res = sum(r**2 for r in residuals)
    ss_tot = sum((yi - mean(y))**2 for yi in y)
    r2 = 1 - ss_res / max(ss_tot, 1e-10)

    # T-stats (simplified: t = beta / (std_err))
    sigma2 = ss_res / max(T - 4, 1)
    XtX_inv_diag = [1.0 / max(sum(X[t][j]**2 for t in range(T)), 1e-10) for j in range(4)]
    se = [math.sqrt(sigma2 * XtX_inv_diag[j]) for j in range(4)]
    t_stats = [beta[j] / max(se[j], 1e-10) for j in range(4)]

    return FFExposure(
        alpha=alpha_per * 252,   # annualize
        beta_mkt=b_mkt,
        beta_smb=b_smb,
        beta_hml=b_hml,
        r_squared=r2,
        t_stats=t_stats,
    )

# ---------------------------------------------------------------------------
# 2. Factor timing via momentum and mean-reversion
# ---------------------------------------------------------------------------
def factor_momentum(factor_returns: list[float], lookback: int = 12,
                     skip: int = 1) -> list[float]:
    """
    Compute rolling factor momentum (past-12-month-skip-1-month return).
    Returns time series of momentum signals.
    """
    T = len(factor_returns)
    signals = []
    for t in range(T):
        start = t - lookback - skip
        end = t - skip
        if start < 0 or end < 0:
            signals.append(0.0)
        else:
            signals.append(sum(factor_returns[start:end]))
    return signals

def factor_mean_reversion(factor_returns: list[float], window: int = 21) -> list[float]:
    """Short-term mean reversion signal: negative of recent return (contrarian)."""
    T = len(factor_returns)
    signals = []
    for t in range(T):
        if t < window:
            signals.append(0.0)
        else:
            recent = sum(factor_returns[t-window:t])
            signals.append(-recent)  # contrarian
    return signals

def factor_timing_weights(momentum: list[float], mean_rev: list[float],
                           w_mom: float = 0.6, w_mr: float = 0.4) -> list[float]:
    """
    Blend momentum and mean-reversion signals. Normalize to [-1, 1].
    Returns factor timing allocation (1 = full long, -1 = full short).
    """
    T = min(len(momentum), len(mean_rev))
    combined = [w_mom * momentum[t] + w_mr * mean_rev[t] for t in range(T)]
    sig_std = max(std(combined), 1e-10)
    return [c / sig_std for c in combined]

# ---------------------------------------------------------------------------
# 3. Factor crowding (correlation-based)
# ---------------------------------------------------------------------------
def factor_crowding_score(stock_factor_betas: list[list[float]]) -> float:
    """
    Crowding: average pairwise correlation of stock factor exposures.
    High correlation → crowded factor.
    stock_factor_betas: list of [beta_mkt, beta_smb, beta_hml] per stock.
    """
    n = len(stock_factor_betas)
    if n < 2:
        return 0.0

    total_corr, count = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            bi, bj = stock_factor_betas[i], stock_factor_betas[j]
            c = cov(bi, bj) / max(std(bi) * std(bj), 1e-10)
            total_corr += c
            count += 1

    return total_corr / max(count, 1)

def factor_dispersion(factor_returns: list[list[float]]) -> list[float]:
    """
    Cross-sectional dispersion of factor returns across stocks.
    High dispersion → factor is discriminating; low → crowded.
    factor_returns[t][i] = return of stock i at time t.
    """
    dispersions = []
    for t_returns in factor_returns:
        dispersions.append(std(t_returns))
    return dispersions

# ---------------------------------------------------------------------------
# 4. Factor-adjusted alpha
# ---------------------------------------------------------------------------
def factor_adjusted_alpha(stock_returns: list[float], factors: FFFactors,
                            risk_free: float = 0.05) -> dict:
    """
    Compute CAPM alpha, FF3 alpha, and factor-hedged alpha.
    """
    # CAPM
    excess_stock = [r - factors.rf[t] for t, r in enumerate(stock_returns[:len(factors.rf)])]
    capm_beta_val = cov(excess_stock, factors.mkt_rf) / max(variance(factors.mkt_rf), 1e-10)
    capm_alpha = (mean(excess_stock) - capm_beta_val * mean(factors.mkt_rf)) * 252

    # FF3
    exp = ff3_regression(stock_returns, factors)

    # Factor-hedged: residual from FF3 regression (pure alpha)
    T = min(len(stock_returns), len(factors.mkt_rf))
    y = [stock_returns[t] - factors.rf[t] for t in range(T)]
    hedged = [y[t] - (exp.beta_mkt * factors.mkt_rf[t] +
                       exp.beta_smb * factors.smb[t] +
                       exp.beta_hml * factors.hml[t]) for t in range(T)]
    hedged_sharpe = (mean(hedged) * 252) / max(std(hedged) * math.sqrt(252), 1e-10)

    return {
        'capm_alpha': capm_alpha,
        'capm_beta': capm_beta_val,
        'ff3_alpha': exp.alpha,
        'ff3_beta_mkt': exp.beta_mkt,
        'ff3_beta_smb': exp.beta_smb,
        'ff3_beta_hml': exp.beta_hml,
        'ff3_r2': exp.r_squared,
        'ff3_t_alpha': exp.t_stats[0],
        'hedged_alpha_ann': mean(hedged) * 252,
        'hedged_sharpe': hedged_sharpe,
    }

# ---------------------------------------------------------------------------
# 5. Smart-beta factor portfolio construction
# ---------------------------------------------------------------------------
def smart_beta_weights(stocks: list[str],
                        factor_scores: dict[str, list[float]],
                        factor_weights: dict[str, float]) -> dict[str, float]:
    """
    Multi-factor smart-beta: weight stocks by composite factor score.
    factor_scores: {factor_name: [score_per_stock]}
    factor_weights: {factor_name: weight}
    Returns normalized long-only weights.
    """
    n = len(stocks)
    composite = [0.0] * n

    for factor, w in factor_weights.items():
        scores = factor_scores.get(factor, [0.0] * n)
        # Z-score normalize
        mu_s = mean(scores)
        std_s = max(std(scores), 1e-10)
        z_scores = [(s - mu_s) / std_s for s in scores]
        for i in range(n):
            composite[i] += w * z_scores[i]

    # Softmax to get positive weights
    composite_shifted = [c - min(composite) for c in composite]
    total = sum(composite_shifted)
    if total <= 0:
        return {s: 1.0/n for s in stocks}

    return {stocks[i]: composite_shifted[i] / total for i in range(n)}

# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 65)
    print("DAY 25: Equity Factor Models")
    print("=" * 65)

    rng = random.Random(42)
    T = 252

    def randn():
        u = max(rng.random(), 1e-15)
        return math.sqrt(-2*math.log(u)) * math.cos(2*math.pi*rng.random())

    # Simulate FF factors
    rf_daily = [0.05 / 252] * T
    mkt_rf = [0.0008 + 0.012 * randn() for _ in range(T)]
    smb = [0.0002 + 0.008 * randn() for _ in range(T)]
    hml = [0.0003 + 0.007 * randn() for _ in range(T)]
    factors = FFFactors(mkt_rf=mkt_rf, smb=smb, hml=hml, rf=rf_daily)

    # Simulate 5 stocks with different exposures
    stock_params = [
        ('GrowthLargeCap', 1.2, -0.3, -0.5),   # high beta, anti-value, anti-size
        ('ValueSmallCap',  0.8,  0.6,  0.7),    # value tilt + small cap
        ('Defensive',      0.4,  0.0,  0.3),    # low beta
        ('Momentum',       1.1, -0.1, -0.2),    # market-like
        ('Blend',          0.9,  0.2,  0.1),    # balanced
    ]

    print("\n1. FF3 Factor Exposures")
    print(f"   {'Stock':20s} | {'α ann':>8} | {'β_mkt':>8} | {'β_smb':>8} | {'β_hml':>8} | {'R²':>6} | {'t(α)':>8}")
    print("   " + "-" * 80)

    all_betas = []
    for name, b_mkt, b_smb, b_hml in stock_params:
        alpha_true = 0.0002
        stock_r = [alpha_true + b_mkt * mkt_rf[t] + b_smb * smb[t] + b_hml * hml[t]
                    + 0.005 * randn() + rf_daily[t]
                    for t in range(T)]
        exp = ff3_regression(stock_r, factors)
        print(f"   {name:20s} | {exp.alpha:>+8.4f} | {exp.beta_mkt:>8.3f} | "
              f"{exp.beta_smb:>8.3f} | {exp.beta_hml:>8.3f} | {exp.r_squared:>6.3f} | "
              f"{exp.t_stats[0]:>+8.2f}")
        all_betas.append([exp.beta_mkt, exp.beta_smb, exp.beta_hml])

    print("\n2. Factor Momentum & Timing")
    mkt_mom = factor_momentum(mkt_rf, lookback=21, skip=1)
    mkt_mr = factor_mean_reversion(mkt_rf, window=5)
    timing = factor_timing_weights(mkt_mom, mkt_mr)
    print(f"   Last 5 market timing signals: {[round(timing[t], 3) for t in range(T-5, T)]}")
    print(f"   Signal range: [{min(timing):.3f}, {max(timing):.3f}]")
    positive_pct = sum(1 for t in timing if t > 0) / max(len(timing), 1)
    print(f"   Long market {positive_pct:.1%} of the time")

    print("\n3. Factor Crowding Score")
    crowding = factor_crowding_score(all_betas)
    print(f"   Average pairwise factor-beta correlation: {crowding:.4f}")
    print(f"   {'High crowding > 0.5' if crowding > 0.5 else 'Low crowding (< 0.5)'}")

    print("\n4. Factor-Adjusted Alpha (GrowthLargeCap)")
    b_mkt_g, b_smb_g, b_hml_g = 1.2, -0.3, -0.5
    stock_g = [0.0005 + b_mkt_g * mkt_rf[t] + b_smb_g * smb[t] + b_hml_g * hml[t]
                + 0.006 * randn() + rf_daily[t] for t in range(T)]
    alpha_result = factor_adjusted_alpha(stock_g, factors)
    print(f"   CAPM alpha (ann):  {alpha_result['capm_alpha']:+.4f}  beta={alpha_result['capm_beta']:.3f}")
    print(f"   FF3  alpha (ann):  {alpha_result['ff3_alpha']:+.4f}  t={alpha_result['ff3_t_alpha']:+.2f}")
    print(f"   FF3  R²         :  {alpha_result['ff3_r2']:.4f}")
    print(f"   Hedged alpha ann:  {alpha_result['hedged_alpha_ann']:+.4f}  Sharpe={alpha_result['hedged_sharpe']:+.3f}")

    print("\n5. Smart-Beta Portfolio")
    stocks_names = ['GrowthLargeCap', 'ValueSmallCap', 'Defensive', 'Momentum', 'Blend']
    # Factor scores: simulate value, momentum, quality scores per stock
    factor_scores_dict = {
        'value':    [0.1, 0.9, 0.5, 0.2, 0.5],
        'momentum': [0.8, 0.3, 0.2, 0.9, 0.5],
        'quality':  [0.6, 0.5, 0.9, 0.4, 0.6],
    }
    factor_wts = {'value': 0.33, 'momentum': 0.33, 'quality': 0.34}
    smart_wts = smart_beta_weights(stocks_names, factor_scores_dict, factor_wts)
    print(f"   {'Stock':20s} | {'Weight':>8}")
    print("   " + "-" * 32)
    for name, w in smart_wts.items():
        print(f"   {name:20s} | {w:>8.4f}")
    print(f"   Sum: {sum(smart_wts.values()):.4f}")

    print("\n[Done] Day 25: Factor Models complete.")
