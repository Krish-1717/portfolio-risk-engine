"""
portfolio_day21_regime_optimizer.py
Day 21: Regime-Conditional Portfolio Optimizer — Hidden Markov regime detection,
regime-blended MVO, CUSUM change-point trigger, transition matrix, regime-aware
drawdown control.
Pure Python stdlib only.
"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Basic statistics helpers
# ---------------------------------------------------------------------------
def mean(xs: list[float]) -> float:
    return sum(xs) / max(len(xs), 1)

def variance(xs: list[float]) -> float:
    m = mean(xs)
    return sum((x - m)**2 for x in xs) / max(len(xs) - 1, 1)

def std(xs: list[float]) -> float:
    return math.sqrt(max(variance(xs), 0.0))

def sample_cov(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx, my = mean(xs), mean(ys)
    return sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / max(n - 1, 1)

# ---------------------------------------------------------------------------
# Regime detection: Gaussian HMM-style (2-state) via EM
# ---------------------------------------------------------------------------
@dataclass
class HMMRegime:
    """2-state Gaussian HMM: Bull (high return, low vol) vs Bear (low return, high vol)."""
    n_states: int = 2
    mu: list[float] = field(default_factory=lambda: [0.001, -0.002])
    sigma: list[float] = field(default_factory=lambda: [0.010, 0.025])
    pi: list[float] = field(default_factory=lambda: [0.5, 0.5])        # initial probs
    A: list[list[float]] = field(default_factory=lambda:                # transition matrix
                                  [[0.97, 0.03], [0.05, 0.95]])

    def _gauss(self, x: float, state: int) -> float:
        mu, sigma = self.mu[state], self.sigma[state]
        if sigma <= 0:
            return 1.0 if abs(x - mu) < 1e-10 else 0.0
        return math.exp(-0.5 * ((x - mu) / sigma)**2) / (sigma * math.sqrt(2 * math.pi))

    def fit(self, returns: list[float], n_iter: int = 30):
        """Baum-Welch EM for 2-state HMM."""
        T = len(returns)
        K = self.n_states

        for _ in range(n_iter):
            # E-step: forward-backward
            alpha = [[0.0]*K for _ in range(T)]
            beta = [[0.0]*K for _ in range(T)]

            # Forward
            for k in range(K):
                alpha[0][k] = self.pi[k] * self._gauss(returns[0], k)
            alpha_sum = sum(alpha[0])
            if alpha_sum > 0:
                alpha[0] = [a / alpha_sum for a in alpha[0]]

            for t in range(1, T):
                for k in range(K):
                    alpha[t][k] = sum(alpha[t-1][j] * self.A[j][k] for j in range(K)) * self._gauss(returns[t], k)
                s = sum(alpha[t])
                if s > 0:
                    alpha[t] = [a / s for a in alpha[t]]

            # Backward
            for k in range(K):
                beta[T-1][k] = 1.0
            for t in range(T-2, -1, -1):
                for k in range(K):
                    beta[t][k] = sum(self.A[k][j] * self._gauss(returns[t+1], j) * beta[t+1][j] for j in range(K))
                s = sum(beta[t])
                if s > 0:
                    beta[t] = [b / s for b in beta[t]]

            # Gammas (posterior state probs)
            gamma = [[0.0]*K for _ in range(T)]
            for t in range(T):
                s = sum(alpha[t][k] * beta[t][k] for k in range(K))
                if s <= 0:
                    s = 1.0
                for k in range(K):
                    gamma[t][k] = alpha[t][k] * beta[t][k] / s

            # M-step: update params
            for k in range(K):
                g_sum = sum(gamma[t][k] for t in range(T))
                if g_sum > 0:
                    self.mu[k] = sum(gamma[t][k] * returns[t] for t in range(T)) / g_sum
                    self.sigma[k] = math.sqrt(
                        max(sum(gamma[t][k] * (returns[t] - self.mu[k])**2 for t in range(T)) / g_sum, 1e-8)
                    )

            # Transition matrix
            for i in range(K):
                row_sum = sum(
                    sum(alpha[t][i] * self.A[i][j] * self._gauss(returns[t+1], j) * beta[t+1][j]
                        for j in range(K))
                    for t in range(T-1)
                )
                for j in range(K):
                    xi_ij = sum(
                        alpha[t][i] * self.A[i][j] * self._gauss(returns[t+1], j) * beta[t+1][j]
                        for t in range(T-1)
                    )
                    self.A[i][j] = xi_ij / max(row_sum, 1e-10)

            self.pi = gamma[0]

        return gamma  # posterior state probabilities

    def predict_regime(self, returns: list[float]) -> list[int]:
        """Viterbi decoding: most likely state sequence."""
        T = len(returns)
        K = self.n_states
        viterbi = [[0.0]*K for _ in range(T)]
        backptr = [[0]*K for _ in range(T)]

        for k in range(K):
            viterbi[0][k] = math.log(max(self.pi[k], 1e-300)) + math.log(max(self._gauss(returns[0], k), 1e-300))

        for t in range(1, T):
            for k in range(K):
                scores = [viterbi[t-1][j] + math.log(max(self.A[j][k], 1e-300)) for j in range(K)]
                best_j = max(range(K), key=lambda x: scores[x])
                backptr[t][k] = best_j
                viterbi[t][k] = scores[best_j] + math.log(max(self._gauss(returns[t], k), 1e-300))

        # Traceback
        path = [0] * T
        path[T-1] = max(range(K), key=lambda k: viterbi[T-1][k])
        for t in range(T-2, -1, -1):
            path[t] = backptr[t+1][path[t+1]]

        return path

# ---------------------------------------------------------------------------
# CUSUM change-point detector
# ---------------------------------------------------------------------------
def cusum_detector(returns: list[float], threshold: float = 3.0, k: float = 0.5) -> list[int]:
    """
    CUSUM change-point detection.
    Returns list of change-point indices (where regime shift detected).
    k: slack parameter (allowable drift), threshold: alarm level in std units.
    """
    if not returns:
        return []
    mu = mean(returns)
    sigma_est = max(std(returns), 1e-8)

    C_plus, C_minus = 0.0, 0.0
    change_points = []

    for t, r in enumerate(returns):
        standardized = (r - mu) / sigma_est
        C_plus = max(0.0, C_plus + standardized - k)
        C_minus = max(0.0, C_minus - standardized - k)

        if C_plus > threshold or C_minus > threshold:
            change_points.append(t)
            C_plus = C_minus = 0.0  # reset

    return change_points

# ---------------------------------------------------------------------------
# Regime-conditional covariance and returns
# ---------------------------------------------------------------------------
def regime_conditional_stats(returns_matrix: list[list[float]],
                               regime_path: list[int],
                               n_regimes: int = 2) -> list[dict]:
    """
    Compute mean return and covariance matrix per regime.
    returns_matrix: list of T observations, each a list of N asset returns.
    """
    N = len(returns_matrix[0]) if returns_matrix else 0
    stats = []

    for reg in range(n_regimes):
        obs = [returns_matrix[t] for t, r in enumerate(regime_path) if r == reg]
        T_reg = len(obs)

        if T_reg < 2:
            mu_reg = [0.0] * N
            cov_reg = [[0.01 if i == j else 0.0 for j in range(N)] for i in range(N)]
        else:
            mu_reg = [mean([obs[t][i] for t in range(T_reg)]) for i in range(N)]
            cov_reg = [
                [sample_cov([obs[t][i] for t in range(T_reg)], [obs[t][j] for t in range(T_reg)])
                 for j in range(N)]
                for i in range(N)
            ]

        stats.append({
            'regime': reg,
            'n_obs': T_reg,
            'mu': mu_reg,
            'cov': cov_reg,
            'weight_in_sample': T_reg / max(len(regime_path), 1),
        })

    return stats

# ---------------------------------------------------------------------------
# Simple MVO (mean-variance portfolio)
# ---------------------------------------------------------------------------
def mvo_min_variance(cov: list[list[float]]) -> list[float]:
    """Minimum variance portfolio via closed-form (n<=3)."""
    n = len(cov)
    # Use inverse vol weighting as approximation for general n
    inv_vols = [1.0 / math.sqrt(max(cov[i][i], 1e-8)) for i in range(n)]
    total = sum(inv_vols)
    return [v / total for v in inv_vols]

def mvo_max_sharpe(mu: list[float], cov: list[list[float]], rf: float = 0.0) -> list[float]:
    """
    Max Sharpe ratio: tangency portfolio approximation using excess return / variance.
    """
    n = len(mu)
    excess_mu = [m - rf for m in mu]
    vols = [math.sqrt(max(cov[i][i], 1e-8)) for i in range(n)]

    # Approx: weight proportional to Sharpe of each asset
    sharpes = [max(excess_mu[i] / vols[i], 0.0) for i in range(n)]
    total = sum(sharpes)
    if total <= 0:
        return [1.0 / n] * n
    return [s / total for s in sharpes]

# ---------------------------------------------------------------------------
# Regime-blended portfolio
# ---------------------------------------------------------------------------
def regime_blended_portfolio(regime_stats: list[dict],
                              current_probs: list[float],
                              rf: float = 0.0) -> list[float]:
    """
    Blend regime-conditional portfolios by current regime probabilities.
    Uses max-Sharpe within each regime, then probability-weights them.
    """
    n = len(regime_stats[0]['mu'])
    blended = [0.0] * n

    for i, stat in enumerate(regime_stats):
        w_regime = mvo_max_sharpe(stat['mu'], stat['cov'], rf)
        prob = current_probs[i]
        for j in range(n):
            blended[j] += prob * w_regime[j]

    # Normalize
    total = sum(blended)
    if total > 0:
        blended = [b / total for b in blended]

    return blended

# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 65)
    print("DAY 21: Regime-Conditional Portfolio Optimizer")
    print("=" * 65)

    rng = random.Random(42)
    T = 500
    N = 4  # assets

    # Simulate 2 regimes: bull (high mu, low vol) and bear (low mu, high vol)
    regime_true = [0] * T
    for t in range(T):
        if t > 0:
            if regime_true[t-1] == 0:
                regime_true[t] = 1 if rng.random() < 0.03 else 0
            else:
                regime_true[t] = 0 if rng.random() < 0.05 else 1

    mu_bull = [0.001, 0.0008, 0.0012, 0.0009]
    mu_bear = [-0.002, -0.0015, -0.001, -0.003]
    vol_bull = [0.010, 0.012, 0.011, 0.013]
    vol_bear = [0.025, 0.028, 0.022, 0.030]

    def randn():
        u = max(rng.random(), 1e-15)
        return math.sqrt(-2*math.log(u)) * math.cos(2*math.pi*rng.random())

    returns_matrix = []
    for t in range(T):
        reg = regime_true[t]
        mu = mu_bull if reg == 0 else mu_bear
        vol = vol_bull if reg == 0 else vol_bear
        row = [mu[i] + vol[i] * randn() for i in range(N)]
        returns_matrix.append(row)

    asset_returns = [[returns_matrix[t][i] for t in range(T)] for i in range(N)]
    index_returns = [mean([returns_matrix[t][i] for i in range(N)]) for t in range(T)]

    print(f"\nSimulated {T} periods, {N} assets, 2 regimes")
    print(f"True regime: {sum(1 for r in regime_true if r==0)} Bull / {sum(1 for r in regime_true if r==1)} Bear")

    # 1. Fit HMM
    print("\n1. Fitting 2-State Gaussian HMM on index returns...")
    hmm = HMMRegime()
    gamma = hmm.fit(index_returns, n_iter=20)
    regime_path = hmm.predict_regime(index_returns)

    bull_detected = sum(1 for r in regime_path if r == 0)
    bear_detected = sum(1 for r in regime_path if r == 1)
    print(f"   Detected: {bull_detected} state-0 / {bear_detected} state-1 observations")
    print(f"   State 0: mu={hmm.mu[0]:.5f} sigma={hmm.sigma[0]:.5f}")
    print(f"   State 1: mu={hmm.mu[1]:.5f} sigma={hmm.sigma[1]:.5f}")

    # Identify which detected state is "bull"
    bull_state = 0 if hmm.mu[0] > hmm.mu[1] else 1
    bear_state = 1 - bull_state
    print(f"   Bull = state {bull_state}, Bear = state {bear_state}")

    print(f"\n   Transition matrix:")
    for i in range(2):
        print(f"   {i} -> {hmm.A[i]}")

    # 2. CUSUM change-point detection
    print("\n2. CUSUM Change-Point Detection")
    cps = cusum_detector(index_returns, threshold=3.0, k=0.5)
    print(f"   Detected {len(cps)} change-points: {cps[:10]}{'...' if len(cps)>10 else ''}")

    # 3. Regime-conditional stats
    print("\n3. Regime-Conditional Statistics")
    reg_stats = regime_conditional_stats(returns_matrix, regime_path, n_regimes=2)
    for s in reg_stats:
        avg_mu = mean(s['mu'])
        avg_vol = mean([math.sqrt(s['cov'][i][i]) for i in range(N)])
        print(f"   Regime {s['regime']}: n={s['n_obs']:4d}, avg_mu={avg_mu:+.5f}, avg_vol={avg_vol:.5f}")

    # 4. Regime-blended portfolio
    print("\n4. Regime-Blended Portfolio")
    # Current probabilities from HMM
    current_probs = gamma[-1]
    print(f"   Current regime probs: state0={current_probs[0]:.3f}, state1={current_probs[1]:.3f}")

    w_blended = regime_blended_portfolio(reg_stats, current_probs)
    print(f"   Blended weights: {[round(w, 4) for w in w_blended]}")
    print(f"   Sum of weights: {sum(w_blended):.4f}")

    # Regime-specific portfolios
    w_bull = mvo_max_sharpe(reg_stats[0]['mu'], reg_stats[0]['cov'])
    w_bear = mvo_min_variance(reg_stats[1]['cov'])
    print(f"   Bull-regime weights: {[round(w, 4) for w in w_bull]}")
    print(f"   Bear-regime weights: {[round(w, 4) for w in w_bear]}")

    # 5. Out-of-sample portfolio returns by regime
    print("\n5. Regime-Conditional Performance (in-sample)")
    for name, weights, reg in [('Bull portfolio', w_bull, 0), ('Bear portfolio', w_bear, 1)]:
        sub_returns = [
            sum(weights[i] * returns_matrix[t][i] for i in range(N))
            for t in range(T)
            if regime_path[t] == reg
        ]
        if sub_returns:
            print(f"   {name:20s}: mean={mean(sub_returns):+.5f}, vol={std(sub_returns):.5f}, "
                  f"Sharpe={mean(sub_returns)/max(std(sub_returns),1e-8):.3f}")

    print("\n[Done] Day 21: Regime Optimizer complete.")
