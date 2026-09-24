"""
regimes/hmm_regime.py -- Regime-switching Hidden Markov Model.
Consolidated from regime-switching-hmm repo.
2-state Gaussian HMM via Baum-Welch EM, Viterbi decoding, regime forecasting.
Pure Python stdlib -- no external dependencies.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class GaussianHMMParams:
    n_states: int
    pi: List[float]             # initial state probs
    A: List[List[float]]        # transition matrix [from][to]
    means: List[float]          # emission means per state
    stds: List[float]           # emission stds per state


def _gauss_pdf(x: float, mu: float, sigma: float) -> float:
    sigma = max(sigma, 1e-10)
    return math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))


def _log_sum_exp(log_probs: List[float]) -> float:
    m = max(log_probs)
    return m + math.log(sum(math.exp(lp - m) for lp in log_probs))


def init_params(n_states: int, observations: List[float]) -> GaussianHMMParams:
    """Heuristic init: split obs into n equal chunks by magnitude."""
    n = len(observations)
    chunk = max(n // n_states, 1)
    sorted_obs = sorted(observations)
    means, stds, pi = [], [], []
    for k in range(n_states):
        chunk_k = sorted_obs[k * chunk: (k + 1) * chunk] or [0.0]
        mu = sum(chunk_k) / len(chunk_k)
        sd = math.sqrt(sum((x - mu) ** 2 for x in chunk_k) / max(len(chunk_k) - 1, 1))
        means.append(mu)
        stds.append(max(sd, 1e-4))
        pi.append(1.0 / n_states)
    A = [[1.0 / n_states] * n_states for _ in range(n_states)]
    return GaussianHMMParams(n_states, pi, A, means, stds)


def forward(params: GaussianHMMParams, obs: List[float]) -> Tuple[List[List[float]], float]:
    """Scaled forward algorithm. Returns alpha (T x K) and log-likelihood."""
    T, K = len(obs), params.n_states
    alpha = [[0.0] * K for _ in range(T)]
    scales = [0.0] * T

    for k in range(K):
        alpha[0][k] = params.pi[k] * _gauss_pdf(obs[0], params.means[k], params.stds[k])
    scales[0] = max(sum(alpha[0]), 1e-300)
    alpha[0] = [a / scales[0] for a in alpha[0]]

    for t in range(1, T):
        for k in range(K):
            alpha[t][k] = sum(alpha[t-1][j] * params.A[j][k] for j in range(K)) * \
                          _gauss_pdf(obs[t], params.means[k], params.stds[k])
        scales[t] = max(sum(alpha[t]), 1e-300)
        alpha[t] = [a / scales[t] for a in alpha[t]]

    log_lik = sum(math.log(s) for s in scales)
    return alpha, log_lik


def backward(params: GaussianHMMParams, obs: List[float], scales: List[float]) -> List[List[float]]:
    T, K = len(obs), params.n_states
    beta = [[0.0] * K for _ in range(T)]
    beta[T-1] = [1.0] * K
    for t in range(T - 2, -1, -1):
        for j in range(K):
            beta[t][j] = sum(params.A[j][k] * _gauss_pdf(obs[t+1], params.means[k], params.stds[k])
                              * beta[t+1][k] for k in range(K))
        s = max(scales[t+1], 1e-300)
        beta[t] = [b / s for b in beta[t]]
    return beta


def baum_welch(
    observations: List[float],
    n_states: int = 2,
    max_iter: int = 50,
    tol: float = 1e-4,
) -> GaussianHMMParams:
    """Fit Gaussian HMM via Baum-Welch EM."""
    params = init_params(n_states, observations)
    T, K = len(observations), n_states
    prev_ll = -math.inf

    for _ in range(max_iter):
        alpha, log_lik = forward(params, observations)
        scales = []
        s = sum(params.pi[k] * _gauss_pdf(observations[0], params.means[k], params.stds[k]) for k in range(K))
        scales.append(max(s, 1e-300))
        for t in range(1, T):
            s = sum(sum(alpha[t-1][j] * params.A[j][k] for j in range(K)) *
                    _gauss_pdf(observations[t], params.means[k], params.stds[k]) for k in range(K))
            scales.append(max(s, 1e-300))
        beta = backward(params, observations, scales)

        gamma = [[alpha[t][k] * beta[t][k] for k in range(K)] for t in range(T)]
        for t in range(T):
            s = max(sum(gamma[t]), 1e-300)
            gamma[t] = [g / s for g in gamma[t]]

        xi = [[[0.0]*K for _ in range(K)] for _ in range(T-1)]
        for t in range(T-1):
            denom = sum(alpha[t][j] * params.A[j][k] *
                        _gauss_pdf(observations[t+1], params.means[k], params.stds[k]) *
                        beta[t+1][k] for j in range(K) for k in range(K))
            denom = max(denom, 1e-300)
            for j in range(K):
                for k in range(K):
                    xi[t][j][k] = (alpha[t][j] * params.A[j][k] *
                                   _gauss_pdf(observations[t+1], params.means[k], params.stds[k]) *
                                   beta[t+1][k]) / denom

        # Update
        params.pi = gamma[0]
        for j in range(K):
            for k in range(K):
                num = sum(xi[t][j][k] for t in range(T-1))
                den = sum(sum(xi[t][j][l] for l in range(K)) for t in range(T-1))
                params.A[j][k] = num / max(den, 1e-300)
        for k in range(K):
            gk = [gamma[t][k] for t in range(T)]
            sg = max(sum(gk), 1e-300)
            params.means[k] = sum(gk[t] * observations[t] for t in range(T)) / sg
            params.stds[k] = max(math.sqrt(sum(gk[t] * (observations[t] - params.means[k])**2
                                               for t in range(T)) / sg), 1e-4)
        if abs(log_lik - prev_ll) < tol:
            break
        prev_ll = log_lik
    return params


def viterbi(params: GaussianHMMParams, obs: List[float]) -> List[int]:
    """Viterbi most-likely state sequence."""
    T, K = len(obs), params.n_states
    dp = [[-math.inf] * K for _ in range(T)]
    ptr = [[0] * K for _ in range(T)]
    for k in range(K):
        p = _gauss_pdf(obs[0], params.means[k], params.stds[k])
        dp[0][k] = math.log(max(params.pi[k], 1e-300)) + math.log(max(p, 1e-300))
    for t in range(1, T):
        for k in range(K):
            scores = [dp[t-1][j] + math.log(max(params.A[j][k], 1e-300)) for j in range(K)]
            best = max(range(K), key=lambda j: scores[j])
            p = _gauss_pdf(obs[t], params.means[k], params.stds[k])
            dp[t][k] = scores[best] + math.log(max(p, 1e-300))
            ptr[t][k] = best
    path = [0] * T
    path[T-1] = max(range(K), key=lambda k: dp[T-1][k])
    for t in range(T-2, -1, -1):
        path[t] = ptr[t+1][path[t+1]]
    return path


if __name__ == "__main__":
    import random
    rng = random.Random(3)
    obs = [rng.gauss(0.001, 0.008) if i % 100 < 70 else rng.gauss(-0.002, 0.02)
           for i in range(500)]
    params = baum_welch(obs, n_states=2, max_iter=30)
    print(f"State 0: mean={params.means[0]:.4f} std={params.stds[0]:.4f}")
    print(f"State 1: mean={params.means[1]:.4f} std={params.stds[1]:.4f}")
    path = viterbi(params, obs[-20:])
    print(f"Last 20 regimes: {path}")
