"""
portfolio_day23_dynamic_risk_parity.py
Day 23: Dynamic Risk Parity — Equal Risk Contribution, vol targeting,
drawdown control overlay, regime-conditional risk parity.
Pure Python stdlib only.
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Covariance estimation
# ---------------------------------------------------------------------------

def sample_covariance(returns_matrix: list[list[float]]) -> list[list[float]]:
    """Returns the sample covariance matrix from a T x N returns matrix."""
    T = len(returns_matrix)
    N = len(returns_matrix[0]) if T > 0 else 0
    means = [sum(returns_matrix[t][n] for t in range(T)) / T for n in range(N)]

    cov = [[0.0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            cov[i][j] = sum(
                (returns_matrix[t][i] - means[i]) * (returns_matrix[t][j] - means[j])
                for t in range(T)
            ) / max(T - 1, 1)
    return cov


def rolling_volatility(returns: list[float], window: int = 20) -> list[float]:
    """Rolling annualized volatility (std * sqrt(252))."""
    vols = []
    for i in range(len(returns)):
        start = max(0, i - window + 1)
        sub = returns[start:i+1]
        if len(sub) < 2:
            vols.append(0.0)
            continue
        mean_r = sum(sub) / len(sub)
        var = sum((r - mean_r)**2 for r in sub) / (len(sub) - 1)
        vols.append(math.sqrt(var * 252))
    return vols


def ewma_covariance(returns_matrix: list[list[float]], lam: float = 0.94) -> list[list[float]]:
    """EWMA (RiskMetrics) covariance matrix."""
    T = len(returns_matrix)
    N = len(returns_matrix[0]) if T > 0 else 0

    cov = [[0.0] * N for _ in range(N)]
    # Initialize with outer product of first observation
    for i in range(N):
        for j in range(N):
            cov[i][j] = returns_matrix[0][i] * returns_matrix[0][j]

    for t in range(1, T):
        for i in range(N):
            for j in range(N):
                cov[i][j] = lam * cov[i][j] + (1 - lam) * returns_matrix[t][i] * returns_matrix[t][j]

    return cov


def portfolio_variance(weights: list[float], cov: list[list[float]]) -> float:
    """w^T Sigma w."""
    N = len(weights)
    total = 0.0
    for i in range(N):
        for j in range(N):
            total += weights[i] * cov[i][j] * weights[j]
    return total


def portfolio_volatility(weights: list[float], cov: list[list[float]]) -> float:
    return math.sqrt(max(portfolio_variance(weights, cov), 0.0))


# ---------------------------------------------------------------------------
# Risk Contribution
# ---------------------------------------------------------------------------

def marginal_risk_contribution(weights: list[float], cov: list[list[float]]) -> list[float]:
    """Marginal risk contribution: (Sigma * w)_i."""
    N = len(weights)
    mrc = []
    for i in range(N):
        mrc_i = sum(cov[i][j] * weights[j] for j in range(N))
        mrc.append(mrc_i)
    return mrc


def risk_contribution(weights: list[float], cov: list[list[float]]) -> list[float]:
    """Absolute risk contribution: w_i * (Sigma*w)_i."""
    mrc = marginal_risk_contribution(weights, cov)
    return [weights[i] * mrc[i] for i in range(len(weights))]


def percent_risk_contribution(weights: list[float], cov: list[list[float]]) -> list[float]:
    """Percentage risk contribution."""
    rc = risk_contribution(weights, cov)
    total = sum(rc)
    if total == 0:
        return [1.0 / len(weights)] * len(weights)
    return [r / total for r in rc]


# ---------------------------------------------------------------------------
# Equal Risk Contribution (ERC) Portfolio
# ---------------------------------------------------------------------------

def erc_portfolio(cov: list[list[float]], max_iter: int = 500, tol: float = 1e-8) -> list[float]:
    """
    Equal Risk Contribution (Maillard, Roncalli, Teiletche 2010).
    Find w such that RC_i = RC_j for all i, j.
    Uses iterative Newton-like approach.
    """
    N = len(cov)
    # Start with inverse-vol weights
    vols = [math.sqrt(max(cov[i][i], 1e-10)) for i in range(N)]
    inv_vols = [1.0 / v for v in vols]
    total_inv_vol = sum(inv_vols)
    weights = [iv / total_inv_vol for iv in inv_vols]

    for iteration in range(max_iter):
        rc = risk_contribution(weights, cov)
        port_vol = portfolio_volatility(weights, cov)

        if port_vol < 1e-10:
            break

        # Target: equal RC = port_vol / N
        target_rc = port_vol**2 / N  # each RC = sigma^2 / N

        # Update: w_i <- w_i * (target_RC / RC_i)^step
        step = 0.5
        new_weights = []
        for i in range(N):
            rc_i = max(rc[i], 1e-12)
            new_w = weights[i] * (target_rc / rc_i) ** step
            new_weights.append(new_w)

        # Normalize
        total = sum(new_weights)
        new_weights = [w / total for w in new_weights]

        # Convergence check
        diff = sum((new_weights[i] - weights[i])**2 for i in range(N))
        weights = new_weights

        if diff < tol**2:
            break

    return weights


# ---------------------------------------------------------------------------
# Inverse-volatility portfolio
# ---------------------------------------------------------------------------

def inverse_vol_portfolio(vols: list[float]) -> list[float]:
    """Weights proportional to 1/vol_i."""
    inv_vols = [1.0 / max(v, 1e-8) for v in vols]
    total = sum(inv_vols)
    return [iv / total for iv in inv_vols]


# ---------------------------------------------------------------------------
# Diversification Ratio
# ---------------------------------------------------------------------------

def diversification_ratio(weights: list[float], cov: list[list[float]]) -> float:
    """
    DR = (sum_i w_i * sigma_i) / sigma_portfolio.
    Higher = more diversified.
    """
    N = len(weights)
    vols = [math.sqrt(max(cov[i][i], 0)) for i in range(N)]
    weighted_avg_vol = sum(weights[i] * vols[i] for i in range(N))
    port_vol = portfolio_volatility(weights, cov)
    if port_vol < 1e-10:
        return 1.0
    return weighted_avg_vol / port_vol


# ---------------------------------------------------------------------------
# Volatility Targeting
# ---------------------------------------------------------------------------

def vol_target_leverage(
    current_vol: float,
    target_vol: float,
    max_leverage: float = 2.0,
    min_leverage: float = 0.1,
) -> float:
    """Scale leverage so portfolio vol matches target."""
    if current_vol < 1e-6:
        return 1.0
    leverage = target_vol / current_vol
    return max(min_leverage, min(max_leverage, leverage))


class VolTargetPortfolio:
    """ERC portfolio with volatility targeting."""

    def __init__(self, target_vol: float = 0.10, lookback: int = 60,
                 max_leverage: float = 1.5):
        self.target_vol = target_vol
        self.lookback = lookback
        self.max_leverage = max_leverage

    def compute_weights(
        self, returns_history: list[list[float]]
    ) -> tuple[list[float], float]:
        """Returns (scaled_weights, leverage)."""
        n_periods = min(len(returns_history), self.lookback)
        window = returns_history[-n_periods:]

        cov = ewma_covariance(window)
        base_weights = erc_portfolio(cov)

        port_vol = portfolio_volatility(base_weights, cov) * math.sqrt(252)
        leverage = vol_target_leverage(port_vol, self.target_vol, self.max_leverage)

        scaled = [w * leverage for w in base_weights]
        return scaled, leverage


# ---------------------------------------------------------------------------
# Drawdown Control Overlay
# ---------------------------------------------------------------------------

class DrawdownControl:
    """
    Reduce position size when drawdown exceeds threshold.
    Linear scale-down: at max_dd, go to min_exposure.
    """

    def __init__(
        self,
        max_dd_threshold: float = 0.10,
        min_exposure: float = 0.20,
        recovery_speed: float = 0.05,
    ):
        self.max_dd_threshold = max_dd_threshold
        self.min_exposure = min_exposure
        self.recovery_speed = recovery_speed
        self._peak = 1.0
        self._current_wealth = 1.0

    def update(self, portfolio_return: float) -> float:
        """Update wealth and return current exposure [min_exposure, 1.0]."""
        self._current_wealth *= (1 + portfolio_return)
        if self._current_wealth > self._peak:
            self._peak = self._current_wealth

        drawdown = 1 - self._current_wealth / self._peak

        if drawdown <= 0:
            return 1.0
        elif drawdown >= self.max_dd_threshold:
            return self.min_exposure
        else:
            # Linear interpolation
            frac = drawdown / self.max_dd_threshold
            return 1.0 - (1.0 - self.min_exposure) * frac

    @property
    def current_drawdown(self) -> float:
        return 1 - self._current_wealth / self._peak


# ---------------------------------------------------------------------------
# Regime-conditional Risk Parity
# ---------------------------------------------------------------------------

def detect_regime(returns: list[float], window: int = 20) -> str:
    """Simple regime: 'high_vol' or 'low_vol' based on rolling vol."""
    if len(returns) < window:
        return 'low_vol'
    recent = returns[-window:]
    mean_r = sum(recent) / len(recent)
    vol = math.sqrt(sum((r - mean_r)**2 for r in recent) / (len(recent) - 1)) * math.sqrt(252)
    # Also check trend
    trend = sum(recent[-5:]) / 5 - sum(recent[:5]) / 5

    if vol > 0.20:
        return 'high_vol'
    elif trend < -0.002:
        return 'stress'
    else:
        return 'low_vol'


class RegimeConditionalRiskParity:
    """
    Switch between risk parity configurations based on detected regime.
    """

    def __init__(
        self,
        vol_target_normal: float = 0.10,
        vol_target_stress: float = 0.05,
        lookback: int = 60,
    ):
        self.vol_target_normal = vol_target_normal
        self.vol_target_stress = vol_target_stress
        self.lookback = lookback

    def compute_weights(
        self,
        returns_history: list[list[float]],
        market_returns: list[float],
    ) -> dict:
        """Compute regime-appropriate weights."""
        regime = detect_regime(market_returns, window=min(20, len(market_returns)))

        n_periods = min(len(returns_history), self.lookback)
        window = returns_history[-n_periods:]
        cov = ewma_covariance(window)

        base_weights = erc_portfolio(cov)
        port_vol = portfolio_volatility(base_weights, cov) * math.sqrt(252)

        target = self.vol_target_stress if regime == 'stress' else self.vol_target_normal
        if regime == 'high_vol':
            target = self.vol_target_normal * 0.7

        leverage = vol_target_leverage(port_vol, target, max_leverage=1.5)
        scaled = [w * leverage for w in base_weights]

        return {
            'weights': scaled,
            'base_weights': base_weights,
            'leverage': leverage,
            'regime': regime,
            'target_vol': target,
            'estimated_vol': port_vol,
            'risk_contributions': percent_risk_contribution(base_weights, cov),
        }


# ---------------------------------------------------------------------------
# Rebalancing threshold trigger
# ---------------------------------------------------------------------------

def needs_rebalance(
    current_weights: list[float],
    target_weights: list[float],
    threshold: float = 0.05,
) -> bool:
    """True if any weight deviates from target by more than threshold."""
    for c, t in zip(current_weights, target_weights):
        if abs(c - t) > threshold:
            return True
    return False


def drift_weights(initial_weights: list[float], returns: list[float]) -> list[float]:
    """Compute drifted weights after a period of returns."""
    new_vals = [w * (1 + r) for w, r in zip(initial_weights, returns)]
    total = sum(new_vals)
    return [v / total for v in new_vals]


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 65)
    print("DAY 23: Dynamic Risk Parity")
    print("=" * 65)

    rng = random.Random(42)
    N_ASSETS = 5
    N_PERIODS = 252

    # Simulate correlated returns
    asset_vols = [0.15, 0.20, 0.12, 0.25, 0.18]

    returns_matrix = []
    for _ in range(N_PERIODS):
        row = [rng.gauss(0.0003, v / math.sqrt(252)) for v in asset_vols]
        returns_matrix.append(row)

    cov = ewma_covariance(returns_matrix, lam=0.94)

    # --- ERC vs inverse-vol vs equal-weight ---
    print("\n1. Portfolio Weights Comparison")
    erc_w = erc_portfolio(cov)
    vols = [math.sqrt(max(cov[i][i], 0)) * math.sqrt(252) for i in range(N_ASSETS)]
    inv_vol_w = inverse_vol_portfolio(vols)
    equal_w = [1.0 / N_ASSETS] * N_ASSETS

    print(f"{'Asset':>7} | {'ERC':>8} | {'InvVol':>8} | {'Equal':>8} | {'Asset Vol':>10}")
    print("-" * 55)
    for i in range(N_ASSETS):
        print(f"  A{i+1:>3}  | {erc_w[i]:>8.4f} | {inv_vol_w[i]:>8.4f} | {equal_w[i]:>8.4f} | {vols[i]:>10.4%}")

    print(f"\nERC Risk Contributions:")
    prc = percent_risk_contribution(erc_w, cov)
    for i, rc in enumerate(prc):
        print(f"  A{i+1}: {rc:.4%}  (target: {1/N_ASSETS:.4%})")

    erc_vol = portfolio_volatility(erc_w, cov) * math.sqrt(252)
    inv_vol_vol = portfolio_volatility(inv_vol_w, cov) * math.sqrt(252)
    equal_vol = portfolio_volatility(equal_w, cov) * math.sqrt(252)
    erc_dr = diversification_ratio(erc_w, cov)
    inv_dr = diversification_ratio(inv_vol_w, cov)

    print(f"\nPortfolio Statistics:")
    print(f"  {'':8} | {'Vol':>8} | {'Div Ratio':>10}")
    print("  " + "-" * 35)
    print(f"  {'ERC':8} | {erc_vol:>8.4%} | {erc_dr:>10.4f}")
    print(f"  {'InvVol':8} | {inv_vol_vol:>8.4%} | {inv_dr:>10.4f}")
    print(f"  {'Equal':8} | {equal_vol:>8.4%} | {diversification_ratio(equal_w, cov):>10.4f}")

    # --- Vol targeting ---
    print("\n2. Volatility Targeting")
    vt_portfolio = VolTargetPortfolio(target_vol=0.10, lookback=60, max_leverage=1.5)
    scaled_w, leverage = vt_portfolio.compute_weights(returns_matrix)
    scaled_vol = portfolio_volatility(scaled_w, cov) * math.sqrt(252)
    print(f"  Target vol : 10.00%")
    print(f"  Raw ERC vol: {erc_vol:.4%}")
    print(f"  Leverage   : {leverage:.4f}x")
    print(f"  Scaled vol : {scaled_vol:.4%}")

    # --- Drawdown control ---
    print("\n3. Drawdown Control Overlay")
    dd_ctrl = DrawdownControl(max_dd_threshold=0.10, min_exposure=0.20)

    # Simulate a market stress period
    stress_returns = [rng.gauss(-0.002, 0.02) for _ in range(50)] + \
                     [rng.gauss(0.001, 0.01) for _ in range(50)]

    exposures = []
    for r in stress_returns:
        exp = dd_ctrl.update(r)
        exposures.append(exp)

    min_exp = min(exposures)
    max_dd_reached = max(1 - min(exposures) / max(exposures[0], 1e-6), 0)
    print(f"  Min exposure during stress: {min_exp:.4%}")
    print(f"  Current drawdown: {dd_ctrl.current_drawdown:.4%}")
    print(f"  Final exposure  : {exposures[-1]:.4%}")

    # Print exposure over time (sampled)
    print(f"\n  Exposure timeline (sampled every 10 days):")
    for idx in range(0, 100, 10):
        bar = '#' * int(exposures[idx] * 20)
        print(f"  Day {idx:>3}: [{bar:<20}] {exposures[idx]:.2%}")

    # --- Regime-conditional ---
    print("\n4. Regime-Conditional Risk Parity")
    regime_model = RegimeConditionalRiskParity(
        vol_target_normal=0.10, vol_target_stress=0.05
    )
    market_rets = [row[0] for row in returns_matrix]  # use first asset as market proxy
    result = regime_model.compute_weights(returns_matrix, market_rets)

    print(f"  Detected regime: {result['regime']}")
    print(f"  Vol target     : {result['target_vol']:.2%}")
    print(f"  Estimated vol  : {result['estimated_vol']:.2%}")
    print(f"  Leverage       : {result['leverage']:.4f}x")
    print(f"  Final weights  :", [f"{w:.4f}" for w in result['weights']])
    print(f"  Risk contribs  :", [f"{rc:.2%}" for rc in result['risk_contributions']])

    # --- Rebalancing trigger ---
    print("\n5. Rebalancing Threshold Triggers")
    target_w = erc_w.copy()
    period_returns = [rng.gauss(0, 0.01) for _ in range(N_ASSETS)]
    drifted_w = drift_weights(target_w, period_returns)
    rebal_needed = needs_rebalance(drifted_w, target_w, threshold=0.05)

    print(f"  Target  : {[f'{w:.4f}' for w in target_w]}")
    print(f"  Drifted : {[f'{w:.4f}' for w in drifted_w]}")
    print(f"  Max drift: {max(abs(d-t) for d,t in zip(drifted_w,target_w)):.4%}")
    print(f"  Rebalance needed (5% threshold): {rebal_needed}")

    print("\n[Done] Day 23: Dynamic Risk Parity complete.")
