"""
ML-Based Return Predictor (Gradient Boosting)
Day 15 â portfolio-risk-engine/ml/return_predictor.py

Pure-Python gradient boosting (decision stumps) for cross-sectional
return prediction. No external ML libraries required.
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Decision stump
# ---------------------------------------------------------------------------

@dataclass
class Stump:
    """Depth-1 regression tree (decision stump)."""
    feature_idx: int
    threshold: float
    left_value: float    # prediction when feature <= threshold
    right_value: float   # prediction when feature >  threshold


def _fit_stump(X: List[List[float]],
               residuals: List[float],
               feature_indices: List[int],
               n_thresholds: int = 10) -> Stump:
    """Find the best stump minimising MSE of residuals."""
    n_samples = len(X)
    best_loss = math.inf
    best_feat = 0
    best_thresh = 0.0
    best_left = 0.0
    best_right = 0.0

    for fi in feature_indices:
        vals = sorted(set(X[i][fi] for i in range(n_samples)))
        # sub-sample thresholds for speed
        step = max(1, len(vals) // n_thresholds)
        thresholds = vals[::step]

        for thresh in thresholds:
            left_res = [residuals[i] for i in range(n_samples) if X[i][fi] <= thresh]
            right_res = [residuals[i] for i in range(n_samples) if X[i][fi] > thresh]

            if not left_res or not right_res:
                continue

            lv = sum(left_res) / len(left_res)
            rv = sum(right_res) / len(right_res)

            loss = (sum((r - lv) ** 2 for r in left_res) +
                    sum((r - rv) ** 2 for r in right_res))

            if loss < best_loss:
                best_loss = loss
                best_feat = fi
                best_thresh = thresh
                best_left = lv
                best_right = rv

    return Stump(best_feat, best_thresh, best_left, best_right)


def _predict_stump(stump: Stump, x: List[float]) -> float:
    return stump.left_value if x[stump.feature_idx] <= stump.threshold else stump.right_value


# ---------------------------------------------------------------------------
# Gradient Boosting Regressor
# ---------------------------------------------------------------------------

@dataclass
class GBMParams:
    n_estimators: int = 100
    learning_rate: float = 0.05
    subsample: float = 0.8           # row sampling fraction
    max_features: float = 0.8        # column sampling fraction
    min_samples_leaf: int = 2
    random_seed: int = 42


@dataclass
class GBMModel:
    params: GBMParams
    stumps: List[Stump] = field(default_factory=list)
    initial_prediction: float = 0.0
    feature_importances_: Optional[List[float]] = None


def fit_gbm(X: List[List[float]],
            y: List[float],
            params: Optional[GBMParams] = None) -> GBMModel:
    """Fit gradient boosting model to (X, y)."""
    params = params or GBMParams()
    rng = random.Random(params.random_seed)

    n_samples = len(X)
    n_features = len(X[0]) if X else 0

    model = GBMModel(params=params)
    model.initial_prediction = sum(y) / n_samples if y else 0.0

    predictions = [model.initial_prediction] * n_samples
    feat_use = [0] * n_features

    n_sub = max(1, int(params.subsample * n_samples))
    n_feat_sub = max(1, int(params.max_features * n_features))

    for _ in range(params.n_estimators):
        residuals = [y[i] - predictions[i] for i in range(n_samples)]

        # Row & column subsampling
        row_idx = rng.sample(range(n_samples), n_sub)
        feat_idx = rng.sample(range(n_features), n_feat_sub)

        X_sub = [X[i] for i in row_idx]
        r_sub = [residuals[i] for i in row_idx]

        stump = _fit_stump(X_sub, r_sub, feat_idx)
        model.stumps.append(stump)

        # Update predictions
        for i in range(n_samples):
            predictions[i] += params.learning_rate * _predict_stump(stump, X[i])

        feat_use[stump.feature_idx] += 1

    # Normalised feature importances
    total = sum(feat_use) or 1
    model.feature_importances_ = [c / total for c in feat_use]
    return model


def predict_gbm(model: GBMModel, X: List[List[float]]) -> List[float]:
    """Predict returns for new samples."""
    preds = [model.initial_prediction] * len(X)
    for stump in model.stumps:
        for i, x in enumerate(X):
            preds[i] += model.params.learning_rate * _predict_stump(stump, x)
    return preds


# ---------------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------------

def build_features(returns: List[List[float]],
                   market_caps: Optional[List[List[float]]] = None
                   ) -> Tuple[List[List[float]], List[str]]:
    """
    Build cross-sectional features from return time series.

    returns: T Ã N matrix (T periods, N assets)
    market_caps: T Ã N matrix (optional)
    Returns (X, feature_names) where X is N Ã F
    """
    T, N = len(returns), len(returns[0])

    def _col(t_idx, asset_idx):
        return returns[t_idx][asset_idx]

    X: List[List[float]] = []
    for n in range(N):
        col = [returns[t][n] for t in range(T)]
        row_feats: List[float] = []

        # Momentum features
        for window in [5, 21, 63, 126, 252]:
            if T >= window:
                row_feats.append(sum(col[-window:]))
            else:
                row_feats.append(0.0)

        # Volatility
        for window in [21, 63]:
            if T >= window:
                sub = col[-window:]
                mu = sum(sub) / len(sub)
                var = sum((r - mu) ** 2 for r in sub) / len(sub)
                row_feats.append(math.sqrt(var * 252))
            else:
                row_feats.append(0.0)

        # Short-term reversal
        row_feats.append(sum(col[-5:]) if T >= 5 else 0.0)

        # Skewness (21-day)
        if T >= 21:
            sub = col[-21:]
            mu = sum(sub) / len(sub)
            sigma = math.sqrt(sum((r - mu) ** 2 for r in sub) / len(sub)) or 1e-10
            skew = sum(((r - mu) / sigma) ** 3 for r in sub) / len(sub)
            row_feats.append(skew)
        else:
            row_feats.append(0.0)

        X.append(row_feats)

    feature_names = (
        ["mom_5d", "mom_21d", "mom_63d", "mom_126d", "mom_252d",
         "vol_21d", "vol_63d", "reversal_5d", "skew_21d"]
    )
    return X, feature_names


# ---------------------------------------------------------------------------
# Cross-sectional ranking and portfolio construction
# ---------------------------------------------------------------------------

def cross_sectional_rank(scores: List[float]) -> List[float]:
    """Convert raw scores to rank (0â1 scaled)."""
    n = len(scores)
    order = sorted(range(n), key=lambda i: scores[i])
    ranks = [0.0] * n
    for rank, idx in enumerate(order):
        ranks[idx] = rank / (n - 1) if n > 1 else 0.5
    return ranks


def long_short_weights(scores: List[float],
                       top_pct: float = 0.2,
                       bottom_pct: float = 0.2) -> List[float]:
    """
    Construct long-short portfolio weights from cross-sectional scores.
    Long top_pct, short bottom_pct, equal weight within each bucket.
    """
    n = len(scores)
    n_long = max(1, int(n * top_pct))
    n_short = max(1, int(n * bottom_pct))

    ranked = sorted(range(n), key=lambda i: scores[i], reverse=True)
    long_set = set(ranked[:n_long])
    short_set = set(ranked[-n_short:])

    weights = [0.0] * n
    for i in long_set:
        weights[i] = 1.0 / n_long
    for i in short_set:
        weights[i] -= 1.0 / n_short
    return weights


def ic_score(predicted: List[float], actual: List[float]) -> float:
    """Pearson information coefficient between predicted and actual returns."""
    n = len(predicted)
    mp = sum(predicted) / n
    ma = sum(actual) / n
    cov = sum((predicted[i] - mp) * (actual[i] - ma) for i in range(n)) / n
    sp = math.sqrt(sum((p - mp) ** 2 for p in predicted) / n)
    sa = math.sqrt(sum((a - ma) ** 2 for a in actual) / n)
    return cov / (sp * sa + 1e-12)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    random.seed(0)
    T, N = 300, 20  # 300 daily returns, 20 assets

    # Synthetic return data
    rets = [[random.gauss(0, 0.01) for _ in range(N)] for _ in range(T)]

    # Build features from first 252 days, predict using last 48
    X_train, feat_names = build_features(rets[:252])
    X_test, _ = build_features(rets[252:])

    # Target: next 21-day forward returns (use last 21 of training window)
    y_train = [sum(rets[231 + t][n] for t in range(21)) for n in range(N)]

    params = GBMParams(n_estimators=50, learning_rate=0.1, subsample=0.7)
    model = fit_gbm(X_train, y_train, params)

    preds = predict_gbm(model, X_test)
    ranks = cross_sectional_rank(preds)
    weights = long_short_weights(preds)

    print("Feature importances:")
    for name, imp in zip(feat_names, model.feature_importances_):
        print(f"  {name:<15}: {imp:.3f}")

    print(f"\nTop 5 predicted assets (by score):")
    top5 = sorted(range(N), key=lambda i: preds[i], reverse=True)[:5]
    for i in top5:
        print(f"  Asset {i:2d}: pred={preds[i]:.4f}, weight={weights[i]:.3f}")
