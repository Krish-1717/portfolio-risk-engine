# Portfolio Risk Engine

A 30-day systematic build of a production-grade portfolio risk system — covering factor models, stress testing, Black-Litterman, market microstructure, and dynamic risk management. Pure Python stdlib only.

## What's Inside

End-to-end portfolio analytics from basic VaR all the way to Barra-style factor attribution, Kelly sizing, and market impact modelling.

### Days 1–19 — Core Risk Framework

| Module | Topic |
|--------|-------|
| `risk.py` | Portfolio VaR, CVaR, Monte Carlo risk |
| `backtesting/` | Event-driven backtester, performance attribution |
| `derivatives/` | Futures, swaps, forward pricing |
| `execution/` | TWAP/VWAP algorithms, slippage models |
| `factor_models/` | Fama-French 3-factor OLS |
| `factors/` | Momentum, value, quality signal construction |
| `fixed_income/` | Duration, convexity, yield curve analytics |
| `microstructure/` | Bid-ask spread, market impact primitives |
| `ml/` | Random forest alpha signals |
| `optimization/` | Mean-variance optimisation, efficient frontier |
| `overfitting/` | Walk-forward validation, deflated Sharpe |
| `portfolio/` | Portfolio construction, rebalancing |
| `regimes/` | HMM regime detection, volatility clustering |
| `reporting/` | Tear sheet generation |
| `yield_curve/` | Nelson-Siegel, bootstrapping |

### Days 20–30 — Advanced Risk (`quant_code/`)

| File | Topic |
|------|-------|
| `portfolio_day20_pca_risk_model.py` | PCA risk decomposition, factor variance explained |
| `portfolio_day21_regime_optimizer.py` | Regime-conditional MVO, HMM state detection |
| `portfolio_day22_risk_summary.py` | Comprehensive risk dashboard, drawdown analytics |
| `portfolio_day23_dynamic_risk_parity.py` | ERC weights, cyclical descent, turnover control |
| `portfolio_day24_transaction_costs.py` | Market impact models, optimal rebalance frequency |
| `portfolio_day25_factor_models.py` | FF3 + momentum OLS, crowding score, smart-beta |
| `portfolio_day26_stress_testing.py` | Student-t fat-tail MC, Gaussian copula, butterfly scenarios |
| `portfolio_day27_black_litterman.py` | Reverse optimisation, view matrix P/Q/Ω, posterior mu/Sigma |
| `portfolio_day28_dynamic_risk.py` | EWMA covariance (λ=0.94), multi-asset Kelly, fractional Kelly |
| `portfolio_day29_factor_attribution.py` | Barra B·F_cov·B' + D, Brinson-Hood-Beebower attribution |
| `portfolio_day30_final_report.py` | Rolling Sharpe/Sortino/Calmar, regime-conditional metrics, ASCII dashboard |
| `microstructure_vpin.py` | VPIN toxicity, volume bucketing, tick rule, Lee-Ready, PIN estimation |
| `microstructure_liquidity.py` | Amihud, Roll spread, Corwin-Schultz, Kyle's lambda, Almgren-Chriss |
| `portfolio_microstructure_impact.py` | Market impact integration into portfolio construction |
| `portfolio_microstructure_liquidity.py` | Liquidity-adjusted portfolio optimisation |

## Key Concepts

- **Factor risk models** — PCA decomposition, Fama-French 3-factor, Barra-style attribution
- **Black-Litterman** — Reverse optimisation, prior/posterior blending, confidence-weighted views
- **Dynamic risk parity** — Equal risk contribution, cyclical descent, EWMA covariance
- **Kelly criterion** — Multi-asset fractional Kelly, turnover-constrained rebalancing
- **Stress testing** — Student-t fat tails, Gaussian copula, historical & hypothetical scenarios
- **Market microstructure** — VPIN, PIN, Kyle's lambda, Almgren-Chriss optimal execution
- **Regime detection** — HMM, volatility clustering, regime-conditional allocation
- **Performance attribution** — Brinson-Hood-Beebower, rolling Sharpe/Sortino/Calmar

## Running the Code

```bash
# Run any module directly
python quant_code/portfolio_day30_final_report.py

# Run all Day 20-30 demos
python run_demos.py
```

## Requirements

Pure Python 3.8+ standard library only — no external packages required.

```
python >= 3.8
# No pip install needed
```

## Architecture

```
portfolio-risk-engine/
├── risk.py                          # Entry point: VaR, CVaR, Monte Carlo
├── backtesting/                     # Event-driven backtester
├── optimization/                    # MVO, efficient frontier
├── factor_models/                   # Fama-French, factor construction
├── regimes/                         # HMM regime detection
├── execution/                       # TWAP/VWAP, slippage
├── microstructure/                  # Bid-ask, market impact
├── fixed_income/                    # Duration, yield curve
├── reporting/                       # Tear sheet generation
├── quant_code/                      # Days 20-30 advanced modules
└── run_demos.py                     # Demo runner for all modules
```
