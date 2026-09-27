"""
Scenario-Based Stress Testing
Day 15 â portfolio-risk-engine/risk/stress_testing.py

Applies historical and hypothetical stress scenarios to a portfolio
and computes P&L impact, VaR contribution, and worst-case drawdown.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class StressScenario:
    """A single stress scenario with asset-level shocks."""
    name: str
    shocks: Dict[str, float]          # asset â return shock (e.g., -0.30)
    description: str = ""


@dataclass
class PortfolioPosition:
    """A position in the portfolio."""
    asset: str
    weight: float                     # portfolio weight (sum to 1)
    notional: float = 1.0             # notional value in portfolio currency


@dataclass
class StressResult:
    """P&L breakdown for one scenario."""
    scenario_name: str
    total_pnl: float                  # total portfolio P&L
    asset_pnl: Dict[str, float]       # per-asset P&L contribution
    worst_asset: str
    best_asset: str
    loss_probability: Optional[float] = None   # if historical frequency known


@dataclass
class StressReport:
    """Full stress-test report across multiple scenarios."""
    scenarios: List[StressResult]
    worst_scenario: str
    expected_tail_loss: float         # average of bottom-quartile scenario losses
    scenario_var: float               # 95th-percentile loss across scenarios


# ---------------------------------------------------------------------------
# Pre-defined historical scenarios
# ---------------------------------------------------------------------------

HISTORICAL_SCENARIOS: List[StressScenario] = [
    StressScenario(
        name="GFC_2008",
        shocks={
            "US_EQ": -0.37, "INTL_EQ": -0.43, "EM_EQ": -0.53,
            "BONDS": 0.06,  "CREDIT": -0.25, "COMMOD": -0.35,
            "REITS": -0.40, "USD": 0.08,
        },
        description="Global Financial Crisis peak-to-trough (2008)",
    ),
    StressScenario(
        name="COVID_2020",
        shocks={
            "US_EQ": -0.34, "INTL_EQ": -0.33, "EM_EQ": -0.29,
            "BONDS": 0.03,  "CREDIT": -0.15, "COMMOD": -0.40,
            "REITS": -0.42, "USD": 0.04,
        },
        description="COVID-19 crash (FebâMar 2020)",
    ),
    StressScenario(
        name="DOTCOM_2000",
        shocks={
            "US_EQ": -0.49, "INTL_EQ": -0.44, "EM_EQ": -0.30,
            "BONDS": 0.12,  "CREDIT": -0.05, "COMMOD": 0.02,
            "REITS": 0.03,  "USD": 0.05,
        },
        description="Dot-com bust (2000â2002)",
    ),
    StressScenario(
        name="RATE_SHOCK_1994",
        shocks={
            "US_EQ": -0.02, "INTL_EQ": -0.05, "EM_EQ": -0.18,
            "BONDS": -0.10, "CREDIT": -0.07, "COMMOD": 0.05,
            "REITS": -0.12, "USD": -0.03,
        },
        description="Fed rate hike shock (1994)",
    ),
    StressScenario(
        name="RUSSIA_1998",
        shocks={
            "US_EQ": -0.15, "INTL_EQ": -0.20, "EM_EQ": -0.35,
            "BONDS": 0.08,  "CREDIT": -0.20, "COMMOD": -0.15,
            "REITS": -0.10, "USD": 0.05,
        },
        description="Russian default / LTCM crisis (1998)",
    ),
]


# ---------------------------------------------------------------------------
# Hypothetical scenario constructors
# ---------------------------------------------------------------------------

def equity_crash_scenario(magnitude: float = 0.30,
                           bond_flight: float = 0.05) -> StressScenario:
    """Generic equity crash with flight-to-quality bond rally."""
    return StressScenario(
        name=f"EQUITY_CRASH_{int(magnitude*100)}pct",
        shocks={
            "US_EQ": -magnitude,
            "INTL_EQ": -magnitude * 1.1,
            "EM_EQ": -magnitude * 1.4,
            "BONDS": bond_flight,
            "CREDIT": -magnitude * 0.6,
            "COMMOD": -magnitude * 0.8,
            "REITS": -magnitude * 1.0,
            "USD": magnitude * 0.2,
        },
        description=f"Hypothetical {int(magnitude*100)}% equity drawdown",
    )


def rate_spike_scenario(rise_bps: int = 200) -> StressScenario:
    """Sudden interest rate spike."""
    bond_shock = -(rise_bps / 100) * 7.0 / 100  # approx 7yr duration
    return StressScenario(
        name=f"RATE_SPIKE_{rise_bps}bps",
        shocks={
            "US_EQ": -0.05 - (rise_bps / 10000),
            "INTL_EQ": -0.04,
            "EM_EQ": -0.12,
            "BONDS": bond_shock,
            "CREDIT": bond_shock * 0.6,
            "COMMOD": 0.03,
            "REITS": -0.15,
            "USD": 0.04,
        },
        description=f"Interest rates rise by {rise_bps} bps",
    )


def inflation_shock_scenario(inflation_pct: float = 5.0) -> StressScenario:
    """Inflationary surprise scenario."""
    return StressScenario(
        name=f"INFLATION_SHOCK_{int(inflation_pct)}pct",
        shocks={
            "US_EQ": -0.10,
            "INTL_EQ": -0.08,
            "EM_EQ": -0.06,
            "BONDS": -0.12,
            "CREDIT": -0.08,
            "COMMOD": 0.20,
            "REITS": 0.05,
            "USD": -0.05,
        },
        description=f"Unexpected {inflation_pct}% inflation spike",
    )


# ---------------------------------------------------------------------------
# Stress test engine
# ---------------------------------------------------------------------------

def run_scenario(positions: List[PortfolioPosition],
                 scenario: StressScenario,
                 total_portfolio_value: float = 1.0) -> StressResult:
    """Compute portfolio P&L under a single stress scenario."""
    asset_pnl: Dict[str, float] = {}
    total_pnl = 0.0

    for pos in positions:
        shock = scenario.shocks.get(pos.asset, 0.0)
        pnl = pos.weight * shock * total_portfolio_value
        asset_pnl[pos.asset] = pnl
        total_pnl += pnl

    worst = min(asset_pnl, key=lambda a: asset_pnl[a]) if asset_pnl else ""
    best = max(asset_pnl, key=lambda a: asset_pnl[a]) if asset_pnl else ""

    return StressResult(
        scenario_name=scenario.name,
        total_pnl=total_pnl,
        asset_pnl=asset_pnl,
        worst_asset=worst,
        best_asset=best,
    )


def run_stress_report(positions: List[PortfolioPosition],
                      scenarios: List[StressScenario],
                      total_portfolio_value: float = 1.0) -> StressReport:
    """Run all scenarios and aggregate into a report."""
    results = [
        run_scenario(positions, s, total_portfolio_value) for s in scenarios
    ]

    # Sort by total P&L ascending (worst first)
    sorted_results = sorted(results, key=lambda r: r.total_pnl)
    worst_scenario = sorted_results[0].scenario_name

    losses = [r.total_pnl for r in sorted_results if r.total_pnl < 0]
    if losses:
        n_tail = max(1, len(losses) // 4)
        expected_tail_loss = sum(sorted(losses)[:n_tail]) / n_tail
    else:
        expected_tail_loss = 0.0

    # Scenario VaR at 95th pct
    all_pnls = sorted([r.total_pnl for r in results])
    var_idx = int(0.05 * len(all_pnls))
    scenario_var = -all_pnls[var_idx] if all_pnls else 0.0

    return StressReport(
        scenarios=results,
        worst_scenario=worst_scenario,
        expected_tail_loss=expected_tail_loss,
        scenario_var=scenario_var,
    )


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------

def factor_sensitivity(positions: List[PortfolioPosition],
                        factor_shocks: Dict[str, float],
                        factor_loadings: Dict[str, Dict[str, float]],
                        total_value: float = 1.0) -> Dict[str, float]:
    """
    Compute portfolio sensitivity to macro factors via factor loadings.

    factor_loadings: {asset: {factor: beta}}
    factor_shocks:   {factor: shock}
    Returns per-asset P&L.
    """
    result: Dict[str, float] = {}
    for pos in positions:
        loadings = factor_loadings.get(pos.asset, {})
        asset_shock = sum(loadings.get(f, 0.0) * s
                          for f, s in factor_shocks.items())
        result[pos.asset] = pos.weight * asset_shock * total_value
    return result


def marginal_stress_contribution(positions: List[PortfolioPosition],
                                  scenario: StressScenario) -> Dict[str, float]:
    """Fraction of total scenario loss attributable to each position."""
    total = sum(
        pos.weight * scenario.shocks.get(pos.asset, 0.0)
        for pos in positions
    )
    if abs(total) < 1e-12:
        return {pos.asset: 0.0 for pos in positions}
    return {
        pos.asset: (pos.weight * scenario.shocks.get(pos.asset, 0.0)) / total
        for pos in positions
    }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    positions = [
        PortfolioPosition("US_EQ",   0.35),
        PortfolioPosition("INTL_EQ", 0.20),
        PortfolioPosition("EM_EQ",   0.10),
        PortfolioPosition("BONDS",   0.25),
        PortfolioPosition("CREDIT",  0.05),
        PortfolioPosition("COMMOD",  0.05),
    ]
    portfolio_value = 1_000_000.0

    scenarios = HISTORICAL_SCENARIOS + [
        equity_crash_scenario(0.40),
        rate_spike_scenario(300),
        inflation_shock_scenario(6.0),
    ]

    report = run_stress_report(positions, scenarios, portfolio_value)

    print(f"{'Scenario':<25} {'P&L':>12} {'Worst Asset':<12}")
    print("-" * 52)
    for r in sorted(report.scenarios, key=lambda x: x.total_pnl):
        print(f"{r.scenario_name:<25} ${r.total_pnl:>11,.0f}  {r.worst_asset}")

    print(f"\nWorst scenario: {report.worst_scenario}")
    print(f"Expected tail loss: ${report.expected_tail_loss:,.0f}")
    print(f"Scenario VaR (95%): ${report.scenario_var:,.0f}")
