"""
risk/stress_tester.py -- portfolio stress testing for portfolio-risk-engine
Day 11: historical & hypothetical scenario shocks with P&L attribution.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# Position

@dataclass
class Position:
    ticker:   str
    quantity: float
    price:    float
    beta:     float = 1.0
    sector:   str  = "unknown"

    @property
    def market_value(self) -> float:
        return self.quantity * self.price

    def shocked_pnl(self, pct_shock: float) -> float:
        return self.market_value * pct_shock


# Scenario

@dataclass
class ScenarioShock:
    name:         str
    market_shock: float
    ticker_overrides: Dict[str, float] = field(default_factory=dict)
    sector_overrides: Dict[str, float] = field(default_factory=dict)

    def shock_for(self, pos: Position) -> float:
        if pos.ticker in self.ticker_overrides:
            return self.ticker_overrides[pos.ticker]
        if pos.sector in self.sector_overrides:
            return self.sector_overrides[pos.sector]
        return self.market_shock * pos.beta


BUILTIN_SCENARIOS: List[ScenarioShock] = [
    ScenarioShock(
        name="2008 GFC Peak",
        market_shock=-0.45,
        sector_overrides={"financials": -0.65, "real_estate": -0.60},
    ),
    ScenarioShock(
        name="COVID Crash 2020",
        market_shock=-0.34,
        sector_overrides={"energy": -0.55, "travel": -0.60, "technology": -0.20},
    ),
    ScenarioShock(
        name="Rate Spike +200bp",
        market_shock=-0.12,
        sector_overrides={"utilities": -0.18, "real_estate": -0.20, "financials": 0.05},
    ),
    ScenarioShock(
        name="Tech Bubble Burst",
        market_shock=-0.20,
        sector_overrides={"technology": -0.65, "telecommunications": -0.40},
    ),
    ScenarioShock(name="Mild Correction -10%", market_shock=-0.10),
    ScenarioShock(name="Bull Run +20%",        market_shock=0.20),
]


# Results

@dataclass
class PositionPnL:
    ticker:       str
    market_value: float
    shock_pct:    float
    pnl:          float
    sector:       str

@dataclass
class ScenarioResult:
    scenario_name:  str
    total_pnl:      float
    total_mv:       float
    pnl_pct:        float
    positions:      List[PositionPnL] = field(default_factory=list)
    sector_pnl:     Dict[str, float]  = field(default_factory=dict)

    @property
    def top_losers(self) -> List[PositionPnL]:
        return sorted(self.positions, key=lambda p: p.pnl)[:5]

    @property
    def top_gainers(self) -> List[PositionPnL]:
        return sorted(self.positions, key=lambda p: p.pnl, reverse=True)[:5]

    def print_summary(self) -> None:
        bar = "=" * 60
        sign = "+" if self.total_pnl >= 0 else ""
        print("
" + bar)
        print(f"Scenario : {self.scenario_name}")
        print(f"Portfolio NAV: {self.total_mv:,.0f}  P&L: {sign}{self.total_pnl:,.0f} ({sign}{self.pnl_pct:.2f}%)")
        print("
Sector attribution:")
        for sec, pnl in sorted(self.sector_pnl.items(), key=lambda x: x[1]):
            ssign = "+" if pnl >= 0 else ""
            print(f"  {sec:<20} {ssign}{pnl:,.0f}")
        print("
Top 3 losers:")
        for p in self.top_losers[:3]:
            print(f"  {p.ticker:<8} shock={p.shock_pct*100:+.1f}%  PnL={p.pnl:,.0f}")
        print(bar)


@dataclass
class StressReport:
    positions: List[Position]
    results:   List[ScenarioResult] = field(default_factory=list)

    def worst_scenario(self) -> Optional[ScenarioResult]:
        return min(self.results, key=lambda r: r.total_pnl) if self.results else None

    def best_scenario(self) -> Optional[ScenarioResult]:
        return max(self.results, key=lambda r: r.total_pnl) if self.results else None

    def pnl_range(self) -> Tuple[float, float]:
        pnls = [r.total_pnl for r in self.results]
        return (min(pnls), max(pnls)) if pnls else (0.0, 0.0)

    def var_estimate(self, confidence: float = 0.95) -> float:
        pnls = sorted(r.total_pnl for r in self.results)
        idx  = max(0, int((1 - confidence) * len(pnls)) - 1)
        return pnls[idx] if pnls else 0.0

    def print_report(self) -> None:
        total_mv = sum(p.market_value for p in self.positions)
        print("
" + "#" * 60)
        print("  STRESS TEST REPORT")
        print(f"  Portfolio NAV: {total_mv:,.0f}  |  Positions: {len(self.positions)}")
        lo, hi = self.pnl_range()
        print(f"  P&L range: {lo:,.0f} .. {hi:,.0f}")
        var95 = self.var_estimate(0.95)
        print(f"  Scenario VaR(95%): {var95:,.0f}  ({var95/total_mv*100:.2f}%)")
        print("#" * 60)
        for r in self.results:
            r.print_summary()
        worst = self.worst_scenario()
        best  = self.best_scenario()
        print(f"
Worst: {worst.scenario_name if worst else 'n/a'}")
        print(f"Best : {best.scenario_name  if best  else 'n/a'}")


# Stress Tester

class StressTester:
    def __init__(self, positions: List[Position]):
        self.positions = positions
        self._total_mv = sum(p.market_value for p in positions)

    def run_scenario(self, scenario: ScenarioShock) -> ScenarioResult:
        pos_pnls: List[PositionPnL] = []
        sector_pnl: Dict[str, float] = {}
        total_pnl = 0.0
        for pos in self.positions:
            shock = scenario.shock_for(pos)
            pnl   = pos.shocked_pnl(shock)
            pos_pnls.append(PositionPnL(
                ticker=pos.ticker, market_value=pos.market_value,
                shock_pct=shock, pnl=pnl, sector=pos.sector,
            ))
            sector_pnl[pos.sector] = sector_pnl.get(pos.sector, 0.0) + pnl
            total_pnl += pnl
        pnl_pct = total_pnl / self._total_mv * 100 if self._total_mv else 0.0
        return ScenarioResult(
            scenario_name=scenario.name,
            total_pnl=total_pnl, total_mv=self._total_mv,
            pnl_pct=pnl_pct, positions=pos_pnls, sector_pnl=sector_pnl,
        )

    def run_all(self, scenarios=None) -> StressReport:
        if scenarios is None:
            scenarios = BUILTIN_SCENARIOS
        report = StressReport(positions=self.positions)
        for sc in scenarios:
            report.results.append(self.run_scenario(sc))
        return report


# CLI demo

if __name__ == "__main__":
    portfolio = [
        Position("AAPL",  500,  185.0, beta=1.15, sector="technology"),
        Position("MSFT",  300,  370.0, beta=0.90, sector="technology"),
        Position("JPM",   400,  195.0, beta=1.10, sector="financials"),
        Position("XOM",   200,   98.0, beta=0.75, sector="energy"),
        Position("PLD",   150,  120.0, beta=1.05, sector="real_estate"),
        Position("JNJ",   250,  160.0, beta=0.55, sector="healthcare"),
        Position("AMZN",  100, 3300.0, beta=1.25, sector="technology"),
        Position("BRK.B", 200,  340.0, beta=0.65, sector="financials"),
    ]
    tester = StressTester(portfolio)
    report = tester.run_all()
    report.print_report()
