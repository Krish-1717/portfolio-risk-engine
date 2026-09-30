"""
run_demos.py — Portfolio Risk Engine
Runs all Day 20-30 demo modules in sequence.
Pure Python stdlib only — no external dependencies.
"""
import subprocess
import sys
import os
import time

MODULES = [
    "quant_code/portfolio_day20_pca_risk_model.py",
    "quant_code/portfolio_day21_regime_optimizer.py",
    "quant_code/portfolio_day22_risk_summary.py",
    "quant_code/portfolio_day23_dynamic_risk_parity.py",
    "quant_code/portfolio_day24_transaction_costs.py",
    "quant_code/portfolio_day25_factor_models.py",
    "quant_code/portfolio_day26_stress_testing.py",
    "quant_code/portfolio_day27_black_litterman.py",
    "quant_code/portfolio_day28_dynamic_risk.py",
    "quant_code/portfolio_day29_factor_attribution.py",
    "quant_code/portfolio_day30_final_report.py",
    "quant_code/microstructure_vpin.py",
    "quant_code/microstructure_liquidity.py",
    "quant_code/portfolio_microstructure_impact.py",
    "quant_code/portfolio_microstructure_liquidity.py",
]

def run_module(path: str) -> bool:
    name = os.path.basename(path)
    print(f"\n{'='*60}")
    print(f"  Running: {name}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run([sys.executable, path], capture_output=False)
    elapsed = time.time() - t0
    if result.returncode == 0:
        print(f"\n  ✓ {name} completed in {elapsed:.1f}s")
        return True
    else:
        print(f"\n  ✗ {name} failed (exit code {result.returncode})")
        return False

if __name__ == "__main__":
    print("Portfolio Risk Engine — Day 20-30 Demo Runner")
    print("=" * 60)

    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)

    passed, failed = 0, 0
    for module in MODULES:
        if run_module(module):
            passed += 1
        else:
            failed += 1

    print(f"\n{'='*60}")
    print(f"  Results: {passed}/{len(MODULES)} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
