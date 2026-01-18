"""
Tree-based NPV demo (invest-now vs invest-later) plus Bellman value with switching.

Run from repo root:
    python run_project_valuation_demo.py
"""

from __future__ import annotations

import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bellman_equation.params import SwitchingParams
from bellman_equation.solver import SwitchingBellmanSolver, default_price_fn
from npv_rule.calc import NPVParams, TreeNPVCalculator
from tree.oil_futures_curve import FuturesCurve
from tree.oil_tree_builder import OilTrinomialTreeBuilder
from tree.oil_tree_calibrator import OilTrinomialFuturesCalibrator


def main() -> None:
    n_steps = 15
    dt = 0.25  # quarters
    a = 0.6
    sigma = 0.0  # set >0 for stochastic tree; 0 collapses to a deterministic path
    futures_curve = FuturesCurve(lambda t: 75.0 * math.exp(0.05 * t))  # adjust as needed

    builder = OilTrinomialTreeBuilder(n_steps=n_steps, a=a, sigma=sigma, dt=dt, jmax=4, jmin=-4)
    base_tree = builder.build()
    shifted_tree = OilTrinomialFuturesCalibrator(base_tree, futures_curve).calibrate()

    common = dict(
        dt=dt,
        discount_rate=0.08,
        production_rate=1.0,
        variable_cost=45.0,
        fixed_on_cost=5.0,
        capex=20.0,
        switch_on_cost=8.0,
        salvage_multiplier=0.0,
    )

    # Tree NPV (no flexibility beyond invest timing)
    npv_params = NPVParams(**common)
    calc = TreeNPVCalculator(shifted_tree, params=npv_params)
    npvs = calc.npv_schedule()  # invest at step k (wait k*dt, then run ON)
    invest_step = TreeNPVCalculator.earliest_invest_step(npvs)
    invest_now = calc.invest_now_npv()

    # Bellman with switching/investment flexibility
    bellman_params = SwitchingParams(
        **common,
        fixed_off_cost=1.0,
        switch_off_cost=4.0,
        allow_start_on=True,
    )
    bellman_solver = SwitchingBellmanSolver(
        tree=shifted_tree,
        params=bellman_params,
        price_fn=default_price_fn,
        terminal_on=lambda price: 0.0,
        terminal_off=lambda price: 0.0,
    )
    bellman_solution = bellman_solver.solve()
    bellman_value = bellman_solution.value_pre[0][0]
    bellman_action = bellman_solution.policy_pre[0][0]

    _print_results(dt, npvs, invest_step, invest_now, bellman_value, bellman_action)


def _print_results(dt, npvs, invest_step, invest_now, bellman_value, bellman_action):
    print("=== NPV (tree expected, invest step vs now) ===")
    print(f" Invest now (t=0): NPV = {invest_now:.4f} -> {'invest' if invest_now >= 0 else 'do not invest'}")
    print(" NPV by invest step (wait k*dt then invest and run ON):")
    for k, v in enumerate(npvs):
        tag = " <= earliest >=0" if invest_step is not None and k == invest_step else ""
        print(f"   step {k}: NPV = {v:.4f}{tag}")
    if invest_step is None:
        print(" Earliest >=0: none within horizon")
    else:
        print(f" Earliest >=0: step {invest_step} (t={invest_step*dt:.2f}y)")

    print("\n=== Bellman (with switching) ===")
    print(f" Start uninvested value: {bellman_value:.4f}")
    print(f" Recommended first action: {bellman_action}")
    print(" (Policies by state are in bellman_solution if needed)")


if __name__ == "__main__":
    main()
