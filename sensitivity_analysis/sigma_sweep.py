"""
Sigma sensitivity: vary price volatility and compare tree NPV vs Bellman.

Run from repo root:
    python sensitivity_analysis/sigma_sweep.py
"""

from __future__ import annotations

import math
import pathlib
import sys
from typing import Iterable

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bellman_equation.params import SwitchingParams
from bellman_equation.solver import SwitchingBellmanSolver, default_price_fn
from npv_rule.calc import NPVParams, TreeNPVCalculator
from tree.oil_futures_curve import FuturesCurve
from tree.oil_tree_builder import OilTrinomialTreeBuilder
from tree.oil_tree_calibrator import OilTrinomialFuturesCalibrator


def run_sigma_sweep(sigmas: Iterable[float], enable_plot: bool = True) -> None:
    n_steps = 40
    dt = 0.25  # quarters
    a = 0.6
    futures_curve = FuturesCurve(lambda t: 75.0 * math.exp(0.02 * t))

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

    print(
        "sigma | NPV_tree_now | NPV_tree_best_wait | tree_earliest_step | Bellman_value | Bellman_first_action"
    )
    print(
        "----- | ------------- | ------------------ | ------------------ | ------------- | --------------------"
    )

    sig_list: list[float] = []
    npv_tree_list: list[float] = []
    npv_wait_list: list[float] = []
    bellman_list: list[float] = []

    for sigma in sigmas:
        builder = OilTrinomialTreeBuilder(n_steps=n_steps, a=a, sigma=sigma, dt=dt, jmax=4, jmin=-4)
        base_tree = builder.build()
        shifted_tree = OilTrinomialFuturesCalibrator(base_tree, futures_curve).calibrate()

        npv_params = NPVParams(**common)
        tree_calc = TreeNPVCalculator(shifted_tree, params=npv_params)
        tree_npvs = tree_calc.npv_schedule()
        tree_invest_step = TreeNPVCalculator.earliest_invest_step(tree_npvs)
        tree_invest_now = tree_calc.invest_now_npv()
        tree_best_wait = max(tree_npvs) if tree_npvs else float("-inf")

        bellman_params = SwitchingParams(
            **common,
            fixed_off_cost=1e6,   # effectively block OFF
            switch_off_cost=1e6,  # effectively block switching OFF
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

        tree_step_text = "none" if tree_invest_step is None else str(tree_invest_step)
        print(
            f"{sigma:5.2f} | {tree_invest_now:13.2f} | {tree_best_wait:18.2f} | {tree_step_text:18} | "
            f"{bellman_value:13.2f} | {bellman_action}"
        )

        sig_list.append(sigma)
        npv_tree_list.append(tree_invest_now)
        npv_wait_list.append(tree_best_wait)
        bellman_list.append(bellman_value)

    if enable_plot:
        _plot_results(sig_list, npv_tree_list, npv_wait_list, bellman_list)


def _plot_results(sigmas, npv_tree, npv_wait, bellman_vals) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    try:
        plt.figure(figsize=(8, 4))
        plt.plot(sigmas, npv_tree, marker="^", label="NPV (tree, invest now)")
        plt.plot(sigmas, npv_wait, marker="o", label="NPV (tree, best wait)")
        plt.plot(sigmas, bellman_vals, marker="s", label="Bellman value")
        plt.xlabel("sigma")
        plt.ylabel("value")
        plt.title("Sigma sensitivity: NPV vs Bellman")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.show()  # block to keep the window open
    except Exception as exc:
        print(f"\nPlot skipped: {exc}")


if __name__ == "__main__":
    sigmas = [0.0, 0.1, 0.2, 0.35, 0.5]
    run_sigma_sweep(sigmas, enable_plot=True)
