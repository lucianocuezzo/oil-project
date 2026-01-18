"""
Bellman demo: solves the ON/OFF switching problem on the calibrated oil tree.
"""

from __future__ import annotations

import math
import pathlib
import sys

# Ensure repo root is on sys.path when running as a script
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tree.oil_tree_builder import OilTrinomialTreeBuilder
from tree.oil_tree_calibrator import OilTrinomialFuturesCalibrator
from tree.oil_futures_curve import FuturesCurve

from bellman_equation.params import SwitchingParams
from bellman_equation.solver import SwitchingBellmanSolver, default_price_fn
from plots import BellmanValuePlotter, BellmanPolicyPlotter


def main() -> None:
    enable_plot = True  # set False to skip plotting
    block_plots = True  # True keeps windows open (blocking); False is non-blocking/auto-close when script exits

    n_steps = 6
    dt = 0.25  # quarters
    a = 0.6
    sigma = 0.35

    base_tree = OilTrinomialTreeBuilder(n_steps=n_steps, a=a, sigma=sigma, dt=dt).build()

    futures_curve = FuturesCurve(lambda t: 75 *math.exp(0.01 * t))
    shifted_tree = OilTrinomialFuturesCalibrator(base_tree, futures_curve).calibrate()

    params = SwitchingParams(
        dt=dt,
        discount_rate=0.05,
        production_rate=100.0,
        variable_cost=5.0,
        fixed_on_cost=5.0,
        fixed_off_cost=1.0,
        switch_on_cost=8.0,
        switch_off_cost=4.0,
        capex=11.0,
        allow_start_on=False,  # force start from Uninvested (no immediate ON start)
    )

    # Salvage value at horizon: e.g., shutdown and scrap payoff proportional to price.
    salvage_multiplier = 5.0  # tweak as needed; salvage = multiplier * spot price at T
    terminal_on = lambda price: salvage_multiplier * price
    terminal_off = lambda price: salvage_multiplier * price

    solver = SwitchingBellmanSolver(
        tree=shifted_tree,
        params=params,
        price_fn=default_price_fn,
        terminal_on=terminal_on,
        terminal_off=terminal_off,
    )

    solution = solver.solve()

    root_j = 0
    print("\n--- Project value assuming you start Uninvested (must decide to invest) ---")
    print(f"Start Uninvested: {solution.value_pre[0][root_j]:.4f}")
    print(f"Recommended first action: {solution.policy_pre[0][root_j]}")

    levels_to_show = min(2, n_steps - 1)
    print("\nFirst levels: t, j, P, V_on, V_off, policy_on, policy_off")
    for t in range(levels_to_show + 1):
        level = shifted_tree.levels[t]
        for j in sorted(level):
            price = default_price_fn(shifted_tree, t, j)
            pol_on = solution.policy_on[t][j]
            pol_off = solution.policy_off[t][j]
            print(
                f"t={t}, j={j:2d}, P={price:7.2f}, "
                f"V_on={solution.value_on[t][j]:8.4f}, "
                f"V_off={solution.value_off[t][j]:8.4f}, "
                f"policy_on={pol_on:10s}, policy_off={pol_off:10s}"
            )

    terminal_level = shifted_tree.n_steps
    print("\nTerminal payoffs (defaults are zero):")
    for j in sorted(shifted_tree.levels[terminal_level]):
        price = default_price_fn(shifted_tree, terminal_level, j)
        print(
            f"t={terminal_level}, j={j:2d}, P={price:7.2f}, "
            f"V_on={solution.value_on[terminal_level][j]:8.4f}, "
            f"V_off={solution.value_off[terminal_level][j]:8.4f}"
        )

    if enable_plot:
        try:
            import matplotlib.pyplot as plt

            BellmanValuePlotter(tree=shifted_tree, solution=solution, title="Value evolution per mode (time on x-axis)").plot()
            BellmanPolicyPlotter(tree=shifted_tree, solution=solution, price_fn=default_price_fn, title="Policy regions (color = action)").plot()
            plt.tight_layout()
            if block_plots:
                plt.show()  # blocking: windows stay until closed
            else:
                plt.show(block=False)
                plt.pause(0.1)  # allow GUI event loop to draw before script exits
        except Exception as exc:  # plot is optional
            print(f"\nPlot skipped: {exc}")


if __name__ == "__main__":
    main()
