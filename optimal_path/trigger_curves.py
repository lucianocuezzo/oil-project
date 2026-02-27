"""
Price trigger curves from the Bellman solution.

At each time step t, finds the boundary oil price that triggers each action:
  - Investment trigger  S*_invest(t) : min price where policy_pre = invest
  - Switch-off trigger  S*_off(t)   : max price where policy_on  = switch_off
  - Switch-on trigger   S*_on(t)    : min price where policy_off = switch_on

Shows how these thresholds evolve over time and shift with volatility.

Run from repo root:
    python optimal_path/trigger_curves.py
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
from bellman_equation.solver import SwitchingBellmanSolver, BellmanSolution, default_price_fn
from tree.hull_white_tree import ShiftedOilTrinomialTree
from tree.oil_futures_curve import FuturesCurve
from tree.oil_tree_builder import OilTrinomialTreeBuilder
from tree.oil_tree_calibrator import OilTrinomialFuturesCalibrator


def extract_trigger_curves(
    tree: ShiftedOilTrinomialTree,
    solution: BellmanSolution,
    dt: float,
) -> tuple[list[float], list, list, list]:
    """
    Returns (times, invest_triggers, switch_off_triggers, switch_on_triggers).
    Each trigger list has one entry per time step; None means no boundary found at that step.
    """
    n_steps = tree.n_steps
    times = [t * dt for t in range(n_steps)]

    invest_triggers: list = []
    switch_off_triggers: list = []
    switch_on_triggers: list = []

    for t in range(n_steps):
        prices = {j: default_price_fn(tree, t, j) for j in tree.levels[t]}

        # Investment trigger: minimum price where invest action appears
        invest_prices = [p for j, p in prices.items()
                         if solution.policy_pre[t].get(j) in ("invest_on", "invest_off")]
        invest_triggers.append(min(invest_prices) if invest_prices else None)

        # Switch-off trigger: maximum price where switch_off appears (above → stay on)
        off_prices = [p for j, p in prices.items()
                      if solution.policy_on[t].get(j) == "switch_off"]
        switch_off_triggers.append(max(off_prices) if off_prices else None)

        # Switch-on trigger: minimum price where switch_on appears (above → restart)
        on_prices = [p for j, p in prices.items()
                     if solution.policy_off[t].get(j) == "switch_on"]
        switch_on_triggers.append(min(on_prices) if on_prices else None)

    return times, invest_triggers, switch_off_triggers, switch_on_triggers


def run_trigger_analysis(sigmas: Iterable[float]) -> None:
    n_steps = 40
    dt = 0.25
    a = 0.6
    futures_curve = FuturesCurve(lambda t: 72.0 + 12.0 * math.exp(-0.5 * t))  # backwardation: spot=84, floor=72 (breakeven=70)

    common = dict(
        dt=dt,
        discount_rate=0.08,
        production_rate=1.0,
        variable_cost=65.0,
        fixed_on_cost=5.0,
        capex=20.0,
        switch_on_cost=5.0,
        salvage_multiplier=0.0,
    )
    bellman_extra = dict(fixed_off_cost=2.0, switch_off_cost=1.0, allow_start_on=True)

    # Forward curve for reference
    fwd_times = [t * dt for t in range(1, n_steps + 1)]
    fwd_prices = [futures_curve.value(t, dt) for t in range(1, n_steps + 1)]

    results = []
    for sigma in sigmas:
        builder = OilTrinomialTreeBuilder(n_steps=n_steps, a=a, sigma=sigma, dt=dt, jmax=6, jmin=-6)
        base_tree = builder.build()
        shifted_tree = OilTrinomialFuturesCalibrator(base_tree, futures_curve).calibrate()

        params = SwitchingParams(**common, **bellman_extra)
        solution = SwitchingBellmanSolver(
            tree=shifted_tree,
            params=params,
            price_fn=default_price_fn,
            terminal_on=lambda price: 0.0,
            terminal_off=lambda price: 0.0,
        ).solve()

        times, inv_trig, off_trig, on_trig = extract_trigger_curves(shifted_tree, solution, dt)
        results.append((sigma, times, inv_trig, off_trig, on_trig))

    _plot_triggers(results, fwd_times, fwd_prices)


def _plot_triggers(results, fwd_times, fwd_prices) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import numpy as np
    except Exception:
        return
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        colors = cm.viridis(np.linspace(0.15, 0.85, len(results)))

        # Forward curve on both panels
        ax1.plot(fwd_times, fwd_prices, color="black", linewidth=1.2,
                 linestyle="--", label="Forward curve")
        ax2.plot(fwd_times, fwd_prices, color="black", linewidth=1.2,
                 linestyle="--", label="Forward curve")

        for (sigma, times, inv_trig, off_trig, on_trig), color in zip(results, colors):
            label = f"σ={sigma:.2f}"

            # --- Left: investment trigger ---
            t_inv = [t for t, v in zip(times, inv_trig) if v is not None]
            v_inv = [v for v in inv_trig if v is not None]
            if t_inv:
                ax1.plot(t_inv, v_inv, marker=".", markersize=4, color=color, label=label)

            # --- Right: switch-off and switch-on triggers ---
            t_off = [t for t, v in zip(times, off_trig) if v is not None]
            v_off = [v for v in off_trig if v is not None]
            t_on  = [t for t, v in zip(times, on_trig)  if v is not None]
            v_on  = [v for v in on_trig  if v is not None]
            if t_off:
                ax2.plot(t_off, v_off, marker=".", markersize=4,
                         linestyle="-",  color=color, label=f"{label} shut-down")
            if t_on:
                ax2.plot(t_on,  v_on,  marker=".",  markersize=4,
                         linestyle="--", color=color, label=f"{label} restart")

        ax1.set_xlabel("Time (years)")
        ax1.set_ylabel("Oil price  S")
        ax1.set_title("Investment trigger  $S^*_{invest}(t)$")
        ax1.grid(True, linestyle="--", alpha=0.35)
        ax1.legend(fontsize=7)

        ax2.set_xlabel("Time (years)")
        ax2.set_ylabel("Oil price  S")
        ax2.set_title("Switching triggers  $S^*_{off}(t)$  and  $S^*_{on}(t)$")
        ax2.grid(True, linestyle="--", alpha=0.35)
        ax2.legend(fontsize=7)

        plt.suptitle("Price trigger curves", fontsize=11, y=1.01)
        plt.tight_layout()
        plt.show()
    except Exception as exc:
        print(f"\nPlot skipped: {exc}")


if __name__ == "__main__":
    sigmas = [0.1, 0.2, 0.35, 0.5]
    run_trigger_analysis(sigmas)
