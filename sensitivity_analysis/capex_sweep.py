"""
Capex sensitivity: how the investment entry trigger S*_invest(t) shifts with capex.

Plot: entry trigger price vs capex, one line per reference time.

Run from repo root:
    python sensitivity_analysis/capex_sweep.py
"""

from __future__ import annotations

import math
import pathlib
import sys
from typing import Iterable

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REF_TIMES = [1.0, 5.0, 9.0]   # years at which to read the trigger

from bellman_equation.params import SwitchingParams
from bellman_equation.solver import SwitchingBellmanSolver, default_price_fn
from optimal_path.trigger_curves import extract_trigger_curves
from tree.oil_futures_curve import FuturesCurve
from tree.oil_tree_builder import OilTrinomialTreeBuilder
from tree.oil_tree_calibrator import OilTrinomialFuturesCalibrator


def run_capex_sweep(capex_values: Iterable[float], enable_plot: bool = True) -> None:
    n_steps = 80
    dt = 10.0 / n_steps   # 0.125
    a = 0.6
    sigma = 0.2
    futures_curve = FuturesCurve(lambda t: 72.0 + 12.0 * math.exp(-0.5 * t))  # backwardation: spot=84, floor=72 (breakeven=70)

    base_params = dict(
        dt=dt,
        discount_rate=0.08,
        production_rate=1.0,
        variable_cost=65.0,
        fixed_on_cost=5.0,
        switch_on_cost=5.0,
        salvage_multiplier=0.0,
        fixed_off_cost=2.0,
        switch_off_cost=1.0,
        allow_start_on=True,
    )

    # Build tree once (same for all capex values)
    builder = OilTrinomialTreeBuilder(n_steps=n_steps, a=a, sigma=sigma, dt=dt, jmax=9, jmin=-9)
    shifted_tree = OilTrinomialFuturesCalibrator(builder.build(), futures_curve).calibrate()

    print(f"{'capex':>8} | {'trigger_t0':>12} | {'first_invest_time':>18}")
    print("-" * 45)

    results = []
    for capex in capex_values:
        params = SwitchingParams(**base_params, capex=capex)
        solution = SwitchingBellmanSolver(
            tree=shifted_tree,
            params=params,
            price_fn=default_price_fn,
            terminal_on=lambda price: 0.0,
            terminal_off=lambda price: 0.0,
        ).solve()

        _, inv_trig, _, _ = extract_trigger_curves(shifted_tree, solution, dt)

        # Collect trigger at each reference time
        trigger_at: dict[float, float | None] = {}
        for t_ref in REF_TIMES:
            step = int(t_ref / dt)
            trigger_at[t_ref] = inv_trig[step] if step < len(inv_trig) else None

        row = "  ".join(
            f"t={t:.0f}:{trigger_at[t]:.1f}" if trigger_at[t] is not None else f"t={t:.0f}:N/A"
            for t in REF_TIMES
        )
        print(f"capex={capex:5.1f}  |  {row}")

        results.append((capex, trigger_at))

    if enable_plot:
        _plot_results(results)


def _plot_results(results) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import numpy as np
    except Exception:
        return
    try:
        fig, ax = plt.subplots(figsize=(7, 5))

        colors = cm.viridis(np.linspace(0.1, 0.85, len(REF_TIMES)))
        capex_vals = [r[0] for r in results]

        for t_ref, color in zip(REF_TIMES, colors):
            trig_vals = [r[1][t_ref] for r in results]
            valid = [(c, v) for c, v in zip(capex_vals, trig_vals) if v is not None]
            if not valid:
                continue
            cx, tv = zip(*valid)
            ax.plot(cx, tv, marker="o", linewidth=1.8, color=color,
                    label=f"Año {t_ref:.0f}")

        ax.set_xlabel("Capex")
        ax.set_ylabel(r"Precio de entrada óptimo  $S^*_{invest}$")
        ax.set_title(r"Precio de entrada vs Capex  ($\sigma$=0.20)")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(title="Año de referencia", fontsize=8)

        plt.tight_layout()
        plt.show()
    except Exception as exc:
        print(f"\nPlot skipped: {exc}")


if __name__ == "__main__":
    capex_values = [5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0]
    run_capex_sweep(capex_values)
