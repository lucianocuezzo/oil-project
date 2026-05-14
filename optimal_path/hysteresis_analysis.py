"""
Hysteresis analysis: switching triggers vs friction costs.

Shows the band [S*_off, S*_on] between the shut-down and restart triggers.
When friction costs = 0, the band collapses to a single threshold: no hysteresis.

fixed_on_cost = fixed_off_cost so the zero-friction threshold is exactly
variable_cost (65), making the collapse unambiguous.

Run from repo root:
    python optimal_path/hysteresis_analysis.py
"""

from __future__ import annotations

import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REF_TIME = 5.0   # year at which to read the triggers

from bellman_equation.params import SwitchingParams
from bellman_equation.solver import SwitchingBellmanSolver, default_price_fn
from optimal_path.trigger_curves import extract_trigger_curves
from tree.oil_futures_curve import FuturesCurve
from tree.oil_tree_builder import OilTrinomialTreeBuilder
from tree.oil_tree_calibrator import OilTrinomialFuturesCalibrator


def run_hysteresis_analysis(
    switching_costs: list[float], enable_plot: bool = True
) -> None:
    n_steps = 1280
    dt = 10.0 / n_steps   # 0.0078125
    a = 0.6
    sigma = 0.2
    variable_cost = 65.0
    futures_curve = FuturesCurve(lambda t: 55.0 + 20.0 * math.exp(-0.3 * t))

    base_params = dict(
        dt=dt,
        discount_rate=0.08,
        production_rate=1.0,
        variable_cost=variable_cost,
        fixed_on_cost=3.0,   # equal to fixed_off_cost so zero-friction
        fixed_off_cost=3.0,  # threshold = variable_cost exactly
        capex=20.0,
        salvage_multiplier=0.0,
        allow_start_on=True,
    )

    builder = OilTrinomialTreeBuilder(
        n_steps=n_steps, a=a, sigma=sigma, dt=dt, jmax=36, jmin=-36
    )
    shifted_tree = OilTrinomialFuturesCalibrator(builder.build(), futures_curve).calibrate()

    ref_step = int(REF_TIME / dt)

    print(f"{'cost':>8} | {'S*_on':>10} | {'S*_off':>10} | {'band':>8}")
    print("-" * 45)

    results = []
    for cost in switching_costs:
        params = SwitchingParams(
            **base_params, switch_on_cost=cost, switch_off_cost=cost
        )
        solution = SwitchingBellmanSolver(
            tree=shifted_tree,
            params=params,
            price_fn=default_price_fn,
            terminal_on=lambda price: 0.0,
            terminal_off=lambda price: 0.0,
        ).solve()

        _, _, off_trig, on_trig = extract_trigger_curves(shifted_tree, solution, dt)

        s_on  = on_trig[ref_step]  if ref_step < len(on_trig)  else None
        s_off = off_trig[ref_step] if ref_step < len(off_trig) else None
        band  = (s_on - s_off) if (s_on is not None and s_off is not None) else None

        print(
            f"{cost:8.1f} | "
            f"{s_on  if s_on  is not None else 'N/A':>10} | "
            f"{s_off if s_off is not None else 'N/A':>10} | "
            f"{band:.1f}" if band is not None else f"{cost:8.1f} | {'N/A':>10} | {'N/A':>10} | N/A"
        )

        results.append((cost, s_on, s_off))

    if enable_plot:
        _plot_results(results, variable_cost, sigma)


def _plot_results(results, variable_cost: float, sigma: float) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    try:
        valid = [(c, von, voff) for c, von, voff in results
                 if von is not None and voff is not None]
        if not valid:
            print("No valid trigger data to plot.")
            return

        cx, tv_on, tv_off = zip(*valid)
        cx = list(cx); tv_on = list(tv_on); tv_off = list(tv_off)

        _, ax = plt.subplots(figsize=(7, 5))

        ax.plot(cx, tv_on,  marker="^", linewidth=1.8, color="steelblue",
                label=r"$S^*_{on}$  (umbral de reactivación)")
        ax.plot(cx, tv_off, marker="v", linewidth=1.8, color="tomato",
                label=r"$S^*_{off}$  (umbral de cierre)")
        ax.fill_between(cx, tv_off, tv_on, alpha=0.15, color="steelblue",
                        label="Banda de histéresis")
        ax.axhline(variable_cost, color="black", linewidth=1.0, linestyle=":",
                   label=f"Punto de equilibrio operativo  ({variable_cost} \$/bbl)")

        ax.set_xlabel(r"Costo de switching  (u.m.)  (C. encendido = C. apagado)")
        ax.set_ylabel(r"Precio umbral del petróleo  ($/bbl)")
        ax.set_title(
            rf"Banda de histéresis vs Costo de Switching  ($\sigma$={sigma:.2f},  t={REF_TIME:.0f} yr)"
        )
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(fontsize=8)

        plt.tight_layout()
        plt.show()
    except Exception as exc:
        print(f"\nPlot skipped: {exc}")


if __name__ == "__main__":
    switching_costs = [0.0, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0]
    run_hysteresis_analysis(switching_costs)
