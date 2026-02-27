"""
Sensibilidad a la velocidad de reversión a la media (a).

Mayor 'a' → menor incertidumbre de largo plazo → menor valor de opción de espera.

Panel izquierdo : valor del proyecto (NPV ahora, Bellman timing, Bellman full)
Panel derecho   : valor de las opciones (inversión y operación)

Ejecutar desde la raíz del repositorio:
    python sensitivity_analysis/project_value/mean_reversion_sweep.py
"""

from __future__ import annotations

import math
import pathlib
import sys
from typing import Iterable

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bellman_equation.params import SwitchingParams
from bellman_equation.solver import SwitchingBellmanSolver, default_price_fn
from npv_rule.calc import NPVParams, TreeNPVCalculator
from tree.oil_futures_curve import FuturesCurve
from tree.oil_tree_builder import OilTrinomialTreeBuilder
from tree.oil_tree_calibrator import OilTrinomialFuturesCalibrator

_NO_SWITCH = 1e9  # switch_off_cost prohibitivo para desactivar switching operacional


def _run_bellman(tree, common: dict, switch_off_cost: float):
    params = SwitchingParams(
        **common,
        fixed_off_cost=2.0,
        switch_off_cost=switch_off_cost,
        allow_start_on=True,
    )
    sol = SwitchingBellmanSolver(
        tree=tree,
        params=params,
        price_fn=default_price_fn,
        terminal_on=lambda price: 0.0,
        terminal_off=lambda price: 0.0,
    ).solve()
    return sol.value_pre[0][0], sol.policy_pre[0][0]


def run_mean_reversion_sweep(
    a_values: Iterable[float], enable_plot: bool = True
) -> None:
    n_steps = 160
    dt = 10.0 / n_steps   # 0.0625
    sigma = 0.2
    futures_curve = FuturesCurve(lambda t: 72.0 + 12.0 * math.exp(-0.5 * t))  # backwardation: spot=84, floor=72 (breakeven=70)

    common = dict(
        dt=dt,
        discount_rate=0.08,
        production_rate=1.0,
        variable_cost=65.0,
        fixed_on_cost=5.0,
        switch_on_cost=5.0,
        capex=20.0,
        salvage_multiplier=0.0,
    )

    print(
        f"{'a':>6} | {'NPV_now':>10} | {'B_noswitch':>10} | {'B_full':>10} | "
        f"{'Inv_opt':>10} | {'Op_opt':>10} | {'action'}"
    )
    print("-" * 85)

    a_list: list[float] = []
    npv_now_list: list[float] = []
    b_noswitch_list: list[float] = []
    b_full_list: list[float] = []
    inv_option_list: list[float] = []
    op_option_list: list[float] = []

    for a in a_values:
        # jmax_min: validez de branching en la frontera (>= 0.184 / (a*dt))
        # jmax_valid: pm >= 0 en rama central  (<= sqrt(2/3) / (a*dt))
        jmax_min   = math.ceil(0.184 / (a * dt))
        jmax_valid = int(math.sqrt(2.0 / 3.0) / (a * dt))
        jmax = max(jmax_min, min(15, jmax_valid))

        builder = OilTrinomialTreeBuilder(
            n_steps=n_steps, a=a, sigma=sigma, dt=dt, jmax=jmax, jmin=-jmax
        )
        shifted_tree = OilTrinomialFuturesCalibrator(
            builder.build(), futures_curve
        ).calibrate()

        # NPV plano: invierte en t=0, siempre encendido, sin opciones
        tree_invest_now = TreeNPVCalculator(
            shifted_tree, params=NPVParams(**common)
        ).invest_now_npv()

        # Bellman sin switching: captura solo el timing de inversión
        b_noswitch, _ = _run_bellman(shifted_tree, common, switch_off_cost=_NO_SWITCH)

        # Bellman completo: timing + switching operacional
        b_full, action = _run_bellman(shifted_tree, common, switch_off_cost=1.0)

        inv_option = b_noswitch - tree_invest_now   # valor de la opción de timing
        op_option  = b_full    - b_noswitch         # valor de la opción operacional

        print(
            f"{a:6.2f} | {tree_invest_now:10.2f} | {b_noswitch:10.2f} | {b_full:10.2f} | "
            f"{inv_option:10.2f} | {op_option:10.2f} | {action}"
        )

        a_list.append(a)
        npv_now_list.append(tree_invest_now)
        b_noswitch_list.append(b_noswitch)
        b_full_list.append(b_full)
        inv_option_list.append(inv_option)
        op_option_list.append(op_option)

    if enable_plot:
        _plot_results(a_list, npv_now_list, b_noswitch_list, b_full_list,
                      inv_option_list, op_option_list, sigma)


def _plot_results(
    a_vals, npv_now, b_noswitch, b_full, inv_option, op_option, sigma: float
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

        # --- Izquierda: valor del proyecto ---
        ax1.plot(a_vals, npv_now,    marker="^", linestyle="--", color="steelblue",
                 label="Valuación con NPV")
        ax1.plot(a_vals, b_noswitch, marker="o", linestyle="-",  color="darkorange",
                 label="Valuación con NPV + Opción Inv.")
        ax1.plot(a_vals, b_full,     marker="s", linestyle="-",  color="seagreen",
                 label="Valuación con NPV + Opción Inv. + Opción Op.")
        ax1.axhline(0, color="black", linewidth=0.7, linestyle=":")
        ax1.set_xlabel(r"Velocidad de reversión a la media  $a$")
        ax1.set_ylabel("Valor del proyecto  $V$")
        ax1.set_title("Valor del proyecto vs reversión a la media")
        ax1.grid(True, linestyle="--", alpha=0.35)
        ax1.legend(fontsize=8)

        # --- Derecha: valor de las opciones ---
        ax2.plot(a_vals, inv_option, marker="o", linestyle="-", color="darkorange",
                 label="Opción de Inversión")
        ax2.plot(a_vals, op_option,  marker="s", linestyle="-", color="seagreen",
                 label="Opción Operacional")
        ax2.axhline(0, color="black", linewidth=0.7, linestyle=":")
        ax2.set_xlabel(r"Velocidad de reversión a la media  $a$")
        ax2.set_ylabel("Valor de la opción")
        ax2.set_title("Valor de las opciones vs reversión a la media")
        ax2.grid(True, linestyle="--", alpha=0.35)
        ax2.legend(fontsize=8)

        plt.suptitle(
            rf"Sensibilidad a la reversión a la media  ($\sigma$={sigma:.2f})",
            fontsize=11, y=1.01
        )
        plt.tight_layout()
        plt.show(block=False)
    except Exception as exc:
        print(f"\nGráfico omitido: {exc}")


if __name__ == "__main__":
    a_values = [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 1.75, 2.0]
    run_mean_reversion_sweep(a_values)
