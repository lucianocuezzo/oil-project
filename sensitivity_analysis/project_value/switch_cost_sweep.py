"""
Sensibilidad al costo de apagado (switch_off_cost).

switch_on_cost se mantiene fijo en el valor base. Solo se varía switch_off_cost
(costo de cerrar temporalmente el pozo). A mayor costo de apagado, la opción de
apagado pierde valor y B_full converge a B_noswitch.

Descomposición de opciones:
  - Opción de inversión  = Bellman_no_switch − NPV_invest_now   (constante: no depende de switch_off_cost)
  - Opción operacional   = Bellman_full      − Bellman_no_switch (decrece con switch_off_cost, ≥ 0 siempre)

Ejecutar desde la raíz del repositorio:
    python sensitivity_analysis/project_value/switch_cost_sweep.py
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


def run_switch_cost_sweep(
    switch_off_costs: Iterable[float], enable_plot: bool = True
) -> None:
    n_steps = 40
    dt = 0.25  # quarters
    a = 0.6
    sigma = 0.2
    futures_curve = FuturesCurve(lambda t: 72.0 + 12.0 * math.exp(-0.5 * t))  # backwardation: spot=84, floor=72 (breakeven=70)

    common = dict(
        dt=dt,
        discount_rate=0.08,
        production_rate=1.0,
        variable_cost=65.0,
        fixed_on_cost=5.0,
        capex=20.0,
        switch_on_cost=5.0,   # fijo: no varía en este sweep
        salvage_multiplier=0.0,
    )

    # Build tree once (switching costs no afectan la dinámica del precio)
    builder = OilTrinomialTreeBuilder(n_steps=n_steps, a=a, sigma=sigma, dt=dt, jmax=6, jmin=-6)
    shifted_tree = OilTrinomialFuturesCalibrator(builder.build(), futures_curve).calibrate()

    # NPV_now y B_noswitch son constantes (no dependen de switch_off_cost)
    tree_invest_now = TreeNPVCalculator(shifted_tree, params=NPVParams(**common)).invest_now_npv()
    b_noswitch, _   = _run_bellman(shifted_tree, common, switch_off_cost=_NO_SWITCH)
    inv_option      = b_noswitch - tree_invest_now

    print(
        f"{'sw_off_cost':>12} | {'NPV_now':>10} | {'B_noswitch':>10} | {'B_full':>10} | "
        f"{'Inv_opt':>10} | {'Op_opt':>10} | {'action'}"
    )
    print("-" * 95)

    sc_list: list[float] = []
    npv_now_list: list[float] = []
    b_noswitch_list: list[float] = []
    b_full_list: list[float] = []
    inv_option_list: list[float] = []
    op_option_list: list[float] = []

    for sc in switch_off_costs:
        # Solo switch_off_cost varía; switch_on_cost permanece fijo
        b_full, action = _run_bellman(shifted_tree, common, switch_off_cost=sc)
        op_option = b_full - b_noswitch   # ≥ 0: opción de apagado vale ≥ 0

        print(
            f"{sc:12.2f} | {tree_invest_now:10.2f} | {b_noswitch:10.2f} | {b_full:10.2f} | "
            f"{inv_option:10.2f} | {op_option:10.2f} | {action}"
        )

        sc_list.append(sc)
        npv_now_list.append(tree_invest_now)
        b_noswitch_list.append(b_noswitch)
        b_full_list.append(b_full)
        inv_option_list.append(inv_option)
        op_option_list.append(op_option)

    if enable_plot:
        _plot_results(sc_list, npv_now_list, b_noswitch_list, b_full_list,
                      inv_option_list, op_option_list)


def _plot_results(sc_vals, npv_now, b_noswitch, b_full, inv_option, op_option) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

        # --- Izquierda: valor del proyecto ---
        ax1.plot(sc_vals, npv_now,    marker="^", linestyle="--", color="steelblue",
                 label="Valuación con NPV")
        ax1.plot(sc_vals, b_noswitch, marker="o", linestyle="-",  color="darkorange",
                 label="Valuación con NPV + Opción Inv.")
        ax1.plot(sc_vals, b_full,     marker="s", linestyle="-",  color="seagreen",
                 label="Valuación con NPV + Opción Inv. + Opción Op.")
        ax1.axhline(0, color="black", linewidth=0.7, linestyle=":")
        ax1.set_xlabel("Costo de apagado  (switch_off_cost)")
        ax1.set_ylabel("Valor del proyecto  $V$")
        ax1.set_title("Valor del proyecto vs costo de apagado")
        ax1.grid(True, linestyle="--", alpha=0.35)
        ax1.legend(fontsize=8)

        # --- Derecha: valor de las opciones ---
        ax2.plot(sc_vals, inv_option, marker="o", linestyle="-", color="darkorange",
                 label="Opción de Inversión")
        ax2.plot(sc_vals, op_option,  marker="s", linestyle="-", color="seagreen",
                 label="Opción Operacional")
        ax2.axhline(0, color="black", linewidth=0.7, linestyle=":")
        ax2.set_xlabel("Costo de apagado  (switch_off_cost)")
        ax2.set_ylabel("Valor de la opción")
        ax2.set_title("Valor de las opciones vs costo de apagado")
        ax2.grid(True, linestyle="--", alpha=0.35)
        ax2.legend(fontsize=8)

        plt.suptitle("Sensibilidad al costo de apagado operacional", fontsize=11, y=1.01)
        plt.tight_layout()
        plt.show(block=False)
    except Exception as exc:
        print(f"\nGráfico omitido: {exc}")


if __name__ == "__main__":
    switch_off_costs = [0.0, 1.0, 2.0, 5.0, 8.0, 12.0, 18.0, 25.0]
    run_switch_cost_sweep(switch_off_costs)
