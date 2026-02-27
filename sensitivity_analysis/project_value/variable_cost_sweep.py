"""
Sensibilidad al costo variable operativo.

Mayor variable_cost → menor margen → proyecto se acerca al breakeven operativo.
Las opciones valen más cerca del breakeven (proyecto "at the money") y menos
cuando el proyecto está profundamente en o fuera del dinero.

Breakeven operativo: variable_cost + fixed_on_cost = variable_cost + 5
  - floor de la curva de futuros = 72 → breakeven_var = 72 - 5 = 67
  - spot = 84 → proyecto rentable para variable_cost < 84 - 5 = 79

Ejecutar desde la raíz del repositorio:
    python sensitivity_analysis/project_value/variable_cost_sweep.py
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

_NO_SWITCH = 1e9
_FIXED_ON_COST = 5.0


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


def run_variable_cost_sweep(
    var_costs: Iterable[float], enable_plot: bool = True
) -> None:
    n_steps = 40
    dt = 0.25  # quarters
    a = 0.6
    sigma = 0.2
    futures_curve = FuturesCurve(lambda t: 72.0 + 12.0 * math.exp(-0.5 * t))  # spot=84, floor=72

    base_common = dict(
        dt=dt,
        discount_rate=0.08,
        production_rate=1.0,
        fixed_on_cost=_FIXED_ON_COST,
        capex=20.0,
        switch_on_cost=5.0,
        salvage_multiplier=0.0,
    )

    # El árbol no depende del costo variable
    builder = OilTrinomialTreeBuilder(n_steps=n_steps, a=a, sigma=sigma, dt=dt, jmax=6, jmin=-6)
    shifted_tree = OilTrinomialFuturesCalibrator(builder.build(), futures_curve).calibrate()

    print(
        f"{'var_cost':>10} | {'breakeven':>10} | {'NPV_now':>10} | {'B_noswitch':>10} | "
        f"{'B_full':>10} | {'Inv_opt':>10} | {'Op_opt':>10} | {'action'}"
    )
    print("-" * 105)

    vc_list: list[float] = []
    be_list: list[float] = []
    npv_now_list: list[float] = []
    b_noswitch_list: list[float] = []
    b_full_list: list[float] = []
    inv_option_list: list[float] = []
    op_option_list: list[float] = []

    for vc in var_costs:
        common = {**base_common, "variable_cost": vc}
        breakeven = vc + _FIXED_ON_COST

        tree_invest_now = TreeNPVCalculator(
            shifted_tree, params=NPVParams(**common)
        ).invest_now_npv()

        b_noswitch, _ = _run_bellman(shifted_tree, common, switch_off_cost=_NO_SWITCH)
        b_full, action = _run_bellman(shifted_tree, common, switch_off_cost=1.0)

        inv_option = b_noswitch - tree_invest_now
        op_option  = b_full    - b_noswitch

        print(
            f"{vc:10.1f} | {breakeven:10.1f} | {tree_invest_now:10.2f} | {b_noswitch:10.2f} | "
            f"{b_full:10.2f} | {inv_option:10.2f} | {op_option:10.2f} | {action}"
        )

        vc_list.append(vc)
        be_list.append(breakeven)
        npv_now_list.append(tree_invest_now)
        b_noswitch_list.append(b_noswitch)
        b_full_list.append(b_full)
        inv_option_list.append(inv_option)
        op_option_list.append(op_option)

    if enable_plot:
        _plot_results(vc_list, npv_now_list, b_noswitch_list, b_full_list,
                      inv_option_list, op_option_list)


def _plot_results(vc_vals, npv_now, b_noswitch, b_full, inv_option, op_option) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    try:
        # Breakeven operativo de largo plazo: floor - fixed_on_cost = 72 - 5 = 67
        breakeven_vc = 72.0 - _FIXED_ON_COST

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

        # --- Izquierda: valor del proyecto ---
        ax1.plot(vc_vals, npv_now,    marker="^", linestyle="--", color="steelblue",
                 label="Valuación con NPV")
        ax1.plot(vc_vals, b_noswitch, marker="o", linestyle="-",  color="darkorange",
                 label="Valuación con NPV + Opción Inv.")
        ax1.plot(vc_vals, b_full,     marker="s", linestyle="-",  color="seagreen",
                 label="Valuación con NPV + Opción Inv. + Opción Op.")
        ax1.axhline(0, color="black", linewidth=0.7, linestyle=":")
        ax1.axvline(breakeven_vc, color="gray", linewidth=0.9, linestyle="--",
                    label=f"Breakeven largo plazo ({breakeven_vc:.0f})")
        ax1.set_xlabel("Costo variable  $c_v$")
        ax1.set_ylabel("Valor del proyecto  $V$")
        ax1.set_title("Valor del proyecto vs costo variable")
        ax1.grid(True, linestyle="--", alpha=0.35)
        ax1.legend(fontsize=8)

        # --- Derecha: valor de las opciones ---
        ax2.plot(vc_vals, inv_option, marker="o", linestyle="-", color="darkorange",
                 label="Opción de Inversión")
        ax2.plot(vc_vals, op_option,  marker="s", linestyle="-", color="seagreen",
                 label="Opción Operacional")
        ax2.axhline(0, color="black", linewidth=0.7, linestyle=":")
        ax2.axvline(breakeven_vc, color="gray", linewidth=0.9, linestyle="--",
                    label=f"Breakeven largo plazo ({breakeven_vc:.0f})")
        ax2.set_xlabel("Costo variable  $c_v$")
        ax2.set_ylabel("Valor de la opción")
        ax2.set_title("Valor de las opciones vs costo variable")
        ax2.grid(True, linestyle="--", alpha=0.35)
        ax2.legend(fontsize=8)

        plt.suptitle("Sensibilidad al costo variable operativo", fontsize=11, y=1.01)
        plt.tight_layout()
        plt.show(block=False)
    except Exception as exc:
        print(f"\nGráfico omitido: {exc}")


if __name__ == "__main__":
    # var_cost de 45 a 80: breakeven de 50 a 85, spot=84, floor=72
    var_costs = [45.0, 50.0, 55.0, 60.0, 65.0, 67.0, 70.0, 73.0, 76.0, 79.0]
    run_variable_cost_sweep(var_costs)
