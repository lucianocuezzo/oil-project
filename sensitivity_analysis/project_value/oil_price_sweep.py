"""
Sensibilidad al nivel de precio del petróleo (futures floor).

Se varía el piso de la curva de futuros F(t) = floor + 12·exp(-0.5·t),
manteniendo la forma de backwardation. El precio spot es floor + 12.

Para cada nivel de precio se recalibra el árbol (los alphas cambian),
pero la estructura de ramificación (pu, pm, pd) se reutiliza.

Breakeven operativo: variable_cost + fixed_on_cost = 65 + 5 = 70.
  - floor < 70 → proyecto mayormente no rentable (NPV < 0)
  - floor ~ 70 → zona de transición (opciones valen más)
  - floor > 70 → proyecto en dinero (NPV > 0)

Ejecutar desde la raíz del repositorio:
    python sensitivity_analysis/project_value/oil_price_sweep.py
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


def run_oil_price_sweep(floors: Iterable[float], enable_plot: bool = True) -> None:
    n_steps = 40
    dt = 0.25  # quarters
    a = 0.6
    sigma = 0.2
    premium = 12.0   # backwardation premium (spot = floor + premium)
    decay   = 0.5    # velocidad de convergencia al piso

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

    # Base tree reutilizable: la estructura no depende del nivel de precios
    base_tree = OilTrinomialTreeBuilder(
        n_steps=n_steps, a=a, sigma=sigma, dt=dt, jmax=6, jmin=-6
    ).build()

    print(
        f"{'floor':>7} | {'spot':>6} | {'NPV_now':>10} | {'B_noswitch':>10} | {'B_full':>10} | "
        f"{'Inv_opt':>10} | {'Op_opt':>10} | {'action'}"
    )
    print("-" * 100)

    floor_list: list[float] = []
    spot_list: list[float] = []
    npv_now_list: list[float] = []
    b_noswitch_list: list[float] = []
    b_full_list: list[float] = []
    inv_option_list: list[float] = []
    op_option_list: list[float] = []

    for floor in floors:
        spot = floor + premium
        futures_curve = FuturesCurve(lambda t, f=floor: f + premium * math.exp(-decay * t))
        shifted_tree = OilTrinomialFuturesCalibrator(base_tree, futures_curve).calibrate()

        # NPV plano
        tree_invest_now = TreeNPVCalculator(
            shifted_tree, params=NPVParams(**common)
        ).invest_now_npv()

        # Bellman sin switching
        b_noswitch, _ = _run_bellman(shifted_tree, common, switch_off_cost=_NO_SWITCH)

        # Bellman completo
        b_full, action = _run_bellman(shifted_tree, common, switch_off_cost=1.0)

        inv_option = b_noswitch - tree_invest_now
        op_option  = b_full    - b_noswitch

        print(
            f"{floor:7.1f} | {spot:6.1f} | {tree_invest_now:10.2f} | {b_noswitch:10.2f} | {b_full:10.2f} | "
            f"{inv_option:10.2f} | {op_option:10.2f} | {action}"
        )

        floor_list.append(floor)
        spot_list.append(spot)
        npv_now_list.append(tree_invest_now)
        b_noswitch_list.append(b_noswitch)
        b_full_list.append(b_full)
        inv_option_list.append(inv_option)
        op_option_list.append(op_option)

    if enable_plot:
        _plot_results(spot_list, npv_now_list, b_noswitch_list, b_full_list,
                      inv_option_list, op_option_list)


def _plot_results(spots, npv_now, b_noswitch, b_full, inv_option, op_option) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    try:
        breakeven_spot = 65.0 + 5.0 + 12.0   # variable_cost + fixed_on_cost + premium (spot cuando floor=breakeven)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

        # --- Izquierda: valor del proyecto ---
        ax1.plot(spots, npv_now,    marker="^", linestyle="--", color="steelblue",
                 label="Valuación con NPV")
        ax1.plot(spots, b_noswitch, marker="o", linestyle="-",  color="darkorange",
                 label="Valuación con NPV + Opción Inv.")
        ax1.plot(spots, b_full,     marker="s", linestyle="-",  color="seagreen",
                 label="Valuación con NPV + Opción Inv. + Opción Op.")
        ax1.axhline(0, color="black", linewidth=0.7, linestyle=":")
        ax1.axvline(breakeven_spot, color="gray", linewidth=0.9, linestyle="--",
                    label=f"Precio spot breakeven ({breakeven_spot:.0f})")
        ax1.set_xlabel("Precio spot  $S_0$  (piso + prima)")
        ax1.set_ylabel("Valor del proyecto  $V$")
        ax1.set_title("Valor del proyecto vs precio del petróleo")
        ax1.grid(True, linestyle="--", alpha=0.35)
        ax1.legend(fontsize=8)

        # --- Derecha: valor de las opciones ---
        ax2.plot(spots, inv_option, marker="o", linestyle="-", color="darkorange",
                 label="Opción de Inversión")
        ax2.plot(spots, op_option,  marker="s", linestyle="-", color="seagreen",
                 label="Opción Operacional")
        ax2.axhline(0, color="black", linewidth=0.7, linestyle=":")
        ax2.axvline(breakeven_spot, color="gray", linewidth=0.9, linestyle="--",
                    label=f"Breakeven ({breakeven_spot:.0f})")
        ax2.set_xlabel("Precio spot  $S_0$  (piso + prima)")
        ax2.set_ylabel("Valor de la opción")
        ax2.set_title("Valor de las opciones vs precio del petróleo")
        ax2.grid(True, linestyle="--", alpha=0.35)
        ax2.legend(fontsize=8)

        plt.suptitle("Sensibilidad al nivel de precio del petróleo", fontsize=11, y=1.01)
        plt.tight_layout()
        plt.show(block=False)
    except Exception as exc:
        print(f"\nGráfico omitido: {exc}")


if __name__ == "__main__":
    # floor va de 55 a 90: spot de 67 a 102, breakeven spot = 82
    floors = [55.0, 58.0, 61.0, 64.0, 67.0, 70.0, 73.0, 76.0, 80.0, 85.0, 90.0]
    run_oil_price_sweep(floors)
