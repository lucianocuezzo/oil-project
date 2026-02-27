"""
Sigma sensitivity: vary price volatility and compare plain NPV vs Bellman.

Decomposition of option values:
  - Investment option  = Bellman_no_switch − NPV_invest_now
  - Operational option = Bellman_full      − Bellman_no_switch

Run from repo root:
    python sensitivity_analysis/project_value/sigma_sweep.py
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
from plots import TreePlotter
from tree.oil_futures_curve import FuturesCurve
from tree.oil_tree_builder import OilTrinomialTreeBuilder
from tree.oil_tree_calibrator import OilTrinomialFuturesCalibrator

_NO_SWITCH = 1e9  # prohibitively high switch_off_cost to disable switching


def _run_bellman(tree, common: dict, switch_off_cost: float) -> float:
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


def run_sigma_sweep(sigmas: Iterable[float], enable_plot: bool = True, show_tree: bool = False) -> None:
    n_steps = 40
    dt = 0.25  # quarters
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

    print(
        f"{'sigma':>6} | {'NPV_now':>10} | {'B_noswitch':>10} | {'B_full':>10} | "
        f"{'Inv_opt':>10} | {'Op_opt':>10} | {'action'}"
    )
    print("-" * 85)

    sig_list: list[float] = []
    npv_now_list: list[float] = []
    b_noswitch_list: list[float] = []
    b_full_list: list[float] = []
    inv_option_list: list[float] = []
    op_option_list: list[float] = []

    for idx, sigma in enumerate(sigmas):
        builder = OilTrinomialTreeBuilder(n_steps=n_steps, a=a, sigma=sigma, dt=dt, jmax=6, jmin=-6)
        base_tree = builder.build()
        shifted_tree = OilTrinomialFuturesCalibrator(base_tree, futures_curve).calibrate()
        if show_tree and idx == 0:
            _plot_tree(shifted_tree)

        # Plain NPV: invest at t=0, always ON, no options
        npv_params = NPVParams(**common)
        tree_invest_now = TreeNPVCalculator(shifted_tree, params=npv_params).invest_now_npv()

        # Bellman with switching disabled: captures investment timing only
        b_noswitch, _ = _run_bellman(shifted_tree, common, switch_off_cost=_NO_SWITCH)

        # Full Bellman: investment timing + on/off switching
        b_full, action = _run_bellman(shifted_tree, common, switch_off_cost=1.0)

        inv_option = b_noswitch - tree_invest_now   # value of investment timing flexibility
        op_option  = b_full    - b_noswitch         # value of operational switching flexibility

        print(
            f"{sigma:6.2f} | {tree_invest_now:10.2f} | {b_noswitch:10.2f} | {b_full:10.2f} | "
            f"{inv_option:10.2f} | {op_option:10.2f} | {action}"
        )

        sig_list.append(sigma)
        npv_now_list.append(tree_invest_now)
        b_noswitch_list.append(b_noswitch)
        b_full_list.append(b_full)
        inv_option_list.append(inv_option)
        op_option_list.append(op_option)

    if enable_plot:
        _plot_results(sig_list, npv_now_list, b_noswitch_list, b_full_list, inv_option_list, op_option_list)


def _plot_results(sigmas, npv_now, b_noswitch, b_full, inv_option, op_option) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

        # --- Left: project value ---
        ax1.plot(sigmas, npv_now,    marker="^", linestyle="--", color="steelblue",   label="Valuación con NPV")
        ax1.plot(sigmas, b_noswitch, marker="o", linestyle="-",  color="darkorange",  label="Valuación con NPV + Opción Inv.")
        ax1.plot(sigmas, b_full,     marker="s", linestyle="-",  color="seagreen",    label="Valuación con NPV + Opción Inv. + Opción Op.")
        ax1.axhline(0, color="black", linewidth=0.7, linestyle=":")
        ax1.set_xlabel(r"$\sigma$ (volatilidad)")
        ax1.set_ylabel("Valor del proyecto  $V$")
        ax1.set_title("Valor del proyecto vs volatilidad")
        ax1.grid(True, linestyle="--", alpha=0.35)
        ax1.legend(fontsize=8)

        # --- Right: option values ---
        ax2.plot(sigmas, inv_option, marker="o", linestyle="-", color="darkorange",
                 label="Opción de Inversión")
        ax2.plot(sigmas, op_option,  marker="s", linestyle="-", color="seagreen",
                 label="Opción Operacional")
        ax2.axhline(0, color="black", linewidth=0.7, linestyle=":")
        ax2.set_xlabel(r"$\sigma$ (volatilidad)")
        ax2.set_ylabel("Valor de la opción")
        ax2.set_title("Valor de las opciones vs volatilidad")
        ax2.grid(True, linestyle="--", alpha=0.35)
        ax2.legend(fontsize=8)

        plt.suptitle("Sensibilidad a la volatilidad", fontsize=11, y=1.01)
        plt.tight_layout()
        plt.show(block=False)
    except Exception as exc:
        print(f"\nPlot skipped: {exc}")


def _plot_tree(shifted_tree) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    try:
        plotter = TreePlotter(shifted_tree, y_axis="r")
        plotter.plot_original_and_shifted(
            annotate_factors=True,
            title="Tree (original vs shifted)",
        )
        plt.tight_layout()
        plt.show()
    except Exception as exc:
        print(f"\nTree plot skipped: {exc}")


if __name__ == "__main__":
    sigmas = [0.0, 0.1, 0.2, 0.35, 0.5]
    run_sigma_sweep(sigmas, enable_plot=True, show_tree=True)
