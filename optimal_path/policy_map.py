"""
Mapa de política óptima en el plano (tiempo × precio).

Zonas pintadas usando las curvas de trigger como fronteras (fill_between):

  Izquierda — Decisión de inversión:
      · Gris  (abajo de S*_invest) : "Esperar"
      · Verde (arriba de S*_invest): "Invertir"

  Derecha — Decisión operacional:
      · Rojo    (abajo de S*_off)            : "Apagar" (shutdown)
      · Amarillo (entre S*_off y S*_on)      : "Histéresis" (mantener estado actual)
      · Azul    (arriba de S*_on)            : "Encender / operar"

Ejecutar desde la raíz del repositorio:
    python optimal_path/policy_map.py
"""

from __future__ import annotations

import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bellman_equation.params import SwitchingParams
from bellman_equation.solver import SwitchingBellmanSolver, default_price_fn
from optimal_path.trigger_curves import extract_trigger_curves
from tree.oil_futures_curve import FuturesCurve
from tree.oil_tree_builder import OilTrinomialTreeBuilder
from tree.oil_tree_calibrator import OilTrinomialFuturesCalibrator


def _ffill(values: list) -> list:
    """Forward-fill None values in a list."""
    filled = list(values)
    last = None
    for i, v in enumerate(filled):
        if v is not None:
            last = v
        elif last is not None:
            filled[i] = last
    # backward-fill leading Nones
    last = None
    for i in range(len(filled) - 1, -1, -1):
        if filled[i] is not None:
            last = filled[i]
        elif last is not None:
            filled[i] = last
    return filled


def build_policy_map(sigma: float = 0.2) -> None:
    n_steps = 40
    dt = 0.25
    a = 0.6
    futures_curve = FuturesCurve(lambda t: 72.0 + 12.0 * math.exp(-0.5 * t))

    params_dict = dict(
        dt=dt,
        discount_rate=0.08,
        production_rate=1.0,
        variable_cost=65.0,
        fixed_on_cost=5.0,
        fixed_off_cost=2.0,
        capex=20.0,
        switch_on_cost=5.0,
        switch_off_cost=1.0,
        salvage_multiplier=0.0,
        allow_start_on=True,
    )

    builder = OilTrinomialTreeBuilder(
        n_steps=n_steps, a=a, sigma=sigma, dt=dt, jmax=6, jmin=-6
    )
    shifted_tree = OilTrinomialFuturesCalibrator(builder.build(), futures_curve).calibrate()

    params = SwitchingParams(**params_dict)
    solution = SwitchingBellmanSolver(
        tree=shifted_tree,
        params=params,
        price_fn=default_price_fn,
        terminal_on=lambda price: 0.0,
        terminal_off=lambda price: 0.0,
    ).solve()

    times, inv_trig, off_trig, on_trig = extract_trigger_curves(shifted_tree, solution, dt)

    # Price bounds from tree nodes
    all_prices = [
        default_price_fn(shifted_tree, t, j)
        for t in range(n_steps)
        for j in shifted_tree.levels[t]
    ]
    p_min = min(all_prices) * 0.95
    p_max = max(all_prices) * 1.05

    # Forward curve
    fwd_times  = [t * dt for t in range(1, n_steps + 1)]
    fwd_prices = [futures_curve.value(t, dt) for t in range(1, n_steps + 1)]

    _plot_policy_map(
        times, inv_trig, off_trig, on_trig,
        fwd_times, fwd_prices,
        p_min, p_max, sigma, params_dict
    )


def _plot_policy_map(times, inv_trig, off_trig, on_trig,
                     fwd_times, fwd_prices, p_min, p_max, sigma, params_dict) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except Exception:
        return
    try:
        # Fill-forward any None gaps in trigger curves
        inv_f = _ffill(inv_trig)
        off_f = _ffill(off_trig)
        on_f  = _ffill(on_trig)

        t_arr    = np.array(times)
        inv_arr  = np.array(inv_f, dtype=float)
        off_arr  = np.array(off_f, dtype=float)
        on_arr   = np.array(on_f,  dtype=float)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        # ------------------------------------------------------------------ #
        # Panel 1 — Decisión de inversión                                     #
        # ------------------------------------------------------------------ #
        # Gray: wait zone (below S*_invest)
        ax1.fill_between(t_arr, p_min, inv_arr,
                         color="#cfd8dc", alpha=0.85, label="Esperar")
        # Green: invest zone (above S*_invest)
        ax1.fill_between(t_arr, inv_arr, p_max,
                         color="#66bb6a", alpha=0.75, label="Invertir")

        # Trigger line
        ax1.plot(t_arr, inv_arr, color="black", linewidth=2.0, zorder=5,
                 label=r"$S^*_{invest}(t)$")
        # NPV threshold (flat: no depende del tiempo)
        r  = params_dict["discount_rate"]
        dt = params_dict["dt"]
        n  = int(round(10.0 / dt))
        annuity = dt * (1 - math.exp(-r * n * dt)) / (1 - math.exp(-r * dt))
        npv_thresh = (params_dict["variable_cost"] + params_dict["fixed_on_cost"]
                      + (params_dict["capex"] + params_dict["switch_on_cost"]) / annuity)
        ax1.axhline(npv_thresh, color="dimgray", linewidth=1.4,
                    linestyle="--", zorder=5, label=r"Umbral VAN  (VAN = 0)")
        # Forward curve
        ax1.plot(fwd_times, fwd_prices, color="white", linewidth=1.5,
                 linestyle="--", zorder=5, label="Curva de futuros")

        ax1.set_xlim(t_arr[0], t_arr[-1])
        ax1.set_ylim(p_min, p_max)
        ax1.set_xlabel("Tiempo (años)")
        ax1.set_ylabel(r"Precio del petróleo  $S$  (\$/bbl)")
        ax1.set_title("Política de inversión")
        ax1.grid(True, linestyle="--", alpha=0.25, zorder=1)
        ax1.legend(fontsize=8, loc="upper right")

        # ------------------------------------------------------------------ #
        # Panel 2 — Decisión operacional                                      #
        # ------------------------------------------------------------------ #
        # Red: always OFF (below S*_off → shutdown)
        ax2.fill_between(t_arr, p_min, off_arr,
                         color="#ef9a9a", alpha=0.85, label="Apagar")
        # Yellow: hysteresis band (between S*_off and S*_on)
        ax2.fill_between(t_arr, off_arr, on_arr,
                         color="#fff59d", alpha=0.90, label="Histéresis")
        # Blue: always ON (above S*_on → operate)
        ax2.fill_between(t_arr, on_arr, p_max,
                         color="#90caf9", alpha=0.85, label="Encender / operar")

        # Trigger lines
        ax2.plot(t_arr, off_arr, color="black", linewidth=2.0, zorder=5,
                 label=r"$S^*_{off}(t)$")
        ax2.plot(t_arr, on_arr,  color="black", linewidth=2.0, linestyle="--",
                 zorder=5, label=r"$S^*_{on}(t)$")
        # Forward curve
        ax2.plot(fwd_times, fwd_prices, color="dimgray", linewidth=1.5,
                 linestyle=":", zorder=5, label="Curva de futuros")

        ax2.set_xlim(t_arr[0], t_arr[-1])
        ax2.set_ylim(p_min, p_max)
        ax2.set_xlabel("Tiempo (años)")
        ax2.set_ylabel(r"Precio del petróleo  $S$  (\$/bbl)")
        ax2.set_title("Política operacional")
        ax2.grid(True, linestyle="--", alpha=0.25, zorder=1)
        ax2.legend(fontsize=8, loc="upper right")

        plt.suptitle(
            rf"Mapa de política óptima  ($\sigma$={sigma:.2f},  $a$=0.6,  capex=20)",
            fontsize=11, y=1.01
        )
        plt.tight_layout()
        plt.show()
    except Exception as exc:
        print(f"\nPlot skipped: {exc}")


if __name__ == "__main__":
    build_policy_map(sigma=0.2)
