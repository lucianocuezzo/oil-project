"""
Mapa de política óptima comparado entre cuatro niveles de volatilidad.

Grid de 4 filas × 2 columnas:
  - Cada fila corresponde a un valor de σ ∈ {0.10, 0.20, 0.35, 0.50}
  - Columna izquierda  → política de inversión  (gris / verde)
  - Columna derecha    → política operacional   (rojo / amarillo / azul)

El eje Y es compartido entre todos los paneles (misma escala de precio),
lo que permite ver directamente cómo σ afecta:
  · La altura de S*_invest  (mayor σ → trigger más alto = invertir más tarde)
  · La anchura de la banda de histéresis  (mayor σ → banda más ancha)

Ejecutar desde la raíz del repositorio:
    python optimal_path/policy_map_sigma.py
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bellman_equation.params import SwitchingParams
from bellman_equation.solver import SwitchingBellmanSolver, default_price_fn
from optimal_path.trigger_curves import extract_trigger_curves
from tree.oil_futures_curve import FuturesCurve
from tree.oil_tree_builder import OilTrinomialTreeBuilder
from tree.oil_tree_calibrator import OilTrinomialFuturesCalibrator


# ---------------------------------------------------------------------------
# Shared parameters
# ---------------------------------------------------------------------------

N_STEPS = 40
DT      = 0.25
A       = 0.6
SIGMAS  = [0.10, 0.20, 0.35, 0.50]

FUTURES = FuturesCurve(lambda t: 72.0 + 12.0 * math.exp(-0.5 * t))

BASE_PARAMS = dict(
    dt=DT,
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ffill(values: list) -> list:
    filled = list(values)
    last = None
    for i, v in enumerate(filled):
        if v is not None:
            last = v
        elif last is not None:
            filled[i] = last
    last = None
    for i in range(len(filled) - 1, -1, -1):
        if filled[i] is not None:
            last = filled[i]
        elif last is not None:
            filled[i] = last
    return filled


def _build_and_solve(sigma: float):
    builder = OilTrinomialTreeBuilder(
        n_steps=N_STEPS, a=A, sigma=sigma, dt=DT, jmax=6, jmin=-6
    )
    tree = OilTrinomialFuturesCalibrator(builder.build(), FUTURES).calibrate()
    params = SwitchingParams(**BASE_PARAMS)
    solution = SwitchingBellmanSolver(
        tree=tree,
        params=params,
        price_fn=default_price_fn,
        terminal_on=lambda p: 0.0,
        terminal_off=lambda p: 0.0,
    ).solve()
    return tree, solution


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_policy_map_sigma(sigmas: list[float] = SIGMAS) -> None:
    fwd_times  = [t * DT for t in range(1, N_STEPS + 1)]
    fwd_prices = [FUTURES.value(t, DT) for t in range(1, N_STEPS + 1)]

    # Solve for each sigma and collect results
    results = []
    for sigma in sigmas:
        print(f"Solving σ={sigma:.2f} ...", flush=True)
        tree, solution = _build_and_solve(sigma)
        times, inv_trig, off_trig, on_trig = extract_trigger_curves(tree, solution, DT)

        tree_prices = [
            default_price_fn(tree, t, j)
            for t in range(N_STEPS)
            for j in tree.levels[t]
        ]
        p_min = min(tree_prices) * 0.95
        p_max = max(tree_prices) * 1.05
        results.append((sigma, times, inv_trig, off_trig, on_trig, p_min, p_max))

    _plot(results, fwd_times, fwd_prices)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _plot(results, fwd_times, fwd_prices) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import matplotlib.lines as mlines
    except Exception:
        return
    try:
        n_rows = len(results)
        fig, axes = plt.subplots(
            n_rows, 2,
            figsize=(13, 4.5 * n_rows),
        )

        for row, (sigma, times, inv_trig, off_trig, on_trig, p_min, p_max) in enumerate(results):
            ax1, ax2 = axes[row]

            t_arr   = np.array(times)
            inv_arr = np.array(_ffill(inv_trig), dtype=float)
            off_arr = np.array(_ffill(off_trig), dtype=float)
            on_arr  = np.array(_ffill(on_trig),  dtype=float)

            # -------------------------------------------------------------- #
            # Left panel — Investment decision                                 #
            # -------------------------------------------------------------- #
            ax1.fill_between(t_arr, p_min, inv_arr,
                             color="#cfd8dc", alpha=0.85, rasterized=True)
            ax1.fill_between(t_arr, inv_arr, p_max,
                             color="#66bb6a", alpha=0.75, rasterized=True)
            ax1.plot(t_arr, inv_arr, color="black", linewidth=2.0, zorder=5)
            ax1.plot(fwd_times, fwd_prices, color="white", linewidth=1.5,
                     linestyle="--", zorder=5)

            ax1.set_xlim(t_arr[0], t_arr[-1])
            ax1.set_ylim(p_min, p_max)
            ax1.set_ylabel("Precio  $S$  ($/bbl)")
            ax1.set_title(rf"Política de inversión  ($\sigma$={sigma:.2f})", fontsize=10)
            ax1.grid(True, linestyle="--", alpha=0.25, zorder=1)

            if row == n_rows - 1:
                ax1.set_xlabel("Tiempo (años)")

            # -------------------------------------------------------------- #
            # Right panel — Operational decision                               #
            # -------------------------------------------------------------- #
            ax2.fill_between(t_arr, p_min, off_arr,
                             color="#ef9a9a", alpha=0.85, rasterized=True)
            ax2.fill_between(t_arr, off_arr, on_arr,
                             color="#fff59d", alpha=0.90, rasterized=True)
            ax2.fill_between(t_arr, on_arr, p_max,
                             color="#90caf9", alpha=0.85, rasterized=True)
            ax2.plot(t_arr, off_arr, color="black", linewidth=2.0, zorder=5)
            ax2.plot(t_arr, on_arr,  color="black", linewidth=2.0, linestyle="--",
                     zorder=5)
            ax2.plot(fwd_times, fwd_prices, color="dimgray", linewidth=1.5,
                     linestyle=":", zorder=5)

            ax2.set_xlim(t_arr[0], t_arr[-1])
            ax2.set_ylim(p_min, p_max)
            ax2.set_title(rf"Política operacional  ($\sigma$={sigma:.2f})", fontsize=10)
            ax2.grid(True, linestyle="--", alpha=0.25, zorder=1)

            if row == n_rows - 1:
                ax2.set_xlabel("Tiempo (años)")

        # ------------------------------------------------------------------
        # Shared legends (placed only on first row for clarity)
        # ------------------------------------------------------------------
        axes[0][0].legend(handles=[
            mpatches.Patch(color="#cfd8dc", alpha=0.85, label="Esperar"),
            mpatches.Patch(color="#66bb6a", alpha=0.75, label="Invertir"),
            mlines.Line2D([], [], color="black", linewidth=2,
                          label=r"$S^*_{invest}$"),
            mlines.Line2D([], [], color="white", linewidth=1.5, linestyle="--",
                          label="Futuros"),
        ], fontsize=8, loc="upper right")

        axes[0][1].legend(handles=[
            mpatches.Patch(color="#ef9a9a", alpha=0.85, label="Apagar"),
            mpatches.Patch(color="#fff59d", alpha=0.90, label="Histéresis"),
            mpatches.Patch(color="#90caf9", alpha=0.85, label="Operar"),
            mlines.Line2D([], [], color="black", linewidth=2,
                          label=r"$S^*_{off}$"),
            mlines.Line2D([], [], color="black", linewidth=2, linestyle="--",
                          label=r"$S^*_{on}$"),
            mlines.Line2D([], [], color="dimgray", linewidth=1.5, linestyle=":",
                          label="Futuros"),
        ], fontsize=8, loc="upper right")

        plt.suptitle(
            r"Mapa de política óptima — efecto de la volatilidad  ($a$=0.6,  capex=20)",
            fontsize=12, y=1.005
        )
        plt.tight_layout()
        plt.show(block=False)
    except Exception as exc:
        print(f"\nPlot skipped: {exc}")


if __name__ == "__main__":
    build_policy_map_sigma()
    input("\nPresioná Enter para cerrar...")
