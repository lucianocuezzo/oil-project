"""
Mapa de calor de las funciones de valor de Bellman.

Muestra V_pre(t, S), V_on(t, S) y V_off(t, S) como mapas de color
en el plano (tiempo × precio), con las curvas trigger superpuestas.

Lectura del mapa:
    · Verde oscuro  → valor alto (proyecto muy rentable en ese estado)
    · Amarillo      → valor cercano a cero
    · Rojo oscuro   → valor negativo (destruye valor)
    · Curva negra   → frontera de decisión óptima (trigger)

Ejecutar desde la raíz del repositorio:
    python optimal_path/value_heatmap.py
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


def build_heatmap() -> None:
    n_steps = 40
    dt      = 0.25
    a       = 0.6
    sigma   = 0.2
    futures_curve = FuturesCurve(lambda t: 72.0 + 12.0 * math.exp(-0.5 * t))

    params = SwitchingParams(
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
    tree = OilTrinomialFuturesCalibrator(builder.build(), futures_curve).calibrate()

    solution = SwitchingBellmanSolver(
        tree=tree,
        params=params,
        price_fn=default_price_fn,
        terminal_on=lambda p: 0.0,
        terminal_off=lambda p: 0.0,
    ).solve()

    times, inv_trig, off_trig, on_trig = extract_trigger_curves(tree, solution, dt)

    # ------------------------------------------------------------------
    # Collect scattered node values: (t, price) → V_pre / V_on / V_off
    # ------------------------------------------------------------------
    pts_t:    list[float] = []
    pts_p:    list[float] = []
    v_pre_sc: list[float] = []
    v_on_sc:  list[float] = []
    v_off_sc: list[float] = []

    for t in range(n_steps + 1):
        for j in tree.levels[t]:
            price = default_price_fn(tree, t, j)
            pts_t.append(t * dt)
            pts_p.append(price)
            # value_pre may be absent at terminal step
            v_pre_sc.append(solution.value_pre[t].get(j, 0.0)
                            if t < len(solution.value_pre) else 0.0)
            v_on_sc.append(solution.value_on[t].get(j, float("nan")))
            v_off_sc.append(solution.value_off[t].get(j, float("nan")))

    p_min = min(pts_p) * 0.98
    p_max = max(pts_p) * 1.02

    # ------------------------------------------------------------------
    # Interpolate onto a regular (time × price) grid
    # ------------------------------------------------------------------
    try:
        from scipy.interpolate import griddata
    except ImportError:
        print("scipy is required: pip install scipy")
        return

    T_grid = np.linspace(0, n_steps * dt, 150)
    P_grid = np.linspace(p_min, p_max, 100)
    TT, PP = np.meshgrid(T_grid, P_grid)

    points = np.column_stack([pts_t, pts_p])
    V_pre  = griddata(points, v_pre_sc, (TT, PP), method="linear")
    V_on   = griddata(points, v_on_sc,  (TT, PP), method="linear")
    V_off  = griddata(points, v_off_sc, (TT, PP), method="linear")

    _plot(T_grid, P_grid, TT, PP,
          V_pre, V_on, V_off,
          times, inv_trig, off_trig, on_trig,
          p_min, p_max)


def _plot(T_grid, P_grid, TT, PP,
          V_pre, V_on, V_off,
          times, inv_trig, off_trig, on_trig,
          p_min, p_max) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import matplotlib.lines as mlines
    except Exception:
        return
    try:
        t_arr   = np.array(times)
        inv_arr = np.array(_ffill(inv_trig), dtype=float)
        off_arr = np.array(_ffill(off_trig), dtype=float)
        on_arr  = np.array(_ffill(on_trig),  dtype=float)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        panels = [
            # (data,  title,                          triggers to overlay)
            (V_pre, r"$V^{pre}(t,\,S)$ — Pre-inversión",
             [(inv_arr, "-",  "black", r"$S^*_{invest}$")]),
            (V_on,  r"$V^{on}(t,\,S)$ — Estado ON",
             [(off_arr, "-",  "black", r"$S^*_{off}$")]),
            (V_off, r"$V^{off}(t,\,S)$ — Estado OFF",
             [(off_arr, "-",  "black", r"$S^*_{off}$"),
              (on_arr,  "--", "black", r"$S^*_{on}$")]),
        ]

        for ax, (V, title, triggers) in zip(axes, panels):
            finite = V[np.isfinite(V)]
            if len(finite) == 0:
                continue

            v_lo = np.percentile(finite, 2)
            v_hi = np.percentile(finite, 98)

            # Diverging norm centred at 0 if values span both signs
            if v_lo < 0 < v_hi:
                bound = max(abs(v_lo), abs(v_hi))
                norm = mcolors.TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)
            else:
                norm = mcolors.Normalize(vmin=v_lo, vmax=v_hi)

            im = ax.pcolormesh(TT, PP, V,
                               cmap="RdYlGn",
                               norm=norm,
                               shading="auto",
                               rasterized=True)
            plt.colorbar(im, ax=ax, label="Valor ($)", pad=0.02)

            # Trigger curve overlays
            legend_handles = []
            for arr, ls, col, lbl in triggers:
                ax.plot(t_arr, arr, color=col, linewidth=2.0,
                        linestyle=ls, zorder=5)
                legend_handles.append(
                    mlines.Line2D([], [], color=col, linewidth=2.0,
                                  linestyle=ls, label=lbl)
                )
            if legend_handles:
                ax.legend(handles=legend_handles, fontsize=8,
                          loc="upper right",
                          framealpha=0.75)

            ax.set_xlim(T_grid[0], T_grid[-1])
            ax.set_ylim(p_min, p_max)
            ax.set_xlabel("Tiempo (años)")
            ax.set_ylabel(r"Precio del petróleo  $S$  (\$/bbl)")
            ax.set_title(title, fontsize=10)
            ax.grid(False)

        plt.suptitle(
            r"Funciones de valor de Bellman  ($\sigma$=0.20,  $a$=0.6,  capex=20)",
            fontsize=12, y=1.01
        )
        plt.tight_layout()
        plt.show(block=False)
    except Exception as exc:
        print(f"\nPlot skipped: {exc}")


if __name__ == "__main__":
    build_heatmap()
    input("\nPresioná Enter para cerrar...")
