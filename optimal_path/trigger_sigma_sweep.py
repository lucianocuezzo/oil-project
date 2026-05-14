"""
Sensibilidad del umbral de inversión S*_invest a la volatilidad σ.

Resultado clásico de opciones reales: mayor incertidumbre → esperar más →
trigger más alto. Para cada σ resuelve Bellman y lee S*_invest en tres
momentos de referencia.

Ejecutar desde la raíz del repositorio:
    python optimal_path/trigger_sigma_sweep.py
"""

from __future__ import annotations

import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REF_TIMES = [1.0, 5.0, 9.0]

from bellman_equation.params import SwitchingParams
from bellman_equation.solver import SwitchingBellmanSolver, default_price_fn
from optimal_path.trigger_curves import extract_trigger_curves
from tree.oil_futures_curve import FuturesCurve
from tree.oil_tree_builder import OilTrinomialTreeBuilder
from tree.oil_tree_calibrator import OilTrinomialFuturesCalibrator


def run_trigger_sigma_sweep(sigmas: list[float], enable_plot: bool = True) -> None:
    n_steps = 80
    dt      = 10.0 / n_steps   # 0.125
    a       = 0.6
    futures_curve = FuturesCurve(lambda t: 72.0 + 12.0 * math.exp(-0.5 * t))

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
        capex=20.0,
        allow_start_on=True,
    )

    print(f"{'sigma':>8} | " + " | ".join(f"t={t:.0f}yr" for t in REF_TIMES))
    print("-" * 50)

    results = []
    for sigma in sigmas:
        # El árbol depende de σ → reconstruir en cada iteración
        builder = OilTrinomialTreeBuilder(
            n_steps=n_steps, a=a, sigma=sigma, dt=dt, jmax=9, jmin=-9
        )
        shifted_tree = OilTrinomialFuturesCalibrator(builder.build(), futures_curve).calibrate()

        params = SwitchingParams(**base_params)
        solution = SwitchingBellmanSolver(
            tree=shifted_tree,
            params=params,
            price_fn=default_price_fn,
            terminal_on=lambda price: 0.0,
            terminal_off=lambda price: 0.0,
        ).solve()

        _, inv_trig, _, _ = extract_trigger_curves(shifted_tree, solution, dt)

        trigger_at: dict[float, float | None] = {}
        for t_ref in REF_TIMES:
            step = int(t_ref / dt)
            trigger_at[t_ref] = inv_trig[step] if step < len(inv_trig) else None

        row = " | ".join(
            f"{trigger_at[t]:8.2f}" if trigger_at[t] is not None else "     N/A"
            for t in REF_TIMES
        )
        print(f"{sigma:8.2f} | {row}")
        results.append((sigma, trigger_at))

    if enable_plot:
        _plot_results(results)


def _npv_threshold(variable_cost=65.0, fixed_on_cost=5.0,
                   switch_on_cost=5.0, discount_rate=0.08,
                   capex=20.0, n_steps=80, dt=0.125,
                   production_rate=1.0) -> float:
    """Precio de entrada implícito por la regla del VAN (VAN = 0)."""
    r = discount_rate
    if r == 0:
        annuity = n_steps * dt * production_rate
    else:
        annuity = dt * (1 - math.exp(-r * n_steps * dt)) / (1 - math.exp(-r * dt)) * production_rate
    return variable_cost + fixed_on_cost + (capex + switch_on_cost) / annuity


def _plot_results(results) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import numpy as np
    except Exception:
        return
    try:
        fig, ax = plt.subplots(figsize=(7, 5))

        colors    = cm.viridis(np.linspace(0.1, 0.85, len(REF_TIMES)))
        sigma_vals = [r[0] for r in results]

        for t_ref, color in zip(REF_TIMES, colors):
            trig_vals = [r[1][t_ref] for r in results]
            valid = [(s, v) for s, v in zip(sigma_vals, trig_vals) if v is not None]
            if not valid:
                continue
            sx, tv = zip(*valid)
            ax.plot(sx, tv, marker="o", linewidth=1.8, color=color,
                    label=f"Año {t_ref:.0f}")

        # Umbral VAN: no depende de σ → línea horizontal
        npv_thresh = _npv_threshold()
        ax.axhline(npv_thresh, color="dimgray", linewidth=1.4,
                   linestyle="--", label=r"Umbral VAN  (VAN = 0)")

        ax.set_xlabel(r"Volatilidad  $\sigma$")
        ax.set_ylabel(r"Precio de entrada óptimo  $S^*_{invest}$  (\$/bbl)")
        ax.set_title("Umbral de inversión vs Volatilidad")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(title="Año de referencia", fontsize=8)

        plt.tight_layout()
        plt.show(block=False)
    except Exception as exc:
        print(f"\nPlot skipped: {exc}")


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    OUTPUT_DIR = pathlib.Path(r"C:\Users\lucia\OneDrive\Documentos\tesis\sensibilidades\optimal_path")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sigmas = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    run_trigger_sigma_sweep(sigmas)

    figs = plt.get_fignums()
    if figs:
        path = OUTPUT_DIR / "trigger_sigma_sweep.png"
        plt.figure(figs[-1]).savefig(path, dpi=300, bbox_inches="tight")
        print(f"  → guardado: {path}")

    input("\nPresioná Enter para cerrar...")
