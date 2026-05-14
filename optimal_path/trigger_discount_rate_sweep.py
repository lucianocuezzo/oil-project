"""
Sensibilidad del umbral de inversión S*_invest a la tasa de descuento.

Mayor tasa de descuento → los flujos futuros valen menos → se necesita un
precio spot más alto para justificar invertir hoy → trigger más elevado.

Ejecutar desde la raíz del repositorio:
    python optimal_path/trigger_discount_rate_sweep.py
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


def run_trigger_discount_rate_sweep(
    discount_rates: list[float], enable_plot: bool = True
) -> None:
    n_steps = 80
    dt      = 10.0 / n_steps   # 0.125
    a       = 0.6
    sigma   = 0.2
    futures_curve = FuturesCurve(lambda t: 72.0 + 12.0 * math.exp(-0.5 * t))

    base_params = dict(
        dt=dt,
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

    # El árbol no depende de la tasa de descuento → construir una sola vez
    builder = OilTrinomialTreeBuilder(
        n_steps=n_steps, a=a, sigma=sigma, dt=dt, jmax=9, jmin=-9
    )
    shifted_tree = OilTrinomialFuturesCalibrator(builder.build(), futures_curve).calibrate()

    print(f"{'rate':>8} | " + " | ".join(f"t={t:.0f}yr" for t in REF_TIMES))
    print("-" * 50)

    results = []
    for r in discount_rates:
        params = SwitchingParams(**base_params, discount_rate=r)
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
        print(f"{r:8.2%} | {row}")
        results.append((r, trigger_at))

    if enable_plot:
        _plot_results(results)


def _npv_threshold(discount_rate, variable_cost=65.0, fixed_on_cost=5.0,
                   switch_on_cost=5.0, capex=20.0,
                   n_steps=80, dt=0.125, production_rate=1.0) -> float:
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

        colors = cm.viridis(np.linspace(0.1, 0.85, len(REF_TIMES)))
        rate_vals = [r[0] * 100 for r in results]   # en %

        for t_ref, color in zip(REF_TIMES, colors):
            trig_vals = [r[1][t_ref] for r in results]
            valid = [(rv, tv) for rv, tv in zip(rate_vals, trig_vals) if tv is not None]
            if not valid:
                continue
            rx, tv = zip(*valid)
            ax.plot(rx, tv, marker="o", linewidth=1.8, color=color,
                    label=f"Año {t_ref:.0f}")

        # Umbral VAN: depende de la tasa de descuento
        raw_rates = [r[0] for r in results]
        npv_thresh = [_npv_threshold(discount_rate=r) for r in raw_rates]
        ax.plot(rate_vals, npv_thresh, color="dimgray", linewidth=1.4,
                linestyle="--", label=r"Umbral VAN  (VAN = 0)")

        ax.set_xlabel(r"Tasa de descuento  (%)")
        ax.set_ylabel(r"Precio de entrada óptimo  $S^*_{invest}$  (\$/bbl)")
        ax.set_title("Umbral de inversión vs Tasa de descuento")
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

    discount_rates = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]
    run_trigger_discount_rate_sweep(discount_rates)

    figs = plt.get_fignums()
    if figs:
        path = OUTPUT_DIR / "trigger_discount_rate_sweep.png"
        plt.figure(figs[-1]).savefig(path, dpi=300, bbox_inches="tight")
        print(f"  → guardado: {path}")

    input("\nPresioná Enter para cerrar...")
