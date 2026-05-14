"""
Volatilidad vs ancho de banda de histéresis.

Para cada σ resuelve la ecuación de Bellman y lee los umbrales operacionales
S*_off y S*_on en t = REF_TIME años.

Un panel — ambos umbrales vs σ con banda sombreada.

Ejecutar desde la raíz del repositorio:
    python optimal_path/hysteresis_sigma_sweep.py
"""

from __future__ import annotations

import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REF_TIME   = 5.0    # año en que se leen los triggers
SWITCH_COST = 5.0   # costo de switching fijo para este barrido

from bellman_equation.params import SwitchingParams
from bellman_equation.solver import SwitchingBellmanSolver, default_price_fn
from optimal_path.trigger_curves import extract_trigger_curves
from tree.oil_futures_curve import FuturesCurve
from tree.oil_tree_builder import OilTrinomialTreeBuilder
from tree.oil_tree_calibrator import OilTrinomialFuturesCalibrator


def run_hysteresis_sigma_sweep(
    sigmas: list[float], enable_plot: bool = True
) -> None:
    n_steps = 40
    dt      = 0.25
    a       = 0.6
    variable_cost = 65.0
    futures_curve = FuturesCurve(lambda t: 72.0 + 12.0 * math.exp(-0.5 * t))

    base_params = dict(
        dt=dt,
        discount_rate=0.08,
        production_rate=1.0,
        variable_cost=variable_cost,
        fixed_on_cost=5.0,
        fixed_off_cost=2.0,
        capex=20.0,
        switch_on_cost=SWITCH_COST,
        switch_off_cost=SWITCH_COST,
        salvage_multiplier=0.0,
        allow_start_on=True,
    )

    ref_step = int(REF_TIME / dt)

    print(f"{'sigma':>8} | {'S*_on':>10} | {'S*_off':>10} | {'banda':>8}")
    print("-" * 47)

    results = []
    for sigma in sigmas:
        builder = OilTrinomialTreeBuilder(
            n_steps=n_steps, a=a, sigma=sigma, dt=dt, jmax=6, jmin=-6
        )
        shifted_tree = OilTrinomialFuturesCalibrator(
            builder.build(), futures_curve
        ).calibrate()

        params = SwitchingParams(**base_params)
        solution = SwitchingBellmanSolver(
            tree=shifted_tree,
            params=params,
            price_fn=default_price_fn,
            terminal_on=lambda price: 0.0,
            terminal_off=lambda price: 0.0,
        ).solve()

        _, _, off_trig, on_trig = extract_trigger_curves(shifted_tree, solution, dt)

        s_on  = on_trig[ref_step]  if ref_step < len(on_trig)  else None
        s_off = off_trig[ref_step] if ref_step < len(off_trig) else None
        band  = (s_on - s_off) if (s_on is not None and s_off is not None) else None

        if band is not None:
            print(f"{sigma:8.2f} | {s_on:10.2f} | {s_off:10.2f} | {band:8.2f}")
        else:
            print(f"{sigma:8.2f} | {'N/A':>10} | {'N/A':>10} | {'N/A':>8}")

        results.append((sigma, s_on, s_off, band))

    if enable_plot:
        _plot_results(results, variable_cost)


def _plot_results(results, variable_cost: float) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    try:
        valid = [(s, von, voff, b) for s, von, voff, b in results
                 if von is not None and voff is not None and b is not None]
        if not valid:
            print("No hay datos válidos para graficar.")
            return

        sigmas_v, tv_on, tv_off, _ = zip(*valid)
        sigmas_v = list(sigmas_v)
        tv_on    = list(tv_on)
        tv_off   = list(tv_off)

        _, ax = plt.subplots(figsize=(7, 5))

        ax.plot(sigmas_v, tv_on,  marker="^", linewidth=1.8, color="steelblue",
                label=r"$S^*_{on}$  (umbral de reactivación)")
        ax.plot(sigmas_v, tv_off, marker="v", linewidth=1.8, color="tomato",
                label=r"$S^*_{off}$  (umbral de cierre)")
        ax.fill_between(sigmas_v, tv_off, tv_on, alpha=0.15, color="steelblue",
                        label="Banda de histéresis")
        ax.axhline(variable_cost, color="black", linewidth=1.0, linestyle=":",
                   label=f"Breakeven operativo  ({variable_cost:.0f} \$/bbl)")

        ax.set_xlabel(r"Volatilidad  $\sigma$")
        ax.set_ylabel(r"Precio umbral  (\$/bbl)")
        ax.set_title(
            rf"Banda de histéresis vs $\sigma$  (t={REF_TIME:.0f} yr,  C_sw={SWITCH_COST:.0f})"
        )
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(fontsize=8)

        plt.tight_layout()
        plt.show(block=False)
    except Exception as exc:
        print(f"\nPlot skipped: {exc}")


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    OUTPUT_DIR = pathlib.Path(r"C:\Users\lucia\OneDrive\Documentos\tesis\sensibilidades\optimal_path")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sigmas = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    run_hysteresis_sigma_sweep(sigmas)

    figs = plt.get_fignums()
    if figs:
        path = OUTPUT_DIR / "hysteresis_sigma_sweep.png"
        plt.figure(figs[-1]).savefig(path, dpi=300, bbox_inches="tight")
        print(f"  → guardado: {path}")

    input("\nPresioná Enter para cerrar...")
