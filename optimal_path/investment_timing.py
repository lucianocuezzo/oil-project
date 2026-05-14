"""
Distribución del timing de inversión óptima.

Simula N_PATHS caminos estocásticos a través del árbol trinomial y registra
el momento en que cada camino decide invertir bajo la política Bellman-óptima.

Panel izquierdo — histograma del timing de inversión (σ base):
    · Barra gris en t=0: caminos que invierten inmediatamente
    · Distribución: caminos que esperan a que el precio cruce S*_invest
    · Anotación: % de caminos que nunca invirtieron en el horizonte

Panel derecho — CDF acumulada para tres valores de σ:
    · Muestra cómo mayor volatilidad → inversión más tardía (valor de esperar)
    · X: tiempo, Y: fracción acumulada de caminos que ya invirtieron

Ejecutar desde la raíz del repositorio:
    python optimal_path/investment_timing.py
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
from tree.oil_futures_curve import FuturesCurve
from tree.oil_tree_builder import OilTrinomialTreeBuilder
from tree.oil_tree_calibrator import OilTrinomialFuturesCalibrator

N_PATHS = 1000
N_STEPS = 40
DT      = 0.25
A       = 0.6
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
# Build tree + solve
# ---------------------------------------------------------------------------

def _build_and_solve(sigma: float, capex: float = 20.0):
    builder = OilTrinomialTreeBuilder(
        n_steps=N_STEPS, a=A, sigma=sigma, dt=DT, jmax=6, jmin=-6
    )
    tree = OilTrinomialFuturesCalibrator(builder.build(), FUTURES).calibrate()
    params = SwitchingParams(**{**BASE_PARAMS, "capex": capex})
    solution = SwitchingBellmanSolver(
        tree=tree,
        params=params,
        price_fn=default_price_fn,
        terminal_on=lambda p: 0.0,
        terminal_off=lambda p: 0.0,
    ).solve()
    return tree, solution


# ---------------------------------------------------------------------------
# Simulate investment times
# ---------------------------------------------------------------------------

def simulate_investment_times(
    tree,
    solution,
    n_paths: int = N_PATHS,
    seed: int = 42,
) -> list[float | None]:
    """
    Returns a list of length n_paths.
    Each entry is the time (in years) at which the path invested,
    or None if it never invested within the horizon.
    """
    rng = np.random.default_rng(seed)
    results: list[float | None] = []

    for _ in range(n_paths):
        j = 0
        invested_at: float | None = None

        for t in range(N_STEPS):
            action = solution.policy_pre[t].get(j, "wait")

            if action in ("invest_on", "invest_off"):
                invested_at = t * DT
                break

            node = tree.levels[t][j]
            j_up, j_mid, j_down = node.children
            pu, pm, pd = node.probabilities
            j = rng.choice([j_up, j_mid, j_down], p=[pu, pm, pd])

        results.append(invested_at)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _solve_scenario(capex: float, sigmas: list, n_paths: int, seed: int) -> dict:
    """Solve and simulate for all sigmas at a given capex. Returns timing dict."""
    timing: dict[float, list[float | None]] = {}
    for sigma in sigmas:
        print(f"  σ={sigma:.2f}, capex={capex:.0f} ...", flush=True)
        tree, solution = _build_and_solve(sigma, capex=capex)
        timing[sigma] = simulate_investment_times(tree, solution, n_paths=n_paths, seed=seed)
    return timing


def build_timing_plot(
    capex_list: list[float] = (20.0, 60.0),
    n_paths: int = N_PATHS,
    seed: int = 42,
) -> None:
    sigmas     = [0.10, 0.20, 0.35, 0.50, 0.60, 0.70]
    sigma_base = 0.20

    scenarios: list[tuple[float, dict]] = []
    for capex in capex_list:
        print(f"\nCapex = {capex:.0f}")
        timing = _solve_scenario(capex, sigmas, n_paths, seed)
        scenarios.append((capex, timing))

        base = timing[sigma_base]
        invested  = [t for t in base if t is not None]
        never     = sum(1 for t in base if t is None)
        pct_never = 100.0 * never / n_paths
        print(f"  σ={sigma_base:.2f}: {n_paths-never} inv ({100-pct_never:.1f}%)"
              + (f"  media={np.mean(invested):.2f}a" if invested else ""))

    _plot(scenarios, sigmas, sigma_base, n_paths)


# ---------------------------------------------------------------------------
# Plot — single figure, one row per capex scenario
# ---------------------------------------------------------------------------

def _plot(
    scenarios: list[tuple[float, dict]],
    sigmas: list[float],
    sigma_base: float,
    n_paths: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    try:
        horizon = N_STEPS * DT
        bins    = np.arange(0, horizon + DT, DT)
        colors  = ["#43a047", "#1e88e5", "#e53935", "#8e24aa", "#f57c00", "#795548"]
        t_axis  = np.linspace(0, horizon, 400)

        n_rows = len(scenarios)
        fig, axes = plt.subplots(n_rows, 2, figsize=(13, 5 * n_rows))
        if n_rows == 1:
            axes = [axes]   # make iterable

        for row, (capex, timing) in enumerate(scenarios):
            ax1, ax2 = axes[row]

            # --- Histogram ---
            base_times = timing[sigma_base]
            invested   = [t for t in base_times if t is not None]
            pct_never  = 100.0 * sum(1 for t in base_times if t is None) / n_paths

            counts, edges = np.histogram(invested, bins=bins)
            fracs = counts / n_paths
            ax1.bar(edges[:-1], fracs, width=DT * 0.85,
                    align="edge", color="#5c85d6", alpha=0.85,
                    label="Invirtió en ese trimestre")

            ax1.text(0.97, 0.95,
                     f"{pct_never:.1f}% nunca invirtió",
                     transform=ax1.transAxes,
                     ha="right", va="top", fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

            if invested:
                mean_t   = np.mean(invested)
                median_t = np.median(invested)
                ax1.axvline(mean_t,   color="black",   linewidth=1.5,
                            linestyle="--", label=f"Media  {mean_t:.1f} a")
                ax1.axvline(median_t, color="dimgray", linewidth=1.5,
                            linestyle=":",  label=f"Mediana {median_t:.1f} a")

            ax1.set_xlim(0, horizon)
            ax1.set_xlabel("Tiempo de inversión (años)")
            ax1.set_ylabel("Fracción de caminos")
            ax1.set_title(
                rf"Histograma  ($\sigma$={sigma_base:.2f},  capex={capex:.0f})"
            )
            ax1.legend(fontsize=8)
            ax1.grid(True, linestyle="--", alpha=0.3)

            # --- CDF ---
            for sigma, col in zip(sigmas, colors):
                times        = timing[sigma]
                invested_arr = np.array([t for t in times if t is not None])
                pct_nv       = 100.0 * sum(1 for t in times if t is None) / n_paths
                if len(invested_arr) == 0:
                    continue
                cdf = np.array([np.sum(invested_arr <= ti) / n_paths for ti in t_axis])
                ax2.plot(t_axis, cdf * 100, color=col, linewidth=2.2,
                         label=rf"$\sigma$={sigma:.2f}  ({100-pct_nv:.0f}%)")

            ax2.set_xlim(0, horizon)
            ax2.set_ylim(0, 100)
            ax2.set_xlabel("Tiempo (años)")
            ax2.set_ylabel("% que ya invirtieron")
            ax2.set_title(rf"CDF por volatilidad  (capex={capex:.0f})")
            ax2.legend(fontsize=8, loc="lower right")
            ax2.grid(True, linestyle="--", alpha=0.3)

        plt.suptitle(
            rf"Timing óptimo de inversión  ($a$=0.6,  {n_paths} caminos)",
            fontsize=12, y=1.01
        )
        plt.tight_layout()
        plt.show(block=False)
    except Exception as exc:
        print(f"\nPlot skipped: {exc}")


if __name__ == "__main__":
    build_timing_plot(capex_list=[20.0, 60.0], n_paths=N_PATHS, seed=42)
    input("\nPresioná Enter para cerrar...")
