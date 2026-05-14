"""
Figuras del Capítulo 9 — Implementación numérica y caso base.

Genera tres figuras:
  fig9_1_arbol_calibrado.png  — Fan de precios del árbol trinomial calibrado
  fig9_2_valor_caso_base.png  — Waterfall de descomposición del valor
  fig9_3_mapa_politica.png    — Mapa de política óptima (caso base)

Ejecutar desde la raíz del repositorio:
    python chapter9/caso_base_figures.py
"""

from __future__ import annotations

import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from bellman_equation.params import SwitchingParams
from bellman_equation.solver import SwitchingBellmanSolver, default_price_fn
from npv_rule.calc import NPVParams, TreeNPVCalculator
from optimal_path.policy_map import build_policy_map
from optimal_path.trigger_curves import extract_trigger_curves
from tree.oil_futures_curve import FuturesCurve
from tree.oil_tree_builder import OilTrinomialTreeBuilder
from tree.oil_tree_calibrator import OilTrinomialFuturesCalibrator

OUTPUT_DIR = pathlib.Path(r"C:\Users\lucia\OneDrive\Documentos\tesis\sensibilidades\chapter9")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Parámetros base ────────────────────────────────────────────────────────────
N_STEPS  = 40
DT       = 0.25        # años (trimestres)
A        = 0.6
SIGMA    = 0.20
_FWD_FN  = lambda t: 72.0 + 12.0 * math.exp(-0.5 * t)
FUTURES  = FuturesCurve(_FWD_FN)

# Parámetros que acepta NPVParams (BaseOperatingParams)
NPV_BASE = dict(
    dt               = DT,
    discount_rate    = 0.08,
    production_rate  = 1.0,
    variable_cost    = 65.0,
    fixed_on_cost    = 5.0,
    capex            = 20.0,
    switch_on_cost   = 5.0,
    salvage_multiplier = 0.0,
)

# Parámetros completos para SwitchingParams
BASE = dict(
    **NPV_BASE,
    fixed_off_cost   = 2.0,
    switch_off_cost  = 1.0,
    allow_start_on   = True,
)
_NO_SWITCH = 1e9


def _build_tree_and_solve():
    builder = OilTrinomialTreeBuilder(
        n_steps=N_STEPS, a=A, sigma=SIGMA, dt=DT, jmax=6, jmin=-6
    )
    tree = OilTrinomialFuturesCalibrator(builder.build(), FUTURES).calibrate()
    params = SwitchingParams(**BASE)
    sol = SwitchingBellmanSolver(
        tree=tree, params=params, price_fn=default_price_fn,
        terminal_on=lambda p: 0.0, terminal_off=lambda p: 0.0,
    ).solve()
    return tree, sol


def _bellman_noswitch(tree):
    params = SwitchingParams(**{**BASE, "switch_off_cost": _NO_SWITCH})
    sol = SwitchingBellmanSolver(
        tree=tree, params=params, price_fn=default_price_fn,
        terminal_on=lambda p: 0.0, terminal_off=lambda p: 0.0,
    ).solve()
    return sol.value_pre[0][0]


# ══════════════════════════════════════════════════════════════════════════════
# Figura 9.1 — Árbol de precios calibrado
# ══════════════════════════════════════════════════════════════════════════════
def fig_arbol(tree) -> plt.Figure:
    """Fan de precios del árbol trinomial, con curva de futuros superpuesta."""
    times = [step * DT for step in range(N_STEPS + 1)]

    # Precio mínimo y máximo en cada paso de tiempo
    p_min, p_max, p_mid = [], [], []
    for step in range(N_STEPS + 1):
        if step >= len(tree.levels):
            break
        prices = [default_price_fn(tree, step, j) for j in tree.levels[step]]
        p_min.append(min(prices))
        p_max.append(max(prices))
        p_mid.append(np.median(prices))

    t_arr = times[: len(p_min)]
    fwd   = [_FWD_FN(t) for t in t_arr]

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.fill_between(t_arr, p_min, p_max, alpha=0.18, color="steelblue",
                    label="Rango de precios del árbol")
    ax.plot(t_arr, p_max, color="steelblue", linewidth=0.8, alpha=0.6)
    ax.plot(t_arr, p_min, color="steelblue", linewidth=0.8, alpha=0.6)
    ax.plot(t_arr, p_mid, color="steelblue", linewidth=1.4, linestyle="--",
            label="Mediana de nodos")
    ax.plot(t_arr, fwd, color="darkorange", linewidth=2.0,
            label=r"Curva de futuros $F(t)$")
    ax.axhline(65.0 + 5.0, color="gray", linewidth=0.9, linestyle=":",
               label="Breakeven operativo (70 \$/bbl)")

    ax.set_xlabel("Tiempo (años)")
    ax.set_ylabel(r"Precio del petróleo  (\$/bbl)")
    ax.set_title(
        rf"Árbol trinomial calibrado  ($\sigma$={SIGMA:.2f},  $a$={A},  $N$={N_STEPS} pasos)"
    )
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Figura 9.1b — Árbol de precios (versión A: bandas por percentiles analíticos)
# ══════════════════════════════════════════════════════════════════════════════
def fig_arbol_percentiles() -> plt.Figure:
    """
    Fan de precios basado en percentiles analíticos del modelo Hull-White.

    En el modelo R(t) = ln S(t) sigue dR = (θ(t)-aR)dt + σ dW.
    La calibración a futuros garantiza E[S(t)] ≈ F(t).
    La desviación estándar de R(t) es: σ_R(t) = σ * sqrt((1-e^{-2at}) / (2a)).
    Las bandas de percentiles en precio: F(t) * exp(±z * σ_R(t)).
    """
    t_arr = np.linspace(0, N_STEPS * DT, 300)
    fwd   = np.array([_FWD_FN(t) for t in t_arr])

    def std_R(t):
        if t == 0:
            return 0.0
        return SIGMA * math.sqrt((1 - math.exp(-2 * A * t)) / (2 * A))

    std_arr = np.array([std_R(t) for t in t_arr])

    # Percentiles: z=1.645 → 5/95, z=1.282 → 10/90, z=0.674 → 25/75
    p05 = fwd * np.exp(-1.645 * std_arr)
    p95 = fwd * np.exp(+1.645 * std_arr)
    p25 = fwd * np.exp(-0.674 * std_arr)
    p75 = fwd * np.exp(+0.674 * std_arr)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.fill_between(t_arr, p05, p95, alpha=0.15, color="steelblue",
                    label="Banda 5%–95%")
    ax.fill_between(t_arr, p25, p75, alpha=0.30, color="steelblue",
                    label="Banda 25%–75%")
    ax.plot(t_arr, fwd, color="darkorange", linewidth=2.0,
            label=r"Curva de futuros $F(t)$  (media condicional)")
    ax.axhline(65.0 + 5.0, color="gray", linewidth=0.9, linestyle=":",
               label=r"Breakeven operativo (70 \$/bbl)")

    ax.set_xlabel("Tiempo (años)")
    ax.set_ylabel(r"Precio del petróleo  (\$/bbl)")
    ax.set_title(
        rf"Distribución de precios del petróleo  ($\sigma$={SIGMA:.2f},  $a$={A})"
    )
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Figura 9.2 — Waterfall de descomposición del valor
# ══════════════════════════════════════════════════════════════════════════════
def fig_waterfall(tree, sol) -> plt.Figure:
    """Descomposición: VAN → VAN + Opción Inv → VAN + Opción Inv + Opción Op."""
    npv_now   = TreeNPVCalculator(tree, params=NPVParams(**NPV_BASE)).invest_now_npv()
    b_noswitch = _bellman_noswitch(tree)
    b_full     = sol.value_pre[0][0]

    inv_opt = b_noswitch - npv_now
    op_opt  = b_full    - b_noswitch

    labels   = ["VAN\n(inversión\ninmediata)", "+ Opción\nde Inversión",
                 "+ Opción\nOperacional", "Valor total\nBellman"]
    bottoms  = [0,          npv_now,              b_noswitch,   0]
    heights  = [npv_now,    inv_opt,               op_opt,       b_full]
    colors   = ["steelblue", "darkorange",         "seagreen",   "seagreen"]
    alphas   = [1.0,         1.0,                  1.0,          0.45]

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = []
    for i, (lbl, bot, h, col, al) in enumerate(
        zip(labels, bottoms, heights, colors, alphas)
    ):
        bar = ax.bar(i, h, bottom=bot, color=col, alpha=al,
                     edgecolor="white", linewidth=1.2)
        bars.append(bar)
        val = bot + h
        ax.text(i, val + 0.3, f"{val:.1f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    # Flechas de incremento
    ax.annotate("", xy=(1, b_noswitch), xytext=(0, npv_now),
                arrowprops=dict(arrowstyle="->", color="darkorange", lw=1.5))
    ax.annotate("", xy=(2, b_full), xytext=(1, b_noswitch),
                arrowprops=dict(arrowstyle="->", color="seagreen", lw=1.5))

    # Etiquetas de incremento
    ax.text(0.5, (npv_now + b_noswitch) / 2, f"+{inv_opt:.1f}",
            ha="center", va="center", fontsize=9, color="darkorange",
            fontweight="bold")
    ax.text(1.5, (b_noswitch + b_full) / 2, f"+{op_opt:.1f}",
            ha="center", va="center", fontsize=9, color="seagreen",
            fontweight="bold")

    ax.axhline(0, color="black", linewidth=0.7, linestyle=":")
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Valor del proyecto  (u.m.)")
    ax.set_title("Descomposición del valor — caso base")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    legend_patches = [
        mpatches.Patch(color="steelblue",   label=f"VAN = {npv_now:.1f} u.m."),
        mpatches.Patch(color="darkorange",  label=f"Opción de Inversión = {inv_opt:.1f} u.m."),
        mpatches.Patch(color="seagreen",    label=f"Opción Operacional = {op_opt:.1f} u.m."),
    ]
    ax.legend(handles=legend_patches, fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Construyendo árbol y resolviendo Bellman (caso base)...")
    tree, sol = _build_tree_and_solve()

    print("\n── Figura 9.1: árbol calibrado (versión original: min/max)")
    f1 = fig_arbol(tree)
    f1.savefig(OUTPUT_DIR / "fig9_1_arbol_calibrado.png", dpi=300, bbox_inches="tight")
    print(f"  → {OUTPUT_DIR / 'fig9_1_arbol_calibrado.png'}")

    print("\n── Figura 9.1b: árbol calibrado (versión A: bandas percentiles)")
    f1b = fig_arbol_percentiles()
    f1b.savefig(OUTPUT_DIR / "fig9_1b_arbol_percentiles.png", dpi=300, bbox_inches="tight")
    print(f"  → {OUTPUT_DIR / 'fig9_1b_arbol_percentiles.png'}")

    print("\n── Figura 9.2: waterfall del valor")
    f2 = fig_waterfall(tree, sol)
    f2.savefig(OUTPUT_DIR / "fig9_2_valor_caso_base.png", dpi=300, bbox_inches="tight")
    print(f"  → {OUTPUT_DIR / 'fig9_2_valor_caso_base.png'}")

    print("\n── Figura 9.3: mapa de política (puede tardar ~30 s)")
    _before = set(plt.get_fignums())
    build_policy_map(sigma=SIGMA)
    new = sorted(set(plt.get_fignums()) - _before)
    if new:
        fig3 = plt.figure(new[0])
        fig3.savefig(OUTPUT_DIR / "fig9_3_mapa_politica.png", dpi=300, bbox_inches="tight")
        print(f"  → {OUTPUT_DIR / 'fig9_3_mapa_politica.png'}")

    print(f"\nTodas las figuras guardadas en:\n  {OUTPUT_DIR}")
    plt.show()
