"""
Diagramas para la sección de Implementación:
  fig_arquitectura_solver.png       — Composición de clases del solver
  fig_pipeline_sensibilidades.png   — Pipeline de los análisis de sensibilidad

Ejecutar desde la raíz del repositorio:
    python chapter9/implementation_diagrams.py
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUTPUT_DIR = ROOT / "figures" / "chapter9"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Paleta de colores (consistente entre los dos diagramas)
C_INPUT  = ("#dae8fc", "#6c8ebf")   # azul claro — entradas
C_CLASS  = ("#d5e8d4", "#82b366")   # verde — clases del modelo
C_DATA   = ("#fff2cc", "#d6b656")   # amarillo — datos intermedios
C_CORE   = ("#f8cecc", "#b85450")   # rojo claro — núcleo
C_OUTPUT = ("#dae8fc", "#6c8ebf")   # azul claro — salida


def _box(ax, x, y, w, h, body, fill, edge,
         title=None, fontsize=10, title_fontsize=12, align_left=False):
    """Caja redondeada con título opcional (bold) y cuerpo."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        linewidth=1.4, edgecolor=edge, facecolor=fill,
    )
    ax.add_patch(box)

    if title is None:
        ha = "left" if align_left else "center"
        tx = x + 0.15 if align_left else x + w / 2
        ax.text(tx, y + h / 2, body, ha=ha, va="center", fontsize=fontsize)
    else:
        ax.text(x + w / 2, y + h - 0.15, title,
                ha="center", va="top",
                fontsize=title_fontsize, fontweight="bold")
        ha = "left" if align_left else "center"
        tx = x + 0.25 if align_left else x + w / 2
        ax.text(tx, y + h - 0.55, body, ha=ha, va="top", fontsize=fontsize)


def _arrow(ax, p1, p2, label=None, fontsize=9, rad=0.0):
    """Flecha con etiqueta opcional sobre el punto medio."""
    arrow = FancyArrowPatch(
        p1, p2, arrowstyle="-|>", mutation_scale=14,
        color="#404040", linewidth=1.2,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arrow)
    if label:
        mx = (p1[0] + p2[0]) / 2
        my = (p1[1] + p2[1]) / 2
        ax.text(mx, my, label, fontsize=fontsize, color="#404040",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.18",
                          facecolor="white", edgecolor="none"))


# ══════════════════════════════════════════════════════════════════════════════
# Figura — Arquitectura del solver
# ══════════════════════════════════════════════════════════════════════════════
def fig_arquitectura_solver() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")

    # FuturesCurve
    _box(ax, 3.3, 8.4, 3.4, 1.2,
         "Curva de futuros del petróleo\nF(0, t) — input de mercado",
         *C_INPUT, title="FuturesCurve",
         fontsize=9, title_fontsize=11)

    # Builder
    _box(ax, 0.2, 6.3, 3.6, 1.4,
         "Construye el árbol trinomial\nHull–White del factor OU\n(sin calibrar)",
         *C_CLASS, title="OilTrinomialTreeBuilder",
         fontsize=9, title_fontsize=10)

    # Calibrator
    _box(ax, 6.2, 6.3, 3.6, 1.4,
         "Calcula el shift α(t) que hace\nque el árbol ajuste la curva\nde futuros (Clewlow–Strickland)",
         *C_CLASS, title="OilTrinomialFuturesCalibrator",
         fontsize=9, title_fontsize=10)

    # Shifted tree
    _box(ax, 6.2, 4.2, 3.6, 1.4,
         "Árbol calibrado: cada nodo\ntiene precio S y probabilidad.\nListo para valuar.",
         *C_DATA, title="ShiftedOilTrinomialTree",
         fontsize=9, title_fontsize=10)

    # SwitchingParams
    _box(ax, 0.2, 4.2, 3.6, 1.4,
         "Parámetros operativos del\nproyecto: CAPEX, costos,\ntasa, cash flows por estado",
         *C_CLASS, title="SwitchingParams",
         fontsize=9, title_fontsize=10)

    # Solver
    solver_body = (
        "Inducción regresiva sobre los tres estados del proyecto:\n"
        "    pre-invest   (esperar  /  invertir)\n"
        "    ON                  (seguir  /  apagar)\n"
        "    OFF                (seguir  /  encender)"
    )
    _box(ax, 1.5, 1.8, 7.0, 1.8, solver_body,
         *C_CORE, title="SwitchingBellmanSolver",
         fontsize=10, title_fontsize=12, align_left=True)

    # Solution
    sol_body = ("Función de valor y política óptima\nen cada nodo del árbol")
    _box(ax, 2.7, 0.1, 4.6, 1.1, sol_body,
         *C_OUTPUT, title="BellmanSolution",
         fontsize=9, title_fontsize=11)

    # Arrows
    _arrow(ax, (5.6, 8.4), (7.5, 7.7), label="futures_curve")   # FC → calibrator
    _arrow(ax, (3.8, 7.0), (6.2, 7.0), label="base_tree")        # builder → calibrator
    _arrow(ax, (8.0, 6.3), (8.0, 5.6), label="α(t)")             # calibrator → shifted tree
    _arrow(ax, (6.8, 4.2), (5.5, 3.6), label="tree")             # shifted tree → solver
    _arrow(ax, (3.2, 4.2), (4.5, 3.6), label="params")           # params → solver
    _arrow(ax, (5.0, 1.8), (5.0, 1.2), label="value, policy")    # solver → solution

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Figura — Pipeline de los análisis de sensibilidad
# ══════════════════════════════════════════════════════════════════════════════
def fig_pipeline_sensibilidades() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")

    # Caso base
    _box(ax, 0.2, 8.0, 4.4, 1.4,
         "Conjunto de parámetros del\nproyecto en su valor central\n(σ, a, CAPEX, c_v, r, ...)",
         *C_INPUT, title="Caso base",
         fontsize=9, title_fontsize=11)

    # Parámetro a variar
    _box(ax, 5.4, 8.0, 4.4, 1.4,
         "Una sola variable del modelo\ny una grilla de valores\nsobre la cual barrerla",
         *C_INPUT, title="Parámetro de sweep",
         fontsize=9, title_fontsize=11)

    # Loop del sweep (el corazón)
    loop_body = (
        "Para cada valor de la grilla:\n"
        "    1.  Reemplazar el valor del parámetro en el caso base\n"
        "    2.  Reconstruir el árbol y volver a calibrar si hace falta\n"
        "    3.  Correr el solver de Bellman con los parámetros mutados\n"
        "    4.  Extraer la métrica de interés"
    )
    _box(ax, 0.8, 4.5, 8.4, 2.6, loop_body,
         *C_CORE, title="Bucle del sweep",
         fontsize=10, title_fontsize=12, align_left=True)

    # Métrica extraída
    _box(ax, 0.8, 2.4, 8.4, 1.4,
         "Valor del proyecto en S₀,  precio umbral de inversión,\n"
         "banda de histéresis (encender / apagar),  etc.\n"
         "Una métrica por cada valor del parámetro",
         *C_DATA, title="Métrica",
         fontsize=10, title_fontsize=11)

    # Resultado
    _box(ax, 1.8, 0.3, 6.4, 1.4,
         "Curva o tabla de la métrica\nen función del parámetro variado",
         *C_OUTPUT, title="Resultado del análisis de sensibilidad",
         fontsize=10, title_fontsize=11)

    # Arrows
    _arrow(ax, (2.4, 8.0), (3.5, 7.1))   # caso base → loop
    _arrow(ax, (7.6, 8.0), (6.5, 7.1))   # parámetro → loop
    _arrow(ax, (5.0, 4.5), (5.0, 3.8))   # loop → métrica
    _arrow(ax, (5.0, 2.4), (5.0, 1.7))   # métrica → resultado

    return fig


if __name__ == "__main__":
    print("Generando diagrama: arquitectura del solver...")
    f1 = fig_arquitectura_solver()
    out1 = OUTPUT_DIR / "fig_arquitectura_solver.png"
    f1.savefig(out1, dpi=300, bbox_inches="tight")
    print(f"  → {out1}")

    print("\nGenerando diagrama: pipeline de sensibilidades...")
    f2 = fig_pipeline_sensibilidades()
    out2 = OUTPUT_DIR / "fig_pipeline_sensibilidades.png"
    f2.savefig(out2, dpi=300, bbox_inches="tight")
    print(f"  → {out2}")

    print(f"\nFiguras guardadas en:\n  {OUTPUT_DIR}")
