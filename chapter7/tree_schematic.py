"""
Diagrama esquemático del árbol trinomial — Capítulo 7.

Fija la notación S_{i,j} y las tres probabilidades de transición (p_u, p_m, p_d)
usadas a lo largo de los Capítulos 7 y 8. Muestra dos pasos para ilustrar la
recombinación de las ramas.

Ejecutar desde la raíz del repositorio:
    python chapter7/tree_schematic.py
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

OUTPUT_DIR = ROOT / "figures" / "chapter7"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


C_NODE_FILL = "#d5e8d4"
C_NODE_EDGE = "#82b366"
C_ORIGIN_FILL = "#f8cecc"
C_ORIGIN_EDGE = "#b85450"
C_ARROW = "#404040"
C_PROB_ARROW = "#b85450"


def _node(ax, x, y, label, fill=C_NODE_FILL, edge=C_NODE_EDGE, radius=0.22, fontsize=11):
    circ = Circle((x, y), radius, facecolor=fill, edgecolor=edge, linewidth=1.5, zorder=3)
    ax.add_patch(circ)
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize, zorder=4)


def _arrow(ax, p1, p2, color=C_ARROW, label=None, label_offset=(0, 0),
           fontsize=11, lw=1.2, radius=0.22):
    """Flecha entre dos nodos, recortada al borde de cada círculo."""
    import math as _m
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    dist = _m.hypot(dx, dy)
    ux, uy = dx / dist, dy / dist
    start = (p1[0] + ux * radius, p1[1] + uy * radius)
    end = (p2[0] - ux * radius, p2[1] - uy * radius)
    arrow = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=12,
        color=color, linewidth=lw, zorder=2,
    )
    ax.add_patch(arrow)
    if label:
        mx = (start[0] + end[0]) / 2 + label_offset[0]
        my = (start[1] + end[1]) / 2 + label_offset[1]
        ax.text(mx, my, label, fontsize=fontsize, color=color,
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.18",
                          facecolor="white", edgecolor="none"))


def fig_arbol_trinomial_esquema() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.set_xlim(-1.2, 7.0)
    ax.set_ylim(-3.0, 3.3)
    ax.set_aspect("equal")
    ax.axis("off")

    # Posiciones de los nodos
    DX = 2.5   # separación horizontal entre pasos
    DY = 1.1   # separación vertical entre niveles j

    nodes = {
        # (i, j) : (x, y, label)
        (0, 0):  (0 * DX, 0 * DY, "$x_{0,0}$"),
        (1, 1):  (1 * DX, 1 * DY, "$x_{1,1}$"),
        (1, 0):  (1 * DX, 0 * DY, "$x_{1,0}$"),
        (1, -1): (1 * DX, -1 * DY, "$x_{1,-1}$"),
        (2, 2):  (2 * DX, 2 * DY, "$x_{2,2}$"),
        (2, 1):  (2 * DX, 1 * DY, "$x_{2,1}$"),
        (2, 0):  (2 * DX, 0 * DY, "$x_{2,0}$"),
        (2, -1): (2 * DX, -1 * DY, "$x_{2,-1}$"),
        (2, -2): (2 * DX, -2 * DY, "$x_{2,-2}$"),
    }

    # Dibujar nodos
    for key, (x, y, label) in nodes.items():
        if key == (0, 0):
            _node(ax, x, y, label, fill=C_ORIGIN_FILL, edge=C_ORIGIN_EDGE)
        else:
            _node(ax, x, y, label)

    # Conexiones paso 0 → paso 1 (con probabilidades etiquetadas)
    p0 = (nodes[(0, 0)][0], nodes[(0, 0)][1])
    _arrow(ax, p0, (nodes[(1, 1)][0],  nodes[(1, 1)][1]),
           color=C_PROB_ARROW, label="$p_u$", label_offset=(-0.05, 0.18), lw=1.6)
    _arrow(ax, p0, (nodes[(1, 0)][0],  nodes[(1, 0)][1]),
           color=C_PROB_ARROW, label="$p_m$", label_offset=(0, 0.18), lw=1.6)
    _arrow(ax, p0, (nodes[(1, -1)][0], nodes[(1, -1)][1]),
           color=C_PROB_ARROW, label="$p_d$", label_offset=(-0.05, -0.18), lw=1.6)

    # Conexiones paso 1 → paso 2 (sin etiquetas, mismo patrón de ramificación)
    for j_parent in (1, 0, -1):
        x1, y1, _ = nodes[(1, j_parent)]
        for j_child in (j_parent + 1, j_parent, j_parent - 1):
            x2, y2, _ = nodes[(2, j_child)]
            _arrow(ax, (x1, y1), (x2, y2), color=C_ARROW, lw=1.0)

    # Eje temporal abajo
    y_axis = -2.55
    ax.annotate("", xy=(2 * DX + 0.9, y_axis), xytext=(-0.4, y_axis),
                arrowprops=dict(arrowstyle="->", color="#606060", lw=1.2))
    ax.text(2 * DX + 0.95, y_axis, "tiempo", ha="left", va="center",
            fontsize=10, color="#606060", style="italic")

    # Eje vertical cualitativo (precio del petróleo, sin escala)
    x_axis = -0.85
    ax.annotate("", xy=(x_axis, 2.55), xytext=(x_axis, y_axis),
                arrowprops=dict(arrowstyle="->", color="#606060", lw=1.2))
    ax.text(x_axis, 2.75, r"$x = \log S$", ha="center", va="bottom",
            fontsize=11, color="#606060", style="italic")

    # Marcas de tiempo
    for i, label in [(0, "$t = 0$"), (1, r"$t = \Delta t$"), (2, r"$t = 2\,\Delta t$")]:
        x = i * DX
        ax.plot([x, x], [y_axis - 0.08, y_axis + 0.08], color="#606060", lw=1.2)
        ax.text(x, y_axis - 0.28, label, ha="center", va="top", fontsize=11)

    # Leyenda explicativa (esquina superior derecha para no chocar con el eje y)
    legend_text = (
        r"$x_{i,j} = \log S_{i,j}$ : log-precio en el paso $i$, nivel $j$" "\n"
        r"$S_{i,j} = e^{x_{i,j}}$ : precio del petróleo correspondiente" "\n"
        r"$p_u, p_m, p_d$ : probabilidades de transición  ($p_u + p_m + p_d = 1$)"
    )
    ax.text(0.98, 0.98, legend_text, transform=ax.transAxes,
            ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="white", edgecolor="#cccccc"))

    return fig


if __name__ == "__main__":
    print("Generando esquema del árbol trinomial...")
    fig = fig_arbol_trinomial_esquema()
    out = OUTPUT_DIR / "fig_arbol_trinomial_esquema.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"  -> {out}")
