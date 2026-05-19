"""
Diagrama esquemático del shift de calibración — Capítulo 7.

Compara dos paneles:
  Izquierda  — árbol sin shift (factor OU R*, simétrico en torno a 0)
  Derecha    — árbol shifteado por alpha(t) para ajustar la curva de futuros

Ejecutar desde la raíz del repositorio:
    python chapter7/tree_shift_schematic.py
"""

from __future__ import annotations

import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch

OUTPUT_DIR = ROOT / "figures" / "chapter7"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


C_NODE_FILL_BASE = "#e0e0e0"
C_NODE_EDGE_BASE = "#909090"
C_NODE_FILL_SHIFT = "#d5e8d4"
C_NODE_EDGE_SHIFT = "#82b366"
C_ORIGIN_FILL = "#f8cecc"
C_ORIGIN_EDGE = "#b85450"
C_ARROW = "#404040"
C_CURVE = "#1f6feb"


N_STEPS = 3
DX = 2.2     # espaciado horizontal
DY = 0.9     # espaciado vertical entre niveles j


# alpha(t) sintetico — creciente y curvado, para ilustrar la calibración
def alpha(t: float) -> float:
    return 0.6 + 1.4 * (1.0 - math.exp(-0.7 * t))


def _node(ax, x, y, fill, edge, radius=0.18, label=None, fontsize=9):
    circ = Circle((x, y), radius, facecolor=fill, edgecolor=edge, linewidth=1.4, zorder=3)
    ax.add_patch(circ)
    if label:
        ax.text(x, y, label, ha="center", va="center", fontsize=fontsize, zorder=4)


def _arrow(ax, p1, p2, color=C_ARROW, lw=1.0, radius=0.18):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    dist = math.hypot(dx, dy)
    ux, uy = dx / dist, dy / dist
    start = (p1[0] + ux * radius, p1[1] + uy * radius)
    end = (p2[0] - ux * radius, p2[1] - uy * radius)
    a = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10,
                        color=color, linewidth=lw, zorder=2)
    ax.add_patch(a)


def _draw_tree(ax, shift_fn, title, fill, edge):
    """
    shift_fn(i) -> desplazamiento vertical aplicado al nivel i.
                   Para el árbol sin calibrar: shift_fn(i) = 0.
                   Para el árbol calibrado:    shift_fn(i) = alpha(i*dt).
    """
    nodes = {}
    for i in range(N_STEPS + 1):
        for j in range(-i, i + 1):
            x = i * DX
            y = j * DY + shift_fn(i)
            nodes[(i, j)] = (x, y)

    # ramas (cada nodo se ramifica a +1, 0, -1 en el siguiente paso)
    for i in range(N_STEPS):
        for j in range(-i, i + 1):
            p1 = nodes[(i, j)]
            for dj in (+1, 0, -1):
                p2 = nodes[(i + 1, j + dj)]
                _arrow(ax, p1, p2, color=C_ARROW, lw=0.9)

    # nodos
    for (i, j), (x, y) in nodes.items():
        if (i, j) == (0, 0):
            _node(ax, x, y, C_ORIGIN_FILL, C_ORIGIN_EDGE)
        else:
            _node(ax, x, y, fill, edge)

    # ejes
    y_min = -N_STEPS * DY - 0.6
    y_max = max(y for _, y in nodes.values()) + 0.9

    ax.set_xlim(-0.7, N_STEPS * DX + 1.2)
    ax.set_ylim(y_min - 0.4, y_max + 0.3)
    ax.set_aspect("equal")
    ax.axis("off")

    # eje temporal
    y_axis = y_min
    ax.annotate("", xy=(N_STEPS * DX + 0.8, y_axis), xytext=(-0.3, y_axis),
                arrowprops=dict(arrowstyle="->", color="#606060", lw=1.0))
    ax.text(N_STEPS * DX + 0.85, y_axis, "tiempo", ha="left", va="center",
            fontsize=9, color="#606060", style="italic")
    for i in range(N_STEPS + 1):
        x = i * DX
        ax.plot([x, x], [y_axis - 0.07, y_axis + 0.07], color="#606060", lw=1.0)
        label = "$t=0$" if i == 0 else (r"$t=\Delta t$" if i == 1 else fr"$t={i}\Delta t$")
        ax.text(x, y_axis - 0.28, label, ha="center", va="top", fontsize=9)

    ax.set_title(title, fontsize=12, pad=10)

    return nodes, y_min, y_max


def fig_tree_shift() -> plt.Figure:
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 6.5))

    # ── Panel izquierdo: árbol sin shift ──────────────────────────────────────
    _draw_tree(
        ax_l,
        shift_fn=lambda i: 0.0,
        title=r"Árbol sin calibrar  —  factor OU $\tilde{x} = x - \alpha(t)$",
        fill=C_NODE_FILL_BASE,
        edge=C_NODE_EDGE_BASE,
    )
    # línea de referencia en 0 (simetría) — la anotación va a la izquierda para
    # no chocar con los nodos del último paso
    ax_l.axhline(0.0, color="#999999", lw=0.8, ls=":", zorder=1,
                 xmin=0.05, xmax=0.95)
    ax_l.text(-0.2, 0.0, r"$\mathbb{E}[\tilde{x}] = 0$ ",
              ha="right", va="center", fontsize=9, color="#666666", style="italic")

    # eje vertical cualitativo (label a la izquierda del tope para no chocar
    # con el título)
    ax_l.annotate("", xy=(-0.5, N_STEPS * DY + 0.3),
                  xytext=(-0.5, -N_STEPS * DY - 0.3),
                  arrowprops=dict(arrowstyle="->", color="#606060", lw=1.0))
    ax_l.text(-0.65, N_STEPS * DY + 0.3, r"$\tilde{x}$",
              ha="right", va="center", fontsize=12, color="#606060", style="italic")

    # ── Panel derecho: árbol shifteado ────────────────────────────────────────
    nodes_r, y_min_r, y_max_r = _draw_tree(
        ax_r,
        shift_fn=lambda i: alpha(i * 1.0),
        title=r"Árbol calibrado  —  ajusta la curva de futuros $F(0,t)$",
        fill=C_NODE_FILL_SHIFT,
        edge=C_NODE_EDGE_SHIFT,
    )

    # curva de futuros F(0,t) suave (que pasa por el centro de cada slice)
    t_grid = np.linspace(0, N_STEPS, 200)
    x_grid = t_grid * DX
    y_curve = np.array([alpha(t) for t in t_grid])
    ax_r.plot(x_grid, y_curve, color=C_CURVE, lw=2.0, ls="--", zorder=1.5,
              label=r"$\alpha(t)$  (shift de calibración)")
    ax_r.legend(loc="upper left", frameon=True, fontsize=10)

    # eje vertical cualitativo — log-precio del petróleo (label a la izquierda
    # del tope de la flecha para no chocar con el título)
    ax_r.annotate("", xy=(-0.5, y_max_r + 0.3),
                  xytext=(-0.5, y_min_r - 0.0),
                  arrowprops=dict(arrowstyle="->", color="#606060", lw=1.0))
    ax_r.text(-0.65, y_max_r + 0.3, r"$x = \log S$",
              ha="right", va="center", fontsize=11, color="#606060", style="italic")

    # leyenda inferior comun
    # fig.text(0.5, 0.02,
    #          r"Calibración: se elige $\alpha(t)$ para que  $\mathbb{E}[\,S(t)\,] = F(0,t)$  "
    #          r"en cada paso del árbol.   $x = \alpha(t) + \tilde{x} = \log S$  "
    #          r"y el precio se recupera como  $S = e^{x}$.",
    #          ha="center", va="bottom", fontsize=10, style="italic", color="#404040")

    plt.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


if __name__ == "__main__":
    print("Generando esquema del shift de calibracion...")
    fig = fig_tree_shift()
    out = OUTPUT_DIR / "fig_arbol_trinomial_shift.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"  -> {out}")
