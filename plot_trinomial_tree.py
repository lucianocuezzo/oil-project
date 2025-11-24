"""
Plotting utilities for OilTrinomialTree.

Provides TreePlotter (no __main__), so plotting and demo logic can live in a
separate runner script.
"""

from __future__ import annotations

from typing import Dict, Tuple

from hull_white_tree import OilTrinomialTree, Probabilities


class TreePlotter:
    """
    Plot a trinomial tree produced by OilTrinomialTree.

    Args:
        tree: OilTrinomialTree instance (built and optionally calibrated).
        y_axis: "j" to plot vertical position by lattice index j,
                "r" to plot by R* value (j * DeltaR).
    """

    def __init__(self, tree: OilTrinomialTree, y_axis: str = "j") -> None:
        self.tree = tree
        self.y_axis = y_axis

    def _node_position(self, time_index: int, j: int) -> Tuple[float, float]:
        y = j if self.y_axis == "j" else j * self.tree.delta_r
        return float(time_index), float(y)

    def _positions(self) -> Dict[Tuple[int, int], Tuple[float, float]]:
        return {
            (node.time_index, node.j): self._node_position(node.time_index, node.j)
            for level in self.tree.levels
            for node in level.values()
        }

    def _rate_label(self, time_index: int, j: int) -> str:
        if not self.tree.alphas:
            return f"j={j}"
        # Last level has no alpha entry; reuse last known alpha for display.
        alpha_idx = min(time_index, len(self.tree.alphas) - 1)
        factor = self.tree.alphas[alpha_idx] + j * self.tree.delta_r
        return f"{factor:.4f}"

    def plot(
        self,
        ax=None,
        annotate_probs: bool = False,
        annotate_rates: bool = False,
        title: str | None = None,
    ):
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError("matplotlib is required for plotting") from exc

        positions = self._positions()
        ax = ax or plt.subplots(figsize=(10, 6))[1]

        # Edges
        for level in self.tree.levels[:-1]:
            for node in level.values():
                src_pos = positions[(node.time_index, node.j)]
                self._draw_edges(ax, src_pos, node.children, node.probabilities, positions, annotate_probs)

        # Nodes
        xs = [p[0] for p in positions.values()]
        ys = [p[1] for p in positions.values()]
        ax.scatter(xs, ys, color="black", zorder=3)

        if annotate_rates:
            for (i, j), (x, y) in positions.items():
                ax.text(x, y, self._rate_label(i, j), ha="center", va="bottom", fontsize=8, color="darkblue")

        ax.set_xlabel("time step")
        ax.set_ylabel("j index" if self.y_axis == "j" else "R*")
        ax.set_title(title or "Trinomial Tree")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_xlim(-0.5, self.tree.n_steps + 0.5)
        ax.set_ylim(min(ys) - 0.5, max(ys) + 0.5)
        return ax

    def _draw_edges(
        self,
        ax,
        src_pos: Tuple[float, float],
        children: Tuple[int, int, int],
        probs: Probabilities,
        positions: Dict[Tuple[int, int], Tuple[float, float]],
        annotate_probs: bool,
    ) -> None:
        for prob, child_j in zip(probs, children):
            dst_pos = positions.get((int(src_pos[0]) + 1, child_j))
            if dst_pos is None:
                continue
            ax.plot([src_pos[0], dst_pos[0]], [src_pos[1], dst_pos[1]], color="gray", linewidth=1, zorder=1)
            if annotate_probs:
                mid_x = 0.6 * src_pos[0] + 0.4 * dst_pos[0]
                mid_y = 0.6 * src_pos[1] + 0.4 * dst_pos[1]
                ax.text(mid_x, mid_y, f"{prob:.3f}", fontsize=7, color="maroon")
