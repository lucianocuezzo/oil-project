"""
Plotting utilities for OilTrinomialTree.

Provides TreePlotter (no __main__), so plotting and demo logic can live in a
separate runner script.
"""

from __future__ import annotations

from typing import Dict, Tuple, Union

from hull_white_tree import BaseOilTrinomialTree, Probabilities, ShiftedOilTrinomialTree

TreeLike = Union[BaseOilTrinomialTree, ShiftedOilTrinomialTree]


class TreePlotter:
    """
    Plot a trinomial tree produced by BaseOilTrinomialTree or a shifted wrapper.

    Args:
        tree: BaseOilTrinomialTree (base) or ShiftedOilTrinomialTree instance.
        y_axis: "j" to plot vertical position by lattice index j,
                "r" to plot by R* value (j * DeltaX).
    """

    def __init__(self, tree: TreeLike, y_axis: str = "j") -> None:
        self.tree = tree
        self.y_axis = y_axis

    def _node_position(self, time_index: int, j: int) -> Tuple[float, float]:
        y = j if self.y_axis == "j" else j * self.tree.delta_x
        return float(time_index), float(y)

    def _shifted_position(self, time_index: int, j: int) -> Tuple[float, float]:
        """Position using the calibrated shift (alpha + j*DeltaR) on the y-axis."""
        if not isinstance(self.tree, ShiftedOilTrinomialTree):
            raise RuntimeError("Shifted positions require a ShiftedOilTrinomialTree.")
        y = j if self.y_axis == "j" else self.tree.adjusted_factor(time_index, j)
        return float(time_index), float(y)

    def _positions(self) -> Dict[Tuple[int, int], Tuple[float, float]]:
        return {
            (node.time_index, node.j): self._node_position(node.time_index, node.j)
            for level in self.tree.levels
            for node in level.values()
        }

    def _factor_label(self, time_index: int, j: int) -> str:
        if not isinstance(self.tree, ShiftedOilTrinomialTree):
            return f"j={j}"
        # Last level has no alpha entry; reuse last known alpha for display.
        alpha_idx = min(time_index, len(self.tree.alphas) - 1)
        factor = self.tree.alphas[alpha_idx] + j * self.tree.delta_x
        return f"{factor:.4f}"

    def plot(
        self,
        ax=None,
        annotate_probs: bool = False,
        annotate_factors: bool = False,
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

        if annotate_factors:
            for (i, j), (x, y) in positions.items():
                ax.text(x, y, self._factor_label(i, j), ha="center", va="bottom", fontsize=8, color="darkblue")

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

    def plot_original_and_shifted(
        self,
        ax=None,
        annotate_factors: bool = False,
        title: str | None = None,
    ):
        """
        Overlay the original (unshifted) and shifted (calibrated) trees.

        Original positions are plotted in gray; shifted (alpha-adjusted) in blue.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError("matplotlib is required for plotting") from exc

        if not isinstance(self.tree, ShiftedOilTrinomialTree):
            raise RuntimeError("Provide a ShiftedOilTrinomialTree to plot shifted positions.")

        ax = ax or plt.subplots(figsize=(10, 6))[1]
        base_tree = self.tree.base_tree
        positions_orig = {
            (node.time_index, node.j): (
                float(node.time_index),
                float(node.j if self.y_axis == "j" else node.j * base_tree.delta_x),
            )
            for level in base_tree.levels
            for node in level.values()
        }
        positions_shifted = {
            (node.time_index, node.j): self._shifted_position(node.time_index, node.j)
            for level in base_tree.levels
            for node in level.values()
        }

        # Draw edges for both sets (original lighter).
        for level in base_tree.levels[:-1]:
            for node in level.values():
                src_o = positions_orig[(node.time_index, node.j)]
                src_s = positions_shifted[(node.time_index, node.j)]
                self._draw_edges(ax, src_o, node.children, node.probabilities, positions_orig, annotate_probs=False)
                self._draw_edges(ax, src_s, node.children, node.probabilities, positions_shifted, annotate_probs=False)

        # Nodes
        xs_o = [p[0] for p in positions_orig.values()]
        ys_o = [p[1] for p in positions_orig.values()]
        xs_s = [p[0] for p in positions_shifted.values()]
        ys_s = [p[1] for p in positions_shifted.values()]

        ax.scatter(xs_o, ys_o, color="gray", alpha=0.5, label="original", zorder=2)
        ax.scatter(xs_s, ys_s, color="navy", alpha=0.8, label="shifted", zorder=3)

        if annotate_factors:
            for (i, j), (x, y) in positions_shifted.items():
                x_adj = self.tree.adjusted_factor(i, j)
                ax.text(x, y, f"{x_adj:.4f}", ha="center", va="bottom", fontsize=7, color="navy")

        ax.set_xlabel("time step")
        ylabel = "j index" if self.y_axis == "j" else "factor"
        ax.set_ylabel(ylabel)
        ax.set_title(title or "Original vs Shifted Tree")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        ax.set_xlim(-0.5, self.tree.n_steps + 0.5)
        margin = max(0.5, 0.1 * (max(ys_o + ys_s) - min(ys_o + ys_s)))
        ax.set_ylim(min(ys_o + ys_s) - margin, max(ys_o + ys_s) + margin)
        return ax
