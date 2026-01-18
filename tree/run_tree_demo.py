"""
Demo runner: build/calibrate the oil trinomial tree and optionally plot it.

Run directly with ``python run_tree_demo.py``. The script only depends on the
standard library; plotting is skipped unless ``matplotlib`` is installed in the
environment.
"""

from __future__ import annotations

import math

from .hull_white_tree import _format_probabilities
from .oil_futures_curve import FuturesCurve
from .oil_tree_builder import OilTrinomialTreeBuilder
from .oil_tree_calibrator import OilTrinomialFuturesCalibrator
from plots.tree_plot import TreePlotter


def main() -> None:
    # Example build: 4 steps, Hull-White style lattice.
    builder = OilTrinomialTreeBuilder(n_steps=4, a=0.1, sigma=0.01, dt=1.0)
    base_tree = builder.build()

    print(f"DeltaX = {base_tree.delta_x:.6f}")
    print(f"j-range: [{base_tree.jmin}, {base_tree.jmax}]")

    # Mock curve (replace with an oil forward/futures curve as needed).
    # Keep it near 1.0 so shifted and unshifted trees sit close together on the plot.
    contango_curve = lambda t: math.exp(0.02 * t)  # mild 2%/year contango from spot ~1.0
    futures_curve = FuturesCurve(contango_curve)
    calibrator = OilTrinomialFuturesCalibrator(base_tree, futures_curve)
    shifted_tree = calibrator.calibrate()
    alphas = shifted_tree.alphas
    q_levels = shifted_tree.q_levels

    print(f"alphas: {[f'{a:.5f}' for a in alphas]}")
    print("Branching probabilities by node (time_index, j):")

    for i, level in enumerate(shifted_tree.levels):
        print(f"\nLevel {i}")
        for j in sorted(level):
            node = level[j]
            probs_text = _format_probabilities(node.probabilities)
            children_text = f"children={node.children}"
            x_adj = shifted_tree.adjusted_factor(node.time_index, node.j)
            print(
                f"  (i={node.time_index}, j={node.j:2d}, x~={node.x_tilde: .5f}, "
                f"x_adj={x_adj: .5f}, S=exp(x_adj)={math.exp(x_adj):.5f}) "
                f"[{node.branch_type:7s}] {probs_text}; {children_text}"
            )

    print("\nReach probabilities Q (per level):")
    for i, q_level in enumerate(q_levels):
        sorted_items = ', '.join(f"j={j}: {q_level[j]:.6f}" for j in sorted(q_level))
        print(f"  level {i}: {sorted_items}")

    # Optional plot (requires matplotlib).
    try:
        plotter = TreePlotter(shifted_tree, y_axis="r")
        ax = plotter.plot_original_and_shifted(
            annotate_factors=True,
            title="Original vs Shifted (Futures-Calibrated) Tree",
        )
        import matplotlib.pyplot as plt

        plt.tight_layout()
        plt.show()
    except RuntimeError as exc:
        print(f"Plot skipped: {exc}")


if __name__ == "__main__":
    main()
