"""
Demo runner: build/calibrate the oil trinomial tree and optionally plot it.

Run directly with ``python run_tree_demo.py``. The script only depends on the
standard library; plotting is skipped unless ``matplotlib`` is installed in the
environment.
"""

from __future__ import annotations

from hull_white_tree import OilTrinomialTree, _format_probabilities
from plot_trinomial_tree import TreePlotter


def main() -> None:
    # Example build: 4 steps, Hull-White style lattice.
    tree = OilTrinomialTree.build(n_steps=4, a=0.1, sigma=0.01, dt=1.0)

    print(f"DeltaR = {tree.delta_r:.6f}")
    print(f"j-range: [{tree.jmin}, {tree.jmax}]")

    # Mock curve (replace with an oil forward/futures curve as needed).
    flat_curve = lambda t: 75.0 + 0.5 * t  # gentle contango for illustration
    alphas, q_levels = tree.calibrate_to_curve(flat_curve)

    print(f"alphas: {[f'{a:.5f}' for a in alphas]}")
    print("Branching probabilities by node (time_index, j):")

    for i, level in enumerate(tree.levels):
        print(f"\nLevel {i}")
        for j in sorted(level):
            node = level[j]
            probs_text = _format_probabilities(node.probabilities)
            children_text = f"children={node.children}"
            print(
                f"  (i={node.time_index}, j={node.j:2d}, R*={node.r_star: .5f}) "
                f"[{node.branch_type:7s}] {probs_text}; {children_text}"
            )

    print("\nDiscounted reach weights Q (per level):")
    for i, q_level in enumerate(q_levels):
        sorted_items = ', '.join(f"j={j}: {q_level[j]:.6f}" for j in sorted(q_level))
        print(f"  level {i}: {sorted_items}")

    # Optional plot (requires matplotlib).
    try:
        plotter = TreePlotter(tree, y_axis="j")
        ax = plotter.plot(annotate_probs=True, annotate_factors=True, title="Oil Trinomial Tree")
        import matplotlib.pyplot as plt

        plt.tight_layout()
        plt.show()
    except RuntimeError as exc:
        print(f"Plot skipped: {exc}")


if __name__ == "__main__":
    main()
