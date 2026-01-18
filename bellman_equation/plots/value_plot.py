from __future__ import annotations

from typing import Any


class ValuePlotter:
    """Scatter plot of value vs time for each operating mode."""

    def __init__(self, tree: Any, solution: Any, title: str = "Value evolution per mode") -> None:
        self.tree = tree
        self.solution = solution
        self.title = title

    def plot(self):
        import matplotlib.pyplot as plt

        modes = [
            ("Uninvested", self.solution.value_pre, "tab:blue"),
            ("OFF", self.solution.value_off, "tab:orange"),
            ("ON", self.solution.value_on, "tab:green"),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True)
        for ax, (label, values, color) in zip(axes, modes):
            for t, level in enumerate(self.tree.levels):
                xs = [t] * len(level)
                ys = [values[t][j] for j in level]
                ax.scatter(xs, ys, s=25, color=color, alpha=0.7, label=f"t={t}")
            ax.set_title(label)
            ax.set_xlabel("time step")
            ax.set_ylabel("value")
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.legend(title="time index", fontsize=8)
        fig.suptitle(self.title)
        return fig, axes
