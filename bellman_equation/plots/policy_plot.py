from __future__ import annotations

from typing import Any, Callable


class PolicyPlotter:
    """Scatter plot of policy regions (action vs time/price) for each mode."""

    def __init__(
        self,
        tree: Any,
        solution: Any,
        price_fn: Callable[[Any, int, int], float],
        title: str = "Policy regions (color = action)",
    ) -> None:
        self.tree = tree
        self.solution = solution
        self.price_fn = price_fn
        self.title = title

    def plot(self):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch

        mode_policies = [
            ("Uninvested", self.solution.policy_pre, {"wait": "tab:blue", "invest_off": "tab:orange", "invest_on": "tab:green"}),
            ("OFF", self.solution.policy_off, {"stay_off": "tab:blue", "switch_on": "tab:green"}),
            ("ON", self.solution.policy_on, {"stay_on": "tab:green", "switch_off": "tab:red"}),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True)
        for ax, (label, policies, cmap) in zip(axes, mode_policies):
            for t, level in enumerate(self.tree.levels[:-1]):  # policies defined up to n_steps-1
                for j in level:
                    action = policies[t][j]
                    price = self.price_fn(self.tree, t, j)
                    ax.scatter(t, price, color=cmap.get(action, "gray"), s=25, alpha=0.9)
            ax.set_title(label)
            ax.set_xlabel("time step")
            ax.set_ylabel("price")
            ax.grid(True, linestyle="--", alpha=0.3)
            legend_handles = [Patch(color=color, label=action) for action, color in cmap.items()]
            ax.legend(handles=legend_handles, title="action", fontsize=8)
        fig.suptitle(self.title)
        return fig, axes
