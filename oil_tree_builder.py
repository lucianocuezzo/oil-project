from __future__ import annotations

import math
from typing import Dict, List

from hull_white_tree import BaseOilTrinomialTree, Branch, Node, Probabilities


class OilTrinomialTreeBuilder:
    """Builder responsible for constructing the Hull-White style lattice."""

    def __init__(
        self,
        n_steps: int,
        a: float,
        sigma: float,
        dt: float,
        jmax: int | None = None,
        jmin: int | None = None,
    ) -> None:
        self.n_steps = n_steps
        self.a = a
        self.sigma = sigma
        self.dt = dt
        self._jmax_input = jmax
        self._jmin_input = jmin

    def build(self) -> BaseOilTrinomialTree:
        """Create and return a BaseOilTrinomialTree instance."""
        self._validate_inputs()
        jmax, jmin = self._resolve_bounds()

        delta_x = self.sigma * math.sqrt(3.0 * self.dt)
        levels: List[Dict[int, Node]] = []

        for i in range(self.n_steps + 1):
            # Natural bounds from the trinomial tree + hard bounds [jmin, jmax]
            j_low = max(jmin, -i)
            j_high = min(jmax, i)
            level_nodes: Dict[int, Node] = {}

            for j in range(j_low, j_high + 1):
                branch = self._branch_type(j, jmin, jmax)
                probs = self._probabilities(branch, self.a, self.dt, j)
                children = self._children_for_branch(branch, j)
                x_tilde = j * delta_x
                level_nodes[j] = Node(
                    time_index=i,
                    j=j,
                    x_tilde=x_tilde,
                    branch_type=branch,
                    children=children,
                    probabilities=probs,
                )

            levels.append(level_nodes)

        return BaseOilTrinomialTree(
            n_steps=self.n_steps,
            a=self.a,
            sigma=self.sigma,
            dt=self.dt,
            delta_x=delta_x,
            jmin=jmin,
            jmax=jmax,
            levels=levels,
        )

    def _validate_inputs(self) -> None:
        if self.n_steps < 1:
            raise ValueError("n_steps must be at least 1")
        if self.a <= 0 or self.sigma <= 0 or self.dt <= 0:
            raise ValueError("a, sigma, and dt must be positive")

    def _resolve_bounds(self) -> tuple[int, int]:
        # If boundaries not supplied, use Hull-White threshold
        if self._jmax_input is None:
            j_switch = 0.184 / (self.a * self.dt)
            jmax = math.ceil(j_switch)
        else:
            jmax = self._jmax_input

        if self._jmin_input is None:
            jmin = -jmax
        else:
            jmin = self._jmin_input

        return jmax, jmin

    @staticmethod
    def _branch_type(j: int, jmin: int, jmax: int) -> str:
        """Choose branching regime based on the node index."""
        if j <= jmin:
            return "lower"
        if j >= jmax:
            return "upper"
        return "central"

    @staticmethod
    def _children_for_branch(branch: str, j: int) -> Branch:
        """Child j-indices for the chosen branching regime."""
        if branch == "central":
            return (j + 1, j, j - 1)
        if branch == "lower":
            return (j + 2, j + 1, j)
        if branch == "upper":
            return (j, j - 1, j - 2)
        raise ValueError(f"Unknown branch type '{branch}'")

    @staticmethod
    def _probabilities(branch: str, a: float, dt: float, j: int) -> Probabilities:
        """
        Probabilities for the three branches, matching the first two moments.

        These follow Hull and White section 30.7 with DeltaX = sigma * sqrt(3*dt),
        and are valid for any OU process (log spot, gas storage factors, etc.).
        """
        aj_dt = a * j * dt
        a2j2_dt2 = (a * a) * (j * j) * (dt * dt)

        if branch == "central":
            pu = 1 / 6 + 0.5 * (a2j2_dt2 - aj_dt)
            pm = 2 / 3 - a2j2_dt2
            pd = 1 / 6 + 0.5 * (a2j2_dt2 + aj_dt)
            return pu, pm, pd

        if branch == "lower":
            pu = 1 / 6 + 0.5 * (a2j2_dt2 + aj_dt)
            pm = -1 / 3 - a2j2_dt2 - 2 * aj_dt
            pd = 7 / 6 + 0.5 * (a2j2_dt2 + 3 * aj_dt)
            return pu, pm, pd

        if branch == "upper":
            pu = 7 / 6 + 0.5 * (a2j2_dt2 - 3 * aj_dt)
            pm = -1 / 3 - a2j2_dt2 + 2 * aj_dt
            pd = 1 / 6 + 0.5 * (a2j2_dt2 - aj_dt)
            return pu, pm, pd

        raise ValueError(f"Unknown branch type '{branch}'")
