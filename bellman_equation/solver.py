from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Literal

from tree.hull_white_tree import BaseOilTrinomialTree, Node, ShiftedOilTrinomialTree

from .params import SwitchingParams

TreeLike = BaseOilTrinomialTree | ShiftedOilTrinomialTree
PriceFunction = Callable[[TreeLike, int, int], float]
TerminalPayoff = Callable[[float], float]
Action = Literal["stay_on", "switch_off", "stay_off", "switch_on"]
ValueLevels = List[Dict[int, float]]
PolicyLevels = List[Dict[int, Action]]


def default_price_fn(tree: TreeLike, time_index: int, j: int) -> float:
    node = tree.levels[time_index][j]
    log_price = tree.adjusted_factor(time_index, j) if hasattr(tree, "adjusted_factor") else node.x_tilde
    return math.exp(log_price)


@dataclass
class BellmanSolution:
    value_on: ValueLevels
    value_off: ValueLevels
    policy_on: PolicyLevels
    policy_off: PolicyLevels


class SwitchingBellmanSolver:
    """Backward induction for the two-state (ON/OFF) Bellman system with switching costs."""

    def __init__(
        self,
        tree: TreeLike,
        params: SwitchingParams,
        price_fn: PriceFunction | None = None,
        terminal_on: TerminalPayoff | None = None,
        terminal_off: TerminalPayoff | None = None,
    ) -> None:
        self.tree = tree
        self.params = params
        self.price_fn = price_fn or default_price_fn
        self.terminal_on = terminal_on or (lambda price: 0.0)
        self.terminal_off = terminal_off or (lambda price: 0.0)

    def solve(self) -> BellmanSolution:
        n_steps = self.tree.n_steps
        discount = self.params.discount_factor

        value_on: ValueLevels = [dict() for _ in range(n_steps + 1)]
        value_off: ValueLevels = [dict() for _ in range(n_steps + 1)]
        policy_on: PolicyLevels = [dict() for _ in range(n_steps)]
        policy_off: PolicyLevels = [dict() for _ in range(n_steps)]

        last_idx = n_steps
        for j in self.tree.levels[last_idx]:
            price = self.price_fn(self.tree, last_idx, j)
            value_on[last_idx][j] = self.terminal_on(price)
            value_off[last_idx][j] = self.terminal_off(price)

        for t in reversed(range(n_steps)):
            level = self.tree.levels[t]
            next_on = value_on[t + 1]
            next_off = value_off[t + 1]

            for j, node in level.items():
                price = self.price_fn(self.tree, t, j)

                cont_on = self._expected_value(next_on, node)
                cont_off = self._expected_value(next_off, node)

                stay_on = self.params.cashflow_on(price) + discount * cont_on
                shut_down = -self.params.switch_off_cost + self.params.cashflow_off(price) + discount * cont_off

                if stay_on >= shut_down:
                    value_on[t][j] = stay_on
                    policy_on[t][j] = "stay_on"
                else:
                    value_on[t][j] = shut_down
                    policy_on[t][j] = "switch_off"

                stay_off = self.params.cashflow_off(price) + discount * cont_off
                start_up = -self.params.switch_on_cost + self.params.cashflow_on(price) + discount * cont_on

                if start_up > stay_off:
                    value_off[t][j] = start_up
                    policy_off[t][j] = "switch_on"
                else:
                    value_off[t][j] = stay_off
                    policy_off[t][j] = "stay_off"

        return BellmanSolution(
            value_on=value_on,
            value_off=value_off,
            policy_on=policy_on,
            policy_off=policy_off,
        )

    @staticmethod
    def _expected_value(next_values: Dict[int, float], node: Node) -> float:
        total = 0.0
        for prob, child_j in zip(node.probabilities, node.children):
            total += prob * next_values[child_j]
        return total
