from __future__ import annotations

import math
from typing import Dict, List

from hull_white_tree import BaseOilTrinomialTree, ShiftedOilTrinomialTree
from oil_futures_curve import FuturesCurve


class OilTrinomialFuturesCalibrator:
    """Calibrates a BaseOilTrinomialTree to an oil futures curve."""

    def __init__(self, base_tree: BaseOilTrinomialTree, futures_curve: FuturesCurve) -> None:
        self.base_tree = base_tree
        self.futures_curve = futures_curve

    def calibrate(self) -> ShiftedOilTrinomialTree:
        """
        Shift the tree so that E[exp(x_{m+1})] matches the supplied futures curve.

        This follows the Clewlow-Strickland commodity construction: propagate
        *probabilities* (no discounting) and choose alpha_m so that:

            sum_j p_{m+1,j} * exp(j * DeltaX + alpha_m) = F(0, t_{m+1})
        """
        alphas: List[float] = []
        q_levels: List[Dict[int, float]] = [{0: 1.0}]  # pure reach probabilities

        for m in range(self.base_tree.n_steps):
            # propagate one step without discounting (commodity-style)
            next_probs: Dict[int, float] = {}
            for j, node in self.base_tree.levels[m].items():
                reach = q_levels[m].get(j, 0.0)
                if reach == 0.0:
                    continue
                for prob, child_j in zip(node.probabilities, node.children):
                    next_probs[child_j] = next_probs.get(child_j, 0.0) + reach * prob

            fwd = self.futures_curve.value(m + 1, self.base_tree.dt)
            if fwd <= 0:
                raise ValueError(f"Forward must be positive at step {m+1}")

            expected_exp_r = sum(prob * math.exp(j * self.base_tree.delta_x) for j, prob in next_probs.items())
            if expected_exp_r <= 0:
                raise RuntimeError(f"Non-positive expected exp(r*) at step {m+1}")

            alpha_m = math.log(fwd / expected_exp_r)
            alphas.append(alpha_m)
            q_levels.append(next_probs)

        return ShiftedOilTrinomialTree(self.base_tree, alphas, q_levels)
