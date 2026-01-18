from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Sequence

from shared.operating_params import BaseOperatingParams
from tree.hull_white_tree import ShiftedOilTrinomialTree
from tree.oil_futures_curve import FuturesCurve


PricePath = Sequence[float] | Callable[[float], float]


@dataclass(frozen=True)
class NPVParams(BaseOperatingParams):
    """Operating parameters for deterministic NPV (no switching)."""


class ForwardNPVCalculator:
    """Deterministic NPV using the futures curve as expected prices."""

    def __init__(self, futures_curve: FuturesCurve, n_steps: int, params: NPVParams) -> None:
        self.futures_curve = futures_curve
        self.n_steps = n_steps
        self.params = params
        self.prices = self._expected_prices()

    def _expected_prices(self) -> List[float]:
        return [self.futures_curve.value(step_idx, self.params.dt) for step_idx in range(1, self.n_steps + 1)]

    def npv_from_step(self, start_step: int) -> float:
        n_steps = len(self.prices)
        beta = self.params.discount_factor
        npv = -(self.params.capex + self.params.switch_on_cost) * (beta ** start_step)

        for idx in range(start_step, n_steps):
            price = self.prices[idx]
            cf = ((price - self.params.variable_cost) * self.params.production_rate - self.params.fixed_on_cost) * self.params.dt
            npv += (beta ** (idx + 1)) * cf

        salvage = self.params.salvage_multiplier * self.prices[-1] if self.params.salvage_multiplier else 0.0
        npv += (beta ** n_steps) * salvage
        return npv

    def npv_schedule(self) -> List[float]:
        return [self.npv_from_step(k) for k in range(len(self.prices))]

    @staticmethod
    def earliest_invest_step(npvs: Sequence[float]) -> int | None:
        for k, v in enumerate(npvs):
            if v >= 0:
                return k
        return None

    def invest_now_npv(self) -> float:
        """NPV if you must decide at t=0 to invest or never invest."""
        return self.npv_from_step(0)


class TreeNPVCalculator:
    """
    Expected NPV over the calibrated tree.

    For linear cashflows this should match the forward-based NPV because the tree is
    calibrated to the futures curve. Still useful for sanity checks.
    """

    def __init__(
        self,
        shifted_tree: ShiftedOilTrinomialTree,
        params: NPVParams,
        price_fn: Callable[[ShiftedOilTrinomialTree, int, int], float] | None = None,
    ) -> None:
        self.tree = shifted_tree
        self.params = params
        self.price_fn = price_fn or (lambda tree, t, j: math.exp(tree.adjusted_factor(t, j)))

    def _expected_price_at_level(self, level_idx: int) -> float:
        # reach probabilities from calibrator: q_levels[level_idx]
        q_level = self.tree.q_levels[level_idx]
        return sum(prob * self.price_fn(self.tree, level_idx, j) for j, prob in q_level.items())

    def npv_from_step(self, start_step: int) -> float:
        n_steps = self.tree.n_steps
        beta = self.params.discount_factor
        npv = -(self.params.capex + self.params.switch_on_cost) * (beta ** start_step)

        for idx in range(start_step, n_steps):
            expected_p = self._expected_price_at_level(idx + 1)  # prices are for step idx+1
            cf = ((expected_p - self.params.variable_cost) * self.params.production_rate - self.params.fixed_on_cost) * self.params.dt
            npv += (beta ** (idx + 1)) * cf

        expected_p_T = self._expected_price_at_level(n_steps)
        salvage = self.params.salvage_multiplier * expected_p_T if self.params.salvage_multiplier else 0.0
        npv += (beta ** n_steps) * salvage
        return npv

    def npv_schedule(self) -> List[float]:
        return [self.npv_from_step(k) for k in range(self.tree.n_steps)]

    @staticmethod
    def earliest_invest_step(npvs: Sequence[float]) -> int | None:
        for k, v in enumerate(npvs):
            if v >= 0:
                return k
        return None

    def invest_now_npv(self) -> float:
        """NPV if you must decide at t=0 to invest or never invest."""
        return self.npv_from_step(0)
