from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SwitchingParams:
    dt: float
    discount_rate: float
    production_rate: float
    variable_cost: float
    fixed_on_cost: float
    fixed_off_cost: float
    switch_on_cost: float
    switch_off_cost: float
    capex: float = 0.0
    allow_start_on: bool = True

    def cashflow_on(self, price: float) -> float:
        margin = (price - self.variable_cost) * self.production_rate
        return (margin - self.fixed_on_cost) * self.dt

    def cashflow_off(self, price: float) -> float:
        return (-self.fixed_off_cost) * self.dt

    @property
    def discount_factor(self) -> float:
        return math.exp(-self.discount_rate * self.dt)
