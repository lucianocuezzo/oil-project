from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BaseOperatingParams:
    dt: float
    discount_rate: float
    production_rate: float
    variable_cost: float
    fixed_on_cost: float
    capex: float
    switch_on_cost: float = 0.0  # startup cost at investment
    salvage_multiplier: float = 0.0  # salvage = multiplier * price at horizon

    @property
    def discount_factor(self) -> float:
        return math.exp(-self.discount_rate * self.dt)
