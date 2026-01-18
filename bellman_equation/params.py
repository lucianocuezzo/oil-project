from __future__ import annotations

from dataclasses import dataclass

from shared.operating_params import BaseOperatingParams


@dataclass(frozen=True)
class SwitchingParams(BaseOperatingParams):
    fixed_off_cost: float = 0.0
    switch_off_cost: float = 0.0
    allow_start_on: bool = True

    def cashflow_on(self, price: float) -> float:
        margin = (price - self.variable_cost) * self.production_rate
        return (margin - self.fixed_on_cost) * self.dt

    def cashflow_off(self, price: float) -> float:
        return (-self.fixed_off_cost) * self.dt
