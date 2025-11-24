from __future__ import annotations

from typing import Callable, List, Sequence


class FuturesCurve:
    """
    Lightweight wrapper for an oil futures curve sampled on the lattice grid.

    Accepts either a discrete sequence (curve[m] -> time (m+1)*dt) or a callable
    of continuous time t. Provides lookup by step index to support calibration.
    """

    def __init__(self, curve: Sequence[float] | Callable[[float], float]) -> None:
        self._callable: Callable[[float], float] | None
        if callable(curve):
            self._callable = curve
            self._values: List[float] | None = None
        else:
            self._values = list(curve)
            self._callable = None

    def value(self, step_index: int, dt: float) -> float:
        """
        Return F(0, t) at t = step_index * dt.

        If initialized with a sequence, step_index must be within range.
        """
        if step_index < 1:
            raise ValueError("step_index must be >= 1")
        if self._values is not None:
            idx = step_index - 1
            if idx >= len(self._values):
                raise IndexError(f"Curve does not cover step_index={step_index}")
            return float(self._values[idx])
        if self._callable is None:
            raise RuntimeError("Curve callable not provided.")
        t = step_index * dt
        return float(self._callable(t))
