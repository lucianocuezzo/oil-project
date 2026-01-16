"""
Hull-White style trinomial tree tailored for a mean-reverting *log spot* factor.

We build the tree for an auxiliary process R* (or x_tilde) following:

    dR* = -a R* dt + sigma dz

In the energy context of Clewlow & Strickland, this R* is the zero-mean part of
x(t) = ln S(t), i.e. the logarithm of the oil spot price after a time-dependent
shift alpha(t). The first stage only cares about the OU dynamics; the structure
is agnostic to the commodity, but this implementation is written with oil in
mind.

Grid spacing is DeltaX = sigma * sqrt(3 * dt). Branching follows the three
regimes from Hull-White (central, lower, upper) to keep probabilities positive.

Calibration to a futures curve is handled outside via a calibrator that returns
a ShiftedOilTrinomialTree, leaving the base lattice untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


Branch = Tuple[int, int, int]  # (up, mid, down) child j-indices
Probabilities = Tuple[float, float, float]  # (pu, pm, pd)


@dataclass(frozen=True)
class Node:
    """A single node of the R* tree."""

    time_index: int
    j: int
    x_tilde: float        # x~ value at this node (zero-mean contribution to log-spot)
    branch_type: str      # "central", "lower", "upper"
    children: Branch      # child j-indices (up, mid, down)
    probabilities: Probabilities  # (pu, pm, pd)


class BaseOilTrinomialTree:
    """
    Hull-White style trinomial tree for a mean-reverting factor (e.g. ln S).

    Use `build` to create the lattice for the OU part R*. Calibration/shift is
    handled externally by a calibrator that returns a ShiftedOilTrinomialTree
    without mutating this instance. For oil, you typically interpret:

        x(t) = alpha(t) + R*(t)
        S(t) = exp(x(t))

    and choose alpha(t) to fit the forward/futures curve.
    """

    def __init__(
        self,
        n_steps: int,
        a: float,
        sigma: float,
        dt: float,
        delta_x: float,
        jmin: int,
        jmax: int,
        levels: List[Dict[int, Node]],
    ) -> None:
        self.n_steps = n_steps
        self.a = a
        self.sigma = sigma
        self.dt = dt
        self.delta_x = delta_x
        self.jmin = jmin
        self.jmax = jmax
        self.levels = levels

    # --------------------- building (first stage) --------------------- #
    @classmethod
    def build(
        cls,
        n_steps: int,
        a: float,
        sigma: float,
        dt: float,
        jmax: int | None = None,
        jmin: int | None = None,
    ) -> "BaseOilTrinomialTree":
        """
        Delegate lattice construction to ``OilTrinomialTreeBuilder``.

        This keeps the tree class focused on calibration/pricing logic while
        the builder encapsulates the Hull-White branching rules.
        """
        from oil_tree_builder import OilTrinomialTreeBuilder

        builder = OilTrinomialTreeBuilder(
            n_steps=n_steps,
            a=a,
            sigma=sigma,
            dt=dt,
            jmax=jmax,
            jmin=jmin,
        )
        return builder.build()


class ShiftedOilTrinomialTree:
    """
    Wrapper containing a base OU tree plus the calibrated shift (alphas).

    This avoids mutating the original lattice while keeping a plotting-friendly
    interface (levels, delta_x, n_steps, adjusted_factor).
    """

    def __init__(self, base_tree: BaseOilTrinomialTree, alphas: List[float], q_levels: List[Dict[int, float]]):
        self.base_tree = base_tree
        self.alphas = alphas
        self.q_levels = q_levels

    # Expose lattice attributes for consumers like TreePlotter.
    @property
    def levels(self) -> List[Dict[int, Node]]:
        return self.base_tree.levels

    @property
    def delta_x(self) -> float:
        return self.base_tree.delta_x

    @property
    def n_steps(self) -> int:
        return self.base_tree.n_steps

    @property
    def jmin(self) -> int:
        return self.base_tree.jmin

    @property
    def jmax(self) -> int:
        return self.base_tree.jmax

    def adjusted_factor(self, time_index: int, j: int) -> float:
        alpha_idx = min(time_index, len(self.alphas) - 1)
        return self.alphas[alpha_idx] + j * self.delta_x


def _format_probabilities(probs: Probabilities) -> str:
    """Helper to pretty-print (pu, pm, pd)."""
    pu, pm, pd = probs
    return f"pu={pu: .4f}, pm={pm: .4f}, pd={pd: .4f}"
