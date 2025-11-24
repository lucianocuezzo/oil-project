"""
Hull-White style trinomial tree tailored for a mean-reverting *log spot* factor.

We build the tree for an auxiliary process R* (or x̃) following:

    dR* = -a R* dt + sigma dz

In the energy context of Clewlow & Strickland, this R* is the zero-mean part of
x(t) = ln S(t), i.e. the logarithm of the oil spot price after a time-dependent
shift alpha(t). The first stage only cares about the OU dynamics; the structure
is agnostic to the commodity, but this implementation is written with oil in
mind.

Grid spacing is DeltaR = sigma * sqrt(3 * dt). Branching follows the three
regimes from Hull-White (central, lower, upper) to keep probabilities positive.

The class also provides a calibration hook for the "second stage", where an
offset alpha(t) is solved so that the tree matches an external curve (for
example, an oil futures curve). The calibration is generic: you can plug any
curve and customize the alpha solver / pricing equation via the
alpha_solver / discount_factor_fn hooks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple, Union


Branch = Tuple[int, int, int]  # (up, mid, down) child j-indices
Probabilities = Tuple[float, float, float]  # (pu, pm, pd)


@dataclass(frozen=True)
class Node:
    """A single node of the R* tree."""

    time_index: int
    j: int
    r_star: float         # R* value at this node (e.g. contribution to log-spot)
    branch_type: str      # "central", "lower", "upper"
    children: Branch      # child j-indices (up, mid, down)
    probabilities: Probabilities  # (pu, pm, pd)


class OilTrinomialTree:
    """
    Hull-White style trinomial tree for a mean-reverting factor (e.g. ln S).

    Use `build` to create the lattice for the OU part R*. Then you can use
    `calibrate_to_curve` as a second stage to solve for a time-dependent shift
    alpha(t) so that the model matches an external curve (oil futures, discount
    factors, etc.). For oil, you typically interpret:

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
        delta_r: float,
        jmin: int,
        jmax: int,
        levels: List[Dict[int, Node]],
    ) -> None:
        self.n_steps = n_steps
        self.a = a
        self.sigma = sigma
        self.dt = dt
        self.delta_r = delta_r
        self.jmin = jmin
        self.jmax = jmax
        self.levels = levels
        # Second-stage calibration outputs:
        self.alphas: List[float] = []              # alpha(t_i)
        self.q_levels: List[Dict[int, float]] = []  # “weights” propagated along the tree

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
    ) -> "OilTrinomialTree":
        """
        Build the R* lattice for the OU process:

            dR* = -a R* dt + sigma dW.

        Args:
            n_steps: number of time steps.
            a: mean reversion speed.
            sigma: volatility.
            dt: time increment (constant step).
            jmax: optional upper boundary for branching switch.
            jmin: optional lower boundary for branching switch.
        """
        if n_steps < 1:
            raise ValueError("n_steps must be at least 1")
        if a <= 0 or sigma <= 0 or dt <= 0:
            raise ValueError("a, sigma, and dt must be positive")

        # If boundaries not supplied, use Hull-White threshold
        if jmax is None:
            j_switch = 0.184 / (a * dt)
            jmax = math.ceil(j_switch)
        if jmin is None:
            jmin = -jmax

        delta_r = sigma * math.sqrt(3.0 * dt)
        levels: List[Dict[int, Node]] = []

        for i in range(n_steps + 1):
            # Natural bounds from the trinomial tree + hard bounds [jmin, jmax]
            j_low = max(jmin, -i)
            j_high = min(jmax, i)
            level_nodes: Dict[int, Node] = {}

            for j in range(j_low, j_high + 1):
                branch = cls._branch_type(j, jmin, jmax)
                probs = cls._probabilities(branch, a, dt, j)
                children = cls._children_for_branch(branch, j)
                r_star = j * delta_r
                level_nodes[j] = Node(
                    time_index=i,
                    j=j,
                    r_star=r_star,
                    branch_type=branch,
                    children=children,
                    probabilities=probs,
                )

            levels.append(level_nodes)

        return cls(
            n_steps=n_steps,
            a=a,
            sigma=sigma,
            dt=dt,
            delta_r=delta_r,
            jmin=jmin,
            jmax=jmax,
            levels=levels,
        )

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

        These follow Hull & White section 30.7 with DeltaR = sigma * sqrt(3*dt),
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

    # --------------------- calibration (second stage) --------------------- #
    def calibrate_to_curve(
        self,
        curve: Union[Sequence[float], Callable[[float], float]],
        alpha_solver: Callable[[Dict[int, float], float, float, float], float] | None = None,
        discount_factor_fn: Callable[[int, float, float, float], float] | None = None,
    ) -> Tuple[List[float], List[Dict[int, float]]]:
        """
        Calibrate alpha(t) so model prices match an external curve.

        Args:
            curve:
                - If it is a sequence: curve[m] is the target for time (m+1)*dt,
                  for m = 0, ..., n_steps-1.
                - If it is callable: curve(t) should return the target at
                  continuous time t (e.g. forward/futures or discount factor).

                For oil, this can be a futures curve F(0, T). It can also be
                repurposed for other markets by supplying the relevant target
                curve.

            alpha_solver:
                Optional callable to compute alpha_m given (Q_level, target,
                dt, delta_r). Defaults to the classic Hull–White bond-matching
                solver:

                    alpha_m = [ln sum_j Q_mj e^{-j DeltaR dt} - ln target] / dt

                For oil you will typically want to pass your own solver that
                enforces E[exp(x_T)] = F(0,T) instead of matching discount
                factors.

            discount_factor_fn:
                Optional callable returning a “discount factor” applied when
                propagating from node j over one step. The default mirrors the
                Hull–White bond setup (exp(-(alpha_m + j DeltaR)*dt)). For oil,
                you might set this to 1.0 (no discounting) or include a
                financing term if needed.

        Returns:
            (alphas, q_levels) where q_levels are the propagated weights
            (state-price-like when using bond-style discounting; probabilities ×
            discounts).
        """
        target_lookup = self._curve_lookup(curve)

        # ---- default: Hull–White style bond calibration ---- #
        def default_alpha_solver(
            Q_level: Dict[int, float],
            target: float,
            dt: float,
            delta_r: float,
        ) -> float:
            numerator = sum(Q * math.exp(-j * delta_r * dt) for j, Q in Q_level.items())
            return (math.log(numerator) - math.log(target)) / dt

        def default_discount_factor(j: int, alpha_m: float, dt: float, delta_r: float) -> float:
            return math.exp(-(alpha_m + j * delta_r) * dt)

        alpha_fn = alpha_solver or default_alpha_solver
        discount_fn = discount_factor_fn or default_discount_factor

        alphas: List[float] = []
        q_levels: List[Dict[int, float]] = [{0: 1.0}]  # Q_0,0 = 1

        for m in range(self.n_steps):
            # step_index = m+1 corresponds to time t = (m+1)*dt
            target = target_lookup(m + 1, self.dt)
            if target <= 0:
                raise ValueError(f"Curve value must be positive at step {m+1}")

            current_Q = q_levels[m]
            alpha_m = alpha_fn(current_Q, target, self.dt, self.delta_r)
            alphas.append(alpha_m)

            next_Q: Dict[int, float] = {}
            for j, node in self.levels[m].items():
                reach_prob = current_Q.get(j, 0.0)
                if reach_prob == 0.0:
                    continue

                discount = discount_fn(j, alpha_m, self.dt, self.delta_r)
                pu, pm, pd = node.probabilities
                children = node.children

                for prob, child_j in zip((pu, pm, pd), children):
                    next_Q[child_j] = next_Q.get(child_j, 0.0) + reach_prob * prob * discount

            q_levels.append(next_Q)

        self.alphas = alphas
        self.q_levels = q_levels
        return alphas, q_levels

    # --------------------- helpers --------------------- #
    @staticmethod
    def _curve_lookup(
        curve: Union[Sequence[float], Callable[[float], float]]
    ) -> Callable[[int, float], float]:
        """
        Normalize curve input into a callable.

        The returned function takes (step_index, dt):

            - step_index: 1, 2, ..., n_steps
            - dt: time step size

        and returns the curve value at time t = step_index * dt.
        """
        if callable(curve):
            # curve is already a function of continuous time t
            def lookup(step_index: int, dt: float) -> float:
                t = step_index * dt
                return curve(t)

        else:
            # curve is a discrete sequence: curve[m] -> time (m+1)*dt
            values = list(curve)

            def lookup(step_index: int, dt: float) -> float:  # dt kept for symmetry
                idx = step_index - 1
                if idx < 0 or idx >= len(values):
                    raise IndexError(f"Curve does not cover step_index={step_index}")
                return values[idx]

        return lookup

    def adjusted_factor(self, time_index: int, j: int) -> float:
        """
        Return the “shifted” factor at node (time_index, j).

        In oil (log-spot model):
            x = alpha(time_index) + R*,
            S = exp(x).
        """
        if not self.alphas:
            raise RuntimeError("Tree has not been calibrated; call calibrate_to_curve first.")
        return self.alphas[time_index] + j * self.delta_r


def _format_probabilities(probs: Probabilities) -> str:
    """Helper to pretty-print (pu, pm, pd)."""
    pu, pm, pd = probs
    return f"pu={pu: .4f}, pm={pm: .4f}, pd={pd: .4f}"
