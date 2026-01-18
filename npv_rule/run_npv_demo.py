"""Deterministic NPV rule (no flexibility): when would you invest?"""

from __future__ import annotations

import math
import pathlib
import sys

# Ensure repo root on path
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tree.oil_futures_curve import FuturesCurve
from tree.oil_tree_builder import OilTrinomialTreeBuilder
from tree.oil_tree_calibrator import OilTrinomialFuturesCalibrator
from npv_rule.calc import (
    ForwardNPVCalculator,
    NPVParams,
    TreeNPVCalculator,
)


def main() -> None:
    n_steps = 6
    dt = 0.25  # quarters

    # Example curve: mild backwardation from $75 with ~5%/yr decay.
    futures_curve = FuturesCurve(lambda t: 75.0 * math.exp(-0.05 * t))

    params = NPVParams(
        dt=dt,
        discount_rate=0.08,
        production_rate=1.0,
        variable_cost=45.0,
        fixed_on_cost=5.0,
        capex=20.0,
        salvage_multiplier=0.0,
    )

    # Forward-based (deterministic) NPV
    fwd_calc = ForwardNPVCalculator(futures_curve, n_steps=n_steps, params=params)
    fwd_npvs = fwd_calc.npv_schedule()
    fwd_invest_step = ForwardNPVCalculator.earliest_invest_step(fwd_npvs)

    # Tree-based expected NPV (should match forward for linear cashflows)
    builder = OilTrinomialTreeBuilder(n_steps=n_steps, a=0.6, sigma=0.35, dt=dt)
    base_tree = builder.build()
    shifted_tree = OilTrinomialFuturesCalibrator(base_tree, futures_curve).calibrate()
    tree_calc = TreeNPVCalculator(shifted_tree, params=params)
    tree_npvs = tree_calc.npv_schedule()
    tree_invest_step = TreeNPVCalculator.earliest_invest_step(tree_npvs)

    print("Forward-curve NPV schedule (no switching flexibility):")
    for k, v in enumerate(fwd_npvs):
        print(f"  invest at step {k}: NPV = {v:.4f}")

    if fwd_invest_step is None:
        print("\nForward: No non-negative NPV within the horizon; do not invest.")
    else:
        t_years = fwd_invest_step * dt
        print(f"\nForward: Earliest non-negative NPV: step {fwd_invest_step} (t={t_years:.2f} years)")

    print("\nTree-expected NPV schedule (should align with forward):")
    for k, v in enumerate(tree_npvs):
        print(f"  invest at step {k}: NPV = {v:.4f}")

    if tree_invest_step is None:
        print("\nTree: No non-negative NPV within the horizon; do not invest.")
    else:
        t_years = tree_invest_step * dt
        print(f"\nTree: Earliest non-negative NPV: step {tree_invest_step} (t={t_years:.2f} years)")


if __name__ == "__main__":
    main()
