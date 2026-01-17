# Bellman Equation (Switching ON/OFF)

Coupled Bellman system for an ON/OFF operating problem with switching costs on top of the existing oil price tree.

At node (t, j) with price P and discount factor beta = exp(-r * dt):

- V_on(t, j) = max( cash_on(P) + beta * E[V_on(t+1)], -switch_off_cost + cash_off(P) + beta * E[V_off(t+1)] )
- V_off(t, j) = max( cash_off(P) + beta * E[V_off(t+1)], -switch_on_cost + cash_on(P) + beta * E[V_on(t+1)] )

Switching costs are paid immediately when taking the action; the cashflow that step uses the post-action operating state.

Files
- params.py: SwitchingParams with costs, discounting, and cashflow helpers.
- solver.py: SwitchingBellmanSolver doing backward induction over a BaseOilTrinomialTree or ShiftedOilTrinomialTree. Includes a pre-invest state (wait vs invest) with CAPEX.
- run_bellman_demo.py: runnable example wiring the solver to the existing tree builder and a mock futures curve.

Run the demo from the repo root:

```
python bellman_equation/run_bellman_demo.py
```

Adapt parameters (costs, rates, production, dt) and the futures curve to your thesis calibration. Provide custom terminal payoffs or price_fn if needed.
