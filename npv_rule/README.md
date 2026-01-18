# NPV Rule (No Flexibility)

Deterministic NPV calculation without switching optionality. Uses the futures curve as the expected price path, computes discounted operating cashflows, optional salvage, and compares against CAPEX.

Files
- calc.py: OOP helpers (ForwardNPVCalculator, TreeNPVCalculator, NPVParams) to compute NPV and earliest invest step.
- run_npv_demo.py: runnable example from repo root showing forward-based and tree-expected NPV.

Run from repo root:
```
python npv_rule/run_npv_demo.py
```
