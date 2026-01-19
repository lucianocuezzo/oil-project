# Sensitivity Analysis

Quick experiments to see how model outputs change under key parameters.

## Sigma sweep

Run `python sensitivity_analysis/sigma_sweep.py` to compare tree NPV (invest-now vs invest-later) and Bellman value/action across a set of volatility levels (`sigmas` list in the script). Adjust the shared economics inside the script as needed.
