"""
Corre todas las sensibilidades de valor del proyecto.

Ejecutar desde la raíz del repositorio:
    python sensitivity_analysis/project_value/run_all.py
"""

import matplotlib.pyplot as plt
from sigma_sweep import run_sigma_sweep
from mean_reversion_sweep import run_mean_reversion_sweep
from capex_sweep import run_capex_sweep
from switch_cost_sweep import run_switch_cost_sweep
from oil_price_sweep import run_oil_price_sweep
from discount_rate_sweep import run_discount_rate_sweep
from variable_cost_sweep import run_variable_cost_sweep

print("=" * 60)
print("  Sensibilidad: volatilidad (sigma)")
print("=" * 60)
run_sigma_sweep([0.0, 0.1, 0.2, 0.35, 0.5])

print()
print("=" * 60)
print("  Sensibilidad: reversión a la media (a)")
print("=" * 60)
run_mean_reversion_sweep([0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0])

print()
print("=" * 60)
print("  Sensibilidad: capex")
print("=" * 60)
run_capex_sweep([5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0])

print()
print("=" * 60)
print("  Sensibilidad: costo de switching")
print("=" * 60)
run_switch_cost_sweep([0.0, 1.0, 2.0, 5.0, 8.0, 12.0, 18.0, 25.0])

print()
print("=" * 60)
print("  Sensibilidad: precio del petróleo")
print("=" * 60)
run_oil_price_sweep([55.0, 58.0, 61.0, 64.0, 67.0, 70.0, 73.0, 76.0, 80.0, 85.0, 90.0])

print()
print("=" * 60)
print("  Sensibilidad: tasa de descuento")
print("=" * 60)
run_discount_rate_sweep([0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20])

print()
print("=" * 60)
print("  Sensibilidad: costo variable")
print("=" * 60)
run_variable_cost_sweep([45.0, 50.0, 55.0, 60.0, 65.0, 67.0, 70.0, 73.0, 76.0, 79.0])

plt.show()  # mantiene todas las ventanas abiertas hasta que las cerrés
