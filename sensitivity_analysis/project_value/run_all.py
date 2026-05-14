"""
Corre todas las sensibilidades de valor del proyecto y guarda los gráficos.

Ejecutar desde la raíz del repositorio:
    python sensitivity_analysis/project_value/run_all.py
"""

import pathlib

import matplotlib.pyplot as plt

from sigma_sweep import run_sigma_sweep
from mean_reversion_sweep import run_mean_reversion_sweep
from capex_sweep import run_capex_sweep
from switch_cost_sweep import run_switch_cost_sweep
from oil_price_sweep import run_oil_price_sweep
from discount_rate_sweep import run_discount_rate_sweep
from variable_cost_sweep import run_variable_cost_sweep

OUTPUT_DIR = pathlib.Path(r"C:\Users\lucia\OneDrive\Documentos\tesis\sensibilidades")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _save_new_figs(before: set, name: str) -> None:
    """Guarda las figuras creadas desde la última llamada."""
    new_nums = sorted(set(plt.get_fignums()) - before)
    for i, num in enumerate(new_nums):
        fig = plt.figure(num)
        suffix = f"_{i + 1}" if len(new_nums) > 1 else ""
        path = OUTPUT_DIR / f"{name}{suffix}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  → guardado: {path}")


# ---------- sigma ----------
print("=" * 60)
print("  Sensibilidad: volatilidad (sigma)")
print("=" * 60)
_before = set(plt.get_fignums())
run_sigma_sweep([0.0, 0.1, 0.2, 0.35, 0.5])
_save_new_figs(_before, "sigma_sweep")

# ---------- reversión a la media ----------
print()
print("=" * 60)
print("  Sensibilidad: reversión a la media (a)")
print("=" * 60)
_before = set(plt.get_fignums())
run_mean_reversion_sweep([0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0])
_save_new_figs(_before, "mean_reversion_sweep")

# ---------- capex ----------
print()
print("=" * 60)
print("  Sensibilidad: capex")
print("=" * 60)
_before = set(plt.get_fignums())
run_capex_sweep([5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0])
_save_new_figs(_before, "capex_sweep")

# ---------- costo de switching ----------
print()
print("=" * 60)
print("  Sensibilidad: costo de switching")
print("=" * 60)
_before = set(plt.get_fignums())
run_switch_cost_sweep([0.0, 1.0, 2.0, 5.0, 8.0, 12.0, 18.0, 25.0])
_save_new_figs(_before, "switch_cost_sweep")

# ---------- precio del petróleo ----------
print()
print("=" * 60)
print("  Sensibilidad: precio del petróleo")
print("=" * 60)
_before = set(plt.get_fignums())
run_oil_price_sweep([55.0, 58.0, 61.0, 64.0, 67.0, 70.0, 73.0, 76.0, 80.0, 85.0, 90.0])
_save_new_figs(_before, "oil_price_sweep")

# ---------- tasa de descuento ----------
print()
print("=" * 60)
print("  Sensibilidad: tasa de descuento")
print("=" * 60)
_before = set(plt.get_fignums())
run_discount_rate_sweep([0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20])
_save_new_figs(_before, "discount_rate_sweep")

# ---------- costo variable ----------
print()
print("=" * 60)
print("  Sensibilidad: costo variable")
print("=" * 60)
_before = set(plt.get_fignums())
run_variable_cost_sweep([45.0, 50.0, 55.0, 60.0, 65.0, 67.0, 70.0, 73.0, 76.0, 79.0])
_save_new_figs(_before, "variable_cost_sweep")

print()
print(f"Todos los gráficos guardados en: {OUTPUT_DIR}")
plt.show()  # mantiene todas las ventanas abiertas hasta que las cerrés
