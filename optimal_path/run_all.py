"""
Genera todas las figuras de optimal_path y las guarda en disco.

Ejecutar desde la raíz del repositorio:
    python optimal_path/run_all.py
"""

from __future__ import annotations

import pathlib
import sys

# Backend no interactivo: plt.show() se vuelve no-op y nada bloquea
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optimal_path.hysteresis_analysis import run_hysteresis_analysis
from optimal_path.hysteresis_sigma_sweep import run_hysteresis_sigma_sweep
from optimal_path.trigger_capex_sweep import run_trigger_capex_sweep
from optimal_path.trigger_sigma_sweep import run_trigger_sigma_sweep
from optimal_path.trigger_variable_cost_sweep import run_trigger_variable_cost_sweep
from optimal_path.trigger_discount_rate_sweep import run_trigger_discount_rate_sweep
from optimal_path.investment_timing import build_timing_plot
from optimal_path.optimal_path_sim import build_optimal_paths
from optimal_path.policy_map import build_policy_map
from optimal_path.policy_map_sigma import build_policy_map_sigma
from optimal_path.value_heatmap import build_heatmap

OUTPUT_DIR = pathlib.Path(r"C:\Users\lucia\OneDrive\Documentos\tesis\sensibilidades\optimal_path")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _save_new_figs(before: set, name: str) -> None:
    """Guarda las figuras creadas desde la última llamada y las cierra."""
    new_nums = sorted(set(plt.get_fignums()) - before)
    for i, num in enumerate(new_nums):
        fig = plt.figure(num)
        suffix = f"_{i + 1}" if len(new_nums) > 1 else ""
        path = OUTPUT_DIR / f"{name}{suffix}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  → guardado: {path}")
        plt.close(fig)


# ---------- umbral de inversión vs capex ----------
print("=" * 60)
print("  Umbral de inversión vs capex")
print("=" * 60)
_before = set(plt.get_fignums())
run_trigger_capex_sweep([5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0])
_save_new_figs(_before, "trigger_capex_sweep")

# ---------- umbral de inversión vs volatilidad ----------
print()
print("=" * 60)
print("  Umbral de inversión vs volatilidad")
print("=" * 60)
_before = set(plt.get_fignums())
run_trigger_sigma_sweep([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50])
_save_new_figs(_before, "trigger_sigma_sweep")

# ---------- umbral de inversión vs costo variable ----------
print()
print("=" * 60)
print("  Umbral de inversión vs costo variable")
print("=" * 60)
_before = set(plt.get_fignums())
run_trigger_variable_cost_sweep([45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0])
_save_new_figs(_before, "trigger_variable_cost_sweep")

# ---------- umbral de inversión vs tasa de descuento ----------
print()
print("=" * 60)
print("  Umbral de inversión vs tasa de descuento")
print("=" * 60)
_before = set(plt.get_fignums())
run_trigger_discount_rate_sweep([0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20])
_save_new_figs(_before, "trigger_discount_rate_sweep")

# ---------- histéresis vs costo de switching ----------
print("=" * 60)
print("  Histéresis vs costo de switching")
print("=" * 60)
_before = set(plt.get_fignums())
run_hysteresis_analysis([0.0, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0])
_save_new_figs(_before, "hysteresis_analysis")

# ---------- volatilidad vs ancho de histéresis ----------
print()
print("=" * 60)
print("  Volatilidad vs ancho de banda de histéresis")
print("=" * 60)
_before = set(plt.get_fignums())
run_hysteresis_sigma_sweep([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50])
_save_new_figs(_before, "hysteresis_sigma_sweep")

# ---------- mapa de política — caso base ----------
print("=" * 60)
print("  Mapa de política — caso base (σ=0.20)")
print("=" * 60)
_before = set(plt.get_fignums())
build_policy_map(sigma=0.2)
_save_new_figs(_before, "policy_map_base")

# ---------- mapa de política — grilla por σ ----------
print()
print("=" * 60)
print("  Mapa de política — grilla por volatilidad")
print("=" * 60)
_before = set(plt.get_fignums())
build_policy_map_sigma()
_save_new_figs(_before, "policy_map_sigma")

# ---------- mapa de calor de funciones de valor ----------
print()
print("=" * 60)
print("  Mapa de calor de funciones de valor")
print("=" * 60)
_before = set(plt.get_fignums())
build_heatmap()
_save_new_figs(_before, "value_heatmap")

# ---------- timing óptimo de inversión ----------
print()
print("=" * 60)
print("  Timing óptimo de inversión")
print("=" * 60)
_before = set(plt.get_fignums())
build_timing_plot(capex_list=[20.0, 60.0], n_paths=1000, seed=42)
_save_new_figs(_before, "investment_timing")

# ---------- simulación de caminos óptimos ----------
print()
print("=" * 60)
print("  Simulación de caminos óptimos")
print("=" * 60)
_before = set(plt.get_fignums())
build_optimal_paths(sigma=0.2, n_paths=8, seed=42)
_save_new_figs(_before, "optimal_path_sim")

print()
print(f"Todas las figuras guardadas en:\n  {OUTPUT_DIR}")
