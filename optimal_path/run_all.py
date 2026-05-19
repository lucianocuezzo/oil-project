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

OUTPUT_DIR = ROOT / "figures" / "optimal_path"
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


# ── Figura 10.1 — Timing óptimo de inversión ──────────────────────────────────
print("=" * 60)
print("  Figura 10.1 — Timing óptimo de inversión")
print("=" * 60)
_before = set(plt.get_fignums())
build_timing_plot(capex_list=[20.0, 60.0], n_paths=1000, seed=42)
_save_new_figs(_before, "fig_10_1_timing_inversion")

# ── Figura 10.2 — Simulación de caminos óptimos ───────────────────────────────
print()
print("=" * 60)
print("  Figura 10.2 — Simulación de caminos óptimos")
print("=" * 60)
_before = set(plt.get_fignums())
build_optimal_paths(sigma=0.2, n_paths=8, seed=42)
_save_new_figs(_before, "fig_10_2_caminos_optimos")

# ── Figura 10.3 — Funciones de valor de Bellman (mapas de calor) ──────────────
print()
print("=" * 60)
print("  Figura 10.3 — Funciones de valor de Bellman (mapas de calor)")
print("=" * 60)
_before = set(plt.get_fignums())
build_heatmap()
_save_new_figs(_before, "fig_10_3_funciones_valor")

# ── Figura 10.4 — Banda de histéresis vs costo de switching ───────────────────
print()
print("=" * 60)
print("  Figura 10.4 — Banda de histéresis vs costo de switching")
print("=" * 60)
_before = set(plt.get_fignums())
run_hysteresis_analysis([0.0, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0])
_save_new_figs(_before, "fig_10_4_histeresis_costo_switching")

# ── Figura 10.5 — Banda de histéresis vs volatilidad ──────────────────────────
print()
print("=" * 60)
print("  Figura 10.5 — Banda de histéresis vs volatilidad")
print("=" * 60)
_before = set(plt.get_fignums())
run_hysteresis_sigma_sweep([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50])
_save_new_figs(_before, "fig_10_5_histeresis_volatilidad")

# ── Figura 10.6 — Umbral óptimo de inversión vs Capex ─────────────────────────
print()
print("=" * 60)
print("  Figura 10.6 — Umbral óptimo de inversión vs Capex")
print("=" * 60)
_before = set(plt.get_fignums())
run_trigger_capex_sweep([5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0])
_save_new_figs(_before, "fig_10_6_umbral_capex")

# ── Figura 10.7 — Umbral óptimo de inversión vs volatilidad ───────────────────
print()
print("=" * 60)
print("  Figura 10.7 — Umbral óptimo de inversión vs volatilidad")
print("=" * 60)
_before = set(plt.get_fignums())
run_trigger_sigma_sweep([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50])
_save_new_figs(_before, "fig_10_7_umbral_volatilidad")

# ── Figura 10.8 — Umbral óptimo de inversión vs costo variable ────────────────
print()
print("=" * 60)
print("  Figura 10.8 — Umbral óptimo de inversión vs costo variable")
print("=" * 60)
_before = set(plt.get_fignums())
run_trigger_variable_cost_sweep([45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0])
_save_new_figs(_before, "fig_10_8_umbral_costo_variable")

# ── Extras (no numeradas en la tesis) ─────────────────────────────────────────
print()
print("=" * 60)
print("  Extra — Umbral óptimo vs tasa de descuento")
print("=" * 60)
_before = set(plt.get_fignums())
run_trigger_discount_rate_sweep([0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20])
_save_new_figs(_before, "extra_umbral_tasa_descuento")

print()
print("=" * 60)
print("  Extra — Mapa de política (caso base, σ=0.20)")
print("=" * 60)
_before = set(plt.get_fignums())
build_policy_map(sigma=0.2)
_save_new_figs(_before, "extra_mapa_politica_base")

print()
print("=" * 60)
print("  Extra — Mapa de política, grilla por volatilidad")
print("=" * 60)
_before = set(plt.get_fignums())
build_policy_map_sigma()
_save_new_figs(_before, "extra_mapa_politica_sigma")

print()
print(f"Todas las figuras guardadas en:\n  {OUTPUT_DIR}")
