"""
Corre todas las sensibilidades de curvas de trigger y guarda los gráficos.

Ejecutar desde la raíz del repositorio:
    python sensitivity_analysis/run_all.py
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt

from optimal_path.hysteresis_analysis import run_hysteresis_analysis
from optimal_path.trigger_capex_sweep import run_trigger_capex_sweep

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


# ---------- trigger vs capex ----------
print("=" * 60)
print("  Sensibilidad: precio de entrada vs capex")
print("=" * 60)
_before = set(plt.get_fignums())
run_trigger_capex_sweep([5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0])
_save_new_figs(_before, "trigger_capex_sweep")

# ---------- histéresis ----------
print()
print("=" * 60)
print("  Sensibilidad: banda de histéresis vs costo de switching")
print("=" * 60)
_before = set(plt.get_fignums())
run_hysteresis_analysis([0.0, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0])
_save_new_figs(_before, "hysteresis_analysis")

print()
print(f"Todos los gráficos guardados en: {OUTPUT_DIR}")
plt.show()  # mantiene todas las ventanas abiertas hasta que las cerrés
