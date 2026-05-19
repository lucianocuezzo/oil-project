"""
Reproduce TODAS las figuras de la tesis ejecutando cada script generador.

Uso desde la raíz del repositorio:
    python run_all_figures.py

Esto regenera, en orden:
  1. Figuras del caso base               (chapter9/caso_base_figures.py)
  2. Sensibilidades de valor del proyecto (sensitivity_analysis/project_value/run_all.py)
  3. Sensibilidades de políticas óptimas  (optimal_path/run_all.py)

Las figuras se guardan dentro del repositorio en:
    <repo>/figures/
con subcarpetas chapter9/, project_value/ y optimal_path/.

Se fuerza el backend Matplotlib "Agg" en los procesos hijos para que no se
abran ventanas y la corrida sea desatendida.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import time

# Windows: forzar UTF-8 en stdout para que los caracteres Unicode no exploten.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

ROOT = pathlib.Path(__file__).resolve().parent
FIGURES_DIR = ROOT / "figures"

SCRIPTS: list[tuple[str, pathlib.Path]] = [
    ("Caso base (capítulo 9)",
        ROOT / "chapter9" / "caso_base_figures.py"),
    ("Sensibilidades — valor del proyecto",
        ROOT / "sensitivity_analysis" / "project_value" / "run_all.py"),
    ("Sensibilidades — políticas óptimas",
        ROOT / "optimal_path" / "run_all.py"),
]


def _run(label: str, script: pathlib.Path) -> tuple[str, float, int]:
    rel = script.relative_to(ROOT)
    print()
    print("█" * 72)
    print(f"  {label}")
    print(f"  ▶ {rel}")
    print("█" * 72)

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"  # sin pop-ups: corrida desatendida
    env["PYTHONIOENCODING"] = "utf-8"  # Windows: caracteres Unicode en consola

    t0 = time.perf_counter()
    result = subprocess.run([sys.executable, str(script)], cwd=ROOT, env=env)
    elapsed = time.perf_counter() - t0
    return label, elapsed, result.returncode


if __name__ == "__main__":
    summary: list[tuple[str, float, int]] = []
    for label, script in SCRIPTS:
        if not script.exists():
            print(f"  [SKIP]  {label} — no existe: {script}")
            summary.append((label, 0.0, -1))
            continue
        summary.append(_run(label, script))

    print()
    print("=" * 72)
    print("  Resumen")
    print("=" * 72)
    for label, elapsed, rc in summary:
        if rc == 0:
            status = "OK"
        elif rc == -1:
            status = "OMITIDO"
        else:
            status = f"FALLÓ (código {rc})"
        print(f"  [{status:>16}]  {label}  ({elapsed:.1f} s)")

    print()
    print("=" * 72)
    print("  Figuras guardadas en:")
    print(f"    {FIGURES_DIR}")
    print("    ├── chapter9/")
    print("    │     fig_9_1_distribucion_precios.png")
    print("    │     fig_9_2_mapa_politica.png")
    print("    │     fig_9_3_valor_caso_base.png")
    print("    ├── optimal_path/")
    print("    │     fig_10_1_timing_inversion.png")
    print("    │     fig_10_2_caminos_optimos.png")
    print("    │     fig_10_3_funciones_valor.png")
    print("    │     fig_10_4_histeresis_costo_switching.png")
    print("    │     fig_10_5_histeresis_volatilidad.png")
    print("    │     fig_10_6_umbral_capex.png")
    print("    │     fig_10_7_umbral_volatilidad.png")
    print("    │     fig_10_8_umbral_costo_variable.png")
    print("    └── project_value/")
    print("          fig_11_1_precio_petroleo.png")
    print("          fig_11_2_volatilidad.png")
    print("          fig_11_3_capex.png")
    print("          fig_11_4_costo_variable.png")
    print("          fig_11_5_costo_switching.png")
    print("          fig_11_6_tasa_descuento.png")
    print("          fig_11_7_reversion_media.png")
    print("=" * 72)
