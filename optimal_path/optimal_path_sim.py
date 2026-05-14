"""
Simulación de caminos óptimos en el árbol trinomial.

Para N caminos estocásticos simulados, aplica la política Bellman-óptima
en cada nodo y muestra la trayectoria del precio con las decisiones tomadas,
superpuesta sobre el mapa de política.

Panel izquierdo — Fase pre-inversión:
    · Zona gris  : esperar
    · Zona verde : invertir
    · Caminos    : líneas de colores
    · Estrella   : momento de inversión

Panel derecho — Fase operacional:
    · Zona roja    : apagar
    · Zona amarilla: histéresis
    · Zona azul    : operar
    · Caminos      : coloreados según estado ON (sólido) / OFF (punteado)
    · Triángulo ▼  : switch_off
    · Triángulo ▲  : switch_on

Ejecutar desde la raíz del repositorio:
    python optimal_path/optimal_path_sim.py
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bellman_equation.params import SwitchingParams
from bellman_equation.solver import SwitchingBellmanSolver, default_price_fn
from optimal_path.trigger_curves import extract_trigger_curves
from tree.oil_futures_curve import FuturesCurve
from tree.oil_tree_builder import OilTrinomialTreeBuilder
from tree.oil_tree_calibrator import OilTrinomialFuturesCalibrator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _npv_threshold(params: dict) -> float:
    """Precio de entrada implícito por la regla del VAN (VAN = 0)."""
    r  = params["discount_rate"]
    dt = params["dt"]
    n  = int(round(10.0 / dt))   # total steps ≈ horizonte 10 años
    q  = params["production_rate"]
    if r == 0:
        annuity = n * dt * q
    else:
        annuity = dt * (1 - math.exp(-r * n * dt)) / (1 - math.exp(-r * dt)) * q
    return (params["variable_cost"] + params["fixed_on_cost"]
            + (params["capex"] + params["switch_on_cost"]) / annuity)


def _ffill(values: list) -> list:
    filled = list(values)
    last = None
    for i, v in enumerate(filled):
        if v is not None:
            last = v
        elif last is not None:
            filled[i] = last
    last = None
    for i in range(len(filled) - 1, -1, -1):
        if filled[i] is not None:
            last = filled[i]
        elif last is not None:
            filled[i] = last
    return filled


# ---------------------------------------------------------------------------
# Path simulation
# ---------------------------------------------------------------------------

def simulate_paths(
    tree,
    solution,
    dt: float,
    n_paths: int = 8,
    seed: int = 42,
) -> list[list[dict]]:
    """
    Simulate n_paths stochastic price paths through the tree, applying the
    Bellman-optimal policy at every node.

    Each step record:
        t      – time in years
        price  – oil price at this node
        state  – "pre" | "on" | "off"
        action – policy decision at this node
    """
    rng = np.random.default_rng(seed)
    n_steps = tree.n_steps
    paths = []

    for _ in range(n_paths):
        path: list[dict] = []
        j = 0            # trinomial tree root always at j=0
        state = "pre"    # all paths start pre-investment

        for t in range(n_steps):
            price = default_price_fn(tree, t, j)

            # Look up optimal action
            if state == "pre":
                action = solution.policy_pre[t].get(j, "wait")
            elif state == "on":
                action = solution.policy_on[t].get(j, "stay_on")
            else:
                action = solution.policy_off[t].get(j, "stay_off")

            path.append({"t": t * dt, "price": price, "state": state, "action": action})

            # Update state according to action
            if state == "pre":
                if action == "invest_on":
                    state = "on"
                elif action == "invest_off":
                    state = "off"
            elif state == "on":
                if action == "switch_off":
                    state = "off"
            else:  # off
                if action == "switch_on":
                    state = "on"

            # Stochastic transition to next node
            node = tree.levels[t][j]
            j_up, j_mid, j_down = node.children
            pu, pm, pd = node.probabilities
            j = rng.choice([j_up, j_mid, j_down], p=[pu, pm, pd])

        # Terminal step (no decision, just record price)
        price = default_price_fn(tree, n_steps, j)
        path.append({"t": n_steps * dt, "price": price, "state": state, "action": "terminal"})
        paths.append(path)

    return paths


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def build_optimal_paths(sigma: float = 0.2, n_paths: int = 8, seed: int = 42) -> None:
    n_steps = 40
    dt = 0.25
    a = 0.6
    futures_curve = FuturesCurve(lambda t: 72.0 + 12.0 * math.exp(-0.5 * t))

    params_dict = dict(
        dt=dt,
        discount_rate=0.08,
        production_rate=1.0,
        variable_cost=65.0,
        fixed_on_cost=5.0,
        fixed_off_cost=2.0,
        capex=20.0,
        switch_on_cost=5.0,
        switch_off_cost=1.0,
        salvage_multiplier=0.0,
        allow_start_on=True,
    )

    builder = OilTrinomialTreeBuilder(
        n_steps=n_steps, a=a, sigma=sigma, dt=dt, jmax=6, jmin=-6
    )
    shifted_tree = OilTrinomialFuturesCalibrator(builder.build(), futures_curve).calibrate()

    params = SwitchingParams(**params_dict)
    solution = SwitchingBellmanSolver(
        tree=shifted_tree,
        params=params,
        price_fn=default_price_fn,
        terminal_on=lambda price: 0.0,
        terminal_off=lambda price: 0.0,
    ).solve()

    times, inv_trig, off_trig, on_trig = extract_trigger_curves(shifted_tree, solution, dt)

    all_prices = [
        default_price_fn(shifted_tree, t, j)
        for t in range(n_steps)
        for j in shifted_tree.levels[t]
    ]
    p_min = min(all_prices) * 0.95
    p_max = max(all_prices) * 1.05

    fwd_times  = [t * dt for t in range(1, n_steps + 1)]
    fwd_prices = [futures_curve.value(t, dt) for t in range(1, n_steps + 1)]

    paths = simulate_paths(shifted_tree, solution, dt, n_paths=n_paths, seed=seed)

    # Print summary statistics
    n_invested   = sum(1 for p in paths if any(s["action"] in ("invest_on", "invest_off") for s in p))
    n_switched_off = sum(
        sum(1 for s in p if s["action"] == "switch_off") for p in paths
    )
    n_switched_on  = sum(
        sum(1 for s in p if s["action"] == "switch_on") for p in paths
    )
    print(f"Paths simulated    : {n_paths}")
    print(f"Paths that invested: {n_invested} / {n_paths}")
    print(f"Total switch_off   : {n_switched_off}")
    print(f"Total switch_on    : {n_switched_on}")

    _plot(times, inv_trig, off_trig, on_trig,
          fwd_times, fwd_prices, p_min, p_max, sigma, paths, params_dict)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _plot(times, inv_trig, off_trig, on_trig,
          fwd_times, fwd_prices, p_min, p_max, sigma, paths, params_dict) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.lines as mlines
        import matplotlib.patches as mpatches
    except Exception:
        return
    try:
        inv_f = _ffill(inv_trig)
        off_f = _ffill(off_trig)
        on_f  = _ffill(on_trig)

        t_arr   = np.array(times)
        inv_arr = np.array(inv_f, dtype=float)
        off_arr = np.array(off_f, dtype=float)
        on_arr  = np.array(on_f,  dtype=float)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        path_colors = plt.cm.tab10.colors

        # ------------------------------------------------------------------ #
        # Panel 1 — Investment decision                                        #
        # ------------------------------------------------------------------ #
        ax1.fill_between(t_arr, p_min, inv_arr, color="#cfd8dc", alpha=0.85,
                         rasterized=True)
        ax1.fill_between(t_arr, inv_arr, p_max, color="#66bb6a", alpha=0.75,
                         rasterized=True)
        ax1.plot(t_arr, inv_arr, color="black", linewidth=2.0, zorder=5)
        ax1.plot(fwd_times, fwd_prices, color="white", linewidth=1.5,
                 linestyle="--", zorder=5)

        # Umbral VAN: precio que hace VAN = 0 (sin opcionalidad)
        npv_thresh = _npv_threshold(params_dict)
        ax1.axhline(npv_thresh, color="dimgray", linewidth=1.4,
                    linestyle="--", zorder=5)

        # Batch scatter: collect all invest events across paths
        inv_ts, inv_ps, inv_cs = [], [], []
        for k, path in enumerate(paths):
            col = path_colors[k % len(path_colors)]
            pre = [s for s in path if s["state"] == "pre"]
            if pre:
                ax1.plot([s["t"] for s in pre], [s["price"] for s in pre],
                         color=col, linewidth=1.8, alpha=0.85, zorder=6)
            for s in path:
                if s["action"] in ("invest_on", "invest_off"):
                    inv_ts.append(s["t"])
                    inv_ps.append(s["price"])
                    inv_cs.append(col)
        if inv_ts:
            ax1.scatter(inv_ts, inv_ps, marker="*", s=150, c=inv_cs,
                        edgecolors="black", linewidths=0.5, zorder=9)

        ax1.set_xlim(t_arr[0], t_arr[-1])
        ax1.set_ylim(p_min, p_max)
        ax1.set_xlabel("Tiempo (años)")
        ax1.set_ylabel(r"Precio del petróleo  $S$  (\$/bbl)")
        ax1.set_title(rf"Política de inversión  ($\sigma$={sigma:.2f})")
        ax1.grid(True, linestyle="--", alpha=0.25, zorder=1)

        ax1.legend(handles=[
            mpatches.Patch(color="#cfd8dc", alpha=0.85, label="Esperar"),
            mpatches.Patch(color="#66bb6a", alpha=0.75, label="Invertir"),
            mlines.Line2D([], [], color="black", linewidth=2,
                          label=r"$S^*_{invest}$  (Bellman)"),
            mlines.Line2D([], [], color="dimgray", linewidth=1.4, linestyle="--",
                          label=r"Umbral VAN  (VAN = 0)"),
            mlines.Line2D([], [], color="white", linewidth=1.5, linestyle="--",
                          label="Futuros"),
            mlines.Line2D([], [], color="gray", linewidth=1.2,
                          label="Caminos (pre-inv.)"),
            mlines.Line2D([], [], color="gray", marker="*", linestyle="none",
                          markersize=10, markeredgecolor="black",
                          label="Momento de inversión"),
        ], fontsize=7.5, loc="upper right")

        # ------------------------------------------------------------------ #
        # Panel 2 — Operational decision                                       #
        # ------------------------------------------------------------------ #
        ax2.fill_between(t_arr, p_min, off_arr, color="#ef9a9a", alpha=0.85,
                         rasterized=True)
        ax2.fill_between(t_arr, off_arr, on_arr, color="#fff59d", alpha=0.90,
                         rasterized=True)
        ax2.fill_between(t_arr, on_arr, p_max, color="#90caf9", alpha=0.85,
                         rasterized=True)
        ax2.plot(t_arr, off_arr, color="black", linewidth=2.0, zorder=5)
        ax2.plot(t_arr, on_arr,  color="black", linewidth=2.0, linestyle="--",
                 zorder=5)
        ax2.plot(fwd_times, fwd_prices, color="dimgray", linewidth=1.5,
                 linestyle=":", zorder=5)

        # Batch scatter: collect switch events across all paths
        soff_ts, soff_ps, soff_cs = [], [], []
        son_ts,  son_ps,  son_cs  = [], [], []

        for k, path in enumerate(paths):
            col = path_colors[k % len(path_colors)]
            post = [s for s in path if s["state"] in ("on", "off")]
            if not post:
                continue

            # NaN-break approach: two arrays per path (ON segments / OFF segments)
            # At each state transition the current point is added to BOTH arrays
            # so the solid and dashed lines share the transition node (no gap).
            on_t,  on_p  = [], []
            off_t, off_p = [], []
            for i, s in enumerate(post):
                if s["state"] == "on":
                    on_t.append(s["t"]); on_p.append(s["price"])
                    if i + 1 < len(post) and post[i + 1]["state"] != "on":
                        # Bridge: dashed line starts from this same point
                        off_t.append(s["t"]); off_p.append(s["price"])
                        on_t.append(float("nan")); on_p.append(float("nan"))
                else:
                    off_t.append(s["t"]); off_p.append(s["price"])
                    if i + 1 < len(post) and post[i + 1]["state"] != "off":
                        # Bridge: solid line starts from this same point
                        on_t.append(s["t"]); on_p.append(s["price"])
                        off_t.append(float("nan")); off_p.append(float("nan"))

            if on_t:
                ax2.plot(on_t, on_p, color=col, linewidth=1.8, alpha=0.85,
                         linestyle="-", zorder=6)
            if off_t:
                ax2.plot(off_t, off_p, color=col, linewidth=1.8, alpha=0.85,
                         linestyle="--", zorder=6)

            # Collect switch events for batch scatter
            for s in post:
                if s["action"] == "switch_off":
                    soff_ts.append(s["t"]); soff_ps.append(s["price"])
                    soff_cs.append(col)
                elif s["action"] == "switch_on":
                    son_ts.append(s["t"]); son_ps.append(s["price"])
                    son_cs.append(col)

        if soff_ts:
            ax2.scatter(soff_ts, soff_ps, marker="v", s=70, c=soff_cs,
                        edgecolors="black", linewidths=0.5, zorder=9)
        if son_ts:
            ax2.scatter(son_ts, son_ps, marker="^", s=70, c=son_cs,
                        edgecolors="black", linewidths=0.5, zorder=9)

        ax2.set_xlim(t_arr[0], t_arr[-1])
        ax2.set_ylim(p_min, p_max)
        ax2.set_xlabel("Tiempo (años)")
        ax2.set_ylabel(r"Precio del petróleo  $S$  (\$/bbl)")
        ax2.set_title(rf"Política operacional  ($\sigma$={sigma:.2f})")
        ax2.grid(True, linestyle="--", alpha=0.25, zorder=1)

        ax2.legend(handles=[
            mpatches.Patch(color="#ef9a9a", alpha=0.85, label="Apagar"),
            mpatches.Patch(color="#fff59d", alpha=0.90, label="Histéresis"),
            mpatches.Patch(color="#90caf9", alpha=0.85, label="Operar"),
            mlines.Line2D([], [], color="black", linewidth=2,
                          label=r"$S^*_{off}$"),
            mlines.Line2D([], [], color="black", linewidth=2, linestyle="--",
                          label=r"$S^*_{on}$"),
            mlines.Line2D([], [], color="gray", linewidth=1.2,
                          label="Camino ON (sólido) / OFF (guiones)"),
            mlines.Line2D([], [], color="gray", marker="v", linestyle="none",
                          markersize=8, markeredgecolor="black",
                          label="Apagar"),
            mlines.Line2D([], [], color="gray", marker="^", linestyle="none",
                          markersize=8, markeredgecolor="black",
                          label="Encender"),
        ], fontsize=7.5, loc="upper right")

        plt.suptitle(
            rf"Simulación de caminos óptimos  ($\sigma$={sigma:.2f},  $a$=0.6,  capex=20)",
            fontsize=11, y=1.01
        )
        plt.tight_layout()
        plt.show(block=False)
    except Exception as exc:
        print(f"\nPlot skipped: {exc}")


if __name__ == "__main__":
    build_optimal_paths(sigma=0.2, n_paths=5, seed=42)
    input("\nPresioná Enter para cerrar...")
