from __future__ import annotations

try:
    # Re-export the existing tree plotter for convenience.
    from tree.plot_trinomial_tree import TreePlotter  # type: ignore
except Exception as exc:  # pragma: no cover - fallback for missing dependency
    raise ImportError("tree.plot_trinomial_tree.TreePlotter not available") from exc

__all__ = ["TreePlotter"]
