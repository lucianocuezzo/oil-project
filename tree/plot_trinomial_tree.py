"""
Backwards compatibility shim: TreePlotter is now defined in plots/tree_plot.py.
"""

from __future__ import annotations

from plots.tree_plot import TreePlotter

__all__ = ["TreePlotter"]
