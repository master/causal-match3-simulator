"""
This is the whole-player bootstrap of a population curve (resampling complete grids and draws per player) with 95% percentile intervals for the curve,
the regret, and P(engine optimum). The run-level mean and SD over independent runs are computed elsewhere and never pooled with this.
"""

from __future__ import annotations

import numpy as np


def whole_player_bootstrap(player_terms: np.ndarray, gold: np.ndarray, grid: np.ndarray, *, n_bootstrap: int = 2000, seed: int = 0) -> dict[str, object]:
    """Inputs: player_terms (P, G) per-player contributions whose mean is the curve (WM-MC: mu_i; WM-DR: DR terms); gold (G,); grid (G,).
    Outputs: pointwise 95 % band, regret interval, P(engine optimum), selected-e frequencies."""
    terms = np.asarray(player_terms, np.float64)
    gold, grid = np.asarray(gold, np.float64), np.asarray(grid, np.float64)
    P, G = terms.shape
    if n_bootstrap < 20:
        raise ValueError("n_bootstrap must be at least 20")
    rng = np.random.default_rng(seed)
    curves = np.empty((n_bootstrap, G))
    for b in range(n_bootstrap):
        rows = rng.integers(0, P, size=P)
        curves[b] = terms[rows].mean(axis=0)
    sel = np.argmin(curves, axis=1)
    opt = int(np.argmin(gold))
    regret = gold[sel] - gold[opt]
    lower, upper = np.quantile(curves, [0.025, 0.975], axis=0)
    point = terms.mean(axis=0)
    return {"n_bootstrap": int(n_bootstrap), "curve_lower": lower.tolist(), "curve_upper": upper.tolist(), "regret_estimate": float(gold[int(np.argmin(point))] - gold[opt]),
            "regret_lower": float(np.quantile(regret, 0.025)), "regret_upper": float(np.quantile(regret, 0.975)), "regret_mean": float(regret.mean()), "p_engine_optimum": float(np.mean(sel == opt)),
            "selected_e_frequencies": {str(float(grid[i])): float(np.mean(sel == i)) for i in np.unique(sel)},
            "integrated_abs_error_interval": [float(v) for v in np.quantile(np.abs(curves - gold).mean(axis=1), [0.025, 0.975])]}


__all__ = ["whole_player_bootstrap"]
