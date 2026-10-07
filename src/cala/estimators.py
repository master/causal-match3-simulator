"""
This script implements the estimators on the saved per-player arrays: WM-MC, WM-DR, Direct-DR, curve metrics against the engine gold, and individual
recommendations with their regret. All three routes read the same players, grid, draws and gold.
"""

from __future__ import annotations

import numpy as np

from match3_simulator.experiments.propensity import BANDWIDTH, SUPPORT_MIN, dr_weights, support_diagnostics


def wm_mc_curve(mu: np.ndarray) -> np.ndarray:
    """Inputs: mu (P, G) imagined risk per player and grid point (already averaged over draws and replicates). Outputs: (G,)."""
    return np.asarray(mu, np.float64).mean(axis=0)


def dr_terms(mu_grid: np.ndarray, mu_logged: np.ndarray, Y: np.ndarray, E_logged: np.ndarray, g_logged: np.ndarray, grid: np.ndarray, *, h: float = BANDWIDTH) -> tuple[np.ndarray, np.ndarray]:
    """Per-player doubly robust terms mu(e, W_i) + K_h(E_i - e) / g(E_i | W_i) * (Y_i - mu(E_i, W_i)). Outputs: (terms (P, G), weights (P, G))."""
    w = dr_weights(E_logged, g_logged, grid, h=h)
    resid = np.asarray(Y, np.float64) - np.asarray(mu_logged, np.float64)
    return np.asarray(mu_grid, np.float64) + w * resid[:, None], w


def dr_curve(mu_grid, mu_logged, Y, E_logged, g_logged, grid, *, h: float = BANDWIDTH) -> dict[str, object]:
    """WM-DR (or Direct-DR with an observational outcome model). Outputs: curve (G,), per-player terms, weights, residuals, support diagnostics."""
    terms, w = dr_terms(mu_grid, mu_logged, Y, E_logged, g_logged, grid, h=h)
    return {"curve": terms.mean(axis=0), "terms": terms, "weights": w, "residual": np.asarray(Y, np.float64) - np.asarray(mu_logged, np.float64), "support": support_diagnostics(w)}


def curve_metrics(curve: np.ndarray, gold: np.ndarray, grid: np.ndarray) -> dict[str, object]:
    """Pointwise and integrated absolute error vs the engine curve, the selected e, the engine optimum and the true regret m(e_hat) - m(e*)."""
    curve, gold, grid = np.asarray(curve, np.float64), np.asarray(gold, np.float64), np.asarray(grid, np.float64)
    if curve.shape != gold.shape or curve.shape != grid.shape:
        raise ValueError("curve, gold and grid must align")
    err = np.abs(curve - gold)
    sel, opt = int(np.argmin(curve)), int(np.argmin(gold))
    return {"pointwise_abs_error": err.tolist(), "integrated_abs_error": float(err.mean()), "max_abs_error": float(err.max()), "selected_e": float(grid[sel]), "engine_optimum_e": float(grid[opt]),
            "regret": float(gold[sel] - gold[opt]), "selected_index": sel, "engine_optimum_index": opt, "mean_signed_error": float((curve - gold).mean())}


def individual_recommendations(mu: np.ndarray, g_grid: np.ndarray, gold_player_hazards: np.ndarray, grid: np.ndarray, *, support_min: float = SUPPORT_MIN) -> dict[str, object]:
    """Per-player argmin over candidates with g(e | W_i) > support_min (fallback: the highest-density candidate); individual regret on the engine's per-player curve.

    Inputs: mu (P, G) per-player risk; g_grid (P, G) unfloored density; gold_player_hazards (G, P) engine churn per player and e; grid (G,).
    Outputs: recommendation (P,), individual_regret (P,), support_fraction, mean regret, fallback count.
    """
    mu, g = np.asarray(mu, np.float64), np.asarray(g_grid, np.float64)
    gold = np.asarray(gold_player_hazards, np.float64).T  # (P, G)
    support = g > support_min
    fallback = ~support.any(axis=1)
    masked = np.where(support, mu, np.inf)
    rec = np.where(fallback, np.argmax(g, axis=1), np.argmin(masked, axis=1))
    engine_opt = np.argmin(gold, axis=1)
    regret = gold[np.arange(len(rec)), rec] - gold[np.arange(len(rec)), engine_opt]
    return {"recommendation_index": rec.astype(np.int64), "recommendation_e": np.asarray(grid)[rec], "individual_regret": regret, "mean_individual_regret": float(regret.mean()),
            "support_fraction": float(support.mean()), "fallback_players": int(fallback.sum()), "engine_optimum_index": engine_opt.astype(np.int64),
            "agreement_with_engine": float(np.mean(rec == engine_opt))}


__all__ = ["curve_metrics", "dr_curve", "dr_terms", "individual_recommendations", "wm_mc_curve"]
