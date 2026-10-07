from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BANDWIDTH = 0.75
G_MIN = 0.02
SUPPORT_MIN = 0.01


def gaussian_kernel(distance: np.ndarray, h: float = BANDWIDTH) -> np.ndarray:
    return np.exp(-0.5 * (distance / h) ** 2) / (h * np.sqrt(2.0 * np.pi))


@dataclass(frozen=True)
class Propensity:
    """Per-level ridge location-scale model on standardised W. Arrays indexed by level."""

    mean: np.ndarray  # (F,) standardisation of W
    scale: np.ndarray  # (F,)
    coef: np.ndarray  # (L, F + 1) intercept first
    sigma: np.ndarray  # (L,)
    ridge: float
    g_min: float = G_MIN

    @classmethod
    def fit(cls, W: np.ndarray, E: np.ndarray, level: np.ndarray, *, n_levels: int = 3, ridge: float = 1.0, g_min: float = G_MIN) -> "Propensity":
        W = np.asarray(W, np.float64)
        mean = W.mean(axis=0)
        scale = W.std(axis=0)
        scale = np.where(scale > 1e-8, scale, 1.0)
        X = np.concatenate([np.ones((len(W), 1)), (W - mean) / scale], axis=1)
        f = X.shape[1]
        coef = np.zeros((n_levels, f))
        sigma = np.ones(n_levels)
        penalty = ridge * np.eye(f)
        penalty[0, 0] = 0.0
        for li in range(n_levels):
            sel = np.asarray(level) == li
            if sel.sum() < 3:
                continue
            Xl, yl = X[sel], np.asarray(E, np.float64)[sel]
            coef[li] = np.linalg.solve(Xl.T @ Xl + penalty, Xl.T @ yl)
            resid = yl - Xl @ coef[li]
            sigma[li] = max(float(np.sqrt(np.mean(resid**2))), 0.05)
        return cls(mean, scale, coef, sigma, float(ridge), float(g_min))

    def location(self, W: np.ndarray, level: np.ndarray) -> np.ndarray:
        X = np.concatenate([np.ones((len(W), 1)), (np.asarray(W, np.float64) - self.mean) / self.scale], axis=1)
        return np.einsum("nf,nf->n", X, self.coef[np.asarray(level)])

    def density(self, e: np.ndarray, W: np.ndarray, level: np.ndarray, *, floor: bool = True) -> np.ndarray:
        """g_l(e | W) for e (N,) or (N, C) broadcast against rows. Outputs: same shape as e, floored at g_min when requested."""
        loc = self.location(W, level)
        sig = self.sigma[np.asarray(level)]
        e = np.asarray(e, np.float64)
        if e.ndim == 2:
            loc, sig = loc[:, None], sig[:, None]
        g = np.exp(-0.5 * ((e - loc) / sig) ** 2) / (sig * np.sqrt(2.0 * np.pi))
        return np.maximum(g, self.g_min) if floor else g

    def to_json(self) -> dict[str, object]:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist(), "coef": self.coef.tolist(), "sigma": self.sigma.tolist(), "ridge": self.ridge, "g_min": self.g_min}

    @classmethod
    def from_json(cls, raw: dict[str, object]) -> "Propensity":
        return cls(np.asarray(raw["mean"]), np.asarray(raw["scale"]), np.asarray(raw["coef"]), np.asarray(raw["sigma"]), float(raw["ridge"]), float(raw["g_min"]))


def dr_weights(E_logged: np.ndarray, g_logged: np.ndarray, grid: np.ndarray, *, h: float = BANDWIDTH) -> np.ndarray:
    """K_h(E_i - e) / g(E_i | W_i) for every player and grid point. Inputs: (P,), (P,), (G,). Outputs: (P, G)."""
    return gaussian_kernel(np.asarray(E_logged)[:, None] - np.asarray(grid)[None, :], h) / np.asarray(g_logged)[:, None]


def support_diagnostics(weights: np.ndarray, *, ess_fraction_min: float = 0.2, max_weight_max: float = 0.01) -> list[dict[str, float | bool]]:
    """Per candidate: effective sample size, its fraction of P, the max normalised weight and the pass flags. Inputs: weights (P, G)."""
    out = []
    P = weights.shape[0]
    for gi in range(weights.shape[1]):
        w = weights[:, gi]
        total = w.sum()
        ess = float(total**2 / max((w**2).sum(), 1e-300)) if total > 0 else 0.0
        max_norm = float(w.max() / total) if total > 0 else 1.0
        out.append({"ess": ess, "ess_fraction": ess / P, "max_normalized_weight": max_norm, "ess_pass": ess / P >= ess_fraction_min, "max_weight_pass": max_norm <= max_weight_max})
    return out


__all__ = ["BANDWIDTH", "G_MIN", "Propensity", "SUPPORT_MIN", "dr_weights", "gaussian_kernel", "support_diagnostics"]
