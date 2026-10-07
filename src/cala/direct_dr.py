"""
This is the implementation of the Direct-DR evaluation: a logged-outcome regression mu_obs(e, W) -> P(C_20 = 1), with a small MLP cross-fitted by
player (5 folds) on training and validation players active at the landmark. Test predictions average the fold models.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def _design(e: np.ndarray, W: np.ndarray, level: np.ndarray) -> np.ndarray:
    onehot = np.eye(3)[np.asarray(level)]
    return np.concatenate([np.asarray(e, np.float64)[:, None], np.asarray(W, np.float64), onehot], axis=1)


def _fit(X: np.ndarray, y: np.ndarray, *, seed: int, device: torch.device, epochs: int = 400, hidden: int = 64, weight_decay: float = 1e-3) -> nn.Module:
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(X.shape[1], hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=weight_decay)
    Xt, yt = torch.as_tensor(X, dtype=torch.float32, device=device), torch.as_tensor(y, dtype=torch.float32, device=device)
    for _ in range(epochs):
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(model(Xt).squeeze(1), yt)
        loss.backward()
        opt.step()
    return model.eval()


def cross_fitted_outcome_model(W: np.ndarray, E: np.ndarray, Y: np.ndarray, level: np.ndarray, *, W_test: np.ndarray, E_test: np.ndarray, level_test: np.ndarray, grid: np.ndarray, folds: int = 5, seed: int = 0,
                               device: torch.device = torch.device("cpu"), epochs: int = 400) -> dict[str, object]:
    """Inputs: training rows (W, E, Y, level) and test rows; grid. Outputs: mu_grid_test (P, G), mu_logged_test (P,), out-of-fold Brier on the training rows, fold count."""
    X = _design(E, W, level)
    mean, scale = X.mean(axis=0), np.where(X.std(axis=0) > 1e-8, X.std(axis=0), 1.0)
    z = lambda x: (x - mean) / scale
    rng = np.random.default_rng(seed)
    fold_of = rng.integers(0, folds, size=len(X))
    oof = np.zeros(len(X))
    grid = np.asarray(grid, np.float64)
    P = len(W_test)
    mu_grid = np.zeros((P, len(grid)))
    mu_logged = np.zeros(P)
    for f in range(folds):
        train = fold_of != f
        model = _fit(z(X[train]), np.asarray(Y, np.float64)[train], seed=seed + f, device=device, epochs=epochs)
        with torch.no_grad():
            pred = lambda x: torch.sigmoid(model(torch.as_tensor(z(x), dtype=torch.float32, device=device)).squeeze(1)).cpu().numpy()
            oof[~train] = pred(X[~train])
            for gi, e in enumerate(grid):
                mu_grid[:, gi] += pred(_design(np.full(P, e), W_test, level_test)) / folds
            mu_logged += pred(_design(E_test, W_test, level_test)) / folds
    return {"mu_grid_test": mu_grid, "mu_logged_test": mu_logged, "oof_brier": float(np.mean((oof - np.asarray(Y)) ** 2)), "folds": folds, "n_train_rows": int(len(X)), "epochs": epochs}


__all__ = ["cross_fitted_outcome_model"]
