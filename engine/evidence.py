"""Typed task evidence emitted from multidimensional player skill."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pyro
import pyro.distributions as dist
import torch

from .spec import LevelContext, PlayerSkill, SKILL_NAMES


@dataclass(frozen=True)
class EvidenceSpec:
    name: str
    distribution: str
    loading: tuple[float, float, float, float]
    intercept: float
    context_loading: float
    direction: str
    timing: str = "post_episode"
    dispersion: float = 12.0


EVIDENCE_SPECS: tuple[EvidenceSpec, ...] = (
    EvidenceSpec("search_latency", "log_normal", (-0.55, 0.0, 0.0, 0.0), 1.0, 0.15, "lower", dispersion=0.35),
    EvidenceSpec("candidate_recall", "beta", (0.85, 0.0, 0.0, 0.0), 0.0, -0.10, "higher"),
    EvidenceSpec("hint_count", "poisson", (-0.45, 0.0, 0.0, 0.0), 0.2, 0.15, "lower"),
    EvidenceSpec("pattern_error_rate", "beta", (0.0, -0.75, 0.0, 0.0), -0.2, 0.10, "lower"),
    EvidenceSpec("immediate_pattern_precision", "beta", (0.0, 0.85, 0.0, 0.0), 0.0, -0.10, "higher"),
    EvidenceSpec("distractor_resistance", "beta", (0.0, 0.70, 0.0, 0.0), 0.0, -0.10, "higher"),
    EvidenceSpec("lookahead_choice_rate", "beta", (0.0, 0.0, 0.80, 0.0), 0.0, -0.10, "higher"),
    EvidenceSpec("setup_value_z", "normal", (0.0, 0.0, 0.75, 0.0), 0.0, 0.15, "higher", dispersion=0.60),
    EvidenceSpec("cascade_preparation", "beta", (0.0, 0.0, 0.65, 0.0), -0.2, 0.10, "higher"),
    EvidenceSpec("goal_clear_share", "beta", (0.0, 0.0, 0.0, 0.85), 0.0, -0.10, "higher"),
    EvidenceSpec("goals_per_move", "log_normal", (0.0, 0.0, 0.0, 0.40), 0.2, -0.10, "higher", dispersion=0.40),
    EvidenceSpec("moves_left_efficiency", "beta", (0.0, 0.0, 0.15, 0.65), 0.0, -0.10, "higher"),
)

EVIDENCE_NAMES: tuple[str, ...] = tuple(spec.name for spec in EVIDENCE_SPECS)
EVIDENCE_Q_MATRIX = np.asarray(
    [spec.loading for spec in EVIDENCE_SPECS], dtype=np.float64
)


def _task_context(level: LevelContext) -> float:
    """One standardized complexity contrast for the current evidence task."""
    return float(level.n_colours - 5)


def evidence_expectations(
    player: PlayerSkill, level: LevelContext
) -> np.ndarray:
    """Return each indicator's conditional mean for diagnostics."""
    skill = player.as_array()
    context = _task_context(level)
    values = []
    for spec in EVIDENCE_SPECS:
        predictor = (
            spec.intercept
            + np.asarray(spec.loading) @ skill
            + spec.context_loading * context
        )
        if spec.distribution == "beta":
            values.append(1.0 / (1.0 + np.exp(-predictor)))
        elif spec.distribution == "log_normal":
            values.append(np.exp(predictor + 0.5 * spec.dispersion**2))
        elif spec.distribution == "poisson":
            values.append(np.exp(predictor))
        elif spec.distribution == "normal":
            values.append(predictor)
        else:
            raise ValueError(f"unsupported evidence distribution {spec.distribution}")
    return np.asarray(values, dtype=np.float64)


def sample_evidence(
    player: PlayerSkill,
    level: LevelContext,
    *,
    name: str = "X",
    value: np.ndarray | None = None,
) -> np.ndarray:
    """Sample heterogeneous evidence coordinates conditional on K and task L."""
    if value is not None:
        observed = np.asarray(value, dtype=np.float64)
        if observed.shape != (len(EVIDENCE_SPECS),):
            raise ValueError("evidence value has the wrong shape")
        pyro.deterministic(name, torch.as_tensor(observed, dtype=torch.float32))
        return observed

    skill = player.as_array()
    context = _task_context(level)
    values: list[float] = []
    for spec in EVIDENCE_SPECS:
        if spec.timing != "post_episode":
            raise ValueError(
                f"unsupported evidence timing {spec.timing} for {spec.name}"
            )
        predictor = float(
            spec.intercept
            + np.asarray(spec.loading) @ skill
            + spec.context_loading * context
        )
        if spec.distribution == "beta":
            mean = 1.0 / (1.0 + np.exp(-predictor))
            concentration = spec.dispersion
            distribution = dist.Beta(
                torch.tensor(mean * concentration),
                torch.tensor((1.0 - mean) * concentration),
            )
        elif spec.distribution == "log_normal":
            distribution = dist.LogNormal(
                torch.tensor(predictor), torch.tensor(spec.dispersion)
            )
        elif spec.distribution == "poisson":
            distribution = dist.Poisson(torch.tensor(np.exp(predictor)))
        elif spec.distribution == "normal":
            distribution = dist.Normal(
                torch.tensor(predictor), torch.tensor(spec.dispersion)
            )
        else:
            raise ValueError(f"unsupported evidence distribution {spec.distribution}")
        values.append(float(pyro.sample(f"{name}/{spec.name}", distribution)))
    evidence = np.asarray(values, dtype=np.float64)
    pyro.deterministic(name, torch.as_tensor(evidence, dtype=torch.float32))
    return evidence


def evidence_metadata() -> list[dict[str, object]]:
    metadata = []
    for spec in EVIDENCE_SPECS:
        values = asdict(spec)
        values["skill_names"] = list(SKILL_NAMES)
        metadata.append(values)
    return metadata


__all__ = [
    "EVIDENCE_NAMES",
    "EVIDENCE_Q_MATRIX",
    "EVIDENCE_SPECS",
    "EvidenceSpec",
    "evidence_expectations",
    "evidence_metadata",
    "sample_evidence",
]