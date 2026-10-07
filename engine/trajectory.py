"""Serialise an episode to plain JSON for replay, inspection, or export.

The pygame renderer currently consumes the in-memory episode directly. This
format provides a stable boundary for a future browser-based renderer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .evidence import evidence_metadata
from .scm import Episode
from .retention import AttemptRecord, PlayerTrajectory, completion_margin


def _board(array: np.ndarray) -> list[list[int]]:
    return np.asarray(array).astype(int).tolist()


def episode_to_dict(episode: Episode) -> dict[str, Any]:
    return {
        "version": 3,
        "level": {
            "name": episode.level.name,
            "height": episode.level.height,
            "width": episode.level.width,
            "n_colours": episode.level.n_colours,
            "spawn_weights": list(episode.level.weights()),
            "colour_entropy": episode.level.colour_entropy(),
        },
        "difficulty": {
            "tier": episode.tier,
            "move_budget": episode.difficulty.move_budget,
            "goal_colour": episode.difficulty.goal_colour,
            "nominal_goal_count": episode.difficulty.goal_count,
            "served_goal_count": episode.served_goal_count,
            "baseline_logit": episode.difficulty.baseline,
        },
        "player": {
            "label": episode.player.label,
            "skill": episode.player.as_dict(),
        },
        "E": episode.E,
        "proxy": list(map(float, episode.proxy)),
        "evidence_metadata": evidence_metadata(),
        "outcome": {
            "R": int(episode.R),
            "completion_margin": completion_margin(episode),
            "moves_used": episode.moves_used,
            "goals_cleared": episode.goals_cleared,
            "reshuffles": episode.reshuffles,
            "striped_tiles_created": sum(
                len(transition.created_specials)
                for transition in episode.transitions
            ),
            "striped_tiles_activated": sum(
                len(transition.activated_specials)
                for transition in episode.transitions
            ),
        },
        "policy_diagnostics": [
            {
                "candidate_count": diagnostic.candidate_count,
                "noticed_count": diagnostic.noticed_count,
                "candidate_recall": diagnostic.candidate_recall,
                "pattern_noise_scale": diagnostic.pattern_noise_scale,
                "selected_total_cleared": diagnostic.selected_total_cleared,
                "selected_goal_cleared": diagnostic.selected_goal_cleared,
                "selected_setup_value": diagnostic.selected_setup_value,
            }
            for diagnostic in episode.action_diagnostics
        ],
        "states": [
            {
                "t": s.t,
                "board": _board(s.board),
                "specials": _board(s.specials),
                "moves_left": s.moves_left,
                "goals_left": s.goals_left,
                "goal_colour": s.goal_colour,
            }
            for s in episode.states
        ],
        "transitions": [
            {
                "t": t_index,
                "action": {
                    "row": tr.action.row,
                    "col": tr.action.col,
                    "drow": tr.action.drow,
                    "dcol": tr.action.dcol,
                }
                if tr.action
                else None,
                "board_before": _board(tr.board_before),
                "board_swapped": _board(tr.board_swapped),
                "specials_swapped": (
                    None
                    if tr.specials_swapped is None
                    else _board(tr.specials_swapped)
                ),
                "reshuffled": bool(tr.reshuffled),
                "steps": [
                    {
                        "matched": [list(c) for c in step.matched],
                        "board_before": _board(step.board_before),
                        "board_after": _board(step.board_after),
                        "specials_before": (
                            None
                            if step.specials_before is None
                            else _board(step.specials_before)
                        ),
                        "specials_after": (
                            None
                            if step.specials_after is None
                            else _board(step.specials_after)
                        ),
                        "created_specials": [
                            list(special) for special in step.created_specials
                        ],
                        "activated_specials": [
                            list(special) for special in step.activated_specials
                        ],
                        "fall": [list(f) for f in step.fall],
                        "spawned": [list(s) for s in step.spawned],
                        "goal_cleared": step.goal_cleared,
                    }
                    for step in tr.steps
                ],
            }
            for t_index, tr in enumerate(episode.transitions)
        ],
    }


def save_episode(episode: Episode, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(episode_to_dict(episode), indent=1) + "\n")
    return path


def attempt_record_to_dict(record: AttemptRecord) -> dict[str, Any]:
    document = episode_to_dict(record.episode)
    document.update(
        {
            "version": 4,
            "player_id": record.player_id,
            "attempt_id": record.attempt_id,
            "active_before": 1,
            "mastery_before": record.mastery_before,
            "mastery_after": record.mastery_after,
            "completion_margin": record.completion_margin,
            "oracle_win_probability": record.win_probability,
            "churn_probability": record.churn_probability,
            "churn_after": record.churn_after,
        }
    )
    return document


def player_trajectory_to_dict(trajectory: PlayerTrajectory) -> dict[str, Any]:
    return {
        "version": 4,
        "player_id": trajectory.player_id,
        "player": {
            "label": trajectory.player.label,
            "skill": trajectory.player.as_dict(),
        },
        "churned": int(trajectory.churned),
        "churn_attempt": trajectory.churn_attempt,
        "attempts": [
            attempt_record_to_dict(record) for record in trajectory.attempts
        ],
    }


def save_player_trajectory(
    trajectory: PlayerTrajectory, path: str | Path
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(player_trajectory_to_dict(trajectory), indent=1) + "\n")
    return path


def load_trajectory(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def summary_row(episode: Episode) -> dict[str, Any]:
    """One flat record per episode, for building training tables."""
    row: dict[str, Any] = {
        "level": episode.level.name,
        "n_colours": episode.level.n_colours,
        "colour_entropy": episode.level.colour_entropy(),
        "tier": episode.tier,
        "move_budget": episode.difficulty.move_budget,
        "nominal_goal_count": episode.difficulty.goal_count,
        "baseline_logit": episode.difficulty.baseline,
        "skill_label": episode.player.label,
        "E": episode.E,
        "served_goal_count": episode.served_goal_count,
        "R": int(episode.R),
        "moves_used": episode.moves_used,
        "goals_cleared": episode.goals_cleared,
        "reshuffles": episode.reshuffles,
        "cascade_depth_mean": float(
            np.mean([t.cascade_depth for t in episode.transitions])
        )
        if episode.transitions
        else 0.0,
        "tiles_per_move": float(
            np.mean([t.tiles_cleared for t in episode.transitions])
        )
        if episode.transitions
        else 0.0,
        "striped_tiles_created": sum(
            len(transition.created_specials)
            for transition in episode.transitions
        ),
        "striped_tiles_activated": sum(
            len(transition.activated_specials)
            for transition in episode.transitions
        ),
    }
    diagnostics = episode.action_diagnostics
    row.update(
        {
            "candidate_recall_mean": float(
                np.mean([diagnostic.candidate_recall for diagnostic in diagnostics])
            )
            if diagnostics
            else 0.0,
            "pattern_noise_scale_mean": float(
                np.mean(
                    [diagnostic.pattern_noise_scale for diagnostic in diagnostics]
                )
            )
            if diagnostics
            else 0.0,
            "selected_setup_value_mean": float(
                np.mean(
                    [diagnostic.selected_setup_value for diagnostic in diagnostics]
                )
            )
            if diagnostics
            else 0.0,
            "selected_goal_cleared_mean": float(
                np.mean(
                    [diagnostic.selected_goal_cleared for diagnostic in diagnostics]
                )
            )
            if diagnostics
            else 0.0,
        }
    )
    for name, value in episode.player.as_dict().items():
        row[f"k_{name}"] = value
    from .scm import PROXY_NAMES

    for name, value in zip(PROXY_NAMES, episode.proxy):
        row[f"x_{name}"] = float(value)
    return row


def attempt_summary_row(record: AttemptRecord) -> dict[str, Any]:
    row = summary_row(record.episode)
    row.update(
        {
            "player_id": record.player_id,
            "attempt_id": record.attempt_id,
            "active_before": 1,
            "mastery_before": record.mastery_before,
            "mastery_after": record.mastery_after,
            "completion_margin": record.completion_margin,
            "oracle_win_probability": record.win_probability,
            "churn_probability": record.churn_probability,
            "churn_after": record.churn_after,
        }
    )
    return row


def logged_attempt_summary_row(record: AttemptRecord) -> dict[str, Any]:
    """One deployable attempt row containing no simulator-only state."""
    row = summary_row(record.episode)
    row.pop("skill_label")
    for name in record.episode.player.as_dict():
        row.pop(f"k_{name}")
    row.update(
        {
            "player_id": record.player_id,
            "attempt_id": record.attempt_id,
            "active_before": 1,
            "completion_margin": record.completion_margin,
            "churn_after": record.churn_after,
        }
    )
    return row


def oracle_attempt_summary_row(record: AttemptRecord) -> dict[str, Any]:
    """Simulator-only attempt state stored outside deployable artifacts."""
    row: dict[str, Any] = {
        "player_id": record.player_id,
        "attempt_id": record.attempt_id,
        "mastery_before": record.mastery_before,
        "mastery_after": record.mastery_after,
        "completion_margin": record.completion_margin,
        "oracle_win_probability": record.win_probability,
        "churn_probability": record.churn_probability,
    }
    for name, value in record.episode.player.as_dict().items():
        row[f"k_{name}"] = value
    return row


__all__ = [
    "attempt_record_to_dict",
    "attempt_summary_row",
    "episode_to_dict",
    "load_trajectory",
    "logged_attempt_summary_row",
    "oracle_attempt_summary_row",
    "save_episode",
    "save_player_trajectory",
    "summary_row",
    "player_trajectory_to_dict",
]
