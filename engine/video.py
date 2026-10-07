"""Render an episode to mp4 or gif by piping frames straight into ffmpeg."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import pygame
import pyro

from .animate import Timing, count_frames, episode_frames
from .render import Theme
from .scm import LEVELS, ground_truth_model
from .spec import SKILL_PROFILES, Difficulty, PlayerSkill
from .trajectory import save_episode


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise RuntimeError("ffmpeg not found on PATH; needed to encode video")
    return exe


def write_video(
    episode,
    path: str | Path,
    theme: Theme | None = None,
    timing: Timing | None = None,
    crf: int = 18,
) -> Path:
    """Encode the episode. Frames are piped as raw RGB, so nothing hits disk."""
    theme = theme or Theme()
    timing = timing or Timing()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    frames = episode_frames(episode, theme, timing)
    first = next(frames)
    width, height = first.get_size()

    gif = path.suffix.lower() == ".gif"
    codec = (
        ["-vf", "split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer"]
        if gif
        else ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf),
              "-movflags", "+faststart"]
    )

    command = [
        _ffmpeg(), "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", str(timing.fps),
        "-i", "-", *codec, str(path),
    ]

    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    try:
        process.stdin.write(pygame.image.tostring(first, "RGB"))
        for surface in frames:
            process.stdin.write(pygame.image.tostring(surface, "RGB"))
    finally:
        process.stdin.close()
        code = process.wait()
    if code != 0:
        raise RuntimeError(f"ffmpeg exited with status {code}")
    return path


def write_thumbnail(episode, path: str | Path, theme: Theme | None = None) -> Path:
    """Save the opening board as a still, for figures and link previews."""
    from .render import Hud, render_board, save_png

    theme = theme or Theme()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = episode.states[0]
    hud = Hud(
        moves_left=state.moves_left,
        goals_left=state.goals_left,
        goal_colour=episode.difficulty.goal_colour,
        caption=f"{episode.level.name} · {episode.tier or 'given'}",
        subcaption=f"K={episode.player.label}  E={episode.E:+.2f}",
    )
    save_png(
        render_board(state.board, hud, theme, specials=state.specials), path
    )
    return path


def main() -> None:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="media/episode.mp4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--level", default=None, help="level name, e.g. orchard")
    parser.add_argument(
        "--profile", default=None, help="demo skill profile, e.g. balanced-expert"
    )
    parser.add_argument("--goal-count", type=int, default=None)
    parser.add_argument("--move-budget", type=int, default=20)
    parser.add_argument("--E", type=float, default=None, help="pin effective difficulty")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--cell", type=int, default=56)
    parser.add_argument("--json", default=None, help="also write the trajectory here")
    parser.add_argument("--thumb", default=None, help="also write a still PNG here")
    args = parser.parse_args()

    pyro.set_rng_seed(args.seed)

    level = None
    if args.level:
        level = next(l for l in LEVELS if l.name == args.level)
    player: PlayerSkill | None = None
    if args.profile:
        player = next(s for s in SKILL_PROFILES if s.label == args.profile)
    difficulty = None
    if args.goal_count is not None:
        difficulty = Difficulty(args.move_budget, 1, args.goal_count)

    episode = ground_truth_model(
        level=level, player=player, difficulty=difficulty, E=args.E
    )

    timing = Timing(fps=args.fps)
    path = write_video(episode, args.out, Theme(cell=args.cell), timing)
    print(
        f"{path}  frames={count_frames(episode, timing)}  moves={episode.moves_used}  "
        f"R={episode.R}  E={episode.E:+.2f}  K={episode.player.label}"
    )
    if args.json:
        print(save_episode(episode, args.json))
    if args.thumb:
        print(write_thumbnail(episode, args.thumb, Theme(cell=args.cell)))


if __name__ == "__main__":  # pragma: no cover
    main()
