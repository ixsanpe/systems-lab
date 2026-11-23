"""Load recorded episodes and turn them into (o_i, a_i, o_{i+1}) transitions.

    from wm.data import load_episodes, split, transitions
    eps = load_episodes()                       # every data/episodes/*/episode.npz
    train, test = split(eps)                    # by time, per mode, never by frame
    i0, a, i1 = transitions(train)              # index arrays into each episode

Everything works on episode objects so later steps (encoders, world model) can
reuse it. Frames stay uint8 [T, H, W, 3] as recorded (256x144 by default).

Split: within each mode (drive / clean) episodes are put in time order and the
last `test_fraction` of the steps become the test set, with a `gap` of steps
dropped at the cut. Adjacent frames are near-duplicates, so a random per-frame
split would leak; a contiguous block in time doesn't (except across the gap).

Dropped steps (never used as o_i or o_{i+1}):
- runs of `stop` lasting >= `still_steps` (the station mop-wash stops, pauses):
  nothing happens there, they would only teach "nothing changes";
- steps labelled `unknown` (no trusted pose to infer the action from).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EPISODES = ROOT / "data" / "episodes"
# Shared action vocabulary (same order as robot/record.py ACTIONS).
ACTIONS = ["stop", "forward", "turnLeft", "turnRight", "backward", "unknown"]


@dataclass
class Episode:
    name: str
    mode: str  # "drive" (commanded actions) or "clean" (inferred from pose)
    frames: np.ndarray  # [T, H, W, 3] uint8
    actions: np.ndarray  # [T] int, index into ACTIONS
    motion: np.ndarray  # [T, 3] forward mm, left mm, dtheta deg to the next step
    pose_valid: np.ndarray  # [T] bool
    keep: np.ndarray  # [T] bool: usable step (see module docstring)
    start: int = 0  # first step of this piece (after split)

    def __len__(self) -> int:
        return len(self.actions)

    def piece(self, lo: int, hi: int) -> Episode:
        """Steps [lo, hi) as their own episode (used by split)."""
        return Episode(self.name, self.mode, self.frames[lo:hi], self.actions[lo:hi],
                       self.motion[lo:hi], self.pose_valid[lo:hi], self.keep[lo:hi],
                       self.start + lo)


def _still_runs(actions: np.ndarray, min_len: int) -> np.ndarray:
    """Mask of steps inside runs of `stop` that last at least min_len steps."""
    stop = actions == ACTIONS.index("stop")
    mask = np.zeros_like(stop)
    i = 0
    while i < len(stop):
        if stop[i]:
            j = i
            while j < len(stop) and stop[j]:
                j += 1
            if j - i >= min_len:
                mask[i:j] = True
            i = j
        else:
            i += 1
    return mask


def load_episodes(root: Path = EPISODES, still_steps: int = 10,
                  modes: tuple[str, ...] = ("drive", "clean")) -> list[Episode]:
    """All episodes with an episode.npz, in time order (folder names are timestamps)."""
    episodes = []
    for d in sorted(p for p in root.iterdir() if (p / "episode.npz").exists()):
        meta = json.loads((d / "meta.json").read_text())
        mode = meta.get("mode", "drive")
        if mode not in modes:
            continue
        ep = np.load(d / "episode.npz")
        names = [str(n) for n in ep["action_names"]]
        # Map this episode's own action indices onto the shared vocabulary.
        actions = np.array([ACTIONS.index(names[a]) for a in ep["actions"]], dtype=np.int64)
        n = len(actions)
        motion = ep["motion"] if "motion" in ep else np.full((n, 3), np.nan, np.float32)
        valid = ep["pose_valid"] if "pose_valid" in ep else np.ones(n, bool)
        keep = ~_still_runs(actions, still_steps) & (actions != ACTIONS.index("unknown"))
        episodes.append(Episode(d.name, mode, ep["frames"], actions, motion, valid, keep))
    return episodes


def split(episodes: list[Episode], test_fraction: float = 0.2,
          gap: int = 20) -> tuple[list[Episode], list[Episode]]:
    """Per mode: the last test_fraction of steps (in time) are test, minus a gap."""
    train, test = [], []
    for mode in sorted({e.mode for e in episodes}):
        eps = [e for e in episodes if e.mode == mode]
        total = sum(len(e) for e in eps)
        cut = int(total * (1 - test_fraction))  # global step index of the cut
        seen = 0
        for e in eps:
            lo, hi = seen, seen + len(e)
            if hi <= cut - gap:
                train.append(e)
            elif lo >= cut:
                test.append(e)
            else:  # the cut falls inside this episode
                if cut - gap - lo > 0:
                    train.append(e.piece(0, cut - gap - lo))
                if hi - cut > 0:
                    test.append(e.piece(cut - lo, len(e)))
            seen = hi
    return train, test


def transitions(episodes: list[Episode], offset: int = 0) -> list[tuple[Episode, np.ndarray]]:
    """Per episode, the steps i where (o_{i+offset}, a_i, o_{i+1+offset}) is usable.

    offset != 0 pairs a_i with a later/earlier frame pair: an alignment check (the
    classifier should do best at offset 0 if the shift in record.py is right).
    """
    out = []
    for e in episodes:
        n = len(e)
        idx = np.arange(n)
        a, b = idx + offset, idx + 1 + offset
        ok = (a >= 0) & (b < n)
        ok[ok] &= e.keep[idx[ok]] & e.keep[a[ok]] & e.keep[b[ok]]
        out.append((e, idx[ok]))
    return out


def describe(episodes: list[Episode], title: str) -> None:
    steps = sum(int(e.keep.sum()) for e in episodes)
    counts = np.zeros(len(ACTIONS), int)
    for e in episodes:
        counts += np.bincount(e.actions[e.keep], minlength=len(ACTIONS))
    mix = ", ".join(f"{a} {c}" for a, c in zip(ACTIONS, counts, strict=True) if c)
    print(f"{title}: {len(episodes)} piece(s), {steps} usable steps ({mix})")


if __name__ == "__main__":
    eps = load_episodes()
    for e in eps:
        print(f"  {e.name} {e.mode:<5} {len(e):5d} steps, {int(e.keep.sum()):5d} usable")
    train, test = split(eps)
    describe(train, "train")
    describe(test, "test ")
