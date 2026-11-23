"""Step 2a: the copy-last-frame baseline.

    uv run python -m wm.copy_last

Prediction: o_{i+k} = o_i for every horizon k (the world never changes). On a slow
robot this is hard to beat at k=1, and every world model has to beat it.

Reported on the test split, per mode, as pixel errors on frames scaled to [0, 1]:
    MAE  mean absolute error        (lower is better)
    PSNR 10*log10(1/MSE) in dB       (higher is better; ~20 dB = rough, ~30 = close)
at horizons k = 1, 2, 4, 8 steps (0.5, 1, 2, 4 s), and per action at k = 1: the
baseline should be good for stop, worse for forward and worst for turns (a turn
shifts the whole picture). These numbers are what Step 3 is compared against.
"""

from __future__ import annotations

import numpy as np

from wm.data import ACTIONS, Episode, load_episodes, split

HORIZONS = (1, 2, 4, 8)


def errors(e: Episode, i: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-pair MAE and MSE between o_i and o_{i+k} (frames in [0, 1])."""
    a = e.frames[i].astype(np.float32) / 255
    b = e.frames[i + k].astype(np.float32) / 255
    d = a - b
    return np.abs(d).mean(axis=(1, 2, 3)), (d**2).mean(axis=(1, 2, 3))


def usable_starts(e: Episode, k: int) -> np.ndarray:
    """Steps i where i .. i+k are all usable (no still stretch / unknown inside)."""
    n = len(e)
    ok = np.array([e.keep[i : i + k + 1].all() for i in range(max(n - k, 0))], dtype=bool)
    return np.flatnonzero(ok)


def psnr(mse: np.ndarray) -> float:
    return float(10 * np.log10(1 / max(mse.mean(), 1e-12)))


def report(test: list[Episode]) -> dict[str, dict[int, tuple[float, float, int]]]:
    results: dict[str, dict[int, tuple[float, float, int]]] = {}
    for mode in sorted({e.mode for e in test}):
        eps = [e for e in test if e.mode == mode]
        print(f"\n== {mode} (test) ==")
        print(f"{'horizon':>8} {'seconds':>8} {'MAE':>8} {'PSNR dB':>8} {'pairs':>6}")
        results[mode] = {}
        for k in HORIZONS:
            maes, mses = [], []
            for e in eps:
                i = usable_starts(e, k)
                if len(i):
                    mae, mse = errors(e, i, k)
                    maes.append(mae)
                    mses.append(mse)
            if not maes:
                continue
            mae, mse = np.concatenate(maes), np.concatenate(mses)
            results[mode][k] = (float(mae.mean()), psnr(mse), len(mae))
            print(f"{k:8d} {k / 2:8.1f} {mae.mean():8.4f} {psnr(mse):8.2f} {len(mae):6d}")
        # Per action at k=1: which motions copy-last handles badly.
        print("  per action at k=1 (action a_i between o_i and o_i+1):")
        for a, name in enumerate(ACTIONS):
            maes, mses = [], []
            for e in eps:
                i = usable_starts(e, 1)
                i = i[e.actions[i] == a]
                if len(i):
                    mae, mse = errors(e, i, 1)
                    maes.append(mae)
                    mses.append(mse)
            if maes:
                mae, mse = np.concatenate(maes), np.concatenate(mses)
                print(f"    {name:<10} MAE {mae.mean():.4f}  PSNR {psnr(mse):5.2f} dB  ({len(mae)} pairs)")
    return results


if __name__ == "__main__":
    _, test = split(load_episodes())
    report(test)
