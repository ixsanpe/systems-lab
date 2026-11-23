"""Plot the Step 2-3 results into one figure.

    uv run python -m wm.plot_results      # -> figures/step3_results.png (+ data/results/step3.json)

Runs the saved world model (data/models/predictor.pt) through the same test
evaluation as `python -m wm.predictor --eval` and draws:

  A  error vs copy-last per horizon (open-loop rollouts)
  B  error vs copy-last per action, one step ahead
  C  action sensitivity: which action's prediction is closest to reality (cleaning test)
  D  Step 2 video shift sweep: inverse-dynamics accuracy on driven episodes per shift

Only metrics are drawn (no camera frames), so the PNG is safe to commit.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from wm.data import ROOT, load_episodes, split  # noqa: E402
from wm.predictor import CANDIDATES, SAVE, Latents, Predictor, device, evaluate  # noqa: E402

OUT = ROOT / "figures" / "step3_results.png"
RESULTS = ROOT / "data" / "results" / "step3.json"
BEFORE = ROOT / "data" / "results" / "step3_v1.json"  # first version, without the action loss

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
MODE_COLOUR = {"clean": "#2a78d6", "drive": "#eb6834"}
MODE_LABEL = {"clean": "cleaning (labels from pose)", "drive": "driven (exact labels)"}
BLUES = LinearSegmentedColormap.from_list("blues", ["#f3f7fd", "#9cc0ee", "#2a78d6", "#123f78"])

# Step 2 shift sweep (wm.inverse_dynamics, 3 seeds each, driven test set n=111), 2026-11-23.
SHIFT_SWEEP = {
    0.0: [62.2, 62.2, 66.7], 0.1: [64.0, 70.3, 64.9], 0.2: [64.9, 64.9, 69.4],
    0.3: [67.6, 64.9, 68.5], 0.42: [65.8, 69.4, 69.4], 0.55: [66.7, 73.9, 68.5],
    0.7: [62.2, 61.3, 63.1],
}
DRIVE_MAJORITY = 47.7


def style(ax: plt.Axes, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10, fontweight="semibold")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def copy_last_line(ax: plt.Axes, x: float) -> None:
    ax.axhline(1.0, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(x, 1.01, "copy-last", color=INK2, fontsize=8.5, va="bottom")


def panel_horizons(ax: plt.Axes, res: dict, before: dict | None) -> None:
    style(ax, "A  World model error vs copy-last, by horizon")
    for mode, rows in (before or {}).get("rollout", {}).items():
        ks = sorted(rows, key=int)
        ax.plot([int(k) / 2 for k in ks], [rows[k]["model"] / rows[k]["copy"] for k in ks],
                color=MODE_COLOUR[mode], linewidth=1.2, linestyle=(0, (2, 2)), alpha=0.55,
                label=f"{mode}, without action loss")
    for mode, rows in res["rollout"].items():
        ks = sorted(rows, key=int)
        x = [int(k) / 2 for k in ks]
        y = [rows[k]["model"] / rows[k]["copy"] for k in ks]
        ax.plot(x, y, color=MODE_COLOUR[mode], linewidth=2, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=MODE_LABEL[mode])
    copy_last_line(ax, 0.45)
    ax.set_xscale("log", base=2)
    ax.set_xticks([0.5, 1, 2, 4], ["0.5 s", "1 s", "2 s", "4 s"])
    ax.set_xlim(0.42, 5.2)
    ax.set_ylim(0.6, 1.08)
    ax.set_ylabel("feature error ÷ copy-last error\n(lower is better)", color=INK2, fontsize=9)
    ax.set_xlabel("prediction horizon (open loop)", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=INK)


def panel_actions(ax: plt.Axes, res: dict) -> None:
    style(ax, "B  Error vs copy-last per action, 0.5 s ahead")
    names = ["stop", "forward", "turnLeft", "turnRight"]
    modes = list(res["per_action"])
    width = 0.38
    for j, mode in enumerate(modes):
        for i, name in enumerate(names):
            row = res["per_action"][mode].get(name)
            if row is None:
                continue
            x = i + (j - (len(modes) - 1) / 2) * (width + 0.02)
            ax.bar(x, row["ratio"], width, color=MODE_COLOUR[mode], edgecolor=SURFACE, linewidth=1,
                   label=MODE_LABEL[mode] if i == names.index("forward") else None)
            ax.text(x, row["ratio"] + 0.015, f"{row['ratio']:.2f}", ha="center", va="bottom",
                    fontsize=8, color=INK)
            ax.text(x, 0.03, f"n={row['n']}", ha="center", va="bottom", fontsize=7, color=SURFACE,
                    rotation=90)
    copy_last_line(ax, len(names) - 1.05)
    ax.set_xticks(range(len(names)), names)
    ax.set_xlim(-0.6, len(names) - 0.4)
    ax.set_ylim(0, 1.3)
    ax.set_ylabel("feature error ÷ copy-last error", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=INK)


def panel_confusion(ax: plt.Axes, res: dict, mode: str = "clean") -> None:
    sens = res["sensitivity"][mode]
    conf = np.array(sens["confusion"], float)
    rows = [i for i in range(len(CANDIDATES)) if conf[i].sum() >= 20]  # drop tiny rows (backward)
    share = conf[rows] / conf[rows].sum(1, keepdims=True)  # all 5 columns: every possible pick
    names = [CANDIDATES[i] for i in rows]
    style(ax, f"C  Does the model use the action? ({mode} test)")
    ax.grid(False)
    ax.imshow(share, cmap=BLUES, vmin=0, vmax=1)
    for i, r in enumerate(rows):
        for j in range(len(CANDIDATES)):
            v = share[i, j]
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=9,
                    color=SURFACE if v > 0.45 else INK, fontweight="semibold" if r == j else None)
    ax.set_xticks(range(len(CANDIDATES)), CANDIDATES, fontsize=8.5)
    ax.set_yticks(range(len(names)), [f"{n} (n={int(conf[r].sum())})" for n, r in zip(names, rows)],
                  fontsize=8.5)
    ax.set_xlabel("action whose prediction is closest to the real next frame", color=INK2, fontsize=9)
    ax.set_ylabel("action really taken", color=INK2, fontsize=9)
    ax.text(0, -0.13, f"overall {sens['accuracy']:.0%} correct vs {sens['majority']:.0%} for "
            f"always guessing the most common action", transform=ax.transAxes, fontsize=8.5,
            color=INK2, va="top")
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)


def panel_shift(ax: plt.Axes) -> None:
    style(ax, "D  Step 2: which video shift aligns frames and actions?")
    shifts = sorted(SHIFT_SWEEP)
    means = [np.mean(SHIFT_SWEEP[s]) for s in shifts]
    colour = MODE_COLOUR["drive"]
    for s in shifts:
        ax.scatter([s] * 3, SHIFT_SWEEP[s], s=18, color=colour, alpha=0.35, linewidths=0)
    ax.plot(shifts, means, color=colour, linewidth=2, marker="o", markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.axhline(DRIVE_MAJORITY, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(0.0, DRIVE_MAJORITY + 0.6, "always guessing the most common action", color=INK2,
            fontsize=8.5, va="bottom")
    best = 0.42
    ax.annotate(f"chosen: {best} s\n(measured latency ≈ 0.42 s)", xy=(best, np.mean(SHIFT_SWEEP[best])),
                xytext=(0.47, 56), fontsize=8.5, color=INK,
                arrowprops={"arrowstyle": "-", "color": INK2, "linewidth": 0.8})
    ax.set_xticks(shifts, [f"{s:g}" for s in shifts])
    ax.set_ylim(44, 78)
    ax.set_xlabel("video shift (s) used to pick frame o_i", color=INK2, fontsize=9)
    ax.set_ylabel("inverse-dynamics accuracy, driven test (%)\nline = mean, dots = 3 seeds",
                  color=INK2, fontsize=9)


def main() -> None:
    dev = device()
    saved = torch.load(SAVE, map_location=dev)
    te = Latents(split(load_episodes())[1], dev, saved["pool"])
    model = Predictor(n_patches=te.z.shape[1]).to(dev)
    model.load_state_dict(saved["state"])
    res = evaluate(model, te)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(res, indent=1))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5), facecolor=SURFACE)
    before = json.loads(BEFORE.read_text()) if BEFORE.exists() else None
    panel_horizons(axes[0, 0], res, before)
    panel_actions(axes[0, 1], res)
    panel_confusion(axes[1, 0], res)
    panel_shift(axes[1, 1])
    fig.suptitle("Rocky world model — Step 2 baselines and Step 3 first results (test split)",
                 x=0.06, ha="left", fontsize=13, color=INK, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=3, w_pad=3)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor=SURFACE)
    print(f"\nwrote {OUT} and {RESULTS}")


if __name__ == "__main__":
    main()
