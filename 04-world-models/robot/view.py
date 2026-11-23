"""Step 1 check: play an episode back with its actions and pose drawn on top.

    uv run robot/view.py data/episodes/<time>             # -> <dir>/episode.mp4
    uv run robot/view.py data/episodes/<time> --fps 1     # half speed
    uv run robot/view.py data/episodes/<time> --sheet     # also a contact sheet PNG

Renders exactly what the model will see (episode.npz, not the raw frames):

    +-----------------------------+-----------+
    | frame o_i (2x)              | top-down  |
    |                             | map: path,|
    |                             | robot +   |
    |                             | heading   |
    +-----------------------------+-----------+
    | step i/T   a_i   next a                                |
    | pose, motion to next step                              |
    | action timeline (colour per action, cursor at i)       |
    +--------------------------------------------------------+

What "aligned" looks like: frame i shows the world when a_i was sent, so the
picture should start turning / moving forward in frame i+1, not in frame i and
not two frames later. "motion to next step" (from the pose) should match a_i.
"""

import argparse
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

COLOURS = {
    "stop": (150, 150, 150),
    "forward": (80, 200, 120),
    "turnLeft": (90, 160, 255),
    "turnRight": (255, 160, 60),
    "backward": (200, 90, 200),
    "unknown": (90, 40, 40),
}
SCALE = 2  # frame upscaling for readability
MAP = 288  # map panel size (px)
BAR = 96  # info bar height (px)
FONT = ImageFont.load_default(size=15)
SMALL = ImageFont.load_default(size=12)


def wrap(deg: float) -> float:
    return (deg + 180) % 360 - 180


class TopDown:
    """Fit all valid poses into a square panel (map x right, map y up)."""

    def __init__(self, pose: np.ndarray, valid: np.ndarray, size: int) -> None:
        self.pose, self.valid, self.size = pose, valid, size
        pts = pose[valid, :2]
        if len(pts):
            lo, hi = pts.min(0), pts.max(0)
            span = max(float((hi - lo).max()), 500.0)  # at least 0.5 m across
            self.centre = (lo + hi) / 2
            self.mm_per_px = span / (size - 40)
        else:
            self.centre, self.mm_per_px = np.zeros(2), 1.0

    def px(self, x: float, y: float) -> tuple[float, float]:
        dx, dy = (np.array([x, y]) - self.centre) / self.mm_per_px
        return self.size / 2 + dx, self.size / 2 - dy  # image y points down

    def draw(self, i: int, actions: np.ndarray, names: list[str]) -> Image.Image:
        img = Image.new("RGB", (self.size, self.size), (24, 24, 28))
        d = ImageDraw.Draw(img)
        if not self.valid.any():
            d.text((12, self.size / 2 - 10), "no valid pose in\nthis episode", font=FONT,
                   fill=(220, 120, 120))
            return img
        # Full path faint, travelled path coloured by action.
        idx = np.flatnonzero(self.valid)
        d.line([self.px(*self.pose[k, :2]) for k in idx], fill=(70, 70, 80), width=2)
        for k0, k1 in zip(idx, idx[1:], strict=False):
            if k1 > i:
                break
            colour = COLOURS[names[actions[k0]]]
            d.line([self.px(*self.pose[k0, :2]), self.px(*self.pose[k1, :2])], fill=colour, width=3)
        # Scale bar: 0.5 m.
        bar = 500 / self.mm_per_px
        d.line([(10, self.size - 14), (10 + bar, self.size - 14)], fill=(200, 200, 200), width=2)
        d.text((12, self.size - 32), "0.5 m", font=SMALL, fill=(200, 200, 200))
        if not self.valid[i]:
            d.text((10, 8), "pose invalid", font=FONT, fill=(220, 120, 120))
            return img
        x, y, a = self.pose[i]
        cx, cy = self.px(x, y)
        r = 9
        d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(255, 255, 255), width=2)
        hx, hy = cx + 22 * np.cos(np.radians(a)), cy - 22 * np.sin(np.radians(a))
        d.line([(cx, cy), (hx, hy)], fill=(255, 255, 255), width=3)
        return img


def render(ep_dir: Path, fps: float, sheet: bool) -> Path:
    ep = np.load(ep_dir / "episode.npz")
    frames, actions, pose, t = ep["frames"], ep["actions"], ep["pose"], ep["t"]
    names = [str(n) for n in ep["action_names"]]
    valid = ep["pose_valid"] if "pose_valid" in ep else ~np.isnan(pose[:, 0])
    inferred = bool(ep["labels_inferred"]) if "labels_inferred" in ep else False
    n, h, w, _ = frames.shape
    fw, fh = w * SCALE, h * SCALE
    width, height = fw + MAP, max(fh, MAP) + BAR
    width, height = width + width % 2, height + height % 2  # yuv420p needs even sizes
    top = TopDown(pose, valid, MAP)

    out_path = ep_dir / "episode.mp4"
    sheet_tiles: list[Image.Image] = []
    with av.open(str(out_path), "w") as container:
        stream = container.add_stream("libx264", rate=Fraction(fps).limit_denominator(1000))
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        stream.options = {"crf": "18"}
        for i in range(n):
            canvas = Image.new("RGB", (width, height), (12, 12, 14))
            canvas.paste(Image.fromarray(frames[i]).resize((fw, fh), Image.NEAREST), (0, 0))
            canvas.paste(top.draw(i, actions, names), (fw, 0))
            draw_bar(ImageDraw.Draw(canvas), i, n, actions, names, pose, valid, t,
                     y0=max(fh, MAP), width=width, inferred=inferred)
            for packet in stream.encode(av.VideoFrame.from_image(canvas)):
                container.mux(packet)
            if sheet and i % max(1, n // 24) == 0:
                sheet_tiles.append(canvas.resize((width // 2, height // 2)))
        for packet in stream.encode():
            container.mux(packet)

    if sheet and sheet_tiles:
        cols = 4
        tw, th = sheet_tiles[0].size
        rows = (len(sheet_tiles) + cols - 1) // cols
        grid = Image.new("RGB", (cols * tw, rows * th), (0, 0, 0))
        for k, tile in enumerate(sheet_tiles):
            grid.paste(tile, ((k % cols) * tw, (k // cols) * th))
        grid.save(ep_dir / "episode_sheet.png")
    return out_path


def draw_bar(d: ImageDraw.ImageDraw, i: int, n: int, actions: np.ndarray, names: list[str],
             pose: np.ndarray, valid: np.ndarray, t: np.ndarray, *, y0: int, width: int,
             inferred: bool = False) -> None:
    act = names[actions[i]]
    nxt = names[actions[i + 1]] if i + 1 < n else "-"
    if inferred:  # clean mode: labels come from the pose, not from commands we sent
        nxt += "   (labels inferred from pose)"
    d.text((10, y0 + 8), f"step {i:3d}/{n - 1}   t={t[i] - t[0]:5.1f}s", font=FONT,
           fill=(230, 230, 230))
    d.rectangle((215, y0 + 6, 330, y0 + 28), fill=COLOURS[act])
    d.text((222, y0 + 8), act, font=FONT, fill=(0, 0, 0))
    d.text((340, y0 + 8), f"next: {nxt}", font=FONT, fill=(170, 170, 170))
    if valid[i]:
        x, y, a = pose[i]
        text = f"pose ({x:6.0f}, {y:6.0f}) mm  {a:4.0f}°"
        if i + 1 < n and valid[i + 1]:
            dist = float(np.hypot(*(pose[i + 1, :2] - pose[i, :2])))
            text += f"   -> next step: {dist:4.0f} mm, {wrap(pose[i + 1, 2] - a):+4.0f}°"
    else:
        text = "pose invalid"
    d.text((10, y0 + 32), text, font=FONT, fill=(200, 200, 200))
    # Action timeline with a cursor.
    ty, th = y0 + 58, 22
    cell = (width - 20) / n
    for k in range(n):
        x0 = 10 + k * cell
        d.rectangle((x0, ty, x0 + max(cell - 1, 1), ty + th), fill=COLOURS[names[actions[k]]])
    cx = 10 + (i + 0.5) * cell
    d.polygon([(cx - 6, ty + th + 9), (cx + 6, ty + th + 9), (cx, ty + th + 1)], fill=(255, 255, 255))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("episode", type=Path, help="data/episodes/<time> directory")
    parser.add_argument("--fps", type=float, default=2.0, help="video steps/s (2 = real time)")
    parser.add_argument("--sheet", action="store_true", help="also write episode_sheet.png")
    args = parser.parse_args()
    path = render(args.episode, args.fps, args.sheet)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
