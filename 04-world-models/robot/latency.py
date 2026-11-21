"""Step 0d: how long from sending a move command until the camera shows motion?

    uv run robot/latency.py                 # 4 trials, 0.6 s turns
    uv run robot/latency.py --trials 6 --hold 0.4
    uv run robot/latency.py --act forward   # forward instead of turns (needs space!)

Starts the live view, then runs trials. Each trial:

    wait until the picture is still -> send turn -> wait --hold -> send stop -> rest

Turns alternate left/right so the robot ends where it started, and a turn shifts
the whole picture, so motion is easy to see. For every frame we compute how much
it differs from the previous one (mean absolute pixel difference on a 160x90
grey thumbnail). "Motion" = the difference rises above the still-camera noise
measured just before the command.

Per trial we report, all on time.monotonic() (arrival in Python):
    ack      send -> robot's "ok" reply (command path through the cloud)
    start    send -> first frame that shows motion
    stop     stop sent -> first frame that is still again

`start` is what matters for the world model: command path + robot reaction +
video delay. We can't separate the video delay from the rest here, because the
stream carries no wall-clock capture time.

Results go to data/latency/<time>/: frames.csv (every frame: t, pts, diff) and
trials.csv. Robot must be off the dock with ~30 cm free around it.
"""

import argparse
import asyncio
import csv
import logging
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp
from camera import encoded_pin
from connect import _device_id, email_and_country
from deebot_client.api_client import ApiClient
from deebot_client.authentication import create_rest_config
from deebot_client.commands.json.custom import CustomCommand
from deebot_client.device import Device
from live_view import LiveViewSession, async_send_app_ping, async_start_watch
from PIL import Image, ImageChops, ImageStat
from probe import login, move_args

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "latency"
THUMB = (160, 90)
LOOK = 3  # start: another moving frame within this many frames
QUIET = 6  # end: this many still frames in a row (~0.4 s at 15 fps)


class MotionMeter:
    """on_frame callback: per-frame difference to the previous frame."""

    def __init__(self) -> None:
        self.rows: list[tuple[float, int, float]] = []  # (t_mono, pts, diff)
        self._prev: Image.Image | None = None

    def __call__(self, frame: Any, t_mono: float) -> None:
        # Scale + grey in libav (C), then a cheap PIL diff.
        small = frame.reformat(width=THUMB[0], height=THUMB[1], format="gray").to_image()
        diff = 0.0
        if self._prev is not None:
            diff = ImageStat.Stat(ImageChops.difference(small, self._prev)).mean[0]
        self._prev = small
        self.rows.append((t_mono, frame.pts or 0, diff))

    def window(self, t0: float, t1: float) -> list[tuple[float, int, float]]:
        return [r for r in self.rows if t0 <= r[0] < t1]

    def fps(self) -> float:
        if len(self.rows) < 2:
            return 0.0
        return (len(self.rows) - 1) / (self.rows[-1][0] - self.rows[0][0])


@dataclass
class Trial:
    act: str
    t_send: float
    t_ack: float = 0.0
    t_stop: float = 0.0
    ok: bool = False
    threshold: float = 0.0
    noise: float = 0.0
    t_start_motion: float | None = None
    t_end_motion: float | None = None
    peak: float = 0.0
    notes: list[str] = field(default_factory=list)


def threshold_from(still: list[float], floor: float) -> tuple[float, float]:
    """Noise level of a still camera and the 'this is motion' threshold above it."""
    if len(still) < 3:
        return 0.0, floor
    noise = statistics.median(still)
    spread = statistics.pstdev(still)
    return noise, max(floor, noise + 6 * spread, 2.5 * noise)


def analyse(trial: Trial, meter: MotionMeter, still_before: float, floor: float) -> None:
    still = [d for _, _, d in meter.window(trial.t_send - still_before, trial.t_send)]
    trial.noise, trial.threshold = threshold_from(still, floor)
    after = meter.window(trial.t_send, trial.t_stop + 5)
    trial.peak = max((d for _, _, d in after), default=0.0)
    moving = [d > trial.threshold for _, _, d in after]
    # The T90 streams 15 fps but only ~every other frame is a new image (the rest
    # are exact repeats, diff ~0), so "N frames in a row" never fires. Instead:
    # Start = first frame over the threshold with another one within the next
    # LOOK frames (a single compression glitch doesn't count).
    for i, is_moving in enumerate(moving):
        if is_moving and any(moving[i + 1 : i + 1 + LOOK]):
            trial.t_start_motion = after[i][0]
            break
    if trial.t_start_motion is None:
        trial.notes.append("no motion seen")
        return
    # End = after the stop, the first frame that starts a run of QUIET still frames.
    for i, (t, _, _) in enumerate(after):
        if t < max(trial.t_stop, trial.t_start_motion):
            continue
        window = moving[i : i + QUIET]
        if len(window) == QUIET and not any(window):
            trial.t_end_motion = t
            break
    if trial.t_end_motion is None:
        trial.notes.append("still moving at end of window")


async def wait_still(meter: MotionMeter, floor: float, seconds: float, limit: float) -> bool:
    """Wait until the last `seconds` of frames all look still (or give up after limit)."""
    deadline = time.monotonic() + limit
    while time.monotonic() < deadline:
        now = time.monotonic()
        recent = [d for _, _, d in meter.window(now - seconds, now)]
        if len(recent) >= 3 and max(recent) < floor:
            return True
        await asyncio.sleep(0.2)
    return False


async def run(args: argparse.Namespace) -> None:
    email, country = email_and_country()
    out = OUT_DIR / f"{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    trials: list[Trial] = []
    meter = MotionMeter()

    async with aiohttp.ClientSession() as http:
        auth = await login(http, email, country)
        credentials = await auth.authenticate()
        devices = await ApiClient(auth).get_devices()
        if not devices.mqtt:
            raise SystemExit("No MQTT-supported device on this account.")
        info = devices.mqtt[0]
        robot = dict(info.api)
        device = Device(info, auth)  # commands go over REST, no MQTT needed here

        async def move(act: str) -> tuple[float, float, bool]:
            t0 = time.monotonic()
            resp = await device.execute_command(CustomCommand("move", move_args(act, 90)))
            body = resp.get("resp", {}).get("body", resp)
            return t0, time.monotonic(), body.get("code") == 0

        session_info = await async_start_watch(
            http,
            rest_config=create_rest_config(http, device_id=_device_id(), alpha_2_country=country),
            login_client_id=_device_id(),
            credentials=credentials,
            country=country,
            robot=robot,
            encoded_pin=encoded_pin(robot),
        )
        await async_send_app_ping(http, credentials=credentials, robot=robot)
        live = LiveViewSession(http, session_info=session_info, on_frame=meter)
        try:
            if not await live.start(timeout=30):
                raise SystemExit(f"No video (status={live.status}, error={live.error}).")
            print("Video up. Measuring still-camera noise...")
            await asyncio.sleep(args.warmup)
            print(f"Stream {meter.fps():.1f} fps.\n")

            for i in range(args.trials):
                act = args.act if args.act != "turn" else ("turnLeft", "turnRight")[i % 2]
                if not await wait_still(meter, args.floor, 1.5, limit=8):
                    print(f"  trial {i + 1}: picture never settled; running anyway")
                t_send, t_ack, ok = await move(act)
                trial = Trial(act=act, t_send=t_send, t_ack=t_ack, ok=ok)
                try:
                    await asyncio.sleep(args.hold)
                finally:
                    trial.t_stop, _, _ = await move("stop")
                trials.append(trial)
                print(f"  trial {i + 1}: {act} ({'ok' if ok else 'REJECTED'}), resting...")
                await asyncio.sleep(args.rest)
        finally:
            if trials:
                await move("stop")
            await live.stop()

    for t in trials:
        analyse(t, meter, still_before=1.5, floor=args.floor)
    report(trials, meter, out)


def ms(value: float | None) -> str:
    return f"{value * 1000:6.0f}" if value is not None else "     -"


def report(trials: list[Trial], meter: MotionMeter, out: Path) -> None:
    with (out / "frames.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_mono", "pts", "diff"])
        w.writerows((f"{t:.4f}", p, f"{d:.3f}") for t, p, d in meter.rows)
    with (out / "trials.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["act", "ok", "t_send", "t_ack", "t_stop", "t_start_motion", "t_end_motion",
                    "noise", "threshold", "peak", "notes"])
        for t in trials:
            w.writerow([t.act, t.ok, t.t_send, t.t_ack, t.t_stop, t.t_start_motion,
                        t.t_end_motion, round(t.noise, 3), round(t.threshold, 3),
                        round(t.peak, 3), "; ".join(t.notes)])

    frame_ms = 1000 / meter.fps() if meter.fps() else 0
    print(f"\nStream: {len(meter.rows)} frames, {meter.fps():.1f} fps "
          f"(one frame every {frame_ms:.0f} ms = the resolution of 'start'/'stop')\n")
    print(f"{'trial':<6}{'act':<11}{'ack ms':>8}{'start ms':>10}{'stop ms':>9}"
          f"{'noise':>7}{'thresh':>8}{'peak':>7}  notes")
    for i, t in enumerate(trials, 1):
        start = t.t_start_motion - t.t_send if t.t_start_motion else None
        stop = t.t_end_motion - t.t_stop if t.t_end_motion else None
        print(f"{i:<6}{t.act:<11}{ms(t.t_ack - t.t_send):>8}{ms(start):>10}{ms(stop):>9}"
              f"{t.noise:7.2f}{t.threshold:8.2f}{t.peak:7.2f}  {'; '.join(t.notes)}")

    starts = [t.t_start_motion - t.t_send for t in trials if t.t_start_motion]
    if starts:
        print(f"\nsend -> first motion frame: median {statistics.median(starts) * 1000:.0f} ms, "
              f"range {min(starts) * 1000:.0f}-{max(starts) * 1000:.0f} ms over {len(starts)} trial(s)")
    print(f"Saved {out}")


def reanalyse(out: Path, floor: float) -> None:
    """Load frames.csv + trials.csv from an earlier run and redo analyse/report."""
    meter = MotionMeter()
    with (out / "frames.csv").open() as f:
        meter.rows = [(float(r["t_mono"]), int(r["pts"]), float(r["diff"])) for r in csv.DictReader(f)]
    with (out / "trials.csv").open() as f:
        trials = [
            Trial(act=r["act"], ok=r["ok"] == "True", t_send=float(r["t_send"]),
                  t_ack=float(r["t_ack"]), t_stop=float(r["t_stop"]))
            for r in csv.DictReader(f)
        ]
    for t in trials:
        analyse(t, meter, still_before=1.5, floor=floor)
    report(trials, meter, out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--act", default="turn", help="turn (alternating L/R) or a move act")
    parser.add_argument("--hold", type=float, default=0.6, help="seconds between move and stop")
    parser.add_argument("--rest", type=float, default=3.0, help="seconds after each stop")
    parser.add_argument("--warmup", type=float, default=4.0, help="seconds of still video first")
    parser.add_argument(
        "--floor", type=float, default=3.0,
        help="minimum frame difference (0-255 grey levels) that counts as motion; "
        "still frames are ~0.05, turns 8-33, post-turn exposure flicker up to ~2.4",
    )
    parser.add_argument(
        "--reanalyse", type=Path, metavar="DIR",
        help="re-run the analysis on a saved data/latency/<time> dir (robot not needed)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    if args.reanalyse:
        reanalyse(args.reanalyse, args.floor)
    else:
        asyncio.run(run(args))


if __name__ == "__main__":
    main()
