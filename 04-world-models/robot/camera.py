"""Step 0c: get the T90's live-view camera frames into Python.

    uv run robot/camera.py                 # 10 s, save 4 frames/s
    uv run robot/camera.py --seconds 30 --fps 2
    uv run robot/camera.py --fps 0         # save every frame

Starts the same live view as the app (AWS Kinesis WebRTC, see live_view.py),
decodes the H.264 stream with aiortc/PyAV and saves JPEGs to
data/camera/<time>/ (gitignored) plus frames.csv with, per saved frame:

    idx, t_mono, t_wall, pts, width, height, file

t_mono is time.monotonic() when the frame arrived in Python (same clock as
probe.py's log), t_wall is Unix time. The robot's own capture time is not in
the stream; the delay between capture and arrival is what we measure next.

The live-view PIN (set in the ECOVACS app when you enabled the camera) is asked
once; only its hash (what the API wants anyway) is saved to .secrets/video_pin.

PRIVACY: frames show your home. They stay in data/ (gitignored). The robot
announces / shows when someone is watching, like with the app.
"""

import argparse
import asyncio
import csv
import getpass
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp
from connect import SECRETS, _device_id, email_and_country
from deebot_client.api_client import ApiClient
from deebot_client.authentication import create_rest_config
from live_view import (
    LiveViewSession,
    async_send_app_ping,
    async_start_watch,
    encode_live_view_pin,
)
from probe import login

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "camera"
PIN_FILE = SECRETS / "video_pin"


def encoded_pin(robot: dict[str, Any]) -> str:
    """Hashed live-view PIN: from .secrets/ if saved, else ask once and save."""
    if PIN_FILE.exists():
        return PIN_FILE.read_text().strip()
    pin = getpass.getpass("Live-view PIN (set in the ECOVACS app): ")
    value = encode_live_view_pin(pin, robot)
    PIN_FILE.write_text(value)
    PIN_FILE.chmod(0o600)
    return value


class FrameSaver:
    """on_frame callback: count every frame, save one every 1/fps seconds."""

    def __init__(self, out: Path, fps: float) -> None:
        self.out = out
        self.min_gap = 1 / fps if fps > 0 else 0.0
        self.received = 0
        self.first_t: float | None = None
        self.last_t: float | None = None
        self.last_saved = -1e9
        self.size: tuple[int, int] | None = None
        out.mkdir(parents=True, exist_ok=True)
        self._csv_file = (out / "frames.csv").open("w", newline="")
        self.csv = csv.writer(self._csv_file)
        self.csv.writerow(["idx", "t_mono", "t_wall", "pts", "width", "height", "file"])
        self.saved = 0

    def __call__(self, frame: Any, t_mono: float) -> None:
        self.received += 1
        self.first_t = self.first_t or t_mono
        self.last_t = t_mono
        self.size = (frame.width, frame.height)
        if t_mono - self.last_saved < self.min_gap:
            return
        self.last_saved = t_mono
        name = f"frame_{self.saved:05d}.jpg"
        frame.to_image().save(self.out / name, quality=90)
        self.csv.writerow(
            [self.saved, f"{t_mono:.4f}", f"{time.time():.4f}", frame.pts,
             frame.width, frame.height, name]
        )
        self.saved += 1
        if self.saved % 10 == 1:
            print(f"  saved {self.saved} frame(s), received {self.received}")

    def close(self) -> None:
        self._csv_file.close()


async def run(args: argparse.Namespace) -> None:
    email, country = email_and_country()
    out = OUT_DIR / f"{datetime.now():%Y%m%d-%H%M%S}"

    async with aiohttp.ClientSession() as http:
        auth = await login(http, email, country)
        credentials = await auth.authenticate()
        devices = await ApiClient(auth).get_devices()
        if not devices.mqtt:
            raise SystemExit("No MQTT-supported device on this account.")
        robot = dict(devices.mqtt[0].api)  # raw device record (did, class, resource, service)
        print(f"Using {robot.get('nick') or robot.get('deviceName')} ({robot['class']})")

        t_request = time.monotonic()
        info = await async_start_watch(
            http,
            # Same config login() builds; start_watch needs its portal_url.
            rest_config=create_rest_config(http, device_id=_device_id(), alpha_2_country=country),
            login_client_id=_device_id(),
            credentials=credentials,
            country=country,
            robot=robot,
            encoded_pin=encoded_pin(robot),
        )
        await async_send_app_ping(http, credentials=credentials, robot=robot)
        print(f"Live-view session granted (region {info.region}); connecting WebRTC...")

        saver = FrameSaver(out, args.fps)
        live = LiveViewSession(http, session_info=info, on_frame=saver)
        try:
            if not await live.start(timeout=args.timeout):
                raise SystemExit(
                    f"No video within {args.timeout:g} s (status={live.status}, "
                    f"error={live.error}). Run with --debug for the WebRTC stages."
                )
            print(f"First frame {saver.first_t - t_request:.1f} s after the request. "
                  f"Recording {args.seconds:g} s...")
            try:
                await asyncio.wait_for(live.stop_event.wait(), timeout=args.seconds)
                print(f"Stream ended early: status={live.status} error={live.error}")
            except TimeoutError:
                pass
        finally:
            await live.stop()
            saver.close()

    span = (saver.last_t or 0) - (saver.first_t or 0)
    rate = (saver.received - 1) / span if span > 0 else 0
    print(
        f"\nReceived {saver.received} frames ({rate:.1f} fps, {saver.size[0]}x{saver.size[1]}), "
        f"saved {saver.saved} to {out}"
        if saver.size
        else f"\nNo frames received. Output dir: {out}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seconds", type=float, default=10, help="how long to record")
    parser.add_argument("--fps", type=float, default=4, help="frames/s to save (0 = all)")
    parser.add_argument("--timeout", type=float, default=30, help="max wait for first frame")
    parser.add_argument("--debug", action="store_true", help="log every WebRTC stage")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    if args.debug:
        logging.getLogger("live_view").setLevel(logging.DEBUG)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
