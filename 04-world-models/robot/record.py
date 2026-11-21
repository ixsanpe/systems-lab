"""Step 1: record an episode (camera frames + actions + pose) with a random policy.

    uv run robot/record.py                   # 60 s at 2 Hz
    uv run robot/record.py --seconds 120 --seed 3
    uv run robot/record.py --rebuild data/episodes/<time> --shift 0.42

While recording, a random policy drives the robot one step at a time (2 Hz):
it usually keeps its last action and sometimes picks a new one (forward is the
most likely). If it has been driving forward but the position stopped changing
(stuck on something), it turns away for a few steps.

Everything raw is kept, so the episode can be rebuilt later with other settings:
    data/episodes/<time>/
        frames/000000.jpg ...   every video frame, 640 px wide
        frames.csv              idx, t_mono, pts, file
        steps.csv               per step: t_step, action, whether a command was
                                sent (only on change), its ack, polled pose
        meta.json               settings and stats
        episode.npz             the aligned episode (see below)

episode.npz, one row per step i:
    frames  [T, H, W, 3] uint8   o_i: the newest frame that arrived by t_i + shift
    actions [T] int              a_i: action during step i (index into action_names)
    pose    [T, 3] float32       x, y (mm, map frame), heading (deg), polled at t_i
    t       [T] float64          t_i (time.monotonic)
Commands take ~0.42 s to show in the video (latency.py), so o_i is taken at
t_i + shift: it shows the world as it was when a_i was sent, and o_{i+1} then
shows the effect of a_i.

Robot off the dock, open floor, app open to stop it by hand; Ctrl-C stops it too.
"""

import argparse
import asyncio
import csv
import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
from camera import encoded_pin
from connect import _device_id, email_and_country
from deebot_client.api_client import ApiClient
from deebot_client.authentication import create_rest_config
from deebot_client.commands.json.custom import CustomCommand
from deebot_client.commands.json.pos import GetPos
from deebot_client.commands.json.relocation import SetRelocationState
from deebot_client.device import Device
from live_view import LiveViewSession, async_send_app_ping, async_start_watch
from PIL import Image
from probe import login, move_args

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "episodes"
ACTIONS = ["stop", "forward", "turnLeft", "turnRight"]
RAW_WIDTH = 640
Pose = tuple[float, float, float]


def raw_pose(response: dict[str, Any]) -> tuple[Pose, bool] | None:
    """(x, y, heading) and whether the robot trusts it, from a getPos reply.

    `invalid: 1` means the robot lost track of where it is (e.g. it was carried
    somewhere): the numbers are then stale and must not be used as pose.
    """
    data = response.get("resp", {}).get("body", {}).get("data", {})
    pos = data.get("deebotPos")
    if isinstance(pos, list):
        pos = pos[0] if pos else None
    if not pos:
        return None
    return (pos["x"], pos["y"], pos.get("a", 0)), not pos.get("invalid", 0)


class RandomPolicy:
    """Sticky random actions, with an 'unstick' turn when forward stops moving."""

    def __init__(self, rng: random.Random, keep: float, weights: dict[str, float]) -> None:
        self.rng = rng
        self.keep = keep
        self.names = list(weights)
        self.weights = list(weights.values())
        self.action = "stop"
        self.forced: list[str] = []  # queued unstick steps
        self.forward_steps = 0  # consecutive steps spent driving forward

    def __call__(self, recent_poses: list[tuple[float, float, float] | None]) -> str:
        self.forward_steps = self.forward_steps + 1 if self.action == "forward" else 0
        if self.forced:
            self.action = self.forced.pop(0)
        elif self.forward_steps >= 4 and self._stuck(recent_poses):
            turn = self.rng.choice(["turnLeft", "turnRight"])
            self.forced = [turn] * self.rng.randint(2, 4)
            self.action = self.forced.pop(0)
        elif self.rng.random() > self.keep:
            self.action = self.rng.choices(self.names, self.weights)[0]
        return self.action

    @staticmethod
    def _stuck(poses: list[tuple[float, float, float] | None], steps: int = 4, mm: float = 30) -> bool:
        """Forward for `steps` steps but moved less than `mm`: probably blocked."""
        last = [p for p in poses[-steps:] if p is not None]
        if len(last) < steps:
            return False
        moved = np.hypot(last[-1][0] - last[0][0], last[-1][1] - last[0][1])
        return bool(moved < mm)


class FrameStore:
    """on_frame callback: save every frame (downscaled) with its arrival time."""

    def __init__(self, out: Path) -> None:
        self.dir = out / "frames"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.rows: list[tuple[int, float, int, str]] = []

    def __call__(self, frame: Any, t_mono: float) -> None:
        h = round(frame.height * RAW_WIDTH / frame.width)
        name = f"{len(self.rows):06d}.jpg"
        frame.reformat(width=RAW_WIDTH, height=h).to_image().save(self.dir / name, quality=90)
        self.rows.append((len(self.rows), t_mono, frame.pts or 0, f"frames/{name}"))


async def record(args: argparse.Namespace) -> Path:
    email, country = email_and_country()
    out = OUT_DIR / f"{datetime.now():%Y%m%d-%H%M%S}"
    store = FrameStore(out)
    rng = random.Random(args.seed)
    policy = RandomPolicy(
        rng,
        keep=args.keep,
        weights={"forward": 0.55, "turnLeft": 0.2, "turnRight": 0.2, "stop": 0.05},
    )
    steps: list[dict[str, Any]] = []
    poses: list[tuple[float, float, float] | None] = []

    async with aiohttp.ClientSession() as http:
        auth = await login(http, email, country)
        credentials = await auth.authenticate()
        devices = await ApiClient(auth).get_devices()
        if not devices.mqtt:
            raise SystemExit("No MQTT-supported device on this account.")
        info = devices.mqtt[0]
        robot = dict(info.api)
        device = Device(info, auth)  # commands + getPos over REST

        async def send(act: str) -> bool:
            resp = await device.execute_command(CustomCommand("move", move_args(act, 90)))
            return resp.get("resp", {}).get("body", resp).get("code") == 0

        async def poll_pose() -> tuple[tuple[Pose, bool] | None, float]:
            pose = raw_pose(await device.execute_command(GetPos()))
            return pose, time.monotonic()

        # The policy's unstick rule and the episode's pose both need a trusted pose.
        pose, _ = await poll_pose()
        if not (pose and pose[1]):
            if args.allow_invalid_pose:
                print("WARNING: pose invalid; recording without pose and without unstick.")
            elif not args.relocate:
                raise SystemExit(
                    "The robot doesn't know where it is (getPos says invalid), so poses "
                    "would be empty and the unstick rule can't work.\n"
                    "  --relocate            let it spin in place to find itself first\n"
                    "  --allow-invalid-pose  record anyway (no pose, no unstick)"
                )
            else:
                print("Pose invalid: asking the robot to relocalize (it spins in place)...")
                await device.execute_command(SetRelocationState())
                for _ in range(45):
                    await asyncio.sleep(2)
                    pose, _ = await poll_pose()
                    if pose and pose[1]:
                        print(f"Relocalized at {pose[0]}.")
                        break
                else:
                    raise SystemExit("Still no valid pose after 90 s; giving up.")

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
        live = LiveViewSession(http, session_info=session_info, on_frame=store)
        try:
            if not await live.start(timeout=30):
                raise SystemExit(f"No video (status={live.status}, error={live.error}).")
            await asyncio.sleep(2)  # let the stream settle (first frames are garbage)
            n_steps = int(args.seconds * args.hz)
            dt = 1 / args.hz
            print(f"Recording {n_steps} steps at {args.hz:g} Hz. Ctrl-C to stop early.")
            t0 = time.monotonic()
            last_sent = "stop"
            for i in range(n_steps):
                t_step = t0 + i * dt
                await asyncio.sleep(max(0.0, t_step - time.monotonic()))
                if live.stop_event.is_set():
                    print(f"Video stopped at step {i} ({live.status}); ending episode.")
                    break
                pose_task = asyncio.create_task(poll_pose())
                act = policy(poses)
                row: dict[str, Any] = {"step": i, "t_step": t_step, "action": act,
                                       "sent": False, "t_send": "", "ok": ""}
                if act != last_sent:  # turns/forward keep going until the next command
                    row["t_send"] = time.monotonic()
                    row["ok"] = await send(act)
                    row["sent"] = True
                    last_sent = act
                pose, t_pose = await pose_task
                xya, valid = pose if pose else (("", "", ""), False)
                poses.append(xya if valid else None)  # policy only sees trusted poses
                row.update(t_pose=t_pose, x=xya[0], y=xya[1], a=xya[2], valid=valid)
                steps.append(row)
                if i % 10 == 0:
                    shown = xya if valid else "INVALID"
                    print(f"  step {i:4d}  {act:<10} pose={shown}  frames={len(store.rows)}")
        except asyncio.CancelledError:  # Ctrl-C: asyncio.run cancels this task
            print("Interrupted; stopping the robot and saving what we have.")
            asyncio.current_task().uncancel()  # finish normally so the episode is built
        finally:
            await send("stop")
            await live.stop()

    write_raw(out, store, steps, args)
    return out


def write_raw(out: Path, store: FrameStore, steps: list[dict[str, Any]], args: argparse.Namespace) -> None:
    with (out / "frames.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "t_mono", "pts", "file"])
        w.writerows((i, f"{t:.4f}", p, name) for i, t, p, name in store.rows)
    fields = ["step", "t_step", "action", "sent", "t_send", "ok", "t_pose", "x", "y", "a", "valid"]
    with (out / "steps.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(steps)
    meta = {"hz": args.hz, "seconds": args.seconds, "seed": args.seed, "keep": args.keep,
            "actions": ACTIONS, "raw_width": RAW_WIDTH, "n_steps": len(steps),
            "n_frames": len(store.rows)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))


def build_episode(out: Path, shift: float, size: tuple[int, int]) -> dict[str, np.ndarray]:
    """Align raw frames to steps and write episode.npz (re-runnable from raw)."""
    with (out / "frames.csv").open() as f:
        frames = [(float(r["t_mono"]), r["file"]) for r in csv.DictReader(f)]
    with (out / "steps.csv").open() as f:
        steps = list(csv.DictReader(f))
    if not frames or not steps:
        raise SystemExit(f"Nothing to build in {out} (frames={len(frames)}, steps={len(steps)})")
    times = np.array([t for t, _ in frames])

    imgs, actions, pose, valid, t, gaps = [], [], [], [], [], []
    for s in steps:
        target = float(s["t_step"]) + shift
        j = int(np.searchsorted(times, target, side="right")) - 1  # newest frame <= target
        if j < 0:
            continue  # no frame yet for this step
        gaps.append(target - times[j])
        img = Image.open(out / frames[j][1]).convert("RGB").resize(size, Image.BILINEAR)
        imgs.append(np.asarray(img))
        actions.append(ACTIONS.index(s["action"]))
        # Old recordings have no "valid" column: a pose there is valid if present.
        ok = bool(s["x"]) and s.get("valid", "True") == "True"
        pose.append((float(s["x"]), float(s["y"]), float(s["a"])) if ok else (np.nan,) * 3)
        valid.append(ok)
        t.append(float(s["t_step"]))
    if not imgs:
        raise SystemExit(f"No step in {out} has a frame at t + shift; nothing to build.")

    ep = {
        "frames": np.stack(imgs),
        "actions": np.array(actions, dtype=np.int64),
        "pose": np.array(pose, dtype=np.float32),  # NaN where pose_valid is False
        "pose_valid": np.array(valid, dtype=bool),
        "t": np.array(t, dtype=np.float64),
        "action_names": np.array(ACTIONS),
        "shift": np.array(shift),
    }
    np.savez_compressed(out / "episode.npz", **ep)
    counts = {name: int((ep["actions"] == k).sum()) for k, name in enumerate(ACTIONS)}
    print(f"\nepisode.npz: {len(t)} steps, frames {ep['frames'].shape}, actions {counts}, "
          f"valid poses {int(ep['pose_valid'].sum())}/{len(t)}")
    print(f"frame age at t+shift: median {np.median(gaps) * 1000:.0f} ms, "
          f"max {np.max(gaps) * 1000:.0f} ms (should stay well under one step)")
    return ep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--hz", type=float, default=2.0, help="policy / episode step rate")
    parser.add_argument("--keep", type=float, default=0.75, help="prob. to repeat last action")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--shift", type=float, default=0.42, help="video latency to undo (s)")
    parser.add_argument("--size", type=int, nargs=2, default=[256, 144], metavar=("W", "H"))
    parser.add_argument("--rebuild", type=Path, metavar="DIR", help="only rebuild episode.npz")
    parser.add_argument("--relocate", action="store_true",
                        help="if the pose is invalid, let the robot spin to relocalize first")
    parser.add_argument("--allow-invalid-pose", action="store_true",
                        help="record even without a valid pose (no unstick rule)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    out = args.rebuild or asyncio.run(record(args))
    build_episode(out, args.shift, tuple(args.size))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
