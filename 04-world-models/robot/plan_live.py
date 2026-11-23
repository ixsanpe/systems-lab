"""Step 4b: Rocky plans with the world model, live.

    uv run robot/plan_live.py --dry-run                 # camera + planner, robot never moves
    uv run robot/plan_live.py --trials 4                # turn-away trials (default)
    uv run robot/plan_live.py --trials 4 --kind back    # pushed-back trials
    uv run robot/plan_live.py --trials 6 --kind mixed --degrees 60 120

One trial:
  1. Goal: Rocky stands still, the current camera view becomes the goal image
     (its DINOv2 features) and the current pose is noted.
  2. Displace: we move it away with plain commands, without the model:
       turn  -> turn left or right (alternating) by --degrees (sampled per trial)
       back  -> drive backward for --back-seconds
  3. Plan: every 0.5 s, the planner imagines all action sequences over
     {stop, forward, turnLeft, turnRight} with the world model and sends the first
     action of the one whose imagined future looks most like the goal (MPC).
     It ends when the planner chooses stop --done-stops times in a row, or after
     --max-steps.
  4. Score with the pose (the model never sees it): heading and position error
     to the goal pose, before and after.

Timing, as in recording: frame o_i is the newest frame by t_i + 0.42 s. At step
boundary t_i the newest such frame is o_{i-1}, and a_{i-1} is still being
executed, so the planner imagines a_{i-1} first (prefix) and then plans.

Everything is saved to data/plan_runs/<time>/ (gitignored): every camera frame,
steps.csv (per step: trial, phase, action, distance to goal, pose), goals.npz
(goal frames), trials.json (per-trial scores). PRIVACY: the frames show your home.

Needs: robot charged, off the dock, pose valid, live view NOT open in the app
(one viewer at a time), ~1 m free around it (and behind it for --kind back).
Ctrl-C stops the robot.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
import torch
from camera import encoded_pin
from connect import _device_id, email_and_country
from deebot_client.api_client import ApiClient
from deebot_client.authentication import create_rest_config
from deebot_client.commands.json.custom import CustomCommand
from deebot_client.commands.json.pos import GetPos
from deebot_client.device import Device
from live_view import LiveViewSession, async_send_app_ping, async_start_watch
from PIL import Image
from probe import login, move_args
from record import FrameStore, raw_pose

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from wm.data import ACTIONS  # noqa: E402
from wm.encode import encode, load_encoder  # noqa: E402
from wm.plan import Planner, load_model  # noqa: E402
from wm.predictor import CONTEXT, device, pool_patches  # noqa: E402

OUT_DIR = ROOT / "data" / "plan_runs"
SHIFT = 0.42  # video latency (s), same as the episodes
SIZE = (256, 144)  # episode frame size
TURN_DEG_PER_S = 53.0  # measured turn rate (probe.py)


def wrap(deg: float) -> float:
    return (deg + 180) % 360 - 180


class Frames:
    """on_frame: save every frame to disk (FrameStore) and keep recent small copies."""

    def __init__(self, out: Path) -> None:
        self.store = FrameStore(out)
        self.recent: list[tuple[float, np.ndarray]] = []

    def __call__(self, frame: Any, t_mono: float) -> None:
        self.store(frame, t_mono)
        img = frame.reformat(width=640, height=360).to_image().resize(SIZE, Image.BILINEAR)
        self.recent.append((t_mono, np.asarray(img)))
        del self.recent[:-40]

    def at(self, t: float) -> np.ndarray | None:
        """Newest frame that arrived by time t (like build_episode)."""
        for tf, img in reversed(self.recent):
            if tf <= t:
                return img
        return None


class Brain:
    """Encoder + world model + planner, on the live frames."""

    def __init__(self, horizon: int) -> None:
        self.dev = device()
        self.encoder = load_encoder(self.dev)
        self.model, self.pool = load_model(self.dev)
        self.planner = Planner(self.model, horizon, self.dev)

    def features(self, img: np.ndarray) -> torch.Tensor:
        patches, _ = encode(self.encoder, img[None], self.dev)
        return pool_patches(patches, self.pool)[0].to(self.dev)  # [P, D] float16

    @staticmethod
    def distance(z: torch.Tensor, goal: torch.Tensor) -> float:
        return float(((z.float() - goal.float()) ** 2).mean())


async def run(args: argparse.Namespace) -> Path:
    email, country = email_and_country()
    out = OUT_DIR / f"{datetime.now():%Y%m%d-%H%M%S}"
    rng = random.Random(args.seed)
    print("Loading DINOv2 + world model...")
    brain = Brain(args.horizon)
    rows: list[dict[str, Any]] = []
    trials: list[dict[str, Any]] = []
    goals: list[np.ndarray] = []

    async with aiohttp.ClientSession() as http:
        auth = await login(http, email, country)
        credentials = await auth.authenticate()
        devices = await ApiClient(auth).get_devices()
        info = devices.mqtt[0]
        robot = dict(info.api)
        dev = Device(info, auth)

        async def send(act: str) -> None:
            if args.dry_run:
                return
            await dev.execute_command(CustomCommand("move", move_args(act, 90)))

        async def pose() -> tuple[float, float, float] | None:
            p = raw_pose(await dev.execute_command(GetPos()))
            return p[0] if p and p[1] else None

        if await pose() is None and not args.dry_run:
            raise SystemExit("Pose invalid (the robot doesn't know where it is): send it to the "
                             "dock from the app, drive it off with the arrows, then retry.")

        frames = Frames(out)
        session_info = await async_start_watch(
            http, rest_config=create_rest_config(http, device_id=_device_id(), alpha_2_country=country),
            login_client_id=_device_id(), credentials=credentials, country=country, robot=robot,
            encoded_pin=encoded_pin(robot))
        await async_send_app_ping(http, credentials=credentials, robot=robot)
        live = LiveViewSession(http, session_info=session_info, on_frame=frames)
        try:
            if not await live.start(timeout=30):
                raise SystemExit(f"No video (status={live.status}, error={live.error}).")
            await asyncio.sleep(2.5)
            kinds = (["turn", "back"] if args.kind == "mixed" else [args.kind]) * args.trials
            for n in range(args.trials):
                kind = kinds[n]
                trial = await one_trial(n, kind, args, rng, brain, frames, send, pose, rows, goals)
                trials.append(trial)
                if live.stop_event.is_set():
                    print("Video stopped; ending.")
                    break
        except asyncio.CancelledError:
            print("Interrupted; stopping the robot and saving.")
            asyncio.current_task().uncancel()
        finally:
            await send("stop")
            await live.stop()

    save(out, frames, rows, trials, goals, args)
    return out


async def one_trial(n, kind, args, rng, brain: Brain, frames: Frames, send, pose, rows, goals):
    dt = 0.5
    print(f"\n=== trial {n + 1}/{args.trials}: {kind}")
    await send("stop")
    await asyncio.sleep(1.5)  # still, so the goal view is sharp and settled
    goal_img = frames.at(time.monotonic())
    goal = brain.features(goal_img)
    goals.append(goal_img)
    goal_pose = await pose()

    # Displace without the model.
    if kind == "turn":
        deg = rng.uniform(*args.degrees)
        act = "turnLeft" if n % 2 == 0 else "turnRight"
        print(f"  displace: {act} ~{deg:.0f} deg")
        await send(act)
        await asyncio.sleep(deg / TURN_DEG_PER_S)
        displacement = f"{act} {deg:.0f} deg"
    else:
        print(f"  displace: backward {args.back_seconds:g} s")
        await send("backward")
        await asyncio.sleep(args.back_seconds)
        displacement = f"backward {args.back_seconds:g} s"
    await send("stop")
    await asyncio.sleep(1.0)
    start_pose = await pose()

    # Plan: CONTEXT stop steps first (to fill the history), then MPC.
    zs: list[torch.Tensor] = []
    acts: list[str] = []
    t0 = time.monotonic()
    stops, i, last_sent = 0, 0, "stop"
    d_start = None
    while i < args.max_steps + CONTEXT:
        t_i = t0 + i * dt
        await asyncio.sleep(max(0.0, t_i - time.monotonic()))
        p_task = asyncio.create_task(pose())
        if i >= 1:  # o_{i-1}: newest frame by t_{i-1} + shift
            zs.append(brain.features(frames.at(t0 + (i - 1) * dt + SHIFT)))
        d = brain.distance(zs[-1], goal) if zs else float("nan")
        if i < CONTEXT:
            act, best_cost = "stop", float("nan")
        else:
            z_ctx = torch.stack(zs[-CONTEXT:])
            a_ctx = torch.tensor([ACTIONS.index(a) for a in acts[-CONTEXT:-1]], device=brain.dev)
            plan, costs = brain.planner.plan(z_ctx, a_ctx, goal, prefix=(ACTIONS.index(acts[-1]),))
            act, best_cost = plan[0], float(costs.min())
            d_start = d if d_start is None else d_start
            stops = stops + 1 if act == "stop" else 0
        if act != last_sent:
            await send(act)
            last_sent = act
        acts.append(act)
        p = await p_task
        rows.append({"trial": n, "kind": kind, "step": i, "t": f"{t_i:.3f}", "planning": i >= CONTEXT,
                     "action": act, "dist_to_goal": f"{d:.4f}", "best_cost": f"{best_cost:.4f}",
                     "x": p[0] if p else "", "y": p[1] if p else "", "a": p[2] if p else ""})
        if i >= CONTEXT:
            print(f"  step {i - CONTEXT:3d}  {act:<10} dist {d:.3f}  plan cost {best_cost:.3f}")
        if i >= CONTEXT and stops >= args.done_stops:
            break
        i += 1
    await send("stop")
    await asyncio.sleep(1.0)
    end_pose = await pose()
    z_end = brain.features(frames.at(time.monotonic()))
    d_end = brain.distance(z_end, goal)

    def errors(p):
        if p is None or goal_pose is None:
            return None, None
        return abs(wrap(p[2] - goal_pose[2])), float(np.hypot(p[0] - goal_pose[0], p[1] - goal_pose[1]))

    h0, m0 = errors(start_pose)
    h1, m1 = errors(end_pose)
    trial = {"trial": n, "kind": kind, "displacement": displacement,
             "steps": i - CONTEXT + 1, "goal_pose": goal_pose, "start_pose": start_pose,
             "end_pose": end_pose, "heading_err_start": h0, "heading_err_end": h1,
             "pos_err_start_mm": m0, "pos_err_end_mm": m1,
             "feature_dist_start": d_start, "feature_dist_end": d_end}
    fmt = lambda v, u: "?" if v is None else f"{v:.0f}{u}"  # noqa: E731
    print(f"  result: heading error {fmt(h0, '°')} -> {fmt(h1, '°')}, position error "
          f"{fmt(m0, ' mm')} -> {fmt(m1, ' mm')}, feature distance {d_start:.3f} -> {d_end:.3f}")
    return trial


def save(out: Path, frames: Frames, rows, trials, goals, args) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / "frames.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "t_mono", "pts", "file"])
        w.writerows((i, f"{t:.4f}", p, name) for i, t, p, name in frames.store.rows)
    if rows:
        with (out / "steps.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    if goals:
        np.savez_compressed(out / "goals.npz", frames=np.stack(goals))
    (out / "trials.json").write_text(json.dumps(trials, indent=1))
    (out / "meta.json").write_text(json.dumps(vars(args), indent=1, default=str))
    print(f"\nSaved {out}")
    for t in trials:
        if t["heading_err_end"] is not None:
            print(f"  trial {t['trial'] + 1} {t['displacement']:<22} heading {t['heading_err_start']:.0f}° -> "
                  f"{t['heading_err_end']:.0f}°, position {t['pos_err_start_mm']:.0f} -> "
                  f"{t['pos_err_end_mm']:.0f} mm, {t['steps']} steps")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--kind", choices=["turn", "back", "mixed"], default="turn")
    parser.add_argument("--degrees", type=float, nargs=2, default=[45, 90], metavar=("MIN", "MAX"))
    parser.add_argument("--back-seconds", type=float, default=1.5)
    parser.add_argument("--horizon", type=int, default=2, help="planned steps after the prefix")
    parser.add_argument("--max-steps", type=int, default=30, help="planning steps per trial (0.5 s)")
    parser.add_argument("--done-stops", type=int, default=3, help="stop after this many stop picks")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="never send move commands")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
