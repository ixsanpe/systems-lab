"""Step 0b: can we drive the T90 from code, and what does the robot report back?

Three modes:

    # Read-only: connect, print the robot's pose, exit. Safe on the dock.
    uv run robot/probe.py

    # Moves: for each act, pose -> move <act> -> wait --hold s -> stop -> pose.
    uv run robot/probe.py --move forward
    uv run robot/probe.py --move turnLeft turnRight --hold 3

    # Listen: send nothing, record every message sent to/from the robot for 30 s.
    # Press the app's arrow buttons meanwhile to see what the app sends.
    uv run robot/probe.py --listen 30

--move takes any string, so you can guess act values. A wrong one is rejected by
the robot (code 20004 "unkown act type") before anything moves. A summary table
at the end shows, per act, whether it was accepted and how the pose changed.

What the T90 app sends (firmware 1.103.0, captured with --listen):
    {"act": "forward",   "a": 0}      {"act": "backward", "a": 0}
    {"act": "turnLeft",  "a": 90}     {"act": "turnRight", "a": 90}
    {"act": "stop"}                   (sent when the arrow is released)
Without `a`, turns are rejected with 20011 "format error", but its value does
nothing: a turn keeps going at ~54 deg/s until stop (1 s -> 55 deg, 2 s -> 112 deg,
even with a=30). So --hold sets the turn angle. Left = heading increases.
SpinLeft/SpinRight/left/right (older models' names) are rejected with 20004.

Only the send/ack/pose lines are printed in --move mode; the full MQTT traffic is
printed in --listen mode. Both always go into the log file.

Every command and every position the robot pushes is timestamped and written to
data/probe/<time>.json (gitignored), so we can measure latency afterwards.

Login reuses the token, email and country cached in .secrets/ by connect.py (or
the first probe run). The password is only asked for if the token stops working.

SAFETY: before --move, take the robot off the dock into open floor and keep the
app open to stop it by hand. The script always sends stop on the way out.
"""

import argparse
import asyncio
import getpass
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp
from connect import (
    ACCOUNT_FILE,
    _device_id,
    _load_account,
    _save_account,
    email_and_country,
)
from deebot_client import mqtt_client as mqtt_module
from deebot_client.api_client import ApiClient
from deebot_client.authentication import Authenticator, create_rest_config
from deebot_client.commands.json.custom import CustomCommand
from deebot_client.commands.json.pos import GetPos
from deebot_client.device import Device
from deebot_client.events import CustomCommandEvent
from deebot_client.events.map import PositionsEvent
from deebot_client.exceptions import AuthenticationError
from deebot_client.mqtt_client import MqttClient, create_mqtt_config
from deebot_client.rs.map import PositionType
from deebot_client.util import md5

LOG_DIR = Path(__file__).resolve().parent.parent / "data" / "probe"


def pose_from_reply(response: dict[str, Any]) -> tuple[int, int, int] | None:
    """Robot pose (x, y, heading) from a getPos reply, or None if it has none."""
    data = response.get("resp", {}).get("body", {}).get("data", {})
    pos = data.get("deebotPos")
    if isinstance(pos, list):
        pos = pos[0] if pos else None
    if not pos or pos.get("invalid", 0):
        return None
    return pos["x"], pos["y"], pos.get("a", 0)


def move_args(act: str, angle: int) -> dict[str, Any]:
    """Move payload in the same shape the ECOVACS app sends."""
    if act == "stop":
        return {"act": "stop"}
    return {"act": act, "a": angle if act.startswith("turn") else 0}


def parse_mqtt(topic: str, payload: bytes) -> dict[str, Any]:
    """Split an ECOVACS MQTT message into the parts we care about (no device ids).

    iot/atr/<cmd>/<did>/<class>/<res>/<type>                       robot -> cloud event
    iot/p2p/<cmd>/<from did>/<from class>/<from res>/<to did>/<to class>/<to res>/
            <q|p>/<request id>/<type>                              q = request, p = reply
    """
    parts = topic.split("/")
    if len(parts) >= 12 and parts[1] == "p2p":
        # The header's "m" is the reliable marker: app commands arrive on topics
        # whose q/p level says "p", but their header says "m": "request".
        header = {}
        try:
            header = json.loads(payload).get("header", {})
        except (ValueError, AttributeError):
            pass
        is_request = header.get("m") == "request" or (
            "m" not in header and parts[9] == "q"
        )
        kind = "request" if is_request else "reply"
        info = {"kind": kind, "cmd": parts[2], "from_class": parts[4], "req": parts[10]}
    else:
        info = {"kind": "event", "cmd": parts[2] if len(parts) > 2 else topic}
    # Last topic level is the payload format: j = JSON, x = XML, p = protobuf (binary).
    info["fmt"] = parts[-1]
    try:
        info["payload"] = json.loads(payload)
    except ValueError:
        try:
            info["payload"] = payload.decode()
        except UnicodeDecodeError:
            info["payload"] = "hex:" + payload.hex()  # binary, e.g. protobuf
    return info


def wide_topics(device_info: Any) -> list[str]:
    """deebot-client's topics, plus the same ones in any format and either direction.

    deebot-client only subscribes to JSON (`/j`) messages to the robot whose q|p level
    is `q`. The app's commands (relayed by the cloud, "ecosys") arrive with that level
    NOT set to `q`, so they never matched; the `+` there is what catches them.
    """
    api = device_info.api
    robot = f"{api['did']}/{api['class']}/{api['resource']}"
    return [
        f"iot/atr/+/{robot}/+",  # robot events, any format
        f"iot/p2p/+/+/+/+/{robot}/+/+/+",  # anything sent to the robot
        f"iot/p2p/+/{robot}/+/+/+/+/+/+",  # anything the robot sends to someone
    ]


class Recorder:
    """Timestamped log of everything we send and everything the robot says."""

    def __init__(self) -> None:
        self.t0 = time.monotonic()
        self.rows: list[dict[str, Any]] = []

    def log(self, kind: str, *, echo: bool = True, **data: Any) -> None:
        """Record a row; print it too unless echo is False (file only)."""
        t = round(time.monotonic() - self.t0, 3)
        self.rows.append({"t": t, "kind": kind, **data})
        if echo:
            print(f"[{t:7.3f}s] {kind}: {json.dumps(data, default=str)}")

    def save(self) -> Path:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        path = LOG_DIR / f"{datetime.now():%Y%m%d-%H%M%S}.json"
        path.write_text(json.dumps(self.rows, indent=2, default=str))
        return path


async def login(session: aiohttp.ClientSession, email: str, country: str) -> Authenticator:
    """Token login with the cached account; fall back to the password if it fails."""
    if not ACCOUNT_FILE.exists():
        raise SystemExit("No cached login. Run `uv run robot/connect.py` first.")

    config = create_rest_config(session, device_id=_device_id(), alpha_2_country=country)
    # The password hash is only used if token renewal fails, so start without it.
    auth = Authenticator(config, email, md5(""), account_credentials=_load_account())
    try:
        await auth.authenticate()
    except AuthenticationError:
        password = getpass.getpass("Cached token rejected. ECOVACS password: ")
        auth = Authenticator(config, email, md5(password), account_credentials=_load_account())
        await auth.authenticate()
    _save_account(auth.account_credentials)
    return auth


async def run(args: argparse.Namespace) -> None:
    email, country = email_and_country()
    rec = Recorder()

    async with aiohttp.ClientSession() as session:
        auth = await login(session, email, country)
        devices = await ApiClient(auth).get_devices()
        if not devices.mqtt:
            raise SystemExit("No MQTT-supported device on this account.")
        info = devices.mqtt[0]
        print(f"Using {info.api.get('nick') or info.api.get('deviceName')} ({info.api['class']})\n")

        if args.listen:
            # Swap in wider subscriptions before the MQTT client connects.
            mqtt_module._get_topics = wide_topics
        device = Device(info, auth)
        mqtt = MqttClient(create_mqtt_config(device_id=_device_id(), country=country), auth)

        # Tap the raw MQTT stream: every message to or from the robot, including
        # requests other clients (the phone app) send it. Private API, but tiny.
        handle_message = mqtt._handle_message
        heard: list[dict[str, Any]] = []

        def tap(message: Any) -> None:
            try:
                msg = parse_mqtt(str(message.topic), message.payload)
                heard.append(msg)
                fields = {k: v for k, v in msg.items() if k != "kind"}
                rec.log(f"mqtt {msg['kind']}", echo=bool(args.listen), **fields)
            except Exception as err:  # noqa: BLE001 - never break the library's handling
                rec.log("warning", msg=f"could not parse MQTT message: {err}")
            handle_message(message)

        mqtt._handle_message = tap

        # Poses the robot pushes on its own (e.g. while moving). deebot-client only
        # fires this when the pose changed, so it can't be used to *read* the pose.
        async def on_pos(event: PositionsEvent) -> None:
            for p in event.positions:
                if p.type == PositionType.DEEBOT:
                    rec.log("pose push", x=p.x, y=p.y, a=p.a)

        async def on_custom(event: CustomCommandEvent) -> None:
            # Same content as the "ack" line, so keep it out of the terminal.
            rec.log("reply", echo=False, name=event.name, response=event.response)

        device.events.subscribe(PositionsEvent, on_pos)
        device.events.subscribe(CustomCommandEvent, on_custom)

        async def send(command: Any, label: str) -> dict[str, Any]:
            rec.log("send", command=label)
            response = await device.execute_command(command)
            rec.log("ack", command=label, ret=response.get("resp", {}).get("body", response))
            return response

        async def read_pose() -> tuple[int, int, int] | None:
            pose = pose_from_reply(await send(GetPos(), "getPos"))
            if pose is None:
                rec.log("warning", msg="getPos returned no valid robot pose")
            else:
                rec.log("pose", x=pose[0], y=pose[1], a=pose[2])
            return pose

        async def stop() -> None:
            # A clean-stop backstop was tried here; the T90 answers it with
            # 20003 "rcp not support" when not cleaning, and move-stop always works.
            await send(CustomCommand("move", move_args("stop", 0)), "move stop")

        results = []
        try:
            # MQTT gives us the robot's own pushes (pose updates while it moves).
            await device.initialize(mqtt)
            pose = await read_pose()

            if args.listen:
                print(
                    f"\nListening {args.listen:g} s. Press the turn arrows in the app now "
                    "(one press each, a few seconds apart).\n"
                )
                await asyncio.sleep(args.listen)
                pose = await read_pose()

            for act in args.move or []:
                before = pose
                payload = move_args(act, args.angle)
                body = await send(CustomCommand("move", payload), f"move {json.dumps(payload)}")
                body = body.get("resp", {}).get("body", body)
                code = body.get("code")
                if code == 0:
                    await asyncio.sleep(args.hold)
                await stop()
                await asyncio.sleep(args.settle if code == 0 else 0.5)
                pose = await read_pose()
                results.append((act, code, body.get("msg"), before, pose))
        finally:
            if args.move:
                await stop()  # again, in case we got here through an exception
            await device.teardown()
            await mqtt.disconnect()
            print(f"\nLog saved to {rec.save()}")
            print_summary(results)
            if args.listen:
                print_heard(heard)


def print_heard(heard: list[dict[str, Any]]) -> None:
    """Requests sent to the robot by others (the app): the command formats we want."""
    requests = [m for m in heard if m["kind"] == "request"]
    if not requests:
        if heard:
            print(f"\nHeard {len(heard)} message(s) from the robot, but no requests sent to it.")
        return
    print("\nRequests the robot received (copy the format of the move ones):")
    for m in requests:
        body = m["payload"].get("body", m["payload"]) if isinstance(m["payload"], dict) else m["payload"]
        print(f"  {m['cmd']:<14} from {m['from_class']:<10} [{m['fmt']}] {json.dumps(body)}")


def print_summary(results: list[tuple[Any, ...]]) -> None:
    if not results:
        return
    print(f"\n{'act':<14}{'result':<28}{'dx':>6}{'dy':>6}{'da':>6}")
    for act, code, msg, before, after in results:
        result = "ok" if code == 0 else f"rejected {code} {msg}"
        if code == 0 and before and after:
            dx, dy = after[0] - before[0], after[1] - before[1]
            da = (after[2] - before[2] + 180) % 360 - 180  # wrap to [-180, 180)
            print(f"{act:<14}{result:<28}{dx:>6}{dy:>6}{da:>6}")
        else:
            print(f"{act:<14}{result:<28}{'-':>6}{'-':>6}{'-':>6}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--listen",
        type=float,
        metavar="SECONDS",
        help="send nothing; record all MQTT traffic to/from the robot, any format",
    )
    parser.add_argument(
        "--move", nargs="+", metavar="ACT", help="move act(s) to try in order (omit = read-only)"
    )
    parser.add_argument(
        "--angle",
        type=int,
        default=90,
        help="value of `a` for turns (required, but ignored by the T90: --hold sets the angle)",
    )
    parser.add_argument("--hold", type=float, default=1.0, help="seconds between each move and its stop")
    parser.add_argument("--settle", type=float, default=2.0, help="seconds to watch after stop")
    parser.add_argument("--debug", action="store_true", help="show deebot-client debug logs")
    args = parser.parse_args()
    # Warnings from deebot-client include the robot's reply when a command fails.
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.WARNING)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
