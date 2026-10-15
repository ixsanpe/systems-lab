"""Synthetic airline event generators.

Domain cheat sheet
------------------
- flight_id: carrier + number + date, e.g. "KA318-2026-10-11". Unique per operated flight.
- A flight moves through a lifecycle; ORDER MATTERS per flight:
      SCHEDULED -> (DELAYED)* -> BOARDING -> DEPARTED -> LANDED
  or  SCHEDULED -> CANCELLED
- Most flights of a hub carrier depart from the hub (here ZRH). That is the
  "centre of London" problem from the course: a key like the departure
  airport is heavily skewed.
"""

from __future__ import annotations

import json
import random
import string
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

CARRIER = "KA"  # "Kafka Airways", fictional
HUB = "ZRH"
DESTINATIONS = [
    "LHR",
    "CDG",
    "FRA",
    "AMS",
    "MAD",
    "BCN",
    "FCO",
    "VIE",
    "CPH",
    "ARN",
    "JFK",
    "ORD",
    "BOS",
    "SFO",
    "YUL",
    "NRT",
    "SIN",
    "BKK",
    "DXB",
    "GVA",
]
LIFECYCLE = ["SCHEDULED", "BOARDING", "DEPARTED", "LANDED"]
BAG_CHECKPOINTS = ["CHECKED_IN", "SORTER", "LOADED", "ARRIVED", "CLAIMED"]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class FlightStatusEvent:
    flight_id: str
    origin: str
    destination: str
    status: str
    seq: int  # per-flight sequence number: lets consumers detect reordering
    delay_minutes: int = 0
    gate: str | None = None
    event_time: str = field(default_factory=_now)

    def to_json(self) -> bytes:
        return json.dumps(asdict(self)).encode()

    @staticmethod
    def from_json(raw: bytes) -> FlightStatusEvent:
        return FlightStatusEvent(**json.loads(raw))


def make_flights(n: int, hub_share: float = 0.7, seed: int = 42) -> list[tuple[str, str, str]]:
    """Return n (flight_id, origin, destination). `hub_share` of them depart from HUB."""
    rng = random.Random(seed)
    day = datetime.now(UTC).date().isoformat()
    flights = []
    for i in range(n):
        number = 100 + i
        dest = rng.choice(DESTINATIONS)
        if rng.random() < hub_share:
            origin, destination = HUB, dest
        else:
            origin, destination = dest, HUB
        flights.append((f"{CARRIER}{number}-{day}", origin, destination))
    return flights


def flight_lifecycles(n_flights: int, seed: int = 42) -> list[FlightStatusEvent]:
    """Events for n flights, interleaved across flights but ordered within each flight."""
    rng = random.Random(seed)
    per_flight: list[list[FlightStatusEvent]] = []
    for flight_id, origin, dest in make_flights(n_flights, seed=seed):
        gate = f"{rng.choice('ABDE')}{rng.randint(1, 60)}"
        steps: list[tuple[str, int]] = [("SCHEDULED", 0)]
        if rng.random() < 0.05:
            steps.append(("CANCELLED", 0))
        else:
            delay = 0
            while rng.random() < 0.3:
                delay += rng.choice([10, 15, 30, 45])
                steps.append(("DELAYED", delay))
            steps += [(status, delay) for status in LIFECYCLE[1:]]
        per_flight.append(
            [
                FlightStatusEvent(flight_id, origin, dest, status, seq, delay, gate)
                for seq, (status, delay) in enumerate(steps)
            ]
        )

    # Interleave: repeatedly pick a random flight and emit its next event.
    out: list[FlightStatusEvent] = []
    queues = [list(reversed(e)) for e in per_flight]
    while queues:
        q = rng.choice(queues)
        out.append(q.pop())
        if not q:
            queues.remove(q)
    return out


@dataclass
class BagScan:
    bag_tag: str  # 10-digit IATA license plate
    flight_id: str
    checkpoint: str
    station: str
    event_time: str = field(default_factory=_now)

    def to_json(self) -> bytes:
        return json.dumps(asdict(self)).encode()


def bag_scans(n_bags: int, seed: int = 7) -> list[BagScan]:
    rng = random.Random(seed)
    flights = make_flights(20, seed=seed)
    scans = []
    for _ in range(n_bags):
        tag = "0724" + "".join(rng.choices(string.digits, k=6))
        flight_id, origin, dest = rng.choice(flights)
        for cp in BAG_CHECKPOINTS:
            station = origin if cp in ("CHECKED_IN", "SORTER", "LOADED") else dest
            scans.append(BagScan(tag, flight_id, cp, station))
    return scans


@dataclass
class Booking:
    pnr: str  # 6-char booking reference
    flight_id: str
    seat: str
    passenger: str
    amount_chf: float

    def to_json(self) -> bytes:
        return json.dumps(asdict(self)).encode()


def bookings(n: int, seed: int = 3) -> list[Booking]:
    rng = random.Random(seed)
    flights = make_flights(10, seed=seed)
    out = []
    for _ in range(n):
        pnr = "".join(rng.choices(string.ascii_uppercase + string.digits, k=6))
        seat = f"{rng.randint(1, 40)}{rng.choice('ABCDEF')}"
        out.append(
            Booking(
                pnr,
                rng.choice(flights)[0],
                seat,
                f"PAX-{rng.randint(1000, 9999)}",
                round(rng.uniform(89, 1450), 2),
            )
        )
    return out
