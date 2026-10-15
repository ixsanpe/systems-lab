"""Connection settings and topic names shared by examples and exercises."""

import os

BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "localhost:29092,localhost:39092,localhost:49092")

# Topic name -> (partitions, extra topic config). Replication factor is the cluster default (3).
TOPICS: dict[str, tuple[int, dict[str, str]]] = {
    # Every status change of every flight, keyed by flight_id.
    "flight-status": (4, {}),
    # Same events, keyed by departure airport -> used to demonstrate key skew.
    "flight-status-by-airport": (4, {}),
    # Baggage scans (check-in, sorter, loaded, arrived), keyed by bag tag.
    "bag-scans": (6, {"retention.ms": str(7 * 24 * 3600 * 1000)}),
    # Seat bookings / payments, keyed by booking reference (PNR).
    "bookings": (3, {}),
    "booking-confirmations": (3, {}),
    # Latest status per flight. Compacted: Kafka keeps only the newest value per key.
    "flight-latest": (
        2,
        {
            "cleanup.policy": "compact",
            "segment.ms": "10000",
            "min.cleanable.dirty.ratio": "0.01",
            "delete.retention.ms": "10000",
        },
    ),
    # Passenger contact details: personal data, keep it short (GDPR).
    "passenger-contacts": (3, {"retention.ms": str(60 * 60 * 1000)}),
    # Derived: delays detected from flight-status, with lineage headers.
    "flight-delays": (4, {}),
    "passenger-notifications": (4, {}),
}


def producer_config(**overrides: str) -> dict:
    return {"bootstrap.servers": BOOTSTRAP, **overrides}


def consumer_config(group_id: str, **overrides: str) -> dict:
    return {
        "bootstrap.servers": BOOTSTRAP,
        "group.id": group_id,
        "auto.offset.reset": "earliest",
        **overrides,
    }
