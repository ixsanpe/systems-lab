"""Exercise 8 - A stream processor with headers: the delay detector.

Read `flight-status`, and whenever a flight reaches delay_minutes >= 30
for the first time, emit a FlightDelay event to `flight-delays` (key=flight_id).

Use Kafka headers for metadata that is NOT part of the business payload:
  - lineage:  source-topic, source-partition, source-offset
  - producer-app: "delay-detector" and a version
  - routing:  notify-channel = "sms" if delay >= 45 else "app-push"
  - (think) encryption-key-id: which key encrypted the payload?

Run (after `uv run python -m examples.producer --flights 200`):
    uv run python -m exercises.ex08_delay_detector detect   # flight-status -> flight-delays
    uv run python -m exercises.ex08_delay_detector notify   # route by header
    uv run python -m exercises.ex08_delay_detector trace    # follow lineage back to the source

Questions:
1. `detect()`: Keep "already notified" state in a dict. What happens to
   that state when the process restarts or a partition moves to another
   instance after a rebalance? Because the input is keyed by flight_id, why is
   per-partition local state safe here?
2. `just tail flight-delays` - check the headers.
3. `notify()` (group "passenger-notifier") routes ONLY by the notify-channel
   header, without deserialising the value. Why is that useful?
4. `trace()` takes one flight-delays record and uses the lineage headers to fetch
   the exact source flight-status record (assign to that partition + offset).
"""

import argparse
import json
import uuid
from collections import Counter
from datetime import UTC, datetime

from confluent_kafka import Consumer, KafkaException, Producer, TopicPartition

from airline.config import consumer_config, producer_config
from airline.events import FlightStatusEvent

IN_TOPIC = "flight-status"
OUT_TOPIC = "flight-delays"
THRESHOLD_MIN = 30
SMS_MIN = 45  # long delays get an SMS, shorter ones an app push


def _drain(consumer: Consumer, handle, idle_s: int = 5) -> None:
    """Poll until the topic has been quiet for idle_s seconds."""
    idle = 0
    while idle < idle_s:
        msg = consumer.poll(1.0)
        if msg is None:
            idle += 1
            continue
        if msg.error():
            raise KafkaException(msg.error())
        idle = 0
        handle(msg)


def detect() -> None:
    consumer = Consumer(consumer_config("delay-detector"))
    producer = Producer(producer_config(**{"enable.idempotence": "true"}))
    consumer.subscribe([IN_TOPIC])
    # Local state. Safe because input is keyed by flight_id: one flight's events always
    # come from the same partition, so only one instance ever needs to know about it.
    # (Lost on restart: a real service would rebuild it, e.g. from a compacted topic.)
    notified: set[str] = set()
    stats = Counter()

    def handle(msg):
        ev = FlightStatusEvent.from_json(msg.value())
        stats["read"] += 1
        if ev.delay_minutes < THRESHOLD_MIN or ev.flight_id in notified:
            return
        notified.add(ev.flight_id)
        delay = {
            "flight_id": ev.flight_id,
            "origin": ev.origin,
            "destination": ev.destination,
            "delay_minutes": ev.delay_minutes,
            "detected_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        headers = [
            ("source-topic", msg.topic().encode()),
            ("source-partition", str(msg.partition()).encode()),
            ("source-offset", str(msg.offset()).encode()),
            ("producer-app", b"delay-detector/1.0"),
            ("notify-channel", b"sms" if ev.delay_minutes >= SMS_MIN else b"app-push"),
        ]
        producer.produce(
            OUT_TOPIC, key=msg.key(), value=json.dumps(delay).encode(), headers=headers
        )
        producer.poll(0)
        stats["delays"] += 1

    _drain(consumer, handle)
    producer.flush(10)
    consumer.close()
    print(f"read {stats['read']} status events, emitted {stats['delays']} delays")


def notify() -> None:
    consumer = Consumer(consumer_config("passenger-notifier"))
    consumer.subscribe([OUT_TOPIC])
    channels = Counter()

    def handle(msg):
        # Route on the header only: no need to know (or be able to decode) the payload.
        headers = dict(msg.headers() or [])
        channel = headers.get("notify-channel", b"unknown").decode()
        channels[channel] += 1
        print(f"-> {channel:8s} {msg.key().decode()}")

    _drain(consumer, handle)
    consumer.close()
    print(f"routed: {dict(channels)}")


def trace() -> None:
    """Take one delay and follow its lineage headers back to the exact source record."""
    reader = Consumer(consumer_config(f"trace-{uuid.uuid4()}"))
    reader.subscribe([OUT_TOPIC])
    msg = None
    while msg is None or msg.error():
        msg = reader.poll(5.0)
    reader.close()
    h = {k: v.decode() for k, v in msg.headers()}
    print(
        f"delay {msg.key().decode()} came from {h['source-topic']}"
        f"[{h['source-partition']}] @ {h['source-offset']}"
    )

    source = Consumer(consumer_config(f"trace-{uuid.uuid4()}"))
    source.assign(
        [TopicPartition(h["source-topic"], int(h["source-partition"]), int(h["source-offset"]))]
    )
    original = source.poll(10.0)
    source.close()
    print(f"source record: {original.value().decode()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["detect", "notify", "trace"])
    {"detect": detect, "notify": notify, "trace": trace}[ap.parse_args().cmd]()
