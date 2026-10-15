"""Exercise 6 - Idempotent producer and transactions: charge the card exactly once.

The "booking confirmer" reads `bookings`, "charges" the passenger, and writes a
confirmation to `booking-confirmations`. Duplicates here mean double charges
or two confirmation e-mails.

Part A - idempotence (duplicates caused by producer retries):
  With enable.idempotence=true the producer gets a producer id + sequence number
  per partition, and the broker drops retried duplicates. Read the course
  slides for acks=all "without / with exactly once" again and explain, in your
  own words, the failure where the broker wrote the batch but the ack was lost.

Part B - transactions (read-process-write exactly once):
Implemented in `confirm(crash_after)`:
  - consumer: group "booking-confirmer", enable.auto.commit=false,
    isolation.level=read_committed
  - producer: transactional.id="booking-confirmer-1"; init_transactions()
  - loop: poll a batch -> begin_transaction() -> produce one confirmation per
    booking (key=PNR) -> send_offsets_to_transaction(consumer.position(...),
    consumer.consumer_group_metadata()) -> commit_transaction()
  - with --crash-after N: os._exit(1) in the middle of a transaction (after
    producing, before commit).

`audit(isolation)` reads `booking-confirmations` from the beginning with
  isolation.level=read_committed or read_uncommitted, count messages and
  duplicate PNRs.

Experiments:
  0. start clean: `uv run python -m examples.admin --recreate` (ex05 also writes to `bookings`)
  1. seed (200 bookings), run confirm --crash-after 50, then confirm again.
  2. audit --isolation read_uncommitted  -> duplicates?   (aborted txn data is visible)
     audit --isolation read_committed    -> duplicates?
  3. Where does exactly-once STOP? (hint: the real payment provider is not a Kafka topic)
"""

import argparse
import json
import os
import uuid
from collections import Counter

from confluent_kafka import Consumer, KafkaException, Producer

from airline.config import consumer_config, producer_config
from airline.events import bookings

IN_TOPIC = "bookings"
OUT_TOPIC = "booking-confirmations"


def seed(n: int) -> None:
    producer = Producer(producer_config(**{"enable.idempotence": "true"}))
    for b in bookings(n):
        producer.produce(IN_TOPIC, key=b.pnr.encode(), value=b.to_json())
    producer.flush(10)
    print(f"produced {n} bookings")


def confirm(crash_after: int | None) -> None:
    consumer = Consumer(
        consumer_config(
            "booking-confirmer",
            **{"enable.auto.commit": "false", "isolation.level": "read_committed"},
        )
    )
    # A fixed transactional.id: a restarted instance with the same id fences off
    # (and aborts) whatever the crashed one left open.
    producer = Producer(producer_config(**{"transactional.id": "booking-confirmer-1"}))
    producer.init_transactions()
    consumer.subscribe([IN_TOPIC])

    processed = idle = 0
    while idle < 5:
        batch = consumer.consume(num_messages=20, timeout=1.0)
        if not batch:
            if consumer.assignment():
                idle += 1
            # else: still waiting for the crashed member's session to time out (~45s)
            continue
        idle = 0

        producer.begin_transaction()
        for msg in batch:
            if msg.error():
                raise KafkaException(msg.error())
            booking = json.loads(msg.value())
            confirmation = {
                "pnr": booking["pnr"],
                "flight_id": booking["flight_id"],
                "seat": booking["seat"],
                "charged_chf": booking["amount_chf"],
                "status": "CONFIRMED",
            }
            producer.produce(OUT_TOPIC, key=msg.key(), value=json.dumps(confirmation).encode())
            processed += 1
            if crash_after is not None and processed >= crash_after:
                producer.flush()  # records reach the log, but the transaction never commits
                print(f"CRASH mid-transaction after {processed} confirmations")
                os._exit(1)

        # The consumed offsets are committed INSIDE the same transaction as the output:
        # either both happen or neither does.
        producer.send_offsets_to_transaction(
            consumer.position(consumer.assignment()), consumer.consumer_group_metadata()
        )
        producer.commit_transaction()

    consumer.close()
    print(f"confirmed {processed} bookings this run")


def audit(isolation: str) -> None:
    consumer = Consumer(consumer_config(f"audit-{uuid.uuid4()}", **{"isolation.level": isolation}))
    consumer.subscribe([OUT_TOPIC])
    pnrs: Counter = Counter()
    idle = 0
    while idle < 5:
        msg = consumer.poll(1.0)
        if msg is None:
            idle += 1
            continue
        if msg.error():
            raise KafkaException(msg.error())
        idle = 0
        pnrs[msg.key().decode()] += 1
    consumer.close()
    dupes = {k: v for k, v in pnrs.items() if v > 1}
    print(
        f"isolation={isolation}: {sum(pnrs.values())} confirmations, "
        f"{len(pnrs)} distinct PNRs, {len(dupes)} charged more than once"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["seed", "confirm", "audit"])
    ap.add_argument("-n", type=int, default=200, help="bookings to seed")
    ap.add_argument("--crash-after", type=int)
    ap.add_argument(
        "--isolation", choices=["read_committed", "read_uncommitted"], default="read_committed"
    )
    a = ap.parse_args()
    if a.cmd == "seed":
        seed(a.n)
    elif a.cmd == "confirm":
        confirm(a.crash_after)
    else:
        audit(a.isolation)
