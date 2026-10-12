"""Exercise 3 - Ordering is per partition, so it's per KEY.

A gate display that receives LANDED before BOARDING for the same flight is
showing garbage. Kafka only guarantees order inside one partition, so the key
decides what is ordered relative to what.

Each FlightStatusEvent carries `seq` (0, 1, 2, ... per flight).

TODO:
1. `produce(keyed)`: send flight_lifecycles(200) to "flight-status".
   keyed=True  -> key = flight_id
   keyed=False -> key = None (records spread over partitions)
   Use a fresh topic state each time: `uv run python -m examples.admin --recreate`.
2. `check_order()`: consume the topic from the beginning with a NEW group id each
   run (e.g. f"order-check-{uuid4()}"), stop after ~5s without messages, and
   count, per flight, how often an event arrives with seq < last seen seq.
3. Run both variants. Explain the result:
       uv run python -m exercises.ex03_ordering produce --keyed && uv run python -m exercises.ex03_ordering check
       uv run python -m exercises.ex03_ordering produce          && uv run python -m exercises.ex03_ordering check
   Tip: if unkeyed shows 0 violations, the sticky partitioner batched everything into one
   partition. Add producer.flush() + a short sleep every few messages to force new batches.
4. Bonus: with keys, can order still break? Look up `max.in.flight.requests.per.connection`,
   retries, and why `enable.idempotence=true` fixes it.

   Answer: yes. With up to 5 batches in flight, batch 1 can fail and be retried AFTER
   batch 2 was written -> reordered; a lost ack + retry -> duplicate. Idempotence adds a
   producer id + per-partition sequence number, so the broker drops duplicates and rejects
   gaps (producer resends in order). Java enables it by default; librdkafka (Python) does
   NOT, so set it. It only covers producer retries; an app-level resend needs transactions (ex06).
"""

import sys
import time
import uuid

from confluent_kafka import Consumer, KafkaException, Producer

from airline.config import consumer_config, producer_config
from airline.events import FlightStatusEvent, flight_lifecycles

TOPIC = "flight-status"


def produce(keyed: bool) -> None:
    producer = Producer(producer_config(partitioner="murmur2_random"))
    for i, ev in enumerate(flight_lifecycles(200)):
        producer.produce(
            TOPIC,
            key=ev.flight_id.encode() if keyed else None,
            value=ev.to_json(),
        )
        if i % 20 == 0:
            # Without a key the sticky partitioner fills one batch per partition.
            # Sending the batch now makes the next events go to another partition.
            producer.flush()
            time.sleep(0.05)
    producer.flush(10)
    print(f"produced {i + 1} events, keyed={keyed}")


def check_order() -> None:
    # New group id each run: no committed offsets, so auto.offset.reset=earliest
    # reads the topic from the beginning.
    consumer = Consumer(consumer_config(f"order-check-{uuid.uuid4()}"))
    consumer.subscribe([TOPIC])

    last_seq: dict[str, tuple[int, int]] = {}  # flight_id -> (highest seq, its partition)
    partitions_per_flight: dict[str, set[int]] = {}
    violations = total = idle = 0

    while idle < 5:  # stop after ~5 empty seconds
        msg = consumer.poll(1.0)
        if msg is None:
            idle += 1
            continue
        if msg.error():
            raise KafkaException(msg.error())
        idle = 0
        total += 1

        ev = FlightStatusEvent.from_json(msg.value())
        partitions_per_flight.setdefault(ev.flight_id, set()).add(msg.partition())
        if ev.flight_id in last_seq and ev.seq < last_seq[ev.flight_id][0]:
            violations += 1
            prev_seq, prev_partition = last_seq[ev.flight_id]
            if violations <= 10:
                print(
                    f"OUT OF ORDER {ev.flight_id}: {ev.status} (seq {ev.seq}, p{msg.partition()}) "
                    f"arrived after seq {prev_seq} (p{prev_partition})"
                )
        else:
            last_seq[ev.flight_id] = (ev.seq, msg.partition())

    consumer.close()
    spread = sum(len(p) > 1 for p in partitions_per_flight.values())
    print(
        f"read {total} events for {len(last_seq)} flights: {violations} out of order, "
        f"{spread} flights spread over more than one partition"
    )


if __name__ == "__main__":
    if sys.argv[1] == "produce":
        produce(keyed="--keyed" in sys.argv)
    else:
        check_order()
