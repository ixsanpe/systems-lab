"""Exercise 7 - Compacted topics and retention: "what is the status of KA318 NOW?"

`flight-status` is a history (every change). The departure board only needs
the LATEST status per flight -> `flight-latest`, cleanup.policy=compact:
Kafka keeps at least the newest record per key and eventually deletes older
ones. A record with value=None is a TOMBSTONE: it deletes the key (after
delete.retention.ms).

Topic config is in airline/config.py (tiny segment.ms / dirty ratio so you can
watch compaction within a minute on the lab cluster).

Implemented below (read the code), then run the experiments:
1. `publish(rounds)`: for 50 flights, send `rounds` status updates each to
   "flight-latest" (key=flight_id). Then send tombstones (value=None) for 5
   CANCELLED flights.
2. `board()`: read "flight-latest" from the beginning with a fresh group, build
   dict flight_id -> status (pop the key on tombstones), print
   "messages read" and "flights on the board".
3. Experiment:
   - publish --rounds 20 ; board   -> ~1'000 messages read
   - wait ~1 min, publish --rounds 1 (compaction never touches the ACTIVE segment;
     new records roll a new segment in every partition), wait ~1 min ; board -> how many now?
   - `just dump flight-latest 0` before/after: look at the offsets. Are they
     renumbered after compaction? Why not?
4. Retention questions (write answers in your notes):
   - `passenger-contacts` has retention.ms=1h. Why is a short retention a
     GDPR tool, and why is it NOT enough to honour a deletion request for one
     passenger? (hint: compaction + tombstones, or crypto-shredding)
   - Why would you keep `bag-scans` for 7 days but a compacted `seat-map`
     forever? Who decides - engineering, legal, or the business?
"""

import argparse
import json
import uuid

from confluent_kafka import Consumer, KafkaException, Producer

from airline.config import consumer_config, producer_config
from airline.events import LIFECYCLE, make_flights

TOPIC = "flight-latest"
N_FLIGHTS = 50
N_CANCELLED = 5


def publish(rounds: int) -> None:
    producer = Producer(producer_config())
    flights = make_flights(N_FLIGHTS)
    for r in range(rounds):
        for flight_id, origin, dest in flights:
            status = {
                "flight_id": flight_id,
                "origin": origin,
                "destination": dest,
                "status": LIFECYCLE[r % len(LIFECYCLE)],
                "update": r,
            }
            producer.produce(TOPIC, key=flight_id.encode(), value=json.dumps(status).encode())
        producer.poll(0)
    # Tombstones: value=None deletes the key once compaction runs.
    for flight_id, _, _ in flights[:N_CANCELLED]:
        producer.produce(TOPIC, key=flight_id.encode(), value=None)
    producer.flush(10)
    print(f"published {rounds * N_FLIGHTS} updates + {N_CANCELLED} tombstones")


def board() -> None:
    consumer = Consumer(consumer_config(f"board-{uuid.uuid4()}"))
    consumer.subscribe([TOPIC])
    departures: dict[str, dict] = {}
    read = tombstones = idle = 0
    while idle < 5:
        msg = consumer.poll(1.0)
        if msg is None:
            idle += 1
            continue
        if msg.error():
            raise KafkaException(msg.error())
        idle = 0
        read += 1
        key = msg.key().decode()
        if msg.value() is None:
            tombstones += 1
            departures.pop(key, None)
        else:
            departures[key] = json.loads(msg.value())
    consumer.close()
    print(
        f"messages read: {read} (tombstones: {tombstones}), flights on the board: {len(departures)}"
    )
    for flight in list(departures.values())[:3]:
        print(
            f"  {flight['flight_id']} {flight['origin']}->{flight['destination']} {flight['status']}"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["publish", "board"])
    ap.add_argument("--rounds", type=int, default=20)
    a = ap.parse_args()
    publish(a.rounds) if a.cmd == "publish" else board()
