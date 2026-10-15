"""Exercise 5 - acks, replication and min.insync.replicas: don't lose a booking.

Topic `bookings`: replication factor 3, min.insync.replicas=2 (cluster default,
see docker-compose.yml). A lost booking = a passenger who paid and has no seat.

Implemented below (read the code), then run the experiments:
1. `produce(acks, n, rate)`: send `n` bookings(…) to "bookings" keyed by PNR at
   ~`rate` msg/s with the given acks (0, 1, all) and enable.idempotence=false.
   Count successful deliveries vs errors in the delivery callback and print
   the error codes you see. Also measure total time.
2. Experiments - run a producer of 2'000 bookings at 100/s and meanwhile:
   a) `just describe bookings` - note leader, replicas and ISR per partition.
   b) `just kill 3`. Describe again: what happened to ISR and leaders?
      Do acks=0/1/all still succeed?
   c) `just kill 2` as well (only 1 broker left). Now:
        acks=all -> brokers reply NOT_ENOUGH_REPLICAS; librdkafka retries until
                    message.timeout.ms and you get _MSG_TIMED_OUT (set it to ~8000
                    to see it quickly). Why is that the *right* behaviour?
        acks=1   -> still "succeeds". What happens to those bookings if broker 1
                    dies before the others come back?
   d) `just revive 2 && just revive 3` and watch ISR recover.
3. Throughput: produce 50'000 bookings as fast as possible with acks=0/1/all.
   How big is the difference on a local cluster? Play with linger.ms and
   batch.size. (It will be much larger across real networks / AZs.)
4. Question: `acks=all` with `min.insync.replicas=1` - what guarantee do you
   actually have?
"""

import argparse
import time
from collections import Counter

from confluent_kafka import Producer

from airline.config import producer_config
from airline.events import bookings

TOPIC = "bookings"


def produce(acks: str, n: int, rate: float, timeout_ms: int = 8000) -> None:
    producer = Producer(
        producer_config(
            **{
                "acks": acks,
                "enable.idempotence": "false",
                # Give up on a message after this long; low so failures show up quickly.
                "message.timeout.ms": str(timeout_ms),
                "linger.ms": "5",
            }
        )
    )
    results: Counter = Counter()

    def on_delivery(err, msg):
        results["OK" if err is None else err.name()] += 1

    start = time.time()
    for i, booking in enumerate(bookings(n)):
        while True:
            try:
                producer.produce(
                    TOPIC,
                    key=booking.pnr.encode(),
                    value=booking.to_json(),
                    on_delivery=on_delivery,
                )
                break
            except BufferError:  # local queue full: let it drain
                producer.poll(0.1)
        producer.poll(0)
        if rate > 0:
            time.sleep(max(0.0, start + (i + 1) / rate - time.time()))
        if rate > 0 and (i + 1) % 500 == 0:
            print(f"  sent {i + 1}/{n}, delivered so far: {dict(results)}")

    producer.flush(timeout_ms / 1000 + 5)
    elapsed = time.time() - start
    print(
        f"acks={acks}: {n} bookings in {elapsed:.2f}s ({n / elapsed:,.0f} msg/s) -> {dict(results)}"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--acks", choices=["0", "1", "all"], default="all")
    ap.add_argument("-n", type=int, default=2000)
    ap.add_argument("--rate", type=float, default=100, help="msg/s, 0 = as fast as possible")
    a = ap.parse_args()
    produce(a.acks, a.n, a.rate)
