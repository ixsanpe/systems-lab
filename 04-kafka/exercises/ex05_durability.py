"""Exercise 5 - acks, replication and min.insync.replicas: don't lose a booking.

Topic `bookings`: replication factor 3, min.insync.replicas=2 (cluster default,
see docker-compose.yml). A lost booking = a passenger who paid and has no seat.

TODO:
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


def produce(acks: str, n: int, rate: float) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--acks", choices=["0", "1", "all"], default="all")
    ap.add_argument("-n", type=int, default=2000)
    ap.add_argument("--rate", type=float, default=100)
    a = ap.parse_args()
    produce(a.acks, a.n, a.rate)
