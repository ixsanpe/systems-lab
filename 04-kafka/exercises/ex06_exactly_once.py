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
TODO `confirm(crash_after)`:
  - consumer: group "booking-confirmer", enable.auto.commit=false,
    isolation.level=read_committed
  - producer: transactional.id="booking-confirmer-1"; init_transactions()
  - loop: poll a batch -> begin_transaction() -> produce one confirmation per
    booking (key=PNR) -> send_offsets_to_transaction(consumer.position(...),
    consumer.consumer_group_metadata()) -> commit_transaction()
  - with --crash-after N: os._exit(1) in the middle of a transaction (after
    producing, before commit).

TODO `audit(isolation)`: read `booking-confirmations` from the beginning with
  isolation.level=read_committed or read_uncommitted, count messages and
  duplicate PNRs.

Experiments:
  1. produce bookings (ex05 or a quick loop), run confirm --crash-after 50, then confirm again.
  2. audit --isolation read_uncommitted  -> duplicates?   (aborted txn data is visible)
     audit --isolation read_committed    -> duplicates?
  3. Where does exactly-once STOP? (hint: the real payment provider is not a Kafka topic)
"""

import argparse


def confirm(crash_after: int | None) -> None:
    raise NotImplementedError


def audit(isolation: str) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["confirm", "audit"])
    ap.add_argument("--crash-after", type=int)
    ap.add_argument(
        "--isolation", choices=["read_committed", "read_uncommitted"], default="read_committed"
    )
    a = ap.parse_args()
    confirm(a.crash_after) if a.cmd == "confirm" else audit(a.isolation)
