"""Exercise 4 - Offsets, delivery semantics and replay: the baggage tracker.

The baggage tracker builds a projection `bag_tag -> last checkpoint` from the
`bag-scans` topic (CHECKED_IN -> SORTER -> LOADED -> ARRIVED -> CLAIMED).
Unlike a classic MQ, consuming does NOT delete scans: they stay for the
retention period (7 days on this topic), so you can rebuild the projection.

Implemented below (read the code), then run the experiments:
1. `produce()`: write bag_scans(300) to "bag-scans" keyed by bag_tag.
2. `track(group, crash_after, commit_mode)`: consume "bag-scans" and maintain the
   projection in a dict. Set `enable.auto.commit=false` and commit yourself:
     - commit_mode="after"  : process, then consumer.commit(msg)   -> at-least-once
     - commit_mode="before" : commit, then process                 -> at-most-once
   With --crash-after N, call os._exit(1) after N messages (a hard crash, no close()).
   Print how many messages this run processed and the first offset it saw per partition.
3. Experiments (write down what you observe):
   a) Run with --crash-after 450, then run again without it. Same group.
      'after' vs 'before': which mode re-processes messages? which one loses some?
      Why does at-least-once mean your projection update must be idempotent?
   b) Use a brand-new group with --reset latest, then earliest, then none.
   c) "We had a bug in the tracker, rebuild it": reset the group to the beginning
        docker exec -it kafka-1 /opt/kafka/bin/kafka-consumer-groups.sh \
          --bootstrap-server kafka-1:19092 --group bag-tracker \
          --reset-offsets --to-earliest --topic bag-scans --execute
      (the group must have no active members). Then try --to-datetime.
   d) Do the same rewind from code: --since-minutes 10 uses consumer.offsets_for_times()
      to replay only the scans of the last 10 minutes.
4. `just group bag-tracker` while it runs: what is LAG?
"""

import argparse
import json
import os
import time
from collections import Counter

from confluent_kafka import Consumer, KafkaError, KafkaException, Producer

from airline.config import consumer_config, producer_config
from airline.events import bag_scans

TOPIC = "bag-scans"


def produce() -> None:
    producer = Producer(producer_config(**{"enable.idempotence": "true"}))
    scans = bag_scans(300)
    for scan in scans:
        producer.produce(TOPIC, key=scan.bag_tag.encode(), value=scan.to_json())
        producer.poll(0)
    producer.flush(10)
    print(f"produced {len(scans)} scans for {len({s.bag_tag for s in scans})} bags")


def track(
    group: str,
    crash_after: int | None,
    commit_mode: str,
    offset_reset: str = "earliest",
    since_minutes: float | None = None,
) -> None:
    # librdkafka calls Java's "none" "error": fail instead of guessing where to start.
    reset = "error" if offset_reset == "none" else offset_reset
    consumer = Consumer(
        consumer_config(group, **{"enable.auto.commit": "false", "auto.offset.reset": reset})
    )

    def on_assign(c, partitions):
        if since_minutes is not None:
            # Rewind by time: ask the brokers for the first offset at/after this timestamp.
            ts_ms = int((time.time() - since_minutes * 60) * 1000)
            for p in partitions:
                p.offset = ts_ms
            partitions = c.offsets_for_times(partitions, timeout=10)
            c.assign(partitions)
        print(f"assigned partitions {[p.partition for p in partitions]}")

    consumer.subscribe([TOPIC], on_assign=on_assign)

    projection: dict[str, tuple[str, str]] = {}  # bag_tag -> (last checkpoint, station)
    first_offset: dict[int, int] = {}
    processed = idle = waited = 0
    done = False

    while idle < 5 and not done:
        batch = consumer.consume(num_messages=100, timeout=1.0)
        if not batch:
            if not consumer.assignment():
                # After a hard crash the dead member still "owns" its partitions until
                # session.timeout.ms (45s) expires; only then does the group rebalance.
                waited += 1
                if waited % 10 == 0:
                    print(f"waiting for partitions... {waited}s (crashed member's session timeout)")
                continue
            idle += 1
            continue
        idle = 0

        for msg in batch:
            if msg.error():
                if msg.error().code() == KafkaError._AUTO_OFFSET_RESET:
                    print(f"no committed offset and auto.offset.reset=none: {msg.error()}")
                    done = True
                    break
                raise KafkaException(msg.error())
        if done:
            break

        if commit_mode == "before":
            # at-most-once: mark the whole batch as done BEFORE processing it
            consumer.commit(asynchronous=False)

        for msg in batch:
            first_offset.setdefault(msg.partition(), msg.offset())
            # "Processing": overwrite with the latest scan. Re-applying the same scan
            # gives the same result, so duplicates from at-least-once are harmless.
            data = json.loads(msg.value())
            projection[data["bag_tag"]] = (data["checkpoint"], data["station"])
            processed += 1
            if crash_after is not None and processed >= crash_after:
                print(f"CRASH after {processed} messages, in the middle of a batch")
                os._exit(1)

        if commit_mode == "after":
            # at-least-once: only mark the batch as done once it is fully processed
            consumer.commit(asynchronous=False)

    consumer.close()
    print(f"processed {processed} messages this run")
    print(f"first offset seen per partition: {dict(sorted(first_offset.items()))}")
    print(f"bags by last checkpoint: {dict(Counter(cp for cp, _ in projection.values()))}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["produce", "track"])
    ap.add_argument("--group", default="bag-tracker")
    ap.add_argument("--crash-after", type=int)
    ap.add_argument("--commit", choices=["after", "before"], default="after")
    ap.add_argument("--reset", choices=["earliest", "latest", "none"], default="earliest")
    ap.add_argument("--since-minutes", type=float, help="replay only the last N minutes")
    a = ap.parse_args()
    if a.cmd == "produce":
        produce()
    else:
        track(a.group, a.crash_after, a.commit, a.reset, a.since_minutes)
