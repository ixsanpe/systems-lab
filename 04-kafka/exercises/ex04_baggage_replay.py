"""Exercise 4 - Offsets, delivery semantics and replay: the baggage tracker.

The baggage tracker builds a projection `bag_tag -> last checkpoint` from the
`bag-scans` topic (CHECKED_IN -> SORTER -> LOADED -> ARRIVED -> CLAIMED).
Unlike a classic MQ, consuming does NOT delete scans: they stay for the
retention period (7 days on this topic), so you can rebuild the projection.

TODO:
1. `produce()`: write bag_scans(300) to "bag-scans" keyed by bag_tag.
2. `track(group, crash_after, commit_mode)`: consume "bag-scans" and maintain the
   projection in a dict. Set `enable.auto.commit=false` and commit yourself:
     - commit_mode="after"  : process, then consumer.commit(msg)   -> at-least-once
     - commit_mode="before" : commit, then process                 -> at-most-once
   With --crash-after N, call os._exit(1) after N messages (a hard crash, no close()).
   Print how many messages this run processed and the first offset it saw per partition.
3. Experiments (write down what you observe):
   a) Run with --crash-after 500, then run again without it. Same group.
      'after' vs 'before': which mode re-processes messages? which one loses some?
      Why does at-least-once mean your projection update must be idempotent?
   b) Use a brand-new group with auto.offset.reset=latest, then =earliest, then =none.
   c) "We had a bug in the tracker, rebuild it": reset the group to the beginning
        docker exec -it kafka-1 /opt/kafka/bin/kafka-consumer-groups.sh \
          --bootstrap-server kafka-1:19092 --group bag-tracker \
          --reset-offsets --to-earliest --topic bag-scans --execute
      (the group must have no active members). Then try --to-datetime.
   d) Do the same rewind from code: consumer.offsets_for_times() + assign()/seek()
      to replay only the scans of the last 10 minutes.
4. `just group bag-tracker` while it runs: what is LAG?
"""

import argparse

TOPIC = "bag-scans"


def produce() -> None:
    raise NotImplementedError


def track(group: str, crash_after: int | None, commit_mode: str) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["produce", "track"])
    ap.add_argument("--group", default="bag-tracker")
    ap.add_argument("--crash-after", type=int)
    ap.add_argument("--commit", choices=["after", "before"], default="after")
    a = ap.parse_args()
    produce() if a.cmd == "produce" else track(a.group, a.crash_after, a.commit)
