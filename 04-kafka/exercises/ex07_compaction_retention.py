"""Exercise 7 - Compacted topics and retention: "what is the status of KA318 NOW?"

`flight-status` is a history (every change). The departure board only needs
the LATEST status per flight -> `flight-latest`, cleanup.policy=compact:
Kafka keeps at least the newest record per key and eventually deletes older
ones. A record with value=None is a TOMBSTONE: it deletes the key (after
delete.retention.ms).

Topic config is in airline/config.py (tiny segment.ms / dirty ratio so you can
watch compaction within a minute on the lab cluster).

TODO:
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

TOPIC = "flight-latest"


def publish(rounds: int) -> None:
    raise NotImplementedError


def board() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["publish", "board"])
    ap.add_argument("--rounds", type=int, default=20)
    a = ap.parse_args()
    publish(a.rounds) if a.cmd == "publish" else board()
