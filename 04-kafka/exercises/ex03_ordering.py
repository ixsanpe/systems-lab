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
"""

import sys

TOPIC = "flight-status"


def produce(keyed: bool) -> None:
    raise NotImplementedError


def check_order() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    if sys.argv[1] == "produce":
        produce(keyed="--keyed" in sys.argv)
    else:
        check_order()
