"""Exercise 1 - How many partitions does `flight-status` need?  (no Kafka needed)

From the course notes:

    single-consumer-events-per-second = 1 second / consumer-processing-time-seconds
    partition-count = events-per-second * growth-multiplier / single-consumer-events-per-second

Scenario: the Ops Control Center ingests flight status events.
- ~1'000 flights/day, each emits ~6 status events, but traffic is bursty:
  during the morning bank at the hub you see peaks of 400 events/s
  (status + ACARS position/fuel messages).
- The network planning team expects the schedule to grow 50% next year.
- The slowest consumer is the passenger-notification service: it calls an
  SMS gateway and needs 20 ms per event.

Implemented below (read the code), then run the experiments:
1. The two functions below (round UP - you can't have 0.3 of a partition).
2. Run `uv run pytest tests/test_ex01_partition_sizing.py`.
3. Answer in PLAN.md / your notes:
   - How many partitions for the scenario above?
   - What happens if the SMS gateway degrades to 100 ms/event? Can you just
     add partitions later? (hint: what happens to hash(key) % numPartitions?)
   - Why is it a bad idea to create 1'000 partitions "just in case"?
"""

import math


def single_consumer_events_per_second(processing_time_s: float) -> float:
    return 1 / processing_time_s


def partition_count(
    events_per_second: float, growth_multiplier: float, processing_time_s: float
) -> int:
    # Round up: 12.1 partitions of work still needs a 13th consumer.
    needed = events_per_second * growth_multiplier
    return math.ceil(needed / single_consumer_events_per_second(processing_time_s))


if __name__ == "__main__":
    print("partitions:", partition_count(400, 1.5, 0.020))
