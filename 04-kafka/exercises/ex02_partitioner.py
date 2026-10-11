"""Exercise 2 - Keys, the default partitioner, and key skew.

Course notes: with a key, partition = hash(key) % numPartitions. Without a key,
records are spread (round-robin / sticky). Poor key choice -> uneven load
("centre of London" problem).

Part A (offline):
  Implement Kafka's default partitioner exactly like the Java client does:
    - murmur2 32-bit hash, seed 0x9747b28c, m = 0x5bd1e995, r = 24
      (see Utils.murmur2 in the Kafka source: clients/src/main/java/org/apache/kafka/common/utils/Utils.java)
    - toPositive(h) = h & 0x7fffffff
    - partition = toPositive(murmur2(key_bytes)) % num_partitions
  Careful: Python ints are unbounded, Java ints overflow at 32 bits. Mask with
  & 0xFFFFFFFF after every multiply, and return a SIGNED 32-bit int at the end
  so the test vectors (copied from Kafka's own UtilsTest) match.

Part B (offline): run `uv run python -m exercises.ex02_partitioner`
  It prints how 500 flights spread over 4 partitions when keyed by
  - flight_id        (high cardinality)
  - origin airport   (low cardinality, 70% are ZRH departures)
  Which partition gets hammered during the ZRH morning bank? What does that
  mean for the consumer that owns it?

Part C (live, cluster running + `just topics`): run with `--live`
  Produce to `flight-status-by-airport` keyed by origin and check that your
  predicted partition == the partition in the delivery report, for every message.
  Then remove `"partitioner": "murmur2_random"` from the producer config and run
  again. Why do predictions break? (librdkafka default = consistent_random, CRC32.)
  Why does this matter when a Java service and a Python service produce to the
  same topic?
"""

import sys
from collections import Counter

from airline.events import make_flights


def murmur2(data: bytes) -> int:
    """Kafka's murmur2. Returns a signed 32-bit int."""
    raise NotImplementedError


def default_partition(key: bytes, num_partitions: int) -> int:
    raise NotImplementedError


def distribution(keys: list[bytes], num_partitions: int) -> Counter:
    return Counter(default_partition(k, num_partitions) for k in keys)


def offline_report(num_partitions: int = 4) -> None:
    flights = make_flights(500)
    for label, keys in [
        ("key=flight_id", [f.encode() for f, _, _ in flights]),
        ("key=origin", [o.encode() for _, o, _ in flights]),
    ]:
        dist = distribution(keys, num_partitions)
        print(f"{label:15s} distinct keys={len(set(keys)):3d}  ", end="")
        for p in range(num_partitions):
            print(f"p{p}={dist[p]:3d} {'#' * (dist[p] // 10)}  ", end="")
        print()


def live_check() -> None:
    # TODO: produce one FlightStatusEvent per flight to "flight-status-by-airport"
    #       with key=origin. In the delivery callback compare msg.partition()
    #       with default_partition(msg.key(), 4) and count mismatches.
    raise NotImplementedError


if __name__ == "__main__":
    live_check() if "--live" in sys.argv else offline_report()
