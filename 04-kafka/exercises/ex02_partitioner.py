"""Exercise 2 - Keys, the default partitioner, and key skew.

Course notes: with a key, partition = hash(key) % numPartitions. Without a key,
records are spread (round-robin / sticky). Poor key choice -> uneven load
("centre of London" problem).

Part A (offline, implemented below - read it):
  Kafka's default partitioner, exactly like the Java client does it:
    - murmur2 32-bit hash, seed 0x9747b28c, m = 0x5bd1e995, r = 24
      (Utils.murmur2 in clients/src/main/java/org/apache/kafka/common/utils/Utils.java)
    - toPositive(h) = h & 0x7fffffff
    - partition = toPositive(murmur2(key_bytes)) % num_partitions
  `uv run pytest tests/test_ex02_partitioner.py` checks it against Kafka's own test vectors.

Part B (offline): run `uv run python -m exercises.ex02_partitioner`
  It prints how 500 flights spread over 4 partitions when keyed by
  - flight_id        (high cardinality)
  - origin airport   (low cardinality, 70% are ZRH departures)
  Which partition gets hammered during the ZRH morning bank? What does that
  mean for the consumer that owns it?

Part C (live, cluster running + `just topics`): run with `--live`
  Produces to `flight-status-by-airport` keyed by origin and checks that the
  predicted partition == the partition in the delivery report, for every message.
  Then run with `--live --librdkafka-default` (no murmur2_random). Why do
  predictions break? (librdkafka default = consistent_random, CRC32.)
  Why does this matter when a Java service and a Python service produce to the
  same topic?
"""

import sys
from collections import Counter

from confluent_kafka import Producer

from airline.config import producer_config
from airline.events import FlightStatusEvent, make_flights

MASK32 = 0xFFFFFFFF  # Java ints are 32 bits and wrap around; Python ints never overflow


def murmur2(data: bytes) -> int:
    """Kafka's murmur2. Returns a signed 32-bit int."""
    seed, m, r = 0x9747B28C, 0x5BD1E995, 24
    length = len(data)
    h = (seed ^ length) & MASK32

    # Mix the key 4 bytes at a time (little-endian, like Java's byte shifting).
    n_blocks = length // 4
    for i in range(n_blocks):
        k = int.from_bytes(data[i * 4 : i * 4 + 4], "little")
        k = (k * m) & MASK32
        k ^= k >> r
        k = (k * m) & MASK32
        h = (h * m) & MASK32
        h ^= k

    # Mix the remaining 1-3 bytes (Java's switch with fall-through).
    tail = data[n_blocks * 4 :]
    if len(tail) == 3:
        h ^= tail[2] << 16
    if len(tail) >= 2:
        h ^= tail[1] << 8
    if len(tail) >= 1:
        h ^= tail[0]
        h = (h * m) & MASK32

    # Final avalanche so similar keys ("KA100", "KA101") end up far apart.
    h ^= h >> 13
    h = (h * m) & MASK32
    h ^= h >> 15

    # Reinterpret as a signed Java int.
    return h - (1 << 32) if h & 0x80000000 else h


def default_partition(key: bytes, num_partitions: int) -> int:
    # toPositive() drops the sign bit (not abs(): abs(Integer.MIN_VALUE) is still negative in Java)
    return (murmur2(key) & 0x7FFFFFFF) % num_partitions


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


def live_check(use_murmur2: bool = True, num_partitions: int = 4) -> None:
    config = producer_config()
    if use_murmur2:
        config["partitioner"] = "murmur2_random"  # same as the Java client
    producer = Producer(config)
    stats = {"ok": 0, "mismatch": 0}

    def on_delivery(err, msg):
        if err is not None:
            print(f"FAILED {msg.key()}: {err}")
            return
        predicted = default_partition(msg.key(), num_partitions)
        if predicted == msg.partition():
            stats["ok"] += 1
        else:
            stats["mismatch"] += 1
            print(f"{msg.key().decode()}: predicted p{predicted}, broker wrote p{msg.partition()}")

    for flight_id, origin, dest in make_flights(100):
        event = FlightStatusEvent(flight_id, origin, dest, "SCHEDULED", seq=0)
        producer.produce(
            "flight-status-by-airport",
            key=origin.encode(),
            value=event.to_json(),
            on_delivery=on_delivery,
        )
        producer.poll(0)
    producer.flush(10)

    label = "murmur2_random" if use_murmur2 else "librdkafka default (consistent_random)"
    print(f"partitioner={label}: {stats['ok']} predicted correctly, {stats['mismatch']} wrong")


if __name__ == "__main__":
    if "--live" in sys.argv:
        live_check(use_murmur2="--librdkafka-default" not in sys.argv)
    else:
        offline_report()
