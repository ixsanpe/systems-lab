"""Minimal producer: publish flight status events keyed by flight_id.

    uv run python -m examples.producer --flights 20

Things to notice in the output:
- every event of the same flight lands on the SAME partition (key -> partition)
- offsets increase per partition, not globally
"""

import argparse

from confluent_kafka import KafkaError, Message, Producer

from airline.config import producer_config
from airline.events import flight_lifecycles


def on_delivery(err: KafkaError | None, msg: Message) -> None:
    # Runs inside producer.poll()/flush() once the broker has answered.
    if err is not None:
        print(f"FAILED {msg.key()}: {err}")
    else:
        print(
            f"{msg.key().decode():20s} -> {msg.topic()}[{msg.partition()}] @ offset {msg.offset()}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flights", type=int, default=10)
    ap.add_argument("--topic", default="flight-status")
    args = ap.parse_args()

    producer = Producer(
        producer_config(
            **{
                "acks": "all",
                "enable.idempotence": "true",
                # librdkafka's default partitioner is CRC32-based; murmur2_random
                # matches the Java client so both languages agree on key -> partition.
                "partitioner": "murmur2_random",
                "linger.ms": "20",  # wait a bit so records are sent in batches
            }
        )
    )

    for ev in flight_lifecycles(args.flights):
        producer.produce(
            args.topic,
            key=ev.flight_id.encode(),
            value=ev.to_json(),
            headers={"producer-app": b"ops-control-center", "event-type": ev.status.encode()},
            on_delivery=on_delivery,
        )
        producer.poll(0)  # serve delivery callbacks without blocking

    remaining = producer.flush(10)
    print(f"done, {remaining} messages still in flight")


if __name__ == "__main__":
    main()
