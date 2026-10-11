"""Minimal consumer: read flight status events as part of a consumer group.

    uv run python -m examples.consumer --group gate-displays
    uv run python -m examples.consumer --group gate-displays --name display-2   # in a 2nd terminal

Start several with the same --group and watch partitions get split between them.
Start one with a different --group and watch it get ALL partitions independently.
"""

import argparse

from confluent_kafka import Consumer, KafkaException

from airline.config import consumer_config
from airline.events import FlightStatusEvent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="gate-displays")
    ap.add_argument("--name", default="display-1")
    ap.add_argument("--topic", default="flight-status")
    args = ap.parse_args()

    consumer = Consumer(consumer_config(args.group, **{"client.id": args.name}))

    def on_assign(c, partitions):
        print(f"[{args.name}] assigned: {[p.partition for p in partitions]}")

    def on_revoke(c, partitions):
        print(f"[{args.name}] revoked:  {[p.partition for p in partitions]}")

    consumer.subscribe([args.topic], on_assign=on_assign, on_revoke=on_revoke)
    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                raise KafkaException(msg.error())
            ev = FlightStatusEvent.from_json(msg.value())
            print(
                f"[{args.name}] p{msg.partition()} off={msg.offset():<4} "
                f"{ev.flight_id} {ev.origin}->{ev.destination} {ev.status:9s} "
                f"gate={ev.gate} delay={ev.delay_minutes}"
            )
    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()  # leaves the group cleanly -> immediate rebalance for the others


if __name__ == "__main__":
    main()
