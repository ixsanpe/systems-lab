"""Create (or recreate) the topics used by the exercises.

uv run python -m examples.admin            # create missing topics
uv run python -m examples.admin --recreate # delete + create (fresh offsets)
"""

import sys
import time

from confluent_kafka.admin import AdminClient, NewTopic

from airline.config import BOOTSTRAP, TOPICS


def main(recreate: bool = False) -> None:
    admin = AdminClient({"bootstrap.servers": BOOTSTRAP})

    if recreate:
        existing = [t for t in admin.list_topics(timeout=10).topics if t in TOPICS]
        for topic, fut in admin.delete_topics(existing).items():
            fut.result()
            print(f"deleted {topic}")
        time.sleep(2)  # deletion is async on the brokers

    existing = admin.list_topics(timeout=10).topics
    new = [
        NewTopic(name, num_partitions=parts, config=cfg)
        for name, (parts, cfg) in TOPICS.items()
        if name not in existing
    ]
    for topic, fut in admin.create_topics(new).items() if new else []:
        fut.result()
        print(f"created {topic}")

    md = admin.list_topics(timeout=10)
    while any(name not in md.topics for name in TOPICS):  # metadata propagates async
        time.sleep(0.5)
        md = admin.list_topics(timeout=10)
    for name in TOPICS:
        t = md.topics[name]
        leaders = {p: t.partitions[p].leader for p in sorted(t.partitions)}
        print(f"{name:28s} partitions={len(t.partitions)} leaders={leaders}")


if __name__ == "__main__":
    main(recreate="--recreate" in sys.argv)
