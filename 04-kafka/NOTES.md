# Kafka
Course notes: [Notion – Kafka](https://app.notion.com/p/Kafka-32573396206880aaa3c7f067d892d638)

* [Kafka documentation](https://kafka.apache.org/documentation/)
* [Design: the log, replication, compaction](https://kafka.apache.org/documentation/#design)
* [confluent-kafka-python API](https://docs.confluent.io/platform/current/clients/confluent-kafka-python/html/index.html)
* [librdkafka configuration](https://github.com/confluentinc/librdkafka/blob/master/CONFIGURATION.md) (all producer/consumer settings used here)
* [KIP-82: Record headers](https://cwiki.apache.org/confluence/display/KAFKA/KIP-82+-+Add+Record+Headers)
* [KIP-500: Replace ZooKeeper with KRaft](https://cwiki.apache.org/confluence/display/KAFKA/KIP-500%3A+Replace+ZooKeeper+with+a+Self-Managed+Metadata+Quorum) — answers the "keyraft(?)" note: it's **KRaft**, and Kafka 4.x no longer supports ZooKeeper at all.

## Gotchas worth remembering
* The Python client (librdkafka) and the Java client use **different default partitioners**
  (CRC32 vs murmur2). Set `partitioner=murmur2_random` when mixing languages on one topic.
* Adding partitions changes `hash(key) % numPartitions` → ordering per key breaks across the change.
* Consumers commit the offset of the **next** message to read (`last processed + 1`).
* `acks=all` only means "all *in-sync* replicas"; pair it with `min.insync.replicas >= 2`.
* Compaction never touches the active segment, so the newest data is always uncompacted.
