Exercises to turn the Kafka course notes into hands-on intuition, using airline operations data.
Claude is used as a mentor: implement the `TODO`s yourself, ask for hints, compare approaches.

Domain: a fictional hub carrier "Kafka Airways" (`KA`) flying out of ZRH. Event generators live in
`airline/events.py` (flight status lifecycle, bag scans, bookings).

### Step 0: Cluster & the log on disk
- `just up && just topics`, then run `examples/producer.py` and `examples/consumer.py`.
- `just describe flight-status`: which broker leads which partition? where are the replicas?
- `just files flight-status` and `just dump flight-status 0`: a topic partition is literally a
  directory of append-only segment files. Find your flight events in the dump.
- Notes: *topics, partitions, brokers, replication, KRaft vs ZooKeeper, "distributed write-ahead log"*.

### Step 1: Partition sizing — `exercises/ex01_partition_sizing.py` (offline + tests)
- Apply the course formula to the Ops Control Center scenario.
- Notes: *how to calculate the number of partitions*.

### Step 2: Keys, partitioner & skew — `exercises/ex02_partitioner.py` (offline + live + tests)
- Implement Kafka's murmur2 partitioner, show key skew with `origin` (ZRH hub) vs `flight_id`.
- Live: predict partitions and verify; discover that the Python and Java clients partition differently by default.
- Notes: *record keys, key cardinality, hash(key) % numPartitions*.

### Step 3: Ordering — `exercises/ex03_ordering.py`
- Prove that ordering holds per key, and breaks without keys.
- Notes: *keys guarantee order for a key*.

### Step 4: Consumer groups — `examples/consumer.py` (no new code)
- Gate displays: start 1, 2, 4 and then 5 consumers in group `gate-displays` on the 4-partition
  `flight-status`. Who gets what? What does the 5th do?
- Ctrl-C one of them, and also `kill -9` one. Compare how fast the rebalance happens (clean leave
  vs `session.timeout.ms`).
- Start a consumer in a second group `delay-notifier`: it reads everything independently.
- `just group gate-displays` before and after producing more events: understand LAG.
- Notes: *consumer groups, group members, failover, max parallelism = #partitions*.

### Step 5: Offsets & replay — `exercises/ex04_baggage_replay.py`
- At-least-once vs at-most-once via manual commits and a hard crash.
- `auto.offset.reset` earliest / latest / none.
- Rewind a group (CLI and `offsets_for_times`) to rebuild the baggage projection after a bug.
- Notes: *auto.offset.reset, rewind & replay — "the key"*.

### Step 6: Durability — `exercises/ex05_durability.py`
- acks=0/1/all while killing brokers; `min.insync.replicas` and `NOT_ENOUGH_REPLICAS`.
- Throughput vs durability trade-off.
- Notes: *producer guarantees, topic replication*.

### Step 7: Exactly once — `exercises/ex06_exactly_once.py`
- Idempotent producer, transactional read-process-write, `read_committed` vs `read_uncommitted`.
- Notes: *acks=all with / without exactly once guarantees*.

### Step 8: Compaction & retention — `exercises/ex07_compaction_retention.py`
- Latest flight status on a compacted topic, tombstones for cancellations, GDPR retention questions.
- Notes: *retention policy, compacted topic*.

### Step 9: Headers & a stream processor — `exercises/ex08_delay_detector.py`
- Delay detector with lineage/routing headers, header-only routing, trace back to the source record.
- Notes: *Kafka headers (lineage, producing app, routing, encryption metadata)*.

### Future steps
#### Serialization & Schema Registry
Add a Schema Registry container, switch `flight-status` to Avro, then evolve the schema (add `aircraft_reg`
with a default). Which changes are backward / forward compatible? What breaks old consumers?

#### Capstone: mini Ops Control Center
`flight-status` → delay detector → `passenger-notifications` (joined with the compacted
`passenger-contacts`), with a rebuild-from-scratch replay script and a lag dashboard. Then redo the
delay detector in Kafka Streams (Java) or Flink and compare with the hand-rolled version.

#### Operations
Add partitions to a live topic and watch key → partition mapping change; reassign partitions after
adding a 4th broker; rack awareness (one rack per AZ).
