# Kafka in practice: an airline

Kafka usually sits in the middle of a company. It doesn't replace the existing systems; it
connects them. Each system publishes what happened once, and any number of other systems react
to it independently.

## The airline picture

```
 SOURCES (publish facts)                 KAFKA                    CONSUMERS (each its own group)
 ───────────────────────            ───────────────            ──────────────────────────────
 Ops control / flight ops ──┐                                ┌─► Gate & airport screens
 Aircraft messages (ACARS) ─┤       flight-status            ├─► App / SMS notifications
 Airport ops database ──────┼──►    bag-scans          ──────┼─► Rebooking (missed connections)
 Reservations (bookings) ───┤       bookings                 ├─► Crew scheduling (duty limits)
 Baggage system ────────────┘       flight-latest (compact)  ├─► Data lake (Snowflake/S3) → BI, delay-prediction ML
                                                             └─► Kafka Streams/Flink: detect delays, KPIs
```

Without Kafka, the ops system would need a direct connection to each of those six consumers.
Every new consumer would mean changing the ops system, and if the notification service is down,
its messages are lost.

## One delay, end to end

The 07:10 ZRH→LHR flight, KA318, gets a technical problem:

1. **Ops control** sets a 45-minute delay. One event goes to `flight-status` with key
   `KA318-2026-10-12`. The key keeps every update for that flight in order (exercise 3).
2. Several independent **consumer groups** react at the same time (Step 4):
   - **gate displays** show "07:55";
   - **notifications** send an SMS to 180 passengers. It has 4 instances, so the load is shared;
   - **rebooking** finds the 23 passengers who will now miss their onward flight at LHR and
     rebooks them;
   - **crew scheduling** checks whether the pilots will now exceed their maximum duty time;
   - **data lake** stores the event for on-time-performance reporting.
3. A **stream processor** (exercise 8) notices the delay is ≥30 minutes and publishes to
   `flight-delays`, which triggers meal vouchers. Compensation rules work the same way: EU261
   entitles passengers to money at ≥3 hours.
4. **The notification service crashes for 10 minutes.** Nothing is lost. When it restarts, it
   continues from its committed offset (exercise 4). With a classic message queue, those
   messages could be gone.
5. **A month later**, data science builds a delay-prediction model. They start a new consumer
   group from the beginning of the stored history and replay all past events. No source system
   had to change (replay is "the key" in the course notes).
6. A **new app screen** needs "the current status of all of today's flights". It reads the
   compacted `flight-latest` topic: one record per flight instead of the whole history
   (exercise 7).

## Common patterns you'll see at work

- **Change data capture:** old systems can't publish events, so a tool such as Debezium reads
  their database's change log and turns every row change into a Kafka event. This is how
  reservation systems usually get into Kafka.
- **Kafka Connect:** ready-made connectors that copy topics into and out of Snowflake, S3,
  Elasticsearch and so on, with no code.
- **Stream processing** (Kafka Streams, Flink): joins and windows, such as "bags loaded vs
  passengers boarded per flight" or "on-time performance per hour".
- **Event sourcing:** the topic is the source of truth, and databases are views you can rebuild
  from it.

## When Kafka is the wrong tool

- **Request/response:** "is seat 12A free?" needs an answer now, so use an API.
- **Low volume with no replay needed:** a simple queue such as SQS or RabbitMQ is easier to run.
- **Strict transactions across systems:** payments still need their own guarantees, the limit
  discussed in exercise 6.

## Question to bring to work

Which of your systems publish to Kafka today, and which teams consume those topics? That shows
the real version of this picture.
