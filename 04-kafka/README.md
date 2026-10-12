# 04 – Kafka through airline operations

Hands-on exercises for the Kafka course notes, using flight status, baggage and booking events.
See `PLAN.md` for the step-by-step exercises, `IN_PRACTICE.md` for how this looks in a real airline, and `NOTES.md` for references.

## Setup
Requires Docker, [uv](https://docs.astral.sh/uv/) and [just](https://just.systems/).

```bash
just up            # 3-broker KRaft cluster (localhost:29092,39092,49092)
uv sync
just topics        # create the exercise topics
just test          # unit tests for the offline exercises (fail until you implement them)
```

Optional UI on http://localhost:8080: `just ui`.

## Layout
```
airline/      domain + config: event generators, topic definitions
examples/     working reference code: admin (topics), producer, consumer
exercises/    stubs with TODOs, one file per concept
tests/        tests for the offline exercises
```

## Quick tour
```bash
uv run python -m examples.producer --flights 20     # watch key -> partition
uv run python -m examples.consumer --group gate-displays --name display-1
uv run python -m examples.consumer --group gate-displays --name display-2   # 2nd terminal: rebalance
just describe flight-status                         # leaders, replicas, ISR
just kill 3                                         # lose a broker, see what happens
```
