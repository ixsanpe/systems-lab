"""Exercise 8 - A stream processor with headers: the delay detector.

Read `flight-status`, and whenever a flight reaches delay_minutes >= 30
for the first time, emit a FlightDelay event to `flight-delays` (key=flight_id).

Use Kafka headers for metadata that is NOT part of the business payload:
  - lineage:  source-topic, source-partition, source-offset
  - producer-app: "delay-detector" and a version
  - routing:  notify-channel = "sms" if delay >= 60 else "app-push"
  - (think) encryption-key-id: which key encrypted the payload?

TODO:
1. Implement `run()`. Keep "already notified" state in a dict. What happens to
   that state when the process restarts or a partition moves to another
   instance after a rebalance? Because the input is keyed by flight_id, why is
   per-partition local state safe here?
2. `just tail flight-delays` - check the headers.
3. Write a second consumer `notifier` (group "passenger-notifier") that reads
   flight-delays and routes ONLY by the notify-channel header, without
   deserialising the value. Why is that useful?
4. Given a single flight-delays record, use the lineage headers to fetch the
   exact source flight-status record (assign + seek to that offset).
"""


def run() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    run()
