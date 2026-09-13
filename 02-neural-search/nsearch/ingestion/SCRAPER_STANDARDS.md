# asyncio scraper standards

Checklist for the concurrent download step of the ingestion pipeline (`pipeline.py`'s
`ingest_raw_files`, and anything downstream that fetches from `entscheidsuche.ch`).

1. **Bound concurrency.** Either an `asyncio.Semaphore` acquired around each request, or
   aiohttp's own `TCPConnector(limit=N, limit_per_host=M)` passed into the `ClientSession`.
   The connector approach caps concurrent connections at the pool level and is arguably
   more idiomatic aiohttp since we're hitting a single host. Never leave concurrency
   unbounded, and never bound it per-folder instead of globally for the whole run. ✅
   * How to choose it: bandwitdh vs. latency vs. concurrency in a server's limit
   * `throughput (files/sec) ≈ concurrency / avg_request_latency`
   * first try a sample with specific concurrency, measure throughput, try to saturate

2. **Explicit timeouts.** aiohttp has no default timeout — a single hung connection can
   block forever and permanently occupy a concurrency slot. Always pass
   `aiohttp.ClientTimeout(total=...)` to the session.

3. **Retry only what's retryable, with backoff.** 5xx, timeouts, and connection errors are
   worth retrying (exponential backoff via `tenacity`, already a dependency). A 404 is not
   transient — retrying it wastes time and hammers the server for nothing. Distinguish the
   two explicitly.

4. **Structured concurrency over manual `gather`.** We're on Python 3.12, so prefer
   `asyncio.TaskGroup` — it propagates cancellation and errors more predictably than
   `asyncio.gather`. ✅

5. **Isolate failures per task.** One file failing to download must not take down the whole
   run. Catch and log inside each task rather than letting one bad file cancel the batch. ✅

6. **One `ClientSession` (and connector) for the entire run.** Never create a new session
   per request or per folder — it defeats connection pooling and can exhaust file
   descriptors at volume. ✅

7. **Resumability / idempotency.** Skip pdf+json pairs already on disk. Track enough state
   (manifest, or just file-existence checks) that a killed run can resume instead of
   restarting from zero.

8. **Task creation is cheap; execution is what you bound.** Creating thousands of
   coroutine/task objects upfront is fine memory-wise — the semaphore or connector limit
   governs how many actually run concurrently, not how many exist. Don't over-engineer
   batching the task creation itself. ✅
   * example: if you want to limit batches of documents, it's the execution you want to cope.

9. **Structured logging over prints.** Per-file outcome (downloaded / skipped / failed)
    plus running totals, so a multi-thousand-file run is observable while it's happening,
    not just at the end.

## Reminders specific to this corpus

- `entscheidsuche.ch/robots.txt` allows `/docs/` (only `/hidden` and `/archiv` are
  disallowed) — confirmed 2026-09-05.
- Build folder names from `EntscheidSuche.areas` (a `dict[str, list[str]]`), never from a
  cross product of separate area/topic lists — not every area has every topic.

# asyncio guide
https://github.com/anordin95/a-conceptual-overview-of-asyncio/tree/main