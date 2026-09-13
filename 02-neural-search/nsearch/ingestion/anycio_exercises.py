"""
asyncio exercises — coroutines, tasks, and who hogs the event loop.

HOW TO USE
  1. Read each exercise. WRITE YOUR PREDICTION in the PREDICT block first.
  2. Run it:  python asyncio_exercises.py 1     (or 2, 3, ... / "all")
  3. Compare. If you were wrong, that's the interesting part — figure out why
     before reading the answer at the bottom of the file.

Rule of thumb you are testing:
  `await X` only cedes control if something inside X awaits an UNFINISHED future.
"""

import asyncio
import sys
import time


# ---------------------------------------------------------------------------
# EX 1 — await coroutine vs await task
# ---------------------------------------------------------------------------
# PREDICT: order of the printed lines?
#   your answer: a a a b
#
async def ex1():
    async def a():
        print("a")

    async def b():
        print("b")

    async def main():
        task_b = asyncio.create_task(b())
        for _ in range(3):
            await a()
        await task_b

    await main()


# ---------------------------------------------------------------------------
# EX 2 — same, but wrap a() in a task
# ---------------------------------------------------------------------------
# PREDICT: does "b" move? why?
#   your answer:  b a a a
#
async def ex2():
    async def a():
        print("a")

    async def b():
        print("b")

    async def main():
        task_b = asyncio.create_task(b())
        for _ in range(3):
            await asyncio.create_task(a())
        await task_b

    await main()


# ---------------------------------------------------------------------------
# EX 3 — does a coroutine that sleeps cede, even without a task?
# ---------------------------------------------------------------------------
# PREDICT: "b" prints before or after "a end"?
#   your answer: a start / b / a end
#
async def ex3():
    async def a():
        print("a start")
        await asyncio.sleep(0.1)   # NOT wrapped in a task -> creates unfinished future that 
        # travels up through every enclosing await to which ever Task is actually running the show
        # suspension is caused by unfinished futures
        print("a end")

    async def b():
        print("b")

    async def main():
        task_b = asyncio.create_task(b())
        await a()
        await task_b

    await main()


# ---------------------------------------------------------------------------
# EX 4 — sequential vs concurrent. TIME IT.
# ---------------------------------------------------------------------------
# TODO: fill in `concurrent()` so it runs the same 5 calls in ~0.2s, not ~1.0s.
#       Do it twice: once with a list of tasks, once with asyncio.gather.
# PREDICT the two elapsed times before running.
#   your answer:
#
async def ex4():
    async def work(i):
        await asyncio.sleep(0.2)
        return i * 10

    async def sequential():
        results = []
        for i in range(5):
            results.append(await work(i))
        return results

    async def concurrent():
        # a) tasks
        tasks =  [asyncio.create_task(work(i)) for i in range(5)]
        return [await t for t in tasks] # this does not make them run!
        # b) gather
        return await asyncio.gather(*(work(i) for i in range(5)))
        #FIXME:
        return asyncio.gather(*(await work(i) for i in range(5))) # wrong: check Ex1

    for fn in (sequential, concurrent):
        t0 = time.perf_counter()
        try:
            out = await fn() # NOTE: await drives the coroutine forward
        except NotImplementedError:
            print(f"{fn.__name__}: not implemented yet")
            continue
        print(f"{fn.__name__}: {out} in {time.perf_counter() - t0:.2f}s")


# ---------------------------------------------------------------------------
# EX 5 — when does a created task ACTUALLY start?
# ---------------------------------------------------------------------------
# PREDICT: where does "task ran" appear relative to the three "main" lines?
#   your answer: main 1 main 2 task ran main 3; it runs at first suspension of main
#
async def ex5():
    async def side():
        print("task ran")

    async def main():
        t = asyncio.create_task(side())
        print("main 1")
        print("main 2")
        await asyncio.sleep(0)
        print("main 3")
        await t

    await main()


# ---------------------------------------------------------------------------
# EX 6 — the greedy coroutine
# ---------------------------------------------------------------------------
# `ticker` wants to print once every 0.1s. `hog` never awaits anything.
# PREDICT: how many ticks appear before "hog done"?
#   your answer: none, hog done runs before because of await and nothing cedes control inside
#
# THEN: fix hog() so the ticker gets to run, WITHOUT reducing the loop count.
#
async def ex6():
    async def ticker():
        for i in range(5):
            print(f"  tick {i}")
            await asyncio.sleep(0.1)

    async def hog():
        total = 0
        for i in range(30_000_000):
            total += i
            await asyncio.sleep(0) #FIXME
        print("hog done")

    t = asyncio.create_task(ticker())
    await hog()
    await t


# ---------------------------------------------------------------------------
# EX 7 — the disappearing task
# ---------------------------------------------------------------------------
# TODO: run this. Does "finished!" always print? Run it several times.
#       Then fix it two ways: (a) keep a list of tasks and await them,
#       (b) use asyncio.TaskGroup.
# PREDICT: what does main() return / what gets printed?
#   your answer:
#
async def ex7():
    async def fire_and_forget(i):
        await asyncio.sleep(0.05)
        print(f"finished! {i}")

    for i in range(5):
        asyncio.create_task(fire_and_forget(i))
    print("main is done, exiting immediately")


# ---------------------------------------------------------------------------
# EX 8 — write your own awaitable
# ---------------------------------------------------------------------------
# TODO: implement `my_sleep(delay)` using ONLY asyncio.get_running_loop(),
#       loop.call_later(), and asyncio.Future — no asyncio.sleep.
#       This is the exercise that proves you understand "await forwards a
#       yield from an unfinished future up to the Task".
#
async def ex8():
    async def my_sleep(delay):
        # TODO:
        #   1. get the running loop
        #   2. create a Future
        #   3. schedule loop.call_later(delay, ...) to set its result
        #   4. await the future
        raise NotImplementedError

    async def worker(name, delay):
        await my_sleep(delay)
        print(f"{name} woke after {delay}s")

    try:
        await asyncio.gather(
            worker("fast", 0.1),
            worker("slow", 0.3),
        )
    except NotImplementedError:
        print("ex8: implement my_sleep first")


# ---------------------------------------------------------------------------
# EX 9 — prove that await on a DONE task does not cede
# ---------------------------------------------------------------------------
# TODO: construct a case where `await some_task` prints nothing in between,
#       because the task had already completed. Hint: sleep(0) enough times
#       first to let it finish, then time/observe the await.
# PREDICT what you expect, then build it.
#
async def ex9():
    print("ex9: build this one yourself")


# ---------------------------------------------------------------------------

EXERCISES = {
    "1": ex1, "2": ex2, "3": ex3, "4": ex4, "5": ex5,
    "6": ex6, "7": ex7, "8": ex8, "9": ex9,
}


async def _run(keys):
    for k in keys:
        print(f"\n===== EX {k} =====")
        await EXERCISES[k]()


if __name__ == "__main__":
    args = sys.argv[1:] or ["1"]
    keys = sorted(EXERCISES) if args == ["all"] else args
    asyncio.run(_run(keys))


# ===========================================================================
# ANSWERS — don't scroll until you've predicted and run.
# ===========================================================================
#
# EX 1  a a a b
#       a() contains no await on a future, so `await a()` is an inline call.
#       main never yields until `await task_b`.
#
# EX 2  b a a a
#       The FIRST `await create_task(a())` suspends main immediately. The loop
#       then drains its queue in FIFO order: task_b was queued first, so "b"
#       wins. Note this is not "tasks are concurrent" — it's just queue order.
#
# EX 3  a start / b / a end
#       The key result: you do NOT need a task to cede. sleep() creates an
#       unfinished future; the yield travels up through `await a()` to main's
#       Task all the same. "await coroutine never cedes" is shorthand for
#       "a coroutine with nothing to wait on never cedes".
#
# EX 4  sequential ~1.0s, concurrent ~0.2s. Two valid concurrent versions:
#           tasks = [asyncio.create_task(work(i)) for i in range(5)]
#           return [await t for t in tasks]
#       or:
#           return await asyncio.gather(*(work(i) for i in range(5)))
#       Both work because all five sleeps are started BEFORE any awaiting.
#       Note `[await work(i) for i in range(5)]` is still sequential — the
#       coroutine isn't started until the await, one at a time.
#
# EX 5  main 1 / main 2 / task ran / main 3
#       create_task schedules; it does not run. The task runs at the first
#       suspension of main — here, sleep(0). Not at `await t`.
#
# EX 6  Zero ticks before "hog done" (the loop is blocked for ~1s), then all
#       five ticks. Fix: yield periodically —
#           for i in range(30_000_000):
#               total += i
#               if i % 100_000 == 0:
#                   await asyncio.sleep(0)
#       Better fix for real CPU work: asyncio.to_thread / ProcessPoolExecutor.
#       asyncio hides WAITING, not COMPUTING.
#
# EX 7  "main is done, exiting immediately" and nothing else. asyncio.run
#       cancels remaining tasks when main returns, and the loop only holds
#       weak references, so unreferenced tasks may also be GC'd mid-flight.
#       Fixes:
#           tasks = [asyncio.create_task(f(i)) for i in range(5)]
#           await asyncio.gather(*tasks)
#       or:
#           async with asyncio.TaskGroup() as tg:
#               for i in range(5):
#                   tg.create_task(fire_and_forget(i))
#       TaskGroup also propagates exceptions instead of swallowing them.
#
# EX 8  async def my_sleep(delay):
#           loop = asyncio.get_running_loop()
#           fut = loop.create_future()
#           loop.call_later(delay, lambda: fut.done() or fut.set_result(None))
#           await fut
#       Expected: "fast woke after 0.1s" then "slow woke after 0.3s", total
#       ~0.3s not 0.4s. `await fut` yields the future up to the Task, which
#       parks the coroutine and adds a done-callback to resume it. That is
#       the entire suspension mechanism — asyncio.sleep is barely more.
#
# EX 9  e.g.
#           t = asyncio.create_task(quick())
#           for _ in range(3):
#               await asyncio.sleep(0)   # t completes during these
#           print("before"); await t; print("after")
#       Nothing can interleave between "before" and "after": Future.__await__
#       sees done() is True, skips its yield, and returns the result inline.