from exercises.ex01_partition_sizing import partition_count, single_consumer_events_per_second


def test_single_consumer_throughput():
    assert single_consumer_events_per_second(0.020) == 50
    assert single_consumer_events_per_second(1.0) == 1


def test_partition_count_rounds_up():
    assert partition_count(400, 1.5, 0.020) == 12
    assert partition_count(120, 1.0, 0.025) == 3  # 120 / 40 = 3.0 -> 3
    assert partition_count(121, 1.0, 0.025) == 4  # needs a 4th
