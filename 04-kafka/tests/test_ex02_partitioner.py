import pytest

from exercises.ex02_partitioner import default_partition, murmur2

# Test vectors from Kafka's own clients/src/test/.../UtilsTest.java
VECTORS = {
    "21": -973932308,
    "foobar": -790332482,
    "a-little-bit-long-string": -985981536,
    "a-little-bit-longer-string": -1486304829,
    "lkjh234lh9fiuh90y23oiuhsafujhadof229phr9h19h89h8": -58897971,
    "abc": 479470107,
}


@pytest.mark.parametrize("key,expected", VECTORS.items())
def test_murmur2_matches_java(key, expected):
    assert murmur2(key.encode()) == expected


def test_partition_is_stable_and_in_range():
    for n in (1, 3, 4, 12):
        p = default_partition(b"KA318-2026-10-11", n)
        assert 0 <= p < n
        assert p == default_partition(b"KA318-2026-10-11", n)


def test_partition_uses_positive_hash():
    # murmur2("foobar") is negative; the partition must still be valid
    assert default_partition(b"foobar", 4) == (-790332482 & 0x7FFFFFFF) % 4
