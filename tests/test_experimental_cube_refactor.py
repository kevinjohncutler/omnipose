from omnipose.experimental.cube import cubestats


def test_cubestats_basic():
    assert cubestats(0) == [1]
    assert cubestats(1) == [2, 1]
    assert cubestats(2) == [4, 4, 1]
