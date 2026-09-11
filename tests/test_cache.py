from omnipose.gpu import empty_cache


def test_empty_cache_callable():
    """empty_cache should be callable without error."""
    empty_cache()
