"""Tests for omnirefactor.data.shm.ShmPool.

ShmPool packs arrays into one POSIX shared memory segment so PyTorch DataLoader
workers can zero-copy attach by name. The pickle round-trip simulates what
happens when ShmPool is sent to a spawned worker process.
"""

import pickle

import numpy as np
import pytest

from omnirefactor.data.shm import ShmPool


@pytest.fixture
def arrays():
    return [
        np.arange(16, dtype=np.float32).reshape(4, 4),
        np.arange(64, dtype=np.uint16).reshape(8, 8),
        np.ones((2, 3, 5), dtype=np.int32),
    ]


class TestShmPoolBasic:
    def test_create_and_get(self, arrays):
        pool = ShmPool(arrays)
        try:
            for i, orig in enumerate(arrays):
                loaded = pool.get(i)
                assert loaded.shape == orig.shape
                assert loaded.dtype == orig.dtype
                np.testing.assert_array_equal(loaded, orig)
        finally:
            pool.close()
            pool.unlink()

    def test_get_returns_view_not_copy(self, arrays):
        pool = ShmPool(arrays)
        try:
            v1 = pool.get(0)
            v2 = pool.get(0)
            # Both views share the same buffer
            assert v1.ctypes.data == v2.ctypes.data
        finally:
            pool.close()
            pool.unlink()

    def test_empty_list(self):
        pool = ShmPool([])
        try:
            # Segment has minimum size 1 for zero arrays
            assert pool._shm.size >= 1
        finally:
            pool.close()
            pool.unlink()

    def test_mixed_dtypes(self):
        arrs = [
            np.zeros((3, 3), dtype=np.float64),
            np.ones((5,), dtype=np.uint8),
            np.full((4, 2), 7, dtype=np.int64),
        ]
        pool = ShmPool(arrs)
        try:
            for i, orig in enumerate(arrs):
                np.testing.assert_array_equal(pool.get(i), orig)
                assert pool.get(i).dtype == orig.dtype
        finally:
            pool.close()
            pool.unlink()

    def test_cache_line_alignment(self, arrays):
        """Each array's offset should be on a 64-byte boundary."""
        pool = ShmPool(arrays)
        try:
            for byte_off, _, _ in pool._meta:
                assert byte_off % ShmPool._ALIGN == 0
        finally:
            pool.close()
            pool.unlink()


class TestShmPoolPickle:
    """Pickle round-trip simulates spawn-worker attach."""

    def test_roundtrip_preserves_data(self, arrays):
        pool = ShmPool(arrays)
        try:
            state = pickle.dumps(pool)
            restored = pickle.loads(state)
            try:
                for i, orig in enumerate(arrays):
                    np.testing.assert_array_equal(restored.get(i), orig)
            finally:
                restored.close()  # non-owner close
        finally:
            pool.close()
            pool.unlink()

    def test_restored_is_not_owner(self, arrays):
        pool = ShmPool(arrays)
        try:
            restored = pickle.loads(pickle.dumps(pool))
            try:
                assert restored._owner is False
            finally:
                restored.close()
        finally:
            pool.close()
            pool.unlink()

    def test_non_owner_unlink_is_noop(self, arrays):
        """unlink() on a restored pool should silently do nothing."""
        pool = ShmPool(arrays)
        try:
            restored = pickle.loads(pickle.dumps(pool))
            try:
                # Should not raise and should not unlink the owner's segment
                restored.unlink()
                # Owner's data is still intact
                np.testing.assert_array_equal(pool.get(0), arrays[0])
            finally:
                restored.close()
        finally:
            pool.close()
            pool.unlink()


class TestShmPoolCleanup:
    def test_close_is_idempotent(self, arrays):
        pool = ShmPool(arrays)
        pool.close()
        # Second close should not raise
        pool.close()
        pool.unlink()

    def test_unlink_is_idempotent(self, arrays):
        pool = ShmPool(arrays)
        pool.close()
        pool.unlink()
        # Second unlink should not raise
        pool.unlink()
