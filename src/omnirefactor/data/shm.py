"""Shared-memory array pool for zero-copy DataLoader workers."""

import numpy as np
from multiprocessing.shared_memory import SharedMemory


class ShmPool:
    """Pack a list of numpy arrays into a single POSIX shared memory segment.

    Uses only **one** file descriptor for the entire list, regardless of how
    many arrays it holds. With 284 training images this reduces open-fd
    usage from 568 to 2 (one pool for data, one for labels).

    Pickle-efficient: only the shm name + offset table is serialized.
    Spawn workers attach to the existing segment by name — zero copies.

    Usage::

        pool = ShmPool(list_of_arrays)   # main process: create + copy
        arr = pool.get(i)                # worker: zero-copy view
        pool.close(); pool.unlink()      # main process: cleanup
    """

    _ALIGN = 64  # cache-line alignment

    def __init__(self, arrays):
        meta, offset = [], 0
        for a in arrays:
            meta.append((offset, a.shape, a.dtype.str))
            offset += a.nbytes
            offset = (offset + self._ALIGN - 1) // self._ALIGN * self._ALIGN

        total = max(offset, 1)
        self._shm = SharedMemory(create=True, size=total)
        self._name = self._shm.name
        self._owner = True

        for (byte_off, shape, dtype_str), a in zip(meta, arrays):
            dst = np.ndarray(shape, dtype=np.dtype(dtype_str),
                             buffer=self._shm.buf, offset=byte_off)
            dst[:] = a

        self._meta = meta

    def __getstate__(self):
        return {'name': self._name, 'meta': self._meta}

    def __setstate__(self, state):
        self._name = state['name']
        self._meta = state['meta']
        import multiprocessing.resource_tracker as _rt_mod
        _orig_register = _rt_mod.register
        _rt_mod.register = lambda *a, **kw: None
        try:
            self._shm = SharedMemory(name=self._name, create=False)
        finally:
            _rt_mod.register = _orig_register
        self._owner = False

    def get(self, i) -> np.ndarray:
        """Return a zero-copy numpy view of the i-th array."""
        byte_off, shape, dtype_str = self._meta[i]
        return np.ndarray(shape, dtype=np.dtype(dtype_str),
                          buffer=self._shm.buf, offset=byte_off)

    def close(self):
        try:
            self._shm.close()
        except Exception:
            pass

    def unlink(self):
        if getattr(self, '_owner', False):
            try:
                self._shm.unlink()
            except Exception:
                pass
