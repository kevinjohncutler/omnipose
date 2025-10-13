import os
import warnings
import tempfile
import shutil

# Try to isolate Numba's cache location for pytest runs. Note: Numba's
# function-level cache writes to module-local __pycache__ directories when
# cache=True, and NUMBA_CACHE_DIR may be ignored in that mode. We still set it
# to keep consistency for any code paths that respect it.
_NUMBA_TMP_DIR = None
if "NUMBA_CACHE_DIR" not in os.environ or not os.environ.get("NUMBA_CACHE_DIR"):
    _NUMBA_TMP_DIR = tempfile.mkdtemp(prefix="omnipose_numba_cache_")
    os.environ["NUMBA_CACHE_DIR"] = _NUMBA_TMP_DIR

# Quiet noisy networkit deprecations originating from its profiling helpers
# The emission site for these comes from IPython.core.display when imported
# by networkit.profiling; filter them by the originating module name.
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r"IPython\.core\.display",
)


def pytest_sessionfinish(session, exitstatus):
    global _NUMBA_TMP_DIR
    if _NUMBA_TMP_DIR:
        shutil.rmtree(_NUMBA_TMP_DIR, ignore_errors=True)
        _NUMBA_TMP_DIR = None


# As a robust test-only workaround for external-volume EBUSY races when
# writing __pycache__ entries, recompile the specific hot function with
# cache=False during test sessions. This keeps the library code clean
# (cache=True) while making tests reliable.
def pytest_configure(config):
    try:
        from numba import njit
        import omnipose.core as oc
        # If already a numba dispatcher, get the underlying Python function
        py_func = getattr(oc._get_link_matrix, 'py_func', oc._get_link_matrix)
        oc._get_link_matrix = njit(cache=False, fastmath=True)(py_func)
    except Exception:
        # Best-effort; tests that don't hit this code path won't care
        pass
