"""Regression test: no top-level import cycles in omnipose.

The check itself lives in :mod:`ocdkit.testing.imports` so downstream
packages can opt in with the same one-liner.
"""
import omnipose
from ocdkit.testing import make_import_cycles_test

test_no_import_cycles = make_import_cycles_test(omnipose)
