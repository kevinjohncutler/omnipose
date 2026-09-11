"""Regression test: no submodule shadows a same-named public callable.

The check itself lives in :mod:`ocdkit.testing.collisions`; this file is
just the per-package opt-in.
"""
import omnipose
from ocdkit.testing import make_module_collision_test

test_no_module_callable_collisions = make_module_collision_test(omnipose)
