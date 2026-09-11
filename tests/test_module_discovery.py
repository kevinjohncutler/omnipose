"""Regression test: ``pkgutil.walk_packages`` discovers every omnipose
module without silently swallowing import errors.

The check itself lives in :mod:`ocdkit.testing.imports` so downstream
packages can opt in with the same one-liner.
"""
import omnipose
from ocdkit.testing import make_no_silent_discovery_test

test_no_silent_discovery_errors = make_no_silent_discovery_test(omnipose)
