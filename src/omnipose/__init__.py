"""Omnipose core package."""

__version__ = "2.0.0"

from .pkg import enable_submodules

# Top-level: lazy. Sub-packages (omnipose.core, omnipose.plot, …) load only
# when accessed, keeping bare ``import omnipose`` near-instant.
enable_submodules(__name__, expose=False)
