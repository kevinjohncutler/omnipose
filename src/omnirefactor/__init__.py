"""Refactored Omnipose core package (WIP)."""

__version__ = "0.0.0"

# One-time silent migration of the legacy ``~/.omnipose/`` dotfolder into
# the platformdirs user-data location (``~/Library/Application Support/
# omnipose/`` on macOS, ``$XDG_DATA_HOME/omnipose/`` on Linux, etc.). After
# the first successful run this costs exactly one ``stat()`` per import.
# Wrapped defensively so a misbehaving filesystem can never break the import.
try:
    from ocdkit.utils.paths import migrate_legacy_dotfolder as _mlf
    _mlf("omnipose")
except Exception:  # pragma: no cover
    pass

from .pkg import enable_submodules

enable_submodules(__name__)
