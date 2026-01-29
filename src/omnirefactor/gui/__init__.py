try:
    from .app import main
except Exception:  # pragma: no cover
    def main(*_args, **_kwargs):
        raise ImportError("GUI dependencies are not installed.")

__all__ = ["main"]
