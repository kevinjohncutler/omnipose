from .imports import *


def none_or_str(value):
    """Custom argparse type to accept either a string or None."""
    if value.lower() == "none":
        return None
    return value
