"""omnipose.logger — re-exported from ocdkit."""

from ocdkit.logging import (
    get_logger, set_color, set_colors, silence, TqdmToLogger,
)

# Backwards compat alias
set_logger_color = set_color
setup_logger = get_logger

# Silence noisy third-party loggers on import.
silence("xmlschema", "bfio", "OpenGL", "qdarktheme", "mip")

# Register omnipose module colors.
set_colors({
    "core": "#5c9edc",
    "models": "#5cd97c",
    "__main__": "#fff44f",
    "io": "#ff7f0e",
    "gpu": "#ff0055",
})
