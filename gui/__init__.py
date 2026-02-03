"""Omnipose GUI package.

This package provides a web-based and desktop GUI for the Omnipose cell segmentation tool.

Usage:
    # Run as web server
    python -m gui --server

    # Run as desktop application
    python -m gui
"""

from .server import (
    create_app,
    run_server,
    run_desktop,
    main,
)

__all__ = [
    "create_app",
    "run_server",
    "run_desktop",
    "main",
]
