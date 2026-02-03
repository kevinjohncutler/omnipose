"""Omnipose GUI server package.

This package provides a FastAPI-based web server and PyWebView desktop application
for the Omnipose cell segmentation tool.

Usage:
    # Run as web server
    python -m gui --server

    # Run as desktop application
    python -m gui

    # Or from Python
    from gui.server import run_server, run_desktop
    run_server()
"""

from .app import create_app, run_server, run_desktop
from .session import SessionManager, SessionState, SESSION_MANAGER
from .segmentation import Segmenter, _SEGMENTER, run_segmentation, run_mask_update
from .assets import (
    GUI_DIR,
    WEB_DIR,
    build_html,
    render_index,
)
from .cli import main, parse_args

__all__ = [
    # App and launchers
    "create_app",
    "run_server",
    "run_desktop",
    # Session management
    "SessionManager",
    "SessionState",
    "SESSION_MANAGER",
    # Segmentation
    "Segmenter",
    "_SEGMENTER",
    "run_segmentation",
    "run_mask_update",
    # Assets
    "GUI_DIR",
    "WEB_DIR",
    "build_html",
    "render_index",
    # CLI
    "main",
    "parse_args",
]
