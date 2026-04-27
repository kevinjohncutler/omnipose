"""omnipose.gui — ocdkit.viewer plugin for Omnipose / omnipose.

Hosts the ocdkit plugin shim and an in-package Segmenter wrapper. The full
GUI machinery lives in ``ocdkit.viewer`` (sibling package); ``ocdkit_plugin.py``
is the adapter that exposes Omnipose segmentation through the ocdkit plugin
contract.

Public entry point:
    main(argv) — launches ``ocdkit.viewer`` with the omnipose plugin
    auto-selected. Used by ``python -m omnipose`` (no args) and the
    ``omnipose`` console script.

Old GUI flag names from the standalone gui server are accepted and translated
into ``ocdkit.viewer.run_server`` / ``run_desktop`` calls, so existing scripts
keep working.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

from .ocdkit_plugin import plugin

__all__ = ["plugin", "main"]


def _omnipose_bundled_test_files_dir() -> Optional[Path]:
    """Locate the omnipose repo's bundled ``docs/test_files/`` directory.

    This is the canonical source of sample images — the repo ships them as
    part of its notebook + GUI examples, and that's the same set the original
    ``cellpose_omni.gui`` downloaded on first run. Walks up from this file
    looking for the directory; works whether omnipose is installed
    editable from its sub-repo or alongside an omnipose clone.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "docs" / "test_files"
        if candidate.is_dir():
            return candidate
        # When omnipose is installed from omnipose/omnipose, docs/
        # lives one level above the omnipose checkout root.
        candidate = parent.parent / "docs" / "test_files"
        if candidate.is_dir():
            return candidate
    return None


def _find_default_sample() -> Optional[Path]:
    """Return the bundled default sample image, if present.

    User overrides via ``OCDKIT_VIEWER_SAMPLE_IMAGE`` (or the legacy
    ``OMNIPOSE_SAMPLE_IMAGE``) env var are handled by the caller — this
    function only picks the baked-in default.
    """
    bundled_dir = _omnipose_bundled_test_files_dir()
    if bundled_dir is None:
        return None
    bundled_default = bundled_dir / "e1t1_crop.tif"
    return bundled_default if bundled_default.is_file() else None


def _apply_default_sample_env() -> None:
    """Seed the viewer's sample-image env var if neither override is set."""
    if os.environ.get("OCDKIT_VIEWER_SAMPLE_IMAGE"):
        return
    if os.environ.get("OMNIPOSE_SAMPLE_IMAGE"):
        return
    sample = _find_default_sample()
    if sample is not None:
        os.environ["OCDKIT_VIEWER_SAMPLE_IMAGE"] = str(sample)
        print(f"[omnipose-gui] default sample: {sample}", flush=True)


def _bundled_icon_path() -> Optional[Path]:
    """Locate the Omnipose icon shipped inside this package."""
    candidate = Path(__file__).resolve().parent / "icon.png"
    return candidate if candidate.is_file() else None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omnipose (gui)",
        description="Launch the ocdkit viewer with the Omnipose plugin pre-selected.",
        # Allow unknown flags to be ignored — keeps compatibility with any
        # forwarded CLI invocations that use legacy flags.
        allow_abbrev=False,
    )
    parser.add_argument(
        "--server",
        action="store_true",
        help="Run as an HTTP/HTTPS web server instead of a desktop window.",
    )
    parser.add_argument("--host", default=None,
                        help="Host interface (default: 127.0.0.1 desktop / 0.0.0.0 server).")
    parser.add_argument("--port", type=int, default=None,
                        help="Port (default: 8765 server / auto desktop).")
    parser.add_argument("--ssl-cert", default=None)
    parser.add_argument("--ssl-key", default=None)
    parser.add_argument("--https-dev", action="store_true",
                        help="Provision a temp self-signed localhost cert (server mode).")
    parser.add_argument(
        "--https-auto",
        nargs="?", const=True, default=False, metavar="HOSTS",
        help="Request a trusted cert via ocdkit.tls (server mode). Optional "
             "comma-separated hostnames override; default uses the machine's hostname.",
    )
    parser.add_argument("--reload", action="store_true",
                        help="Enable uvicorn auto-reload.")
    parser.add_argument("--no-reload", dest="reload", action="store_false")
    # Default ON, matching the ocdkit viewer CLI. Edits to plugin sources
    # propagate without manual restarts.
    parser.set_defaults(reload=True)
    # Desktop-specific flags (legacy spellings preserved for compat).
    parser.add_argument("--desktop-host", default=None,
                        help="(deprecated) alias for --host in desktop mode.")
    parser.add_argument("--desktop-port", type=int, default=None,
                        help="(deprecated) alias for --port in desktop mode.")
    parser.add_argument("--desktop-reload", dest="reload", action="store_true")
    parser.add_argument("--no-desktop-reload", dest="reload", action="store_false")
    parser.add_argument("--snapshot", default=None,
                        help="Capture canvas to PNG and exit (desktop mode).")
    parser.add_argument("--snapshot-timeout", type=float, default=4.0)
    parser.add_argument("--eval-js", dest="eval_js", default=None)
    parser.add_argument(
        "--no-omnipose",
        action="store_true",
        help="Skip auto-selecting the omnipose plugin (use threshold default).",
    )
    parser.add_argument(
        "--title",
        default="Omnipose",
        help="Window/tab/docs title (default: 'Omnipose'). "
        "Pass empty string to fall back to the ocdkit default.",
    )
    parser.add_argument(
        "--icon",
        default=None,
        help="Path to a PNG to use as the window/dock icon. "
        "Defaults to the omnipose icon bundled with this package.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Launch the ocdkit viewer with the Omnipose plugin selected."""
    args = _build_parser().parse_args(list(argv) if argv is not None else None)

    # Seed the omnipose default sample image (BEFORE importing ocdkit.viewer
    # so its sample_image module sees the env var on first import).
    _apply_default_sample_env()

    try:
        from ocdkit.viewer.app import _autoload_plugins, run_desktop, run_server
        from ocdkit.viewer.plugins.registry import REGISTRY
        from ocdkit.viewer.segmentation import ACTIVE_PLUGIN
    except ImportError as exc:
        print(
            "ocdkit is required to launch the viewer. "
            "Install with: pip install ocdkit",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    # Make sure our plugin is registered (entry-point discovery + always-on).
    _autoload_plugins()
    if plugin.name not in REGISTRY:
        REGISTRY.register(plugin)

    if not args.no_omnipose:
        try:
            ACTIVE_PLUGIN.select(plugin.name)
        except KeyError:
            print(f"[omnipose] could not auto-select plugin {plugin.name!r}",
                  file=sys.stderr)

    title = args.title or None  # empty string falls back to ocdkit default
    icon = args.icon or (str(_bundled_icon_path()) if _bundled_icon_path() else None)
    if args.server:
        https_auto = args.https_auto
        if isinstance(https_auto, str):
            https_auto = [h.strip() for h in https_auto.split(",") if h.strip()]
        run_server(
            host=args.host or "0.0.0.0",
            port=args.port or 8765,
            ssl_cert=args.ssl_cert,
            ssl_key=args.ssl_key,
            reload=args.reload,
            https_dev=args.https_dev,
            https_auto=https_auto,
            title=title,
        )
        return

    # Desktop mode: pass our own AppIdentity so pinning.py's setup_platform
    # creates ``~/Applications/Omnipose.app`` with the bundled squircle icon.
    # LaunchServices then picks that up for dock rendering, giving the proper
    # macOS rounded-rectangle app-icon presentation rather than the flat
    # ``setApplicationIconImage_`` fallback (which doesn't get the squircle).
    try:
        from ocdkit.desktop.pinning import AppIdentity
    except ImportError:
        AppIdentity = None

    app_identity = None
    if AppIdentity is not None:
        app_identity = AppIdentity(
            name="Omnipose",
            gui_entry_point="omnipose-gui",
            windows_app_id="Omnipose.Viewer.Launcher",
            linux_app_id="omnipose",
            macos_bundle_id="com.omnipose.viewer",
            description="Omnipose segmentation viewer",
            categories="Science;Graphics",
            icon_png=icon,
        )

    host = args.host or args.desktop_host or "127.0.0.1"
    port = args.port or args.desktop_port or 0
    run_desktop(
        host=host,
        port=port if port and port > 0 else None,
        ssl_cert=args.ssl_cert,
        ssl_key=args.ssl_key,
        reload=args.reload,
        snapshot_path=args.snapshot,
        snapshot_timeout=args.snapshot_timeout,
        eval_js=args.eval_js,
        title=title,
        icon=icon,
        app_identity=app_identity,
    )
