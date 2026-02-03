"""Command-line interface for the Omnipose GUI server."""

from __future__ import annotations

import argparse


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Omnipose PyWebView Viewer")
    parser.add_argument(
        "--server",
        action="store_true",
        help="Launch as an HTTPS-capable FastAPI server instead of a desktop window.",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Server host when using --server (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port when using --server (default: 8000)",
    )
    parser.add_argument(
        "--ssl-cert",
        default=None,
        help="Path to SSL certificate for HTTPS server mode",
    )
    parser.add_argument(
        "--ssl-key",
        default=None,
        help="Path to SSL private key for HTTPS server mode",
    )
    parser.add_argument(
        "--reload",
        dest="reload",
        action="store_true",
        default=False,
        help="Enable uvicorn auto-reload for web server mode.",
    )
    parser.add_argument(
        "--no-reload",
        dest="reload",
        action="store_false",
        help="Disable uvicorn auto-reload for web server mode.",
    )
    parser.add_argument(
        "--https-dev",
        action="store_true",
        help="Serve over HTTPS using a temporary self-signed localhost certificate (requires openssl).",
    )
    parser.add_argument(
        "--desktop-host",
        default="127.0.0.1",
        help="Host interface for the embedded desktop server (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--desktop-port",
        type=int,
        default=0,
        help="Port for the embedded desktop server (default: auto)",
    )
    parser.add_argument(
        "--desktop-reload",
        dest="desktop_reload",
        action="store_true",
        default=False,
        help="Enable uvicorn --reload for the embedded desktop server (development only).",
    )
    parser.add_argument(
        "--no-desktop-reload",
        dest="desktop_reload",
        action="store_false",
        help="Disable uvicorn --reload for the embedded desktop server.",
    )
    parser.add_argument(
        "--snapshot",
        metavar="PNG_PATH",
        help="Capture the viewer canvas to the given PNG file and exit.",
    )
    parser.add_argument(
        "--snapshot-timeout",
        type=float,
        default=4.0,
        help="Seconds to wait for the first draw before giving up on --snapshot (default: 4).",
    )
    parser.add_argument(
        "--eval-js",
        dest="eval_js",
        default=None,
        help="JavaScript snippet to evaluate after the viewer loads (testing/automation).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Main entry point for the Omnipose GUI."""
    args = parse_args(argv)

    # Import here to avoid circular imports and speed up --help
    from .app import run_server, run_desktop

    if args.server:
        run_server(
            host=args.host,
            port=args.port,
            ssl_cert=args.ssl_cert,
            ssl_key=args.ssl_key,
            reload=args.reload,
            https_dev=args.https_dev,
        )
    else:
        run_desktop(
            host=args.desktop_host,
            port=args.desktop_port if args.desktop_port and args.desktop_port > 0 else None,
            ssl_cert=args.ssl_cert,
            ssl_key=args.ssl_key,
            reload=args.desktop_reload,
            snapshot_path=args.snapshot,
            snapshot_timeout=args.snapshot_timeout,
            eval_js=args.eval_js,
        )
