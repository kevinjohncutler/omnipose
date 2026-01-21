from __future__ import annotations

import sys

from .cli.runner import main as cli_main
from .gui import main as gui_main

GUI_FLAGS = {
    "--server",
    "--host",
    "--port",
    "--ssl-cert",
    "--ssl-key",
    "--https-dev",
    "--desktop-host",
    "--desktop-port",
    "--desktop-reload",
    "--no-desktop-reload",
    "--snapshot",
    "--snapshot-timeout",
    "--eval-js",
}

CLI_FLAGS = {"--dir", "--train", "--train_size"}


def _wants_gui(argv: list[str]) -> bool:
    if not argv:
        return True
    if any(flag in argv for flag in GUI_FLAGS):
        return True
    if any(flag in argv for flag in CLI_FLAGS):
        return False
    return True


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if _wants_gui(args):
        gui_main(args)
    else:
        cli_main(args)


if __name__ == "__main__":
    main()
