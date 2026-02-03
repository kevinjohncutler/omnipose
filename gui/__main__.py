"""Entry point for running the Omnipose GUI as a module.

Usage:
    python -m gui [OPTIONS]

Options:
    --server        Run as a web server instead of desktop app
    --host HOST     Server host (default: 0.0.0.0)
    --port PORT     Server port (default: 8000)
    --reload        Enable auto-reload for development
    --help          Show all available options
"""

from gui.server.cli import main

if __name__ == "__main__":
    main()
