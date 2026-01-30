"""Logging setup utilities for file-based logging.

This module provides logger_setup() for creating file-based loggers
that write to ~/.omnipose/run.log.

Note: For console logging with colors, use the logger module instead:
    from omnirefactor.logger import get_logger
"""

import logging
import pathlib
import sys


def logger_setup(verbose=False):
    """
    Set up a logger that writes to ~/.omnipose/run.log.

    Parameters:
        verbose: If True, set log level to DEBUG; otherwise INFO

    Returns:
        tuple: (logger, log_file_path)
    """
    omni_dir = pathlib.Path.home().joinpath('.omnipose')
    omni_dir.mkdir(exist_ok=True)
    log_file = omni_dir.joinpath('run.log')
    try:
        log_file.unlink()
    except Exception:
        print('creating new log file')
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)

    return logger, log_file
