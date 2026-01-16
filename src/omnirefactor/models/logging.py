import logging

from ..logger import TqdmToLogger

models_logger = logging.getLogger(__name__)
core_logger = logging.getLogger(__name__)
tqdm_out = TqdmToLogger(core_logger, level=logging.INFO)

__all__ = ["models_logger", "core_logger", "tqdm_out"]
