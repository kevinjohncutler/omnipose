import logging

from ..logger import TqdmToLogger

models_logger = logging.getLogger(__name__)
core_logger = logging.getLogger(__name__)
tqdm_out = TqdmToLogger(core_logger, level=logging.INFO)
