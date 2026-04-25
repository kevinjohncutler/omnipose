import logging

from omnipose import logger as logger_mod


def test_get_logger_and_color():
    lg = logger_mod.get_logger("test_logger", color="#00ff00")
    assert isinstance(lg, logging.Logger)


def test_tqdm_to_logger_flush():
    lg = logger_mod.get_logger("tqdm_test")
    handler = logging.StreamHandler()
    lg.addHandler(handler)
    lg.setLevel(logging.INFO)

    adapter = logger_mod.TqdmToLogger(lg)
    adapter.write("progress")
    adapter.flush()
