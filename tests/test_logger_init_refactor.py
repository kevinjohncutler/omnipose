import logging

from omnirefactor import logger as logger_mod


def test_hex_to_ansi_and_replace_url():
    ansi = logger_mod.hex_to_ansi("#ffffff")
    assert ansi.startswith("\033[38;5;")

    msg = "open file://tmp/test.txt now"
    out = logger_mod.replace_url(msg)
    assert "file://" not in out


def test_get_logger_and_color():
    lg = logger_mod.get_logger("test_logger", color="#00ff00")
    assert isinstance(lg, logging.Logger)


def test_colored_formatter_format():
    fmt = logger_mod.ColoredFormatter()
    record = logging.LogRecord(
        name="core",
        level=logging.INFO,
        pathname="/tmp/core.py",
        lineno=10,
        msg="hello",
        args=(),
        exc_info=None,
        func="func",
    )
    out = fmt.format(record)
    assert "hello" in out


def test_tqdm_to_logger_flush():
    lg = logger_mod.get_logger("tqdm_test")
    handler = logging.StreamHandler()
    lg.addHandler(handler)
    lg.setLevel(logging.INFO)

    adapter = logger_mod.TqdmToLogger(lg)
    adapter.write("progress")
    adapter.flush()
