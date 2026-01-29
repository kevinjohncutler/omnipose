import pathlib

from omnirefactor.io import logging as log_mod


def test_logger_setup_creates_logfile(tmp_path, monkeypatch):
    monkeypatch.setattr(pathlib.Path, "home", lambda: tmp_path)
    logger, log_file = log_mod.logger_setup(verbose=False)
    assert log_file.parent.exists()
    assert log_file.name == "run.log"
    assert log_file.exists()
    assert logger.name == log_mod.__name__


def test_logger_setup_handles_unlink_failure(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pathlib.Path, "home", lambda: tmp_path)
    log_dir = tmp_path / ".omnipose"
    log_dir.mkdir()
    # Make a directory where the log file would be so unlink fails.
    bad_log = log_dir / "run.log"
    bad_log.mkdir()

    monkeypatch.setattr(log_mod.logging, "FileHandler", lambda *_args, **_kwargs: log_mod.logging.NullHandler())
    _logger, log_file = log_mod.logger_setup(verbose=False)
    captured = capsys.readouterr()
    assert "creating new log file" in captured.out
    assert log_file.name == "run.log"
