import runpy
import sys

import omnirefactor.__main__ as main_mod


def test_wants_gui_dispatch():
    assert main_mod._wants_gui([]) is True
    assert main_mod._wants_gui(["--server"]) is True
    assert main_mod._wants_gui(["--dir", "data"]) is False
    assert main_mod._wants_gui(["--train"]) is False
    assert main_mod._wants_gui(["--train_size"]) is False


def test_main_calls_cli_or_gui(monkeypatch):
    calls = []

    def fake_gui(args):
        calls.append(("gui", list(args)))

    def fake_cli(args):
        calls.append(("cli", list(args)))

    monkeypatch.setattr(main_mod, "gui_main", fake_gui)
    monkeypatch.setattr(main_mod, "cli_main", fake_cli)

    main_mod.main(["--dir", "data"])
    main_mod.main(["--server", "--port", "8000"])
    main_mod.main([])

    assert calls[0][0] == "cli"
    assert calls[1][0] == "gui"
    assert calls[2][0] == "gui"


def test_module_entrypoint_executes(monkeypatch):
    calls = []

    def fake_cli(args):
        calls.append(("cli", list(args)))

    import omnirefactor.cli.runner

    monkeypatch.setattr(omnirefactor.cli.runner, "main", fake_cli)

    old_argv = sys.argv
    try:
        sys.modules.pop("omnirefactor.__main__", None)
        sys.argv = ["-m", "--dir", "data"]
        runpy.run_module("omnirefactor.__main__", run_name="__main__")
    finally:
        sys.argv = old_argv

    assert calls[0][0] == "cli"
