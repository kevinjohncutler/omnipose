import ntpath # allows us to simulate windows handling even if not on windows, os.path uses this under the hood on windows

import cellpose_omni.io as io


def _set_platform(monkeypatch, name):
    """Force adjust_file_path to think it is running on a specific OS."""
    monkeypatch.setattr(io.platform, "system", lambda: name)


def test_adjust_file_path_macos_rewrites_home(monkeypatch):
    _set_platform(monkeypatch, "Darwin")
    assert io.adjust_file_path("/home/alice/project") == "/Volumes/project"


def test_adjust_file_path_linux_rewrites_volumes(monkeypatch):
    _set_platform(monkeypatch, "Linux")
    home_dir = "/home/tester"
    monkeypatch.setattr(io.os.path, "expanduser", lambda _: home_dir)
    adjusted = io.adjust_file_path("/Volumes/datasets/run1")
    assert adjusted == f"{home_dir}/datasets/run1"


def _setup_windows(monkeypatch, home_dir=None):
    _set_platform(monkeypatch, "Windows")
    home_dir = home_dir or r"C:\\Users\\Tester"
    monkeypatch.setattr(io.os.path, "expanduser", lambda _: home_dir)
    win_normpath = ntpath.normpath
    # Use the real ntpath.normpath instead of a lambda to avoid recursive re-patching.
    monkeypatch.setattr(io.os.path, "normpath", win_normpath)
    return home_dir


def test_adjust_file_path_windows_from_home(monkeypatch):
    home_dir = _setup_windows(monkeypatch)
    adjusted = io.adjust_file_path("/home/alice/project/data.txt")
    expected = ntpath.normpath(rf"{home_dir}/project/data.txt")
    assert adjusted == expected


def test_adjust_file_path_windows_from_volumes(monkeypatch):
    home_dir = _setup_windows(monkeypatch)
    adjusted = io.adjust_file_path("/Volumes/share/results")
    expected = ntpath.normpath(rf"{home_dir}/share/results")
    assert adjusted == expected


def test_adjust_file_path_windows_home_with_spaces(monkeypatch):
    home_dir = r"C:\\Users\\Primary User\\OneDrive - Example Company"
    home_dir = ntpath.normpath(home_dir)
    home_dir = _setup_windows(monkeypatch, home_dir=home_dir)
    adjusted = io.adjust_file_path("/home/alice/OneDrive - Example Company/data")
    expected = ntpath.normpath(rf"{home_dir}/OneDrive - Example Company/data")
    assert adjusted == expected
