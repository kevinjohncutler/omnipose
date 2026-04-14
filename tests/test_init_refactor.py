import importlib


def _assert_lazy_init(mod):
    # enable_submodules should install these
    assert isinstance(getattr(mod, "__all__", None), list)
    assert callable(getattr(mod, "__getattr__", None))
    assert callable(getattr(mod, "__dir__", None))


def test_package_init_enable_submodules():
    for name in [
        "omnirefactor",
        "omnirefactor.core",
        "omnirefactor.transforms",
        "omnirefactor.utils",
        "omnirefactor.io",
        "omnirefactor.data",
        "omnirefactor.cli",
        "omnirefactor.metrics",
        "omnirefactor.plot",
    ]:
        mod = importlib.import_module(name)
        _assert_lazy_init(mod)


def test_load_init_exports():
    mod = importlib.import_module("omnirefactor.load")
    exported = set(mod.__all__)
    assert "enable_submodules" in exported
    assert "attach_helpers" in exported
    assert "load_submodules" in exported
    assert "attach_function_to_object" in exported
