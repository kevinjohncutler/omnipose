import importlib


def _assert_lazy_init(mod):
    # enable_submodules should install these
    assert isinstance(getattr(mod, "__all__", None), list)
    assert callable(getattr(mod, "__getattr__", None))
    assert callable(getattr(mod, "__dir__", None))


def test_package_init_enable_submodules():
    for name in [
        "omnipose",
        "omnipose.core",
        "omnipose.transforms",
        "omnipose.utils",
        "omnipose.io",
        "omnipose.data",
        "omnipose.cli",
        "omnipose.metrics",
        "omnipose.plot",
    ]:
        mod = importlib.import_module(name)
        _assert_lazy_init(mod)


def test_pkg_init_exports():
    mod = importlib.import_module("omnipose.pkg")
    assert hasattr(mod, "enable_submodules")
    assert hasattr(mod, "attach_helpers")
    assert hasattr(mod, "load_submodules")
    assert hasattr(mod, "attach_function_to_object")
