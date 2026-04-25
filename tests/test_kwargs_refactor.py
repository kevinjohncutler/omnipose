import numpy as np
import pytest

from omnipose import kwargs as kw


def test_base_kwargs_exclude_and_kwargs_merge():
    local_vars = {"a": 1, "b": 2, "kwargs": {"c": 3}}
    out = kw.base_kwargs(local_vars, exclude={"b"})
    assert out == {"a": 1, "kwargs": {"c": 3}, "c": 3}


def test_split_kwargs_strict_unknown():
    def f(a):
        return a

    with pytest.raises(TypeError):
        kw.split_kwargs([f], {"a": 1, "b": 2}, strict=True)


def test_split_kwargs_catchall_and_defaults():
    def f(a, b=5, **kwargs):
        return a, b, kwargs

    def g(c=7):
        return c

    out = kw.split_kwargs([f, g], {"a": 1, "x": 9})
    f_kwargs, g_kwargs = out
    assert f_kwargs["a"] == 1
    assert f_kwargs["b"] == 5
    assert f_kwargs["x"] == 9
    assert g_kwargs["c"] == 7


def test_split_kwargs_shared_param_to_multiple_funcs():
    def f(a, b=1):
        return a + b

    def g(a, c=2):
        return a + c

    out = kw.split_kwargs([f, g], {"a": 3})
    f_kwargs, g_kwargs = out
    assert f_kwargs["a"] == 3
    assert g_kwargs["a"] == 3


def test_listify_args_auto_and_named():
    @kw.listify_args
    def f(channels, rounds):
        return channels, rounds

    @kw.listify_args("foo")
    def g(foo, bar):
        return foo, bar

    ch, rd = f(np.array([1, 2]), 3)
    assert ch == [1, 2]
    assert rd == [3]

    foo, bar = g(5, "x")
    assert foo == [5]
    assert bar == "x"

    assert kw._to_list([1, 2]) == [1, 2]
