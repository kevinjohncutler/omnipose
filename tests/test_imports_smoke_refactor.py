def test_smoke_imports():
    import omnirefactor
    import omnirefactor.cli
    import omnirefactor.data
    import omnirefactor.models
    import omnirefactor.transforms
    import omnirefactor.io

    assert omnirefactor is not None
