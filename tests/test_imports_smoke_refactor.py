def test_smoke_imports():
    """Verify core subpackages import without error."""
    import omnirefactor
    import omnirefactor.cli
    import omnirefactor.data
    import omnirefactor.models
    import omnirefactor.transforms
    import omnirefactor.io
