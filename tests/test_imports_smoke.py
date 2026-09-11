def test_smoke_imports():
    """Verify core subpackages import without error."""
    import omnipose
    import omnipose.cli
    import omnipose.data
    import omnipose.models
    import omnipose.transforms
    import omnipose.io
