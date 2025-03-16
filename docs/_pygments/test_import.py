def test_import():
    try:
        import omnipose
    except ImportError as e:
        print(f"ImportError: {e}")