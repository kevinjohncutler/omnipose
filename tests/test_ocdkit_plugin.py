"""Verify the ocdkit.viewer plugin shim is well-formed.

These tests do not run actual segmentation (which requires cellpose_omni and
a model checkpoint); they verify the contract surface only.
"""

from __future__ import annotations

import pytest


def test_plugin_imports_and_validates():
    """Importing omnipose.gui materializes a SegmentationPlugin."""
    pytest.importorskip("ocdkit.viewer")
    from omnipose.gui import plugin
    from ocdkit.viewer import SegmentationPlugin

    assert isinstance(plugin, SegmentationPlugin)
    assert plugin.name == "omnipose"
    assert plugin.version


def test_plugin_widgets_have_no_duplicates():
    pytest.importorskip("ocdkit.viewer")
    from omnipose.gui import plugin
    names = [w.name for w in plugin.widgets]
    assert len(names) == len(set(names)), f"duplicate widgets: {names}"


def test_plugin_manifest_serializable():
    import json
    pytest.importorskip("ocdkit.viewer")
    from omnipose.gui import plugin
    manifest = plugin.manifest()
    json.dumps(manifest)  # must not raise
    assert manifest["name"] == "omnipose"
    # The model selector is rendered by the host from manifest["models"];
    # the plugin no longer declares a `model` widget itself.
    assert "mask_threshold" in [w["name"] for w in manifest["widgets"]]
    assert manifest["models"], "expected at least one built-in model"


def test_plugin_capabilities_advertise_all_hooks():
    pytest.importorskip("ocdkit.viewer")
    from omnipose.gui import plugin
    caps = plugin.manifest()["capabilities"]
    assert caps["resegment"] is True
    assert caps["relabel_from_affinity"] is True
    assert caps["set_use_gpu"] is True
    assert caps["get_use_gpu"] is True
    assert caps["clear_cache"] is True
    assert caps["warmup"] is True


def test_plugin_models_includes_built_ins():
    pytest.importorskip("ocdkit.viewer")
    from omnipose.gui import plugin
    models = plugin.manifest()["models"]
    assert "bact_phase_affinity" in models
    assert "cyto2_omni" in models


def test_segmenter_import_does_not_fail():
    """The lifted Segmenter should at least import (omnipose runtime not required)."""
    from omnipose.gui._segmenter import Segmenter
    assert Segmenter is not None


def test_entry_point_resolves(tmp_path):
    """If omnipose was installed (entry point registered), the discovery
    via importlib.metadata should find our plugin."""
    pytest.importorskip("ocdkit.viewer")
    from importlib import metadata
    try:
        eps = metadata.entry_points(group="ocdkit.plugins")
    except TypeError:
        eps = metadata.entry_points().get("ocdkit.plugins", [])
    names = [ep.name for ep in eps]
    if "omnipose" not in names:
        pytest.skip(
            "entry point not yet registered (run `pip install -e .` to enable)"
        )
    omnipose_ep = next(ep for ep in eps if ep.name == "omnipose")
    plugin = omnipose_ep.load()
    assert plugin.name == "omnipose"
