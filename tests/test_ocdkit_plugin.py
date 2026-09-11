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


def test_plugin_exposes_area_and_rescale_sliders():
    """The cell-area cutoff and pre-inference rescale knobs are user-facing."""
    pytest.importorskip("ocdkit.viewer")
    from omnipose.gui import plugin
    specs = {w.name: w for w in plugin.widgets}

    area = specs["min_size"]
    assert area.kind == "slider_log"
    assert area.min == 1 and area.max > area.min
    assert area.default == 15

    rescale = specs["rescale_factor"]
    assert rescale.kind == "slider_log"
    assert rescale.min < 1.0 < rescale.max
    assert rescale.default == 1.0


def test_parse_options_defaults_and_coercion():
    from omnipose.gui._segmenter import Segmenter

    seg = Segmenter.__new__(Segmenter)  # no model / GPU probe needed
    parsed, _ = seg._parse_options(None, None)
    assert parsed["min_size"] == 15
    assert parsed["rescale_factor"] == 1.0

    parsed, _ = seg._parse_options({"min_size": "250", "rescale_factor": "0.5"}, None)
    assert parsed["min_size"] == 250
    assert parsed["rescale_factor"] == 0.5

    # Junk and non-positive rescale factors fall back to the default rather
    # than propagating a zero/negative zoom into eval.
    parsed, _ = seg._parse_options({"min_size": -5, "rescale_factor": 0}, None)
    assert parsed["min_size"] == 0
    assert parsed["rescale_factor"] == 1.0


def test_inference_key_tracks_only_network_options():
    """Reconstruction knobs must not invalidate the cached network output."""
    from omnipose.gui._segmenter import Segmenter

    seg = Segmenter.__new__(Segmenter)
    base, _ = seg._parse_options({"model": "bact_phase_affinity"}, None)
    key = seg._inference_key(base)

    for name, value in (("min_size", 500), ("mask_threshold", 1.0),
                        ("flow_threshold", 2.0), ("cluster", False)):
        other, _ = seg._parse_options({"model": "bact_phase_affinity", name: value}, None)
        assert seg._inference_key(other) == key, f"{name} should stay interactive"

    for name, value in (("rescale_factor", 2.0), ("resample", False),
                        ("tile", True), ("augment", True), ("model", "cyto2_omni")):
        other, _ = seg._parse_options({"model": "bact_phase_affinity", name: value}, None)
        assert seg._inference_key(other) != key, f"{name} must force re-inference"


@pytest.mark.parametrize("rescale", [0.1, 0.15, 0.35, 0.5, 2.0, 4.0, 10.0])
def test_rescale_preserves_image_shape(rescale):
    """A rescaled run must return a mask the viewer can overlay on the image.

    The forward rescale truncates with int(), so resampling back by
    1/rescale_factor lands short (384*0.15 -> 57 -> 380) and the mask no
    longer matches the displayed image.
    """
    np = pytest.importorskip("numpy")
    pytest.importorskip("torch")
    from omnipose.data.eval import eval_set as EvalSet

    shape = (384, 392)
    img = np.zeros((1,) + shape, dtype=np.float32)
    ds = EvalSet([img], dim=2, normalize=False, invert=False,
                 rescale_factor=rescale, channel_axis=1)
    assert tuple(ds._native_shapes[0]) == shape
