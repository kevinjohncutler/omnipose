"""ocdkit.viewer plugin for Omnipose / omnipose.

Wraps the in-package :class:`omnipose.gui._segmenter.Segmenter` and
exposes it through the generic :class:`ocdkit.viewer.SegmentationPlugin`
contract. Discovered via the ``ocdkit.plugins`` entry point in
``omnipose/setup.py``::

    entry_points={
        ...
        "ocdkit.plugins": ["omnipose = omnipose.gui:plugin"],
    }

A user with both ``omnipose`` and ``ocdkit`` installed can launch the
viewer with ``python -m ocdkit.viewer serve`` and the Omnipose pane appears
automatically.

Design note: the omnipose package itself is unmodified. The Segmenter
*source* lives in ``_segmenter.py`` next to this plugin; it imports
``cellpose_omni`` and ``omnipose`` only as runtime dependencies.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ocdkit.viewer import SegmentationPlugin, WidgetSpec


# Lazy module-level singleton — instantiating the Segmenter triggers heavy
# imports (torch, cellpose_omni). We defer until first run() / warmup.
_segmenter = None


def _get_segmenter():
    global _segmenter
    if _segmenter is None:
        from ._segmenter import Segmenter
        _segmenter = Segmenter()
    return _segmenter


# ---------------------------------------------------------------------------
# Widget spec — names map 1:1 to keys consumed by Segmenter._parse_options().
# ---------------------------------------------------------------------------

WIDGETS: list[WidgetSpec] = [
    # --- Model -------------------------------------------------------------
    WidgetSpec(
        name="model",
        label="Model",
        kind="dropdown",
        default="bact_phase_affinity",
        choices=[
            "bact_phase_affinity",
            "bact_fluor_affinity",
            "bact_phase_omni",
            "bact_fluor_omni",
            "cyto2_omni",
            "plant_omni",
            "worm_omni",
        ],
        help="Pretrained Omnipose model.",
        group="Model",
    ),
    WidgetSpec(
        name="model_path",
        label="Custom model path",
        kind="text",
        default="",
        help="Optional. Overrides 'Model' if set.",
        group="Model",
    ),
    # --- Detection ---------------------------------------------------------
    WidgetSpec(
        name="mask_threshold",
        label="Mask threshold",
        kind="slider",
        default=-2.0, min=-6.0, max=6.0, step=0.1,
        help="Lower → more pixels included.",
        group="Detection",
    ),
    WidgetSpec(
        name="flow_threshold",
        label="Flow threshold",
        kind="slider",
        default=0.0, min=0.0, max=3.0, step=0.05,
        help="Max flow error per cell. 0 = disabled.",
        group="Detection",
    ),
    WidgetSpec(
        name="niter",
        label="Iterations",
        kind="number",
        default=0, min=0, max=2000, step=1,
        help="Flow-following iterations. 0 = auto.",
        group="Detection",
    ),
    # --- Algorithm ---------------------------------------------------------
    WidgetSpec(
        name="cluster", label="Cluster", kind="toggle", default=True,
        help="DBSCAN-cluster trajectory endpoints.",
        group="Algorithm",
    ),
    WidgetSpec(
        name="affinity_seg", label="Affinity segmentation", kind="toggle", default=True,
        help="Use affinity graph (required for graph editing).",
        group="Algorithm",
    ),
    WidgetSpec(
        name="omni", label="Omni mode", kind="toggle", default=True,
        help="Omnipose flow following (vs. classical Cellpose).",
        group="Algorithm",
    ),
    WidgetSpec(
        name="resample", label="Resample", kind="toggle", default=True,
        help="Resample to model's expected scale before inference.",
        group="Algorithm",
    ),
    WidgetSpec(
        name="tile", label="Tile", kind="toggle", default=False,
        help="Tiled inference for large images.",
        group="Algorithm",
    ),
    WidgetSpec(
        name="augment", label="Augment (TTA)", kind="toggle", default=False,
        help="Test-time augmentation (4× slower).",
        group="Algorithm",
    ),
    WidgetSpec(
        name="transparency", label="Transparency", kind="toggle", default=True,
        help="Render flows with alpha channel.",
        group="Algorithm",
    ),
    WidgetSpec(
        name="verbose", label="Verbose", kind="toggle", default=False,
        help="Print extra diagnostics to the server log.",
        group="Algorithm",
    ),
]


# ---------------------------------------------------------------------------
# Plugin callbacks
# ---------------------------------------------------------------------------


def _coerce_settings(params: Mapping[str, Any]) -> dict[str, Any]:
    """Drop empty optional fields so Segmenter sees clean defaults."""
    out: dict[str, Any] = dict(params)
    if not out.get("model_path"):
        out.pop("model_path", None)
    if isinstance(out.get("niter"), (int, float)) and int(out["niter"]) <= 0:
        out["niter"] = None
    return out


def _build_extras(seg) -> dict[str, Any]:
    """Pull Omnipose-specific extras (overlays, affinity, points) from cache."""
    extras: dict[str, Any] = {}
    flow_overlay, dist_overlay = seg.get_overlays()
    if flow_overlay:
        extras["flowOverlay"] = flow_overlay
    if dist_overlay:
        extras["distanceOverlay"] = dist_overlay
    affinity = seg.get_affinity_graph_payload()
    if affinity is not None:
        extras["affinityGraph"] = affinity
    points = seg.get_points_payload()
    if points is not None:
        extras["points"] = points
    extras["canRebuild"] = bool(seg.has_cache)
    return extras


def _run(image: np.ndarray, params: Mapping[str, Any]):
    seg = _get_segmenter()
    settings = _coerce_settings(params)
    mask = seg.segment(image, settings=settings)
    return mask, _build_extras(seg)


def _resegment(params: Mapping[str, Any]):
    seg = _get_segmenter()
    if not seg.has_cache:
        raise RuntimeError("no cached Omnipose flows; run a full segmentation first")
    settings = _coerce_settings(params)
    mask = seg.resegment(settings=settings)
    return mask, _build_extras(seg)


def _set_use_gpu(enabled: bool) -> None:
    _get_segmenter().set_use_gpu(bool(enabled))


def _get_use_gpu() -> bool:
    return bool(_get_segmenter().get_use_gpu())


def _clear_cache() -> None:
    _get_segmenter().clear_cache()


def _warmup(model_id: str) -> None:
    _get_segmenter().preload_modules_async(delay=0.0)


def _relabel_from_affinity(
    mask: np.ndarray, spatial: np.ndarray, steps: np.ndarray
) -> np.ndarray:
    return _get_segmenter().relabel_from_affinity(mask, spatial, steps)


def _list_models() -> list[str]:
    builtins = next(w.choices for w in WIDGETS if w.name == "model") or ()
    try:
        from ocdkit.viewer.model_registry import list_models
        custom = [m["name"] for m in list_models(plugin="omnipose")]
    except Exception:
        custom = []
    return list(builtins) + custom


# ---------------------------------------------------------------------------
# Plugin object — discovered via entry_points
# ---------------------------------------------------------------------------

plugin = SegmentationPlugin(
    name="omnipose",
    version="1.0.0",
    description=(
        "Omnipose: distance-field + flow-based instance segmentation. "
        "Backend: omnipose (migrated from cellpose_omni/omnipose)."
    ),
    homepage="https://github.com/kevinjohncutler/omnipose",
    widgets=WIDGETS,
    run=_run,
    resegment=_resegment,
    warmup=_warmup,
    set_use_gpu=_set_use_gpu,
    get_use_gpu=_get_use_gpu,
    clear_cache=_clear_cache,
    relabel_from_affinity=_relabel_from_affinity,
    load_models=_list_models,
)
