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

# Model selection is rendered by the host (from manifest.models). The plugin
# declares the catalog here once so both the host dropdown and load_models()
# share the same source of truth.
_BUILTIN_MODELS: list[str] = [
    "bact_phase_affinity",
    "bact_fluor_affinity",
    "bact_phase_omni",
    "bact_fluor_omni",
    "cyto2_omni",
    "plant_omni",
    "worm_omni",
]


WIDGETS: list[WidgetSpec] = [
    # --- Parameters --------------------------------------------------------
    WidgetSpec(
        name="niter",
        label="niter",
        kind="slider",
        default=-1, min=-1, max=400, step=1,
        help="Flow-following iterations. -1 = auto.",
        group="Parameters",
    ),
    WidgetSpec(
        name="mask_threshold",
        label="mask",
        kind="slider",
        default=-2.0, min=-5.0, max=5.0, step=0.1,
        help="Lower → more pixels included.",
        group="Parameters",
    ),
    WidgetSpec(
        name="flow_threshold",
        label="flow",
        kind="slider",
        default=0.0, min=0.0, max=5.0, step=0.1,
        help="Max flow error per cell. 0 = disabled.",
        group="Parameters",
    ),
    # --- Segmentation mode -------------------------------------------------
    # Tri-state segmented control replacing separate cluster + affinity_seg
    # toggles. The plugin's _coerce_settings translates this back to the
    # underlying pair of booleans the segmenter expects.
    WidgetSpec(
        name="seg_mode",
        label="Segmentation Mode",
        kind="segmented",
        default="affinity",
        choices=["cluster", "affinity", "none"],
        choice_icons={
            "cluster": "seg-mode-icon-dbscan",
            "affinity": "seg-mode-icon-affinity",
            "none": "seg-mode-icon-cc",
        },
        help="DBSCAN cluster endpoints | affinity graph | neither (connected components only).",
    ),
    # --- Advanced (collapsed by default) -----------------------------------
    # `as_header=True` makes the host render this toggle as a chevron in the
    # group's heading row; clicking the heading flips the value and the
    # widgets below (gated via `visible_when={"advanced": True}`) collapse
    # or expand. No standalone toggle row is rendered.
    WidgetSpec(
        name="advanced",
        label="advanced options",
        kind="toggle",
        default=False,
        as_header=True,
        help="Reveal Omnipose algorithm flags rarely changed during normal use.",
        group="Advanced",
    ),
    WidgetSpec(
        name="use_gpu", label="Use GPU", kind="toggle", default=True,
        help=(
            "Run inference and reconstruction on the GPU when available. "
            "Disable to force CPU (much slower; only useful for debugging). "
            "Status is reported in the Server Info panel as 'Torch GPU ✓'."
        ),
        group="Advanced", visible_when={"advanced": True},
    ),
    WidgetSpec(
        name="omni", label="Omni mode", kind="toggle", default=True,
        help=(
            "Omnipose pipeline (on) vs. classical Cellpose (off). Affects: "
            "iscell threshold (hysteresis vs hard threshold on the distance "
            "field), flow normalization (div_rescale vs raw / 5), and the "
            "mask reconstruction algorithm. NOTE: only takes full effect in "
            "Segmentation Mode = none (connected components). In affinity "
            "and cluster modes the reconstruction goes through the affinity "
            "branch which is omnipose-only — only the threshold step changes."
        ),
        group="Advanced", visible_when={"advanced": True},
    ),
    WidgetSpec(
        name="resample", label="Resample", kind="toggle", default=True,
        help=(
            "Resample to the model's expected scale before inference. "
            "Inference-only — toggling here only takes effect after pressing "
            "Segment (not on slider drag, which only re-runs reconstruction)."
        ),
        group="Advanced", visible_when={"advanced": True},
    ),
    WidgetSpec(
        name="tile", label="Tile", kind="toggle", default=False,
        help=(
            "Tile the image and run inference on each tile, stitching the "
            "outputs. Use for large images that don't fit in GPU memory. "
            "Inference-only — re-press Segment after toggling."
        ),
        group="Advanced", visible_when={"advanced": True},
    ),
    WidgetSpec(
        name="augment", label="Augment", kind="toggle", default=False,
        help=(
            "Test-time augmentation: average the model's predictions over "
            "rotations and flips of the input. Smoother flows + cleaner "
            "masks at cell boundaries, ~4× slower per inference. "
            "Inference-only — re-press Segment after toggling."
        ),
        group="Advanced", visible_when={"advanced": True},
    ),
    WidgetSpec(
        name="transparency", label="Transparency", kind="toggle", default=True,
        help=(
            "Render flow overlays with an alpha channel (transparent "
            "background). Inference-only — re-press Segment after toggling."
        ),
        group="Advanced", visible_when={"advanced": True},
    ),
    WidgetSpec(
        name="verbose", label="Verbose", kind="toggle", default=False,
        help="Print extra diagnostics to the server log.",
        group="Advanced", visible_when={"advanced": True},
    ),
]


# ---------------------------------------------------------------------------
# Plugin callbacks
# ---------------------------------------------------------------------------


def _coerce_settings(params: Mapping[str, Any]) -> dict[str, Any]:
    """Drop empty optional fields so Segmenter sees clean defaults.

    Translates the host's tri-state ``seg_mode`` into the pair of booleans
    Segmenter actually consumes (``cluster``, ``affinity_seg``), and turns the
    sentinel ``niter == -1`` into ``None`` (auto). Drops the host-only
    ``advanced`` UI flag before forwarding.
    """
    out: dict[str, Any] = dict(params)
    if not out.get("model_path"):
        out.pop("model_path", None)
    if isinstance(out.get("niter"), (int, float)) and int(out["niter"]) < 0:
        out["niter"] = None
    mode = out.pop("seg_mode", None)
    if mode is not None:
        out["cluster"] = (mode == "cluster")
        out["affinity_seg"] = (mode == "affinity")
    out.pop("advanced", None)
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
    try:
        from ocdkit.viewer.model_registry import list_models
        custom = [m["name"] for m in list_models(plugin="omnipose")]
    except Exception:
        custom = []
    return list(_BUILTIN_MODELS) + custom


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
    # Host-managed display toggles to surface for this plugin. The host
    # renders an OFF toggle for each key here and enables it once the
    # corresponding extras payload arrives in a segment response.
    display_overlays=[
        "affinityGraph",
        "points",
        "vector",
        "flowOverlay",
        "distanceOverlay",
    ],
)
