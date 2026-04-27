"""Segmentation model management and inference for the Omnipose ocdkit plugin.

All inference paths use the in-package ``omnipose`` modules
(``omnipose.models.OmniModel``, ``omnipose.core``, ``omnipose.utils``,
``omnipose.gpu``, ``omnipose.transforms``). The legacy ``omnipose`` and
``cellpose_omni`` packages are no longer imported.

Lifted from the original ``gui/server/segmentation.py`` and adapted to the
omnipose API.
"""

from __future__ import annotations

import base64
import io
import sys as _sys
import threading
import time
import traceback as _tb
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from imageio import v2 as imageio


def _log_unexpected(label: str) -> None:
    """Log an unexpected exception traceback (issue #8: silent swallowing)."""
    print(f"[_segmenter] unexpected error in {label}:", file=_sys.stderr)
    _tb.print_exc(file=_sys.stderr)


def _is_array(x: Any) -> bool:
    """Is ``x`` a numpy array (used to disambiguate batched vs flat flows)."""
    return isinstance(x, np.ndarray)


# Stub for the gui-log helper. Routes to ocdkit.viewer's logger when available,
# else silently no-ops.
def append_gui_log(message: str) -> None:
    try:
        from ocdkit.viewer.assets import append_gui_log as _impl
    except Exception:
        return
    try:
        _impl(message)
    except Exception:
        pass


def load_image_uint8(as_rgb: bool = False) -> np.ndarray:
    """Fallback sample image loader (only used by `run_segmentation`, which
    the plugin path doesn't call — kept for API parity)."""
    try:
        from ocdkit.viewer.sample_image import load_image_uint8 as _impl
        return _impl(as_rgb=as_rgb)
    except Exception:
        return np.zeros((256, 256), dtype=np.uint8) if not as_rgb else np.zeros(
            (256, 256, 3), dtype=np.uint8
        )


class Segmenter:
    """Manages ML model loading and segmentation inference."""

    def __init__(self) -> None:
        self._model = None
        self._model_type: str | None = None
        self._model_path: str | None = None
        self._model_lock = threading.Lock()
        self._eval_lock = threading.Lock()
        self._preload_thread: threading.Thread | None = None
        self._modules_preloaded = False
        self._cache: dict[str, Any] | None = None
        self._core_module = None
        self._magma_lut: Optional[np.ndarray] = None
        self._utils_module = None
        self._kernel_cache: dict[int, tuple[Any, Any, Any, Any, Any]] = {}
        try:
            from omnipose.gpu import get_device

            device, available = get_device(gpu_number=0)
            self._use_gpu = bool(available)
            # Cache the actual torch.device so resegment can pass it to
            # compute_masks. Without this, the affinity branch falls back to
            # CPU (compute_masks defaults: use_gpu=False, device=None) and
            # _get_affinity_torch becomes 100× slower than it should be.
            self._device = device if available else None
        except Exception:
            self._use_gpu = False
            self._device = None

    def _ensure_model(self, model_type: str | None = None, model_path: str | None = None) -> None:
        requested_type = model_type or "bact_phase_affinity"
        requested_path = model_path or None
        if self._model is None or requested_type != self._model_type or requested_path != self._model_path:
            with self._model_lock:
                if (
                    self._model is None
                    or requested_type != self._model_type
                    or requested_path != self._model_path
                ):
                    from omnipose.models import OmniModel

                    if requested_path:
                        self._model = OmniModel(
                            gpu=self._use_gpu,
                            pretrained_model=requested_path,
                            model_type=None,
                        )
                    else:
                        self._model = OmniModel(
                            gpu=self._use_gpu,
                            model_type=requested_type,
                        )
                    self._model_type = requested_type
                    self._model_path = requested_path

    def preload_modules_async(self, delay: float = 0.0) -> None:
        if self._modules_preloaded:
            return
        if self._preload_thread is not None and self._preload_thread.is_alive():
            return

        def _target() -> None:
            if delay > 0:
                time.sleep(delay)
            try:
                # Warm the omnipose inference modules. These are heavy
                # (torch + numba JIT) so loading them upfront eliminates the
                # multi-second first-segment delay.
                import omnipose.models  # noqa: F401
                import omnipose.core  # noqa: F401
                import omnipose.utils  # noqa: F401
                self._modules_preloaded = True
                print("[viewer] segmenter module preload completed", flush=True)
            except Exception as exc:  # pragma: no cover - diagnostics only
                print(f"[viewer] segmenter module preload failed: {exc}", flush=True)
            finally:
                self._preload_thread = None

        self._preload_thread = threading.Thread(target=_target, name="SegmenterModulePreload", daemon=True)
        self._preload_thread.start()

    def segment(
        self,
        image: np.ndarray,
        settings: Mapping[str, Any] | None = None,
        **overrides: Any,
    ) -> np.ndarray:
        from omnipose.transforms import normalize99

        # Hold _eval_lock for the entire method so segment/resegment cannot
        # interleave their cache reads/writes (issue #1: cache race).
        with self._eval_lock:
            self._cache = None
            parsed, merged_options = self._parse_options(settings, overrides)
            self._ensure_model(parsed.get("model"), parsed.get("model_path"))
            arr = np.asarray(image, dtype=np.float32)
            if arr.ndim == 3:
                arr = arr.mean(axis=-1)
            arr = normalize99(arr)
            try:
                nchan = int(getattr(self._model, "nchan", 1) or 1)
            except Exception:
                nchan = 1
            channels = [0, 0] if nchan > 1 else None
            # OmniModel.eval returns a Result(masks=..., flows=...) container
            # which supports tuple unpacking; the trailing *rest is empty.
            masks, flows, *rest = self._model.eval(
                arr,
                channels=channels,
                rescale_factor=None,
                mask_threshold=parsed["mask_threshold"],
                flow_threshold=parsed["flow_threshold"],
                transparency=parsed["transparency"],
                omni=parsed["omni"],
                cluster=parsed["cluster"],
                resample=parsed["resample"],
                verbose=parsed["verbose"],
                tile=parsed["tile"],
                niter=parsed["niter"],
                augment=parsed["augment"],
                affinity_seg=parsed["affinity_seg"],
            )
            mask = self._select_first(masks)
            # OmniModel.eval returns masks with a leading batch dimension;
            # drop it so the mask is 2D (H, W) for downstream ncolor / affinity.
            while mask.ndim > 2 and mask.shape[0] == 1:
                mask = mask[0]
            mask_uint32 = np.ascontiguousarray(mask.astype(np.uint32, copy=False))
            flow_components = self._extract_flows(flows)
            self._cache = self._build_cache(arr, flow_components, parsed, merged_options, mask_uint32.shape)
            ncolor_mask = self._compute_ncolor_mask(mask_uint32, expand=True)
            if self._cache is None:
                self._cache = {}
            cache = self._cache
            cache["mask"] = mask_uint32
            cache["ncolor_mask"] = ncolor_mask
            cache["points_payload"] = None
            cache.pop("encoded_mask", None)  # invalidate base64 cache (#7)
            if parsed.get("affinity_seg"):
                # `_affinity_from_flows` expects the unwrapped per-image item
                # list (flows[6] = affinity), so pass the same list we already
                # extracted via `_extract_flows`.
                affinity_data = self._affinity_from_flows(flow_components, mask_uint32)
                if affinity_data is not None:
                    cache["affinity_graph"] = affinity_data
            else:
                cache.pop("affinity_graph", None)
            return mask_uint32

    def resegment(self, settings: Mapping[str, Any] | None = None, **overrides: Any) -> np.ndarray:
        import time as _time, os as _os
        _profile = bool(_os.environ.get("OMNIPOSE_PROFILE_RESEGMENT"))
        _t0 = _time.perf_counter() if _profile else None
        # Snapshot fallback decision under lock so segment/resegment cannot
        # interleave their cache reads/writes (issue #1: cache race).
        required = ("dP", "dist", "bd", "mask_shape", "nclasses", "dim")
        with self._eval_lock:
            if self._cache is None:
                raise RuntimeError("no cached segmentation data available")
            needs_fallback = any(key not in self._cache for key in required)
            fallback_image = self._cache.get("image") if needs_fallback else None

        if needs_fallback:
            if fallback_image is None:
                raise RuntimeError("cached flows missing and no image available; cannot resegment")
            import sys as _sys
            print("[resegment] cache missing flows; falling back to full segment", file=_sys.stderr)
            return self.segment(fallback_image, settings=settings, **overrides)

        parsed, merged_options = self._parse_options(settings, overrides)
        with self._eval_lock:
            cache = self._cache
            if cache is None:
                # Cleared between snapshot and reacquire; treat as no-cache.
                raise RuntimeError("no cached segmentation data available")
            previous_threshold = cache.get("last_mask_threshold", parsed["mask_threshold"])
            have_enough_pixels = parsed["mask_threshold"] > previous_threshold
            dP = np.array(cache["dP"], dtype=np.float32, copy=True)
            dist = np.array(cache["dist"], dtype=np.float32, copy=True)
            bd = np.array(cache["bd"], dtype=np.float32, copy=True)
            p_cache = cache.get("p")
            p = p_cache.copy() if (parsed["affinity_seg"] and p_cache is not None and have_enough_pixels) else None
            rescale_value = cache.get("rescale")
            if rescale_value is None:
                rescale_value = 1.0
            core_module = self._ensure_core()
            mask, p_out, _, bounds, augmented_affinity = core_module.compute_masks(
                dP=dP,
                dist=dist,
                bd=bd,
                p=p,
                niter=parsed["niter"],
                mask_threshold=parsed["mask_threshold"],
                flow_threshold=parsed["flow_threshold"],
                resize=cache["mask_shape"],
                rescale_factor=rescale_value,
                cluster=parsed["cluster"],
                affinity_seg=parsed["affinity_seg"],
                omni=True,
                nclasses=cache["nclasses"],
                dim=cache["dim"],
                # Critical: route compute_masks through GPU. Without these
                # the affinity branch silently falls back to CPU and
                # _get_affinity_torch takes 2 seconds per slider tick.
                use_gpu=self._use_gpu,
                device=self._device,
            )
            _t1 = _time.perf_counter() if _profile else None
            mask_uint32 = np.ascontiguousarray(mask.astype(np.uint32, copy=False))
            ncolor_mask = self._compute_ncolor_mask(mask_uint32, expand=True)
            _t2 = _time.perf_counter() if _profile else None
            cache["mask"] = mask_uint32
            cache["ncolor_mask"] = ncolor_mask
            cache["points_payload"] = None
            cache.pop("encoded_mask", None)  # invalidate base64 cache (#7)
            cache["mask_shape"] = tuple(mask_uint32.shape)
            cache["last_mask_threshold"] = parsed["mask_threshold"]
            cache["last_flow_threshold"] = parsed["flow_threshold"]
            cache["last_niter"] = parsed["niter"]
            cache["last_options"] = merged_options
            if parsed["affinity_seg"]:
                cache["bounds"] = bounds
                affinity_data = self._affinity_from_augmented(augmented_affinity, mask_uint32)
                if affinity_data is not None:
                    cache["affinity_graph"] = affinity_data
            else:
                cache.pop("bounds", None)
                cache.pop("affinity_graph", None)
            if p_out is not None:
                cache["p"] = p_out
        self._cache = cache
        if _profile:
            _t3 = _time.perf_counter()
            import sys as _sys
            print(
                f"[resegment.profile] compute_masks={(_t1-_t0)*1000:.0f}ms "
                f"ncolor={(_t2-_t1)*1000:.0f}ms cache={(_t3-_t2)*1000:.0f}ms "
                f"total={(_t3-_t0)*1000:.0f}ms",
                file=_sys.stderr, flush=True,
            )
        return mask_uint32

    def get_ncolor_mask(self) -> Optional[np.ndarray]:
        cache = self._cache or {}
        ncolor_mask = cache.get("ncolor_mask")
        if ncolor_mask is None:
            return None
        return np.asarray(ncolor_mask, dtype=np.uint32)

    @property
    def has_cache(self) -> bool:
        return self._cache is not None

    def set_use_gpu(self, enabled: bool) -> None:
        next_value = bool(enabled)
        if next_value:
            try:
                from omnipose.gpu import get_device

                _, available = get_device(gpu_number=0)
                next_value = bool(available)
            except Exception:
                next_value = False
        if next_value == self._use_gpu:
            return
        self._use_gpu = next_value
        # Force model rebuild on next use
        self._model = None
        self._model_type = None
        self._model_path = None

    def get_use_gpu(self) -> bool:
        return bool(self._use_gpu)

    def _parse_options(
        self,
        settings: Mapping[str, Any] | None,
        overrides: Mapping[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        merged: dict[str, Any] = {}
        if settings:
            merged.update(dict(settings))
            nested = merged.pop("settings", None)
            if isinstance(nested, Mapping):
                merged.update(dict(nested))
        if overrides:
            merged.update(dict(overrides))

        def _get_float(name: str, default: float) -> float:
            value = merged.get(name, default)
            try:
                return float(value)
            except (TypeError, ValueError):
                return float(default)

        def _get_bool(name: str, default: bool) -> bool:
            value = merged.get(name, default)
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"true", "1", "yes", "on"}:
                    return True
                if lowered in {"false", "0", "no", "off"}:
                    return False
            return bool(value)

        def _get_int(name: str, default: int) -> int:
            value = merged.get(name, default)
            try:
                return int(value)
            except (TypeError, ValueError):
                return int(default)

        def _get_optional_int(name: str) -> int | None:
            value = merged.get(name, None)
            if value is None:
                return None
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"auto", "none"}:
                    return None
            try:
                numeric = int(value)
            except (TypeError, ValueError):
                return None
            if numeric <= -1:
                return None
            return numeric

        parsed = {
            "model": merged.get("model"),
            "model_path": merged.get("model_path"),
            "mask_threshold": _get_float("mask_threshold", -2.0),
            "flow_threshold": _get_float("flow_threshold", 0.0),
            "cluster": _get_bool("cluster", True),
            "affinity_seg": _get_bool("affinity_seg", True),
            "transparency": _get_bool("transparency", True),
            "omni": _get_bool("omni", True),
            "resample": _get_bool("resample", True),
            "verbose": _get_bool("verbose", False),
            "tile": _get_bool("tile", False),
            "niter": _get_optional_int("niter"),
            "augment": _get_bool("augment", False),
        }
        return parsed, merged

    def _ensure_core(self):
        if self._core_module is None:
            from omnipose import core as core_module

            self._core_module = core_module
        return self._core_module

    def _ensure_utils(self):
        if self._utils_module is None:
            from omnipose import utils as utils_module

            self._utils_module = utils_module
        return self._utils_module

    def _get_kernel_info(self, dim: int) -> tuple[np.ndarray, Any, Any, Any, Any]:
        cached = self._kernel_cache.get(dim)
        if cached is not None:
            return cached
        utils_module = self._ensure_utils()
        steps, inds, idx, fact, sign = utils_module.kernel_setup(dim)
        steps_arr = np.asarray(steps)
        cached = (steps_arr, inds, idx, fact, sign)
        self._kernel_cache[dim] = cached
        return cached

    def _normalize_affinity_graph(self, affinity_graph: np.ndarray, mask: np.ndarray) -> Optional[tuple[np.ndarray, np.ndarray]]:
        mask_int = np.asarray(mask, dtype=np.int32)
        if mask_int.ndim != 2:
            append_gui_log('[affinity] mask ndim !=2')
            return None
        coords = np.nonzero(mask_int > 0)
        if coords[0].size == 0:
            append_gui_log('[affinity] no foreground coords')
            return None
        steps, inds, idx, fact, sign = self._get_kernel_info(mask_int.ndim)
        steps_arr = np.asarray(steps)
        center_index = int(idx)
        if center_index <= 0:
            append_gui_log('[affinity] center_index invalid')
            return None
        affinity = np.asarray(affinity_graph)
        if affinity.ndim == 3 and affinity.shape[1:] == mask_int.shape:
            # Already a spatial affinity graph (S, H, W).
            affinity = (affinity > 0).astype(np.uint8, copy=False)
            if affinity.shape[0] == steps_arr.shape[0]:
                non_center_mask = np.ones(steps_arr.shape[0], dtype=bool)
                non_center_mask[center_index] = False
                step_subset = np.ascontiguousarray(steps_arr[non_center_mask].astype(np.int8, copy=False))
                affinity = affinity[non_center_mask]
            elif affinity.shape[0] == steps_arr.shape[0] - 1:
                non_center_mask = np.ones(steps_arr.shape[0], dtype=bool)
                non_center_mask[center_index] = False
                step_subset = np.ascontiguousarray(steps_arr[non_center_mask].astype(np.int8, copy=False))
            else:
                return None
            spatial_subset = np.ascontiguousarray(affinity.astype(np.uint8, copy=False))
            return step_subset, spatial_subset
        if affinity.ndim > 2:
            affinity = np.squeeze(affinity)
        if affinity.ndim != 2:
            return None
        if affinity.shape[1] != coords[0].size:
            # affinity graph must align with foreground coords
            return None
        affinity = (affinity > 0).astype(np.uint8, copy=False)
        if affinity.shape[0] == steps_arr.shape[0]:
            non_center_mask = np.ones(steps_arr.shape[0], dtype=bool)
            non_center_mask[center_index] = False
            step_subset = np.ascontiguousarray(steps_arr[non_center_mask].astype(np.int8, copy=False))
            affinity = affinity[non_center_mask]
        elif affinity.shape[0] == steps_arr.shape[0] - 1:
            non_center_mask = np.ones(steps_arr.shape[0], dtype=bool)
            non_center_mask[center_index] = False
            step_subset = np.ascontiguousarray(steps_arr[non_center_mask].astype(np.int8, copy=False))
        else:
            return None
        core_module = self._ensure_core()
        spatial = core_module.spatial_affinity(affinity, coords, mask_int.shape)
        spatial_subset = np.ascontiguousarray(spatial.astype(np.uint8, copy=False))
        return step_subset, spatial_subset

    def _affinity_from_flows(self, flows: list[np.ndarray] | None, mask: np.ndarray) -> Optional[tuple[np.ndarray, np.ndarray]]:
        if not flows or len(flows) <= 6:
            append_gui_log('[affinity] flows missing or too short: ' + str(0 if flows is None else len(flows)))
            return None
        affinity = flows[6]
        if affinity is None:
            append_gui_log('[affinity] flows[6] is None')
            return None
        try:
            append_gui_log('[affinity] flows[6] shape=' + str(getattr(affinity, 'shape', None)))
        except Exception:
            pass
        try:
            arr = np.asarray(affinity)
        except Exception:
            _log_unexpected("_affinity_from_flows: np.asarray")
            return None
        if arr.ndim == 3:
            dim = int(np.asarray(mask).ndim)
            try:
                if arr.shape[0] == dim + 1 and arr.shape[1] == 3 ** dim:
                    append_gui_log('[affinity] flows[6] appears augmented; using last plane')
                    affinity = arr[-1]
            except Exception:
                _log_unexpected("_affinity_from_flows: shape check")
        return self._normalize_affinity_graph(affinity, mask)

    def _affinity_from_augmented(self, augmented_affinity: Any, mask: np.ndarray) -> Optional[tuple[np.ndarray, np.ndarray]]:
        if augmented_affinity is None:
            append_gui_log('[affinity] augmented affinity is None')
            return None
        try:
            arr = np.asarray(augmented_affinity)
        except Exception:
            _log_unexpected("_affinity_from_augmented: np.asarray")
            append_gui_log('[affinity] augmented affinity not array')
            return None
        if arr.size == 0 or arr.ndim < 3:
            append_gui_log('[affinity] augmented affinity empty/ndim<' + str(getattr(arr, 'ndim', None)) + ')')
            return None
        affinity = arr[-1]
        try:
            append_gui_log('[affinity] augmented[-1] shape=' + str(getattr(affinity, 'shape', None)))
        except Exception:
            pass
        return self._normalize_affinity_graph(affinity, mask)

    def _compute_affinity_graph(self, mask: np.ndarray) -> Optional[tuple[np.ndarray, np.ndarray]]:
        mask_int = np.asarray(mask, dtype=np.int32)
        if mask_int.ndim != 2:
            append_gui_log('[affinity] mask ndim !=2')
            return None
        coords = np.nonzero(mask_int)
        if coords[0].size == 0:
            append_gui_log('[affinity] no foreground coords')
            return None
        steps, inds, idx, fact, sign = self._get_kernel_info(mask_int.ndim)
        center_index = int(idx)
        if center_index <= 0:
            append_gui_log('[affinity] center_index invalid')
            return None
        core_module = self._ensure_core()
        affinity_graph = core_module.masks_to_affinity(
            mask_int,
            coords,
            steps,
            inds,
            idx,
            fact,
            sign,
            mask_int.ndim,
        )
        spatial = core_module.spatial_affinity(affinity_graph, coords, mask_int.shape)
        non_center_mask = np.ones(steps.shape[0], dtype=bool)
        non_center_mask[center_index] = False
        step_subset = np.ascontiguousarray(steps[non_center_mask].astype(np.int8, copy=False))
        spatial_subset = np.ascontiguousarray(spatial[non_center_mask].astype(np.uint8, copy=False))
        return step_subset, spatial_subset

    def relabel_from_affinity(self, mask: np.ndarray, spatial_affinity: np.ndarray, steps: np.ndarray) -> np.ndarray:
        """Relabel using a provided spatial affinity graph (S,H,W) and its step offsets."""
        core_module = self._ensure_core()
        utils_module = self._ensure_utils()
        mask_int = np.asarray(mask, dtype=np.int32)
        if mask_int.ndim != 2:
            return mask_int.astype(np.int32, copy=False)
        shape = mask_int.shape
        coords = np.nonzero(mask_int > 0)
        if coords[0].size == 0:
            return mask_int.astype(np.int32, copy=False)
        dim = mask_int.ndim
        steps_arr = np.asarray(steps, dtype=np.int16)
        spatial = np.asarray(spatial_affinity, dtype=np.uint8)
        if spatial.shape[1:] != shape:
            raise ValueError("spatial affinity shape mismatch")
        if spatial.shape[0] != steps_arr.shape[0]:
            raise ValueError(f"spatial steps mismatch: S={spatial.shape[0]} vs steps={steps_arr.shape[0]}")
        k_steps, _, center_idx, _, _ = utils_module.kernel_setup(dim)
        k_steps = np.asarray(k_steps, dtype=np.int16)
        S_full = k_steps.shape[0]
        spatial_full = np.zeros((S_full, shape[0], shape[1]), dtype=np.uint8)
        step_to_idx = {(int(s[0]), int(s[1])): i for i, s in enumerate(k_steps)}
        for i in range(steps_arr.shape[0]):
            key = (int(steps_arr[i, 0]), int(steps_arr[i, 1]))
            j = step_to_idx.get(key)
            if j is not None:
                spatial_full[j] = spatial[i]
        neighbors = utils_module.get_neighbors(coords, k_steps, dim, shape)
        _, neigh_inds, _ = utils_module.get_neigh_inds(tuple(neighbors), coords, shape)
        aff_sn = spatial_full[(Ellipsis,) + coords]
        iscell = mask_int > 0
        relabeled = core_module.affinity_to_masks(
            aff_sn,
            neigh_inds,
            iscell,
            coords,
            cardinal=False,
            exclude_interior=False,
            return_edges=False,
            verbose=False,
        )
        relabeled = np.asarray(relabeled, dtype=np.int32)
        if relabeled.size == 0 or int(np.max(relabeled)) == 0:
            print('[relabel_from_affinity] WARNING: empty relabel result; returning original mask')
            return mask_int
        return relabeled

    def _compute_ncolor_mask(self, mask: np.ndarray, *, expand: bool = True) -> Optional[np.ndarray]:
        try:
            import ncolor
        except ImportError:
            return None
        if mask.size == 0:
            return None
        mask_int = np.asarray(mask, dtype=np.int32)
        mask_for_label = mask_int
        try:
            import fastremap  # type: ignore
            unique = fastremap.unique(mask_int)
            if unique.size:
                unique = unique[unique > 0]
            if unique.size:
                mapping = {int(value): idx + 1 for idx, value in enumerate(unique)}
                mask_for_label = fastremap.remap(mask_int, mapping, preserve_missing_labels=True, in_place=False)
        except Exception:
            _log_unexpected("_compute_ncolor_mask: fastremap path")
            mask_for_label = mask_int
        try:
            labeled, ngroups = ncolor.label(
                mask_for_label,
                max_depth=20,
                expand=expand,
                return_n=True,
                format_input=False,
            )
        except TypeError:
            try:
                labeled = ncolor.label(mask_for_label, max_depth=20, expand=expand, format_input=False)
            except TypeError:
                labeled = ncolor.label(mask_for_label, max_depth=20, format_input=False)
            ngroups = int(np.unique(labeled[labeled > 0]).size)
        try:
            max_label = int(np.max(mask_int)) if mask_int.size else 0
            report_max = min(max_label, 10)
            mapping = {}
            for label in range(1, report_max + 1):
                coords = np.argwhere(mask_int == label)
                if coords.size == 0:
                    continue
                y, x = coords[0]
                mapping[label] = int(labeled[y, x])
            print(f"[ncolor] label->group (1..{report_max}): {mapping}")
            missing = (mask_int > 0) & (labeled == 0)
            missing_count = int(np.count_nonzero(missing))
            if missing_count:
                missing_labels = np.unique(mask_int[missing])
                sample_labels = missing_labels[:10].astype(int).tolist()
                print(f"[ncolor] missing group pixels={missing_count} labels={sample_labels}")
        except Exception:
            _log_unexpected("_compute_ncolor_mask: diagnostics block")
        labeled_uint32 = np.ascontiguousarray(labeled.astype(np.uint32, copy=False))
        print(f"[ncolor] groups={ngroups}")
        return labeled_uint32

    @staticmethod
    def _select_first(obj: Any) -> np.ndarray:
        if isinstance(obj, (list, tuple)):
            if not obj:
                raise ValueError("empty result from model")
            return np.asarray(obj[0])
        return np.asarray(obj)

    @staticmethod
    def _extract_flows(flows: Any) -> list[Any] | None:
        """Unwrap OmniModel's ``flows`` into a flat per-image item list.

        ``OmniModel.eval()`` returns a :class:`ocdkit.io.result.Result` whose
        ``flows`` attribute is a batched list; each element is itself a
        :class:`Result` with fields ``(rgb, dP, dist, p, bd, traces, affinity,
        bounds)``. That field order matches the legacy cellpose_omni flat
        tuple, so downstream indexed access (flows[1] = dP, flows[2] = dist,
        flows[6] = affinity, etc.) still works once we unwrap one batch layer.

        ``Result`` isn't a list or tuple, but it supports iteration and has a
        ``_fields`` attribute — duck-typed here so namedtuples would also work.
        Items are kept in their original type; downstream consumers call
        ``np.array(item)`` on specific indices themselves.
        """
        if flows is None:
            return None
        candidate = flows
        # Unwrap one layer of batching ([Result] or [[...]]).
        if isinstance(candidate, (list, tuple)) and candidate and not _is_array(candidate[0]):
            candidate = candidate[0]
        if candidate is None:
            return None
        # Result / namedtuple duck-typing via `_fields`; list/tuple supported.
        if isinstance(candidate, (list, tuple)) or hasattr(candidate, "_fields"):
            return list(candidate)
        return [candidate]

    def _build_cache(
        self,
        image: np.ndarray,
        flows: list[np.ndarray] | None,
        parsed: Mapping[str, Any],
        merged_options: Mapping[str, Any],
        mask_shape: tuple[int, ...],
    ) -> dict[str, Any] | None:
        if not flows or len(flows) < 3:
            return None
        dP_raw = np.array(flows[1], dtype=np.float32, copy=True)
        if dP_raw.ndim == 4:
            dP = dP_raw[0]
        else:
            dP = dP_raw
        if dP.ndim != 3:
            dP = np.squeeze(dP)
        dist_raw = np.array(flows[2], dtype=np.float32, copy=True)
        dist = dist_raw[0] if dist_raw.ndim == 3 else np.squeeze(dist_raw)
        bd = None
        if len(flows) > 4 and flows[4] is not None:
            bd_raw = np.array(flows[4], dtype=np.float32, copy=True)
            bd = bd_raw[0] if bd_raw.ndim == 3 else np.squeeze(bd_raw)
        if bd is None:
            bd = np.zeros_like(dist, dtype=np.float32)
        p = None
        if len(flows) > 3 and flows[3] is not None:
            p_raw = np.array(flows[3], dtype=np.float32, copy=True)
            p = p_raw[0] if p_raw.ndim == 4 else np.squeeze(p_raw)
        cache = {
            "image": np.asarray(image, dtype=np.float32),
            "dP": dP,
            "dist": dist,
            "bd": bd,
            "p": p,
            "mask_shape": tuple(mask_shape),
            "dim": getattr(self._model, "dim", dP.shape[0] if dP.ndim > 2 else 2),
            "nclasses": getattr(self._model, "nclasses", 3),
            "last_mask_threshold": parsed["mask_threshold"],
            "last_flow_threshold": parsed["flow_threshold"],
            "last_options": dict(merged_options),
            "mask": None,
            "points_payload": None,
            "rescale": merged_options.get("rescale"),
        }
        flow_overlay, dist_overlay = self._generate_overlays(flows)
        cache["flow_overlay"] = flow_overlay
        cache["dist_overlay"] = dist_overlay
        return cache

    def get_overlays(self) -> tuple[Optional[str], Optional[str]]:
        cache = self._cache or {}
        return cache.get("flow_overlay"), cache.get("dist_overlay")

    def get_affinity_graph_payload(self) -> Optional[dict[str, object]]:
        cache = self._cache or {}
        stored = cache.get("affinity_graph")
        if not stored:
            return None
        steps, data = stored
        if steps is None or data is None:
            return None
        if data.ndim != 3 or data.size == 0:
            return None
        step_array = np.ascontiguousarray(steps.astype(np.int8, copy=False))
        data_array = np.ascontiguousarray(data.astype(np.uint8, copy=False))
        encoded = base64.b64encode(data_array.tobytes()).decode("ascii")
        return {
            "width": int(data_array.shape[2]),
            "height": int(data_array.shape[1]),
            "steps": step_array.tolist(),
            "encoded": encoded,
        }

    def _generate_overlays(self, flows: list[np.ndarray] | None) -> tuple[Optional[str], Optional[str]]:
        if not flows:
            return None, None
        flow_overlay = None
        dist_overlay = None
        try:
            flow_overlay = self._encode_png(self._prepare_flow_image(flows))
        except Exception:
            _log_unexpected("_generate_overlays: flow")
            flow_overlay = None
        try:
            dist_overlay = self._encode_png(self._prepare_distance_image(flows))
        except Exception:
            _log_unexpected("_generate_overlays: distance")
            dist_overlay = None
        return flow_overlay, dist_overlay

    def _prepare_flow_image(self, flows: Sequence[Any]) -> np.ndarray:
        if not flows:
            raise ValueError("no flow data available")
        rgb = np.array(flows[0], dtype=np.float32, copy=True)
        if rgb.ndim >= 4:
            rgb = rgb[0]
        rgb = np.squeeze(rgb)
        if rgb.ndim != 3:
            raise ValueError("unexpected RGB flow shape")
        if rgb.shape[0] in (3, 4) and rgb.shape[-1] not in (3, 4):
            rgb = np.moveaxis(rgb, 0, -1)
        if rgb.shape[-1] == 4:
            alpha = rgb[..., 3]
            fg = np.clip(rgb[..., :3], 0, 255)
            fg_uint8 = fg.astype(np.uint8)
            alpha_norm = np.clip(alpha / 255.0, 0.0, 1.0)
            bg = np.zeros_like(fg_uint8)
            blended = fg_uint8 * alpha_norm[..., None] + bg * (1.0 - alpha_norm[..., None])
            rgb_uint8 = blended.astype(np.uint8)
        else:
            rgb_uint8 = np.clip(rgb, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(rgb_uint8)

    def _prepare_distance_image(self, flows: Sequence[Any]) -> np.ndarray:
        if len(flows) < 3 or flows[2] is None:
            raise ValueError("no distance data available")
        dist = np.array(flows[2], dtype=np.float32, copy=True)
        if dist.ndim >= 3:
            dist = dist[0]
        dist = np.squeeze(dist)
        if dist.size == 0:
            raise ValueError("empty distance map")
        finite = np.isfinite(dist)
        if finite.any():
            min_val = float(dist[finite].min())
            max_val = float(dist[finite].max())
            if max_val > min_val:
                norm = (dist - min_val) / (max_val - min_val)
            else:
                norm = np.zeros_like(dist)
        else:
            norm = np.zeros_like(dist)
        norm = np.clip(norm, 0.0, 1.0)
        lut = self._get_magma_lut()
        indices = np.round(norm * (len(lut) - 1)).astype(int)
        rgba = lut[indices]
        rgb_uint8 = (rgba[..., :3] * 255.0).astype(np.uint8)
        return np.ascontiguousarray(rgb_uint8)

    def _encode_png(self, array: np.ndarray) -> str:
        buffer = io.BytesIO()
        imageio.imwrite(buffer, array, format="png")
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

    def _get_magma_lut(self) -> np.ndarray:
        if self._magma_lut is None:
            try:
                from matplotlib import pyplot as plt
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("matplotlib is required for distance colormap") from exc
            cmap = plt.get_cmap("magma")
            lut = cmap(np.linspace(0, 1, cmap.N))[:, :4]
            self._magma_lut = lut
        return self._magma_lut

    def _prepare_points_payload(self, mask: np.ndarray, points: np.ndarray) -> tuple[str, int, int, int]:
        mask_arr = np.asarray(mask)
        if mask_arr.ndim != 2:
            mask_arr = np.squeeze(mask_arr)
        if mask_arr.ndim != 2:
            raise ValueError("invalid mask shape for points")
        pts = np.asarray(points)
        if pts.ndim == 4 and pts.shape[0] == 1:
            pts = pts[0]
        if pts.ndim == 3:
            if pts.shape[0] >= 2 and pts.shape[0] < pts.shape[-1]:
                py = pts[0]
                px = pts[1]
            elif pts.shape[-1] >= 2:
                py = pts[..., 0]
                px = pts[..., 1]
            else:
                raise ValueError("invalid points array")
        else:
            raise ValueError("invalid points array")
        h, w = mask_arr.shape
        if py.shape != (h, w):
            # The affinity branch in compute_masks pads by 1 on each side,
            # so cached `p` has shape (H+2, W+2) and its values are coords
            # in *padded* space. Center-crop to the unpadded grid AND
            # subtract the pad offset from the trajectory values themselves —
            # otherwise every mask pixel is read from a slot that's shifted
            # one row/col towards top-left in padded coords, which lands on
            # background for any pixel touching a cell's left/top edge and
            # makes those points appear "stuck" at their initial position.
            pad_y = max(0, (py.shape[0] - h) // 2)
            pad_x = max(0, (py.shape[1] - w) // 2)
            py = py[pad_y:pad_y + h, pad_x:pad_x + w] - pad_y
            px = px[pad_y:pad_y + h, pad_x:pad_x + w] - pad_x
        ys, xs = np.nonzero(mask_arr > 0)
        if ys.size == 0:
            return "", w, h, 0
        yf = np.clip(py[ys, xs], 0, h - 1).astype(np.float32)
        xf = np.clip(px[ys, xs], 0, w - 1).astype(np.float32)
        coords = np.empty((ys.size * 2,), dtype=np.float32)
        coords[0::2] = yf
        coords[1::2] = xf
        encoded = base64.b64encode(coords.tobytes()).decode("ascii")
        return encoded, w, h, ys.size

    def get_points_payload(self) -> Optional[dict[str, object]]:
        cache = self._cache or {}
        cached = cache.get("points_payload")
        if isinstance(cached, dict):
            return cached
        mask = cache.get("mask")
        points = cache.get("p")
        if mask is None or points is None:
            return None
        try:
            encoded, width, height, count = self._prepare_points_payload(mask, points)
        except Exception:
            return None
        payload = {
            "encoded": encoded,
            "width": int(width),
            "height": int(height),
            "count": int(count),
            "dtype": "float32",
        }
        cache["points_payload"] = payload
        self._cache = cache
        return payload

    def clear_cache(self) -> None:
        with self._eval_lock:
            self._cache = None


# Global singleton
_SEGMENTER = Segmenter()


def run_segmentation(
    settings: Mapping[str, Any] | None = None,
    *,
    state: "SessionState | None" = None,
) -> dict[str, object]:
    """Run full segmentation on the current image."""
    from .session import SESSION_MANAGER

    if state is None:
        state = SESSION_MANAGER.get_or_create(None)
    image = state.current_image if state.current_image is not None else load_image_uint8(as_rgb=True)
    if isinstance(settings, Mapping) and 'use_gpu' in settings:
        _SEGMENTER.set_use_gpu(bool(settings.get('use_gpu')))
    mask = _SEGMENTER.segment(image, settings=settings or {})
    mask_uint32 = np.ascontiguousarray(mask.astype(np.uint32, copy=False))
    encoded = base64.b64encode(mask_uint32.tobytes()).decode("ascii")
    height, width = mask_uint32.shape
    flow_overlay, dist_overlay = _SEGMENTER.get_overlays()
    ncolor_mask = _SEGMENTER.get_ncolor_mask()
    encoded_ncolor = None
    if ncolor_mask is not None:
        encoded_ncolor = base64.b64encode(np.ascontiguousarray(ncolor_mask).tobytes()).decode("ascii")
        try:
            print(f"[segment] ncolor groups={int(np.unique(ncolor_mask[ncolor_mask>0]).size)}")
        except Exception:
            pass
    affinity_graph = _SEGMENTER.get_affinity_graph_payload()
    if affinity_graph is None:
        try:
            append_gui_log(f"[segment] affinity_graph missing; affinity_seg={bool(settings and settings.get('affinity_seg'))} cache={_SEGMENTER.has_cache}")
        except Exception:
            pass
    points_payload = _SEGMENTER.get_points_payload()
    return {
        "mask": encoded,
        "width": int(width),
        "height": int(height),
        "canRebuild": _SEGMENTER.has_cache,
        "flowOverlay": flow_overlay,
        "distanceOverlay": dist_overlay,
        "nColorMask": encoded_ncolor,
        "affinityGraph": affinity_graph,
        "points": points_payload,
    }


def run_mask_update(
    settings: Mapping[str, Any] | None = None,
    *,
    state: "SessionState | None" = None,
) -> dict[str, object]:
    """Re-run mask computation with updated settings (uses cached flows)."""
    from .session import SESSION_MANAGER

    if state is None:
        state = SESSION_MANAGER.get_or_create(None)
    if isinstance(settings, Mapping) and 'use_gpu' in settings:
        _SEGMENTER.set_use_gpu(bool(settings.get('use_gpu')))
    mask = _SEGMENTER.resegment(settings=settings or {})
    mask_uint32 = np.ascontiguousarray(mask.astype(np.uint32, copy=False))
    encoded = base64.b64encode(mask_uint32.tobytes()).decode("ascii")
    height, width = mask_uint32.shape
    flow_overlay, dist_overlay = _SEGMENTER.get_overlays()
    ncolor_mask = _SEGMENTER.get_ncolor_mask()
    encoded_ncolor = None
    if ncolor_mask is not None:
        encoded_ncolor = base64.b64encode(np.ascontiguousarray(ncolor_mask).tobytes()).decode("ascii")
        try:
            print(f"[rebuild] ncolor groups={int(np.unique(ncolor_mask[ncolor_mask>0]).size)}")
        except Exception:
            pass
    affinity_graph = _SEGMENTER.get_affinity_graph_payload()
    if affinity_graph is None:
        try:
            append_gui_log(f"[rebuild] affinity_graph missing; affinity_seg={bool(settings and settings.get('affinity_seg'))} cache={_SEGMENTER.has_cache}")
        except Exception:
            pass
    points_payload = _SEGMENTER.get_points_payload()
    return {
        "mask": encoded,
        "width": int(width),
        "height": int(height),
        "canRebuild": _SEGMENTER.has_cache,
        "flowOverlay": flow_overlay,
        "distanceOverlay": dist_overlay,
        "nColorMask": encoded_ncolor,
        "affinityGraph": affinity_graph,
        "points": points_payload,
    }
