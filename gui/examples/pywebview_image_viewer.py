"""Minimal PyWebView viewer replicating core Omnipose image interactions."""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import threading
from contextlib import closing
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence
import secrets

from starlette.requests import Request

SCRIPT_START = time.perf_counter()

import numpy as np
import webview
from imageio import v2 as imageio

import types

current_module = sys.modules[__name__]
sys.modules.setdefault("pywebview_image_viewer", current_module)

gui_pkg = sys.modules.setdefault("gui", types.ModuleType("gui"))
if not getattr(gui_pkg, "__path__", None):
    gui_pkg.__path__ = [str(Path(__file__).resolve().parents[1])]

examples_pkg = sys.modules.setdefault("gui.examples", types.ModuleType("gui.examples"))
if not getattr(examples_pkg, "__path__", None):
    examples_pkg.__path__ = [str(Path(__file__).resolve().parent)]

sys.modules["gui.examples.pywebview_image_viewer"] = current_module

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent))
    from sample_image import (  # type: ignore
        DEFAULT_BRUSH_RADIUS,
        get_instance_color_table,
        load_image_uint8,
        _ensure_spatial_last,
        _normalize_uint8,
        get_preload_image_path,
    )
else:
    from .sample_image import (
        DEFAULT_BRUSH_RADIUS,
        get_instance_color_table,
        load_image_uint8,
        _ensure_spatial_last,
        _normalize_uint8,
        get_preload_image_path,
    )


WEB_DIR = (Path(__file__).resolve().parent.parent / "web").resolve()
INDEX_HTML = WEB_DIR / "index.html"
APP_JS = WEB_DIR / "app.js"
HTML_DIR = WEB_DIR / "html"
CSS_DIR = WEB_DIR / "css"
POINTER_JS = WEB_DIR / "js" / "pointer-state.js"
LOGGING_JS = WEB_DIR / "js" / "logging.js"
HISTORY_JS = WEB_DIR / "js" / "history.js"
BRUSH_JS = WEB_DIR / "js" / "brush.js"
PAINTING_JS = WEB_DIR / "js" / "painting.js"
INTERACTIONS_JS = WEB_DIR / "js" / "interactions.js"
HTML_FRAGMENTS = [
    HTML_DIR / "left-panel.html",
    HTML_DIR / "viewer.html",
    HTML_DIR / "sidebar.html",
]
CSS_FILES = [
    CSS_DIR / "layout.css",
    CSS_DIR / "tools.css",
    CSS_DIR / "controls.css",
    CSS_DIR / "viewer.css",
]
CSS_LINKS = (
    '    <link rel="stylesheet" href="/static/css/layout.css" />',
    '    <link rel="stylesheet" href="/static/css/tools.css" />',
    '    <link rel="stylesheet" href="/static/css/controls.css" />',
    '    <link rel="stylesheet" href="/static/css/viewer.css" />',
)
JS_FILES = [POINTER_JS, LOGGING_JS, HISTORY_JS, BRUSH_JS, PAINTING_JS, INTERACTIONS_JS, APP_JS]
JS_STATIC_PATHS = (
    "/static/js/pointer-state.js",
    "/static/js/logging.js",
    "/static/js/history.js",
    "/static/js/brush.js",
    "/static/js/painting.js",
    "/static/js/interactions.js",
    "/static/app.js",
)
_INDEX_HTML_CACHE: dict[str, object] = {"content": "", "mtime": None}
_LAYOUT_MARKUP_CACHE: dict[str, object] = {"markup": "", "mtimes": {}}
_INLINE_CSS_CACHE: dict[str, object] = {"text": "", "mtimes": {}}
_INLINE_JS_CACHE: dict[str, object] = {"text": "", "mtimes": {}}
_CACHE_BUSTER = str(int(time.time()))
_DEV_CERT_DIR = Path(tempfile.gettempdir()) / "omnipose_pywebview_dev_ssl"
WEBGL_LOG_PATH = Path("/tmp/webgl_log.txt")
try:
    WEBGL_LOG_PATH.write_text("", encoding="utf-8")
except OSError:
    pass

CAPTURE_LOG_SCRIPT = """<script>
(function(){
  if (window.__omniLogPush) { return; }
  var queue = [];
  var endpoint = '/api/log';
  var maxBatch = 25;
  var flushTimer = null;
  function flush(){
    if (!queue.length) { return; }
    var payload = queue.slice();
    queue.length = 0;
    try {
      fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ entries: payload })
      }).catch(function(){ });
    } catch (err) {
      console.warn('[log] flush failed', err);
    }
  }
  function schedule(){
    if (flushTimer) { return; }
    flushTimer = setTimeout(function(){ flushTimer = null; flush(); }, 300);
  }
  window.__omniLogPush = function(kind, data){
    try {
      queue.push({ kind: kind, data: data, ts: Date.now() });
      if (queue.length >= maxBatch) {
        if (flushTimer) { clearTimeout(flushTimer); flushTimer = null; }
        flush();
      } else {
        schedule();
      }
    } catch (err) {
      console.warn('[log] push failed', err);
    }
  };
  window.addEventListener('error', function(evt){
    window.__omniLogPush('JS_ERROR', {
      message: evt.message || '',
      filename: evt.filename || '',
      lineno: evt.lineno || 0,
      colno: evt.colno || 0,
      stack: evt.error && evt.error.stack ? String(evt.error.stack) : ''
    });
  });
})();
</script>"""

SUPPORTED_IMAGE_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".gif",
}

SESSION_COOKIE_NAME = "OMNISESSION"


def _ensure_dev_certificate() -> tuple[str, str]:
    """Return paths to a localhost self-signed certificate and key, generating if needed."""

    _DEV_CERT_DIR.mkdir(exist_ok=True)
    cert_path = _DEV_CERT_DIR / "localhost.pem"
    key_path = _DEV_CERT_DIR / "localhost.key"

    if cert_path.exists() and key_path.exists():
        return str(cert_path), str(key_path)

    openssl = shutil.which("openssl")
    if openssl is None:
        raise RuntimeError("openssl executable not found; install openssl or provide --ssl-cert/--ssl-key")

    cmd = [
        openssl,
        "req",
        "-x509",
        "-nodes",
        "-newkey",
        "rsa:2048",
        "-keyout",
        str(key_path),
        "-out",
        str(cert_path),
        "-days",
        "7",
        "-subj",
        "/CN=localhost",
    ]
    subprocess.run(cmd, check=True)
    return str(cert_path), str(key_path)


def _session_path_key(path: Optional[Path]) -> str:
    return str(path.resolve()) if path else "__sample__"


@dataclass
class SessionState:
    session_id: str
    current_path: Optional[Path]
    directory: Optional[Path]
    files: list[Path] = field(default_factory=list)
    saved_states: dict[str, Any] = field(default_factory=dict)
    current_image: Optional[np.ndarray] = None
    image_is_rgb: bool = False
    encoded_image: Optional[str] = None

    def path_key(self, path: Optional[Path] = None) -> str:
        return _session_path_key(path if path is not None else self.current_path)


class SessionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}
        self._lock = threading.Lock()

    def _create_session_unlocked(self) -> SessionState:
        session_id = secrets.token_urlsafe(16)
        initial_path = get_preload_image_path()
        if initial_path and initial_path.exists():
            image, is_rgb = self._load_image_from_path(initial_path)
            directory = initial_path.parent
            files = self._list_directory_images(directory)
        else:
            image = load_image_uint8(as_rgb=True)
            is_rgb = image.ndim == 3 and image.shape[-1] >= 3
            directory = None
            files: list[Path] = []
            initial_path = None
        state = SessionState(
            session_id=session_id,
            current_path=initial_path,
            directory=directory,
            files=files,
            current_image=np.ascontiguousarray(image, dtype=np.uint8),
            image_is_rgb=is_rgb,
            encoded_image=None,
        )
        self._sessions[session_id] = state
        state.encoded_image = self._encode_image(state.current_image, is_rgb=is_rgb)
        return state

    def get_or_create(self, session_id: Optional[str]) -> SessionState:
        with self._lock:
            if session_id and session_id in self._sessions:
                return self._sessions[session_id]
            return self._create_session_unlocked()

    def get(self, session_id: str) -> SessionState:
        with self._lock:
            return self._sessions[session_id]

    def clear_saved_states(self, state: SessionState) -> None:
        with self._lock:
            existing = self._sessions.get(state.session_id)
            if existing:
                existing.saved_states.clear()

    def _load_image_from_path(self, path: Path) -> tuple[np.ndarray, bool]:
        arr = imageio.imread(path)
        arr = _ensure_spatial_last(arr)
        arr = _normalize_uint8(arr)
        is_rgb = arr.ndim == 3 and arr.shape[-1] >= 3
        return arr, is_rgb

    def _list_directory_images(self, directory: Path) -> list[Path]:
        try:
            files = [
                p for p in sorted(directory.iterdir())
                if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS
            ]
        except FileNotFoundError:
            files = []
        return files

    def set_image(self, state: SessionState, path: Optional[Path]) -> None:
        if path is not None:
            path = path.expanduser().resolve()
        if path is not None and not path.exists():
            raise FileNotFoundError(path)
        if path is None:
            image = load_image_uint8(as_rgb=True)
            is_rgb = image.ndim == 3 and image.shape[-1] >= 3
            directory = None
            files: list[Path] = []
        else:
            image, is_rgb = self._load_image_from_path(path)
            directory = path.parent
            files = self._list_directory_images(directory)
        state.current_path = path
        state.directory = directory
        state.files = files
        state.current_image = np.ascontiguousarray(image, dtype=np.uint8)
        state.image_is_rgb = is_rgb
        state.encoded_image = self._encode_image(state.current_image, is_rgb=is_rgb)

    def build_config(self, state: SessionState) -> dict[str, Any]:
        image = state.current_image if state.current_image is not None else load_image_uint8(as_rgb=True)
        is_rgb = state.image_is_rgb
        height, width = image.shape[:2]
        if not state.encoded_image:
            state.encoded_image = self._encode_image(image, is_rgb=is_rgb)
        encoded = state.encoded_image
        directory_entries: list[dict[str, Any]] = []
        index = None
        if state.current_path and state.files:
            for i, item in enumerate(state.files):
                is_current = item == state.current_path
                if is_current:
                    index = i
                directory_entries.append(
                    {
                        "name": item.name,
                        "path": str(item),
                        "isCurrent": is_current,
                    },
                )
        config: dict[str, Any] = {
            "sessionId": state.session_id,
            "width": int(width),
            "height": int(height),
            "imageDataUrl": encoded,
            "colorTable": get_instance_color_table().tolist(),
            # Use frontend fallback for initial brush size; omit explicit default.
            "maskOpacity": 0.8,
            "maskThreshold": -2.0,
            "flowThreshold": 0.0,
            "cluster": True,
            "affinitySeg": True,
            "imagePath": str(state.current_path) if state.current_path else None,
            "imageName": state.current_path.name if state.current_path else "Sample Image",
            "directoryEntries": directory_entries,
            "directoryIndex": index,
            "directoryPath": str(state.directory) if state.directory else None,
            "hasPrev": bool(index is not None and index > 0),
            "hasNext": bool(index is not None and index < len(state.files) - 1),
            "isRgb": is_rgb,
            "useWebglPipeline": True,
        }
        saved_state = state.saved_states.get(state.path_key())
        if saved_state:
            try:
                sanitized = json.loads(json.dumps(saved_state))
            except Exception:
                sanitized = saved_state
            config["savedViewerState"] = sanitized
            state.saved_states[state.path_key()] = sanitized
        return config

    def _encode_image(self, array: np.ndarray, *, is_rgb: bool) -> str:
        buffer = io.BytesIO()
        if is_rgb and array.ndim == 3 and array.shape[-1] == 2:
            # promote 2-channel images to 3 for PNG compatibility
            rgb = np.empty((*array.shape[:-1], 3), dtype=array.dtype)
            rgb[..., :2] = array
            rgb[..., 2] = 0
            imageio.imwrite(buffer, rgb, format="png")
        else:
            imageio.imwrite(buffer, array, format="png")
        data = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{data}"

    def navigate(self, state: SessionState, delta: int) -> Optional[Path]:
        if not state.files or state.current_path is None:
            return None
        try:
            idx = state.files.index(state.current_path)
        except ValueError:
            return None
        target = idx + delta
        if target < 0 or target >= len(state.files):
            return None
        return state.files[target]

    def save_viewer_state(self, state: SessionState, image_path: Optional[Path], viewer_state: dict[str, Any]) -> None:
        key = _session_path_key(image_path if image_path is not None else state.current_path)
        try:
            state.saved_states[key] = json.loads(json.dumps(viewer_state))
        except Exception:
            state.saved_states[key] = viewer_state


SESSION_MANAGER = SessionManager()


# for a general UI, could make a window that lets users specify a kwarg dict
# or could have a config file that allows specification of title. slider types
class Segmenter:
    def __init__(self) -> None:
        self._model = None
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
            from omnipose import gpu as omni_gpu  # type: ignore

            _, available = omni_gpu.use_gpu(0, use_torch=True)
            self._use_gpu = bool(available)
        except Exception:
            self._use_gpu = False

    def _ensure_model(self) -> None:
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    from cellpose_omni import models  # local import to avoid startup cost

                    self._model = models.CellposeModel(
                        gpu=self._use_gpu,
                        model_type="bact_phase_affinity",
                    )

    def preload_modules_async(self, delay: float = 0.0) -> None:
        if self._modules_preloaded:
            return
        if self._preload_thread is not None and self._preload_thread.is_alive():
            return

        def _target() -> None:
            if delay > 0:
                time.sleep(delay)
            try:
                import cellpose_omni.models  # noqa: F401  (import for side effects)
                import omnipose.utils  # noqa: F401
                self._modules_preloaded = True
                print("[pywebview] segmenter module preload completed", flush=True)
            except Exception as exc:  # pragma: no cover - diagnostics only
                print(f"[pywebview] segmenter module preload failed: {exc}", flush=True)
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
        from omnipose.utils import normalize99

        self._cache = None
        self._ensure_model()
        arr = np.asarray(image, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr.mean(axis=-1)
        arr = normalize99(arr)
        parsed, merged_options = self._parse_options(settings, overrides)
        with self._eval_lock:
            masks, flows, *rest = self._model.eval(
                [arr],
                channels=None,
                rescale=None,
                mask_threshold=parsed["mask_threshold"], # should make these use kwarg dicts 
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
        mask_uint32 = np.ascontiguousarray(mask.astype(np.uint32, copy=False))
        flow_components = self._extract_flows(flows)
        self._cache = self._build_cache(arr, flow_components, parsed, merged_options, mask_uint32.shape)
        ncolor_mask = self._compute_ncolor_mask(mask_uint32, expand=True)
        if self._cache is None:
            self._cache = {}
        cache = self._cache
        cache["mask"] = mask_uint32
        cache["ncolor_mask"] = ncolor_mask
        if parsed.get("affinity_seg"):
            affinity_data = self._compute_affinity_graph(mask_uint32)
            if affinity_data is not None:
                cache["affinity_graph"] = affinity_data
            else:
                cache.pop("affinity_graph", None)
        else:
            cache.pop("affinity_graph", None)
        return mask_uint32

    def resegment(self, settings: Mapping[str, Any] | None = None, **overrides: Any) -> np.ndarray:
        if not self.has_cache:
            raise RuntimeError("no cached segmentation data available")
        parsed, merged_options = self._parse_options(settings, overrides)
        cache = self._cache or {}
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
        mask, p_out, _, bounds, _ = core_module.compute_masks(
            dP=dP,
            dist=dist,
            bd=bd,
            p=p,
            mask_threshold=parsed["mask_threshold"],
            flow_threshold=parsed["flow_threshold"],
            resize=cache["mask_shape"],
            rescale=rescale_value,
            cluster=parsed["cluster"],
            affinity_seg=parsed["affinity_seg"],
            omni=True,
            nclasses=cache["nclasses"],
            dim=cache["dim"],
        )
        mask_uint32 = np.ascontiguousarray(mask.astype(np.uint32, copy=False))
        ncolor_mask = self._compute_ncolor_mask(mask_uint32, expand=True)
        cache["mask"] = mask_uint32
        cache["ncolor_mask"] = ncolor_mask
        cache["mask_shape"] = tuple(mask_uint32.shape)
        cache["last_mask_threshold"] = parsed["mask_threshold"]
        cache["last_flow_threshold"] = parsed["flow_threshold"]
        cache["last_options"] = merged_options
        if parsed["affinity_seg"]:
            cache["bounds"] = bounds
            affinity_data = self._compute_affinity_graph(mask_uint32)
            if affinity_data is not None:
                cache["affinity_graph"] = affinity_data
            else:
                cache.pop("affinity_graph", None)
        else:
            cache.pop("bounds", None)
            cache.pop("affinity_graph", None)
        if p_out is not None:
            cache["p"] = p_out
        self._cache = cache
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
                from omnipose import gpu as omni_gpu  # type: ignore

                _, available = omni_gpu.use_gpu(0, use_torch=True)
                next_value = bool(available)
            except Exception:
                next_value = False
        if next_value == self._use_gpu:
            return
        self._use_gpu = next_value
        # Force model rebuild on next use
        self._model = None

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

        parsed = {
            "mask_threshold": _get_float("mask_threshold", -2.0),
            "flow_threshold": _get_float("flow_threshold", 0.0),
            "cluster": _get_bool("cluster", True),
            "affinity_seg": _get_bool("affinity_seg", True),
            "transparency": _get_bool("transparency", True),
            "omni": _get_bool("omni", True),
            "resample": _get_bool("resample", True),
            "verbose": _get_bool("verbose", False),
            "tile": _get_bool("tile", False),
            "niter": merged.get("niter"),
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

    def _compute_affinity_graph(self, mask: np.ndarray) -> Optional[tuple[np.ndarray, np.ndarray]]:
        mask_int = np.asarray(mask, dtype=np.int32)
        if mask_int.ndim != 2:
            return None
        coords = np.nonzero(mask_int)
        if coords[0].size == 0:
            return None
        steps, inds, idx, fact, sign = self._get_kernel_info(mask_int.ndim)
        center_index = int(idx)
        if center_index <= 0:
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
        """Relabel using a provided spatial affinity graph (S,H,W) and its step offsets.

        Converts spatial (S,H,W) to (S,N) using current mask foreground coords and
        runs affinity_to_masks to obtain consistent instance labels.
        """
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
        # steps is (S,2) for 2D; canonicalize to kernel order (center + all neighbors)
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
        step_to_idx = { (int(s[0]), int(s[1])): i for i, s in enumerate(k_steps) }
        for i in range(steps_arr.shape[0]):
            key = (int(steps_arr[i, 0]), int(steps_arr[i, 1]))
            j = step_to_idx.get(key)
            if j is not None:
                spatial_full[j] = spatial[i]
        # neighbors / neigh_inds built against canonical steps
        neighbors = utils_module.get_neighbors(coords, k_steps, dim, shape)
        _, neigh_inds, _ = utils_module.get_neigh_inds(tuple(neighbors), coords, shape)
        # Convert spatial (S_full,H,W) to (S_full,N) by indexing at coords
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
        relabeled = np.array(relabeled, dtype=np.int32, copy=False)
        # Fallback: if relabel produced empty mask, keep the original
        if relabeled.size == 0 or int(np.max(relabeled)) == 0:
            print('[relabel_from_affinity] WARNING: empty relabel result; returning original mask')
            return mask_int
        return relabeled

    # def relabel_by_affinity(self, mask: np.ndarray, links: list[tuple[int, int]] | None = None) -> np.ndarray:
    #     """Relabel by recomputing an affinity graph from mask + optional label links, then affinity_to_masks."""
    #     core_module = self._ensure_core()
    #     utils_module = self._ensure_utils()
    #     mask_int = np.asarray(mask, dtype=np.int32)
    #     if mask_int.ndim != 2:
    #         return mask_int.astype(np.int32, copy=False)
    #     shape = mask_int.shape
    #     coords = np.nonzero(mask_int)
    #     if coords[0].size == 0:
    #         return mask_int.astype(np.int32, copy=False)
    #     dim = mask_int.ndim
    #     steps, inds, idx, fact, sign = self._get_kernel_info(dim)
    #     affinity_graph = core_module.masks_to_affinity(
    #         mask_int,
    #         coords,
    #         steps,
    #         inds,
    #         idx,
    #         fact,
    #         sign,
    #         dim,
    #         links=links or [],
    #     )
    #     neighbors = utils_module.get_neighbors(coords, steps, dim, shape)
    #     _, neigh_inds, _ = utils_module.get_neigh_inds(tuple(neighbors), coords, shape)
    #     iscell = mask_int > 0
    #     relabeled = core_module.affinity_to_masks(
    #         affinity_graph,
    #         neigh_inds,
    #         iscell,
    #         coords,
    #         cardinal=True,
    #         exclude_interior=False,
    #         return_edges=False,
    #         verbose=False,
    #     )
    #     relabeled = np.array(relabeled, dtype=np.int32, copy=False)
    #     if relabeled.size == 0 or int(np.max(relabeled)) == 0:
    #         print('[relabel_by_affinity] WARNING: empty relabel (recompute); returning original mask')
    #         return mask_int
    #     return relabeled

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
        # Debug: report mapping for the first few labels.
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
            pass
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
    def _extract_flows(flows: Any) -> list[np.ndarray] | None:
        if flows is None:
            return None
        candidate = flows
        if isinstance(candidate, (list, tuple)) and candidate and isinstance(candidate[0], (list, tuple)):
            candidate = candidate[0]
        if candidate is None:
            return None
        if isinstance(candidate, (list, tuple)):
            return [np.asarray(item) for item in candidate]
        return [np.asarray(candidate)]

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
        # ensure arrays are contiguous for serialization
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
            flow_overlay = None
        try:
            dist_overlay = self._encode_png(self._prepare_distance_image(flows))
        except Exception:
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

    def clear_cache(self) -> None:
        with self._eval_lock:
            self._cache = None

_SEGMENTER = Segmenter()


def _load_fragment(path: Path) -> str:
    """Return fragment content with leading extract comments removed."""

    lines = path.read_text(encoding="utf-8").splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines and lines[0].lstrip().startswith("<!--"):
        lines.pop(0)
        while lines and not lines[0].strip():
            lines.pop(0)
    return "\n".join(lines)


def _get_index_template() -> str:
    global _INDEX_HTML_CACHE
    try:
        mtime = INDEX_HTML.stat().st_mtime_ns
    except FileNotFoundError:
        mtime = -1
    cached_content = _INDEX_HTML_CACHE.get("content")
    if cached_content and _INDEX_HTML_CACHE.get("mtime") == mtime:
        return cached_content  # type: ignore[return-value]
    content = INDEX_HTML.read_text(encoding="utf-8")
    _INDEX_HTML_CACHE = {"content": content, "mtime": mtime}
    return content


def _snapshot_mtimes(paths: Sequence[Path]) -> dict[str, int]:
    mtimes: dict[str, int] = {}
    for path in paths:
        try:
            mtimes[str(path)] = path.stat().st_mtime_ns
        except FileNotFoundError:
            mtimes[str(path)] = -1
    return mtimes


def _get_layout_markup() -> str:
    global _LAYOUT_MARKUP_CACHE
    mtimes = _snapshot_mtimes(HTML_FRAGMENTS)
    cached_markup = _LAYOUT_MARKUP_CACHE.get("markup")
    cached_mtimes = _LAYOUT_MARKUP_CACHE.get("mtimes")
    if cached_markup and cached_mtimes == mtimes:
        return cached_markup  # type: ignore[return-value]
    markup = "\n".join(_load_fragment(path) for path in HTML_FRAGMENTS)
    _LAYOUT_MARKUP_CACHE = {"markup": markup, "mtimes": mtimes}
    return markup


def _concat_cached_text(paths: Sequence[Path], cache: dict[str, object]) -> str:
    mtimes = _snapshot_mtimes(paths)
    cached_text = cache.get("text")
    if cached_text and cache.get("mtimes") == mtimes:
        return cached_text  # type: ignore[return-value]
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    cache["text"] = text
    cache["mtimes"] = mtimes
    return text


def _prime_static_caches() -> None:
    try:
        _INDEX_HTML_CACHE["content"] = INDEX_HTML.read_text(encoding="utf-8")
        _INDEX_HTML_CACHE["mtime"] = INDEX_HTML.stat().st_mtime_ns
    except FileNotFoundError:
        _INDEX_HTML_CACHE["content"] = ""
        _INDEX_HTML_CACHE["mtime"] = None
    _LAYOUT_MARKUP_CACHE["markup"] = "\n".join(_load_fragment(path) for path in HTML_FRAGMENTS)
    _LAYOUT_MARKUP_CACHE["mtimes"] = _snapshot_mtimes(HTML_FRAGMENTS)
    _INLINE_CSS_CACHE["text"] = "\n".join(path.read_text(encoding="utf-8") for path in CSS_FILES)
    _INLINE_CSS_CACHE["mtimes"] = _snapshot_mtimes(CSS_FILES)
    _INLINE_JS_CACHE["text"] = "\n\n".join(path.read_text(encoding="utf-8") for path in JS_FILES)
    _INLINE_JS_CACHE["mtimes"] = _snapshot_mtimes(JS_FILES)


_prime_static_caches()


def render_index(
    config: dict[str, object],
    *,
    inline_assets: bool,
    cache_buster: str | None = None,
) -> str:
    html = _get_index_template()
    layout_markup = _get_layout_markup()
    placeholder = '    <div id="app"></div>'
    if placeholder in html:
        html = html.replace(
            placeholder,
            f"    <div id=\"app\">\n{layout_markup}\n    </div>",
        )
    config_json = json.dumps(config).replace('</', '<\/')
    debug_webgl = bool(config.get("debugWebgl"))
    config_script = (
        f"<script>window.__OMNI_CONFIG__ = {config_json}; "
        f"window.__OMNI_WEBGL_LOGGING__ = {json.dumps(debug_webgl)};</script>"
    )
    capture_script = CAPTURE_LOG_SCRIPT.strip()
    capture_script = "    " + capture_script.replace("\n", "\n    ")
    css_links = list(CSS_LINKS)
    script_tag = '    <script src="/static/app.js"></script>'
    keep_order_comment = (
        "<!-- IMPORTANT: Viewer scripts must remain classic scripts in this order. "
        "Switching to type=\"module\" breaks PyWebView image loading. -->"
    )

    if inline_assets:
        css_text = _concat_cached_text(CSS_FILES, _INLINE_CSS_CACHE)
        html = html.replace(css_links[0], f"    <style>{css_text}</style>")
        for link in css_links[1:]:
            html = html.replace(f"{link}\n", "")
            html = html.replace(link, "")
        js_bundle = _concat_cached_text(JS_FILES, _INLINE_JS_CACHE)
        bundled_script = f"<script>\n/* {keep_order_comment[5:-4]} */\n{js_bundle}\n</script>"
        bundled_script = "    " + bundled_script.replace("\n", "\n    ")
        html = html.replace(
            script_tag,
            "\n".join([
                config_script,
                capture_script,
                bundled_script,
            ]),
        )
    else:
        suffix = f"?v={cache_buster}" if cache_buster else ""
        for link in css_links:
            html = html.replace(
                link,
                link.replace(".css\"", f".css{suffix}\""),
            )
        script_parts = [config_script, capture_script, f'    {keep_order_comment}']
        script_parts.extend(
            f'    <script src="{path}{suffix}"></script>'
            for path in JS_STATIC_PATHS
        )
        html = html.replace(script_tag, "\n".join(script_parts))
    return html


def run_segmentation(
    settings: Mapping[str, Any] | None = None,
    *,
    state: SessionState | None = None,
) -> dict[str, object]:
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
    return {
        "mask": encoded,
        "width": int(width),
        "height": int(height),
        "canRebuild": _SEGMENTER.has_cache,
        "flowOverlay": flow_overlay,
        "distanceOverlay": dist_overlay,
        "nColorMask": encoded_ncolor,
        "affinityGraph": affinity_graph,
    }


def run_mask_update(
    settings: Mapping[str, Any] | None = None,
    *,
    state: SessionState | None = None,
) -> dict[str, object]:
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
            print(f"[resegment] ncolor groups={int(np.unique(ncolor_mask[ncolor_mask>0]).size)}")
        except Exception:
            pass
    affinity_graph = _SEGMENTER.get_affinity_graph_payload()
    return {
        "mask": encoded,
        "width": int(width),
        "height": int(height),
        "canRebuild": _SEGMENTER.has_cache,
        "flowOverlay": flow_overlay,
        "distanceOverlay": dist_overlay,
        "nColorMask": encoded_ncolor,
        "affinityGraph": affinity_graph,
    }





class DebugAPI:
    def __init__(self, log_path: Path | None = None) -> None:
        self._log_path = log_path or WEBGL_LOG_PATH

    def log(self, message: str) -> None:
        message = str(message)
        try:
            with self._log_path.open("a", encoding="utf-8") as fh:
                fh.write(message + "\n")
        except OSError:
            # Ignore logging errors (e.g., reloader closing descriptors)
            return

    def segment(self, settings: Mapping[str, Any] | None = None) -> dict[str, object]:
        mode = None
        if isinstance(settings, Mapping):
            mode = settings.get("mode")
        state = SESSION_MANAGER.get_or_create(None)
        if mode == "recompute":
            return run_mask_update(settings, state=state)
        return run_segmentation(settings, state=state)

    def resegment(self, settings: Mapping[str, Any] | None = None) -> dict[str, object]:
        state = SESSION_MANAGER.get_or_create(None)
        return run_mask_update(settings, state=state)

    def get_ncolor(self) -> dict[str, object]:
        """Return only the current N-color mask from the cached segmentation, if available."""
        try:
            ncolor_mask = _SEGMENTER.get_ncolor_mask()
        except Exception as exc:  # pragma: no cover - diagnostics only
            return {"error": str(exc)}
        if ncolor_mask is None:
            return {"nColorMask": None}
        encoded = base64.b64encode(np.ascontiguousarray(ncolor_mask).tobytes()).decode("ascii")
        try:
            print(f"[api] get_ncolor groups={int(np.unique(ncolor_mask[ncolor_mask>0]).size)}")
        except Exception:
            pass
        return {"nColorMask": encoded}

    def relabel_from_affinity(self, payload: Mapping[str, Any]) -> dict[str, object]:
        try:
            mask_b64 = payload.get("mask")
            width = int(payload.get("width"))
            height = int(payload.get("height"))
        except Exception:
            return {"error": "invalid payload"}
        if not mask_b64 or width <= 0 or height <= 0:
            return {"error": "missing mask/shape"}
        try:
            raw = base64.b64decode(mask_b64)
            arr = np.frombuffer(raw, dtype=np.uint32)
            if arr.size != width * height:
                return {"error": "mask size mismatch"}
            mask = arr.reshape((height, width)).astype(np.int32, copy=False)
        except Exception as exc:
            return {"error": f"decode failed: {exc}"}

        ag = payload.get("affinityGraph")
        if not isinstance(ag, Mapping):
            return {"error": "affinityGraph required"}
        try:
            w = int(ag.get("width")); h = int(ag.get("height"))
            steps_list = ag.get("steps")
            enc = ag.get("encoded")
            if w <= 0 or h <= 0:
                return {"error": "invalid affinityGraph size"}
            if not isinstance(steps_list, list) or not isinstance(enc, str):
                return {"error": "invalid affinityGraph payload"}
            raw_aff = base64.b64decode(enc)
            arr_aff = np.frombuffer(raw_aff, dtype=np.uint8)
            s = len(steps_list)
            if arr_aff.size != s * h * w:
                return {"error": "affinityGraph data size mismatch"}
            spatial = arr_aff.reshape((s, h, w))
            steps = np.asarray(steps_list, dtype=np.int16)
        except Exception as exc:
            return {"error": f"invalid affinityGraph: {exc}"}

        try:
            print(f"[relabel_from_affinity] using provided spatial affinity S={spatial.shape[0]}")
            before = int(np.unique(mask[mask > 0]).size)
            new_labels = _SEGMENTER.relabel_from_affinity(mask, spatial, steps)
            after = int(np.unique(new_labels[new_labels > 0]).size)
            print(f"[relabel_from_affinity] labels_before={before} labels_after={after}")
        except Exception as exc:
            import traceback, sys
            print("[relabel_from_affinity] EXCEPTION:", file=sys.stderr)
            traceback.print_exc()
            return {"error": f"{type(exc).__name__}: {exc}"}

        cache = _SEGMENTER._cache or {}
        cache["mask"] = np.ascontiguousarray(new_labels.astype(np.uint32, copy=False))
        cache["ncolor_mask"] = _SEGMENTER.get_ncolor_mask()
        # Preserve the provided spatial affinity graph exactly; do not recompute
        try:
            cache["affinity_graph"] = (
                np.ascontiguousarray(steps.astype(np.int8, copy=False)),
                np.ascontiguousarray(spatial.astype(np.uint8, copy=False)),
            )
        except Exception:
            pass
        _SEGMENTER._cache = cache
        encoded_mask = base64.b64encode(cache["mask"].tobytes()).decode("ascii")
        ncolor_mask = cache.get("ncolor_mask")
        encoded_ncolor = None
        if ncolor_mask is not None:
            encoded_ncolor = base64.b64encode(np.ascontiguousarray(ncolor_mask).tobytes()).decode("ascii")
        affinity_payload = _SEGMENTER.get_affinity_graph_payload()
        return {
            "mask": encoded_mask,
            "nColorMask": encoded_ncolor,
            "affinityGraph": affinity_payload,
            "width": int(width),
            "height": int(height),
        }

    # def relabel_by_affinity(self, payload: Mapping[str, Any]) -> dict[str, object]:
    #     try:
    #         mask_b64 = payload.get("mask")
    #         width = int(payload.get("width"))
    #         height = int(payload.get("height"))
    #     except Exception:
    #         return {"error": "invalid payload"}
    #     if not mask_b64 or width <= 0 or height <= 0:
    #         return {"error": "missing mask/shape"}
    #     try:
    #         raw = base64.b64decode(mask_b64)
    #         arr = np.frombuffer(raw, dtype=np.uint32)
    #         if arr.size != width * height:
    #             return {"error": "mask size mismatch"}
    #         mask = arr.reshape((height, width)).astype(np.int32, copy=False)
    #     except Exception as exc:
    #         return {"error": f"decode failed: {exc}"}
    #     # Optional links from frontend to force connectivity between labels touched by the stroke
    #     links_payload = payload.get("links")
    #     links_list: list[tuple[int, int]] | None = None
    #     if isinstance(links_payload, list):
    #         try:
    #             links_list = [(int(a), int(b)) for a, b in links_payload if int(a) > 0 and int(b) > 0 and int(a) != int(b)]
    #         except Exception:
    #             links_list = None
    #     # If client provided a spatial affinity graph, use it directly
    #     spatial = None
    #     steps = None
    #     ag = payload.get("affinityGraph")
    #     if isinstance(ag, Mapping):
    #         try:
    #             w = int(ag.get("width")); h = int(ag.get("height"))
    #             steps_list = ag.get("steps")
    #             enc = ag.get("encoded")
    #             if w > 0 and h > 0 and isinstance(steps_list, list) and isinstance(enc, str):
    #                 raw = base64.b64decode(enc)
    #                 arr = np.frombuffer(raw, dtype=np.uint8)
    #                 s = len(steps_list)
    #                 if arr.size == s * h * w:
    #                     spatial = arr.reshape((s, h, w))
    #                     steps = np.asarray(steps_list, dtype=np.int16)
    #         except Exception:
    #             spatial = None
    #             steps = None

    #     # if spatial is not None and steps is not None:
    #     print(f"[relabel_by_affinity] using provided spatial affinity S={spatial.shape[0]}")
    #     new_labels = _SEGMENTER.relabel_from_affinity(mask, spatial, steps)
    #     # else:
    #     #     print(f"[relabel_by_affinity] recomputing affinity; links={len(links_list) if links_list else 0}")
    #     #     new_labels = _SEGMENTER.relabel_by_affinity(mask, links=links_list)
    #     # Update cache with new labels and overlays
    #     cache = _SEGMENTER._cache or {}
    #     cache["mask"] = np.ascontiguousarray(new_labels.astype(np.uint32, copy=False))
    #     cache["ncolor_mask"] = _SEGMENTER.get_ncolor_mask()
    #     affinity = _SEGMENTER._compute_affinity_graph(cache["mask"]) if cache.get("mask") is not None else None
    #     _SEGMENTER._cache = cache
    #     if affinity is not None:
    #         cache["affinity_graph"] = affinity
    #     else:
    #         cache.pop("affinity_graph", None)
    #     # Build response
    #     encoded_mask = base64.b64encode(cache["mask"].tobytes()).decode("ascii")
    #     ncolor_mask = cache.get("ncolor_mask")
    #     encoded_ncolor = None
    #     if ncolor_mask is not None:
    #         encoded_ncolor = base64.b64encode(np.ascontiguousarray(ncolor_mask).tobytes()).decode("ascii")
    #     # Return the current affinity graph payload from the Segmenter cache
    #     affinity_payload = _SEGMENTER.get_affinity_graph_payload()
    #     try:
    #         print(f"[relabel_by_affinity] labels_before={int(np.unique(mask[mask>0]).size)} labels_after={int(np.unique(new_labels[new_labels>0]).size)}")
    #     except Exception:
    #         pass
    #     return {
    #         "mask": encoded_mask,
    #         "nColorMask": encoded_ncolor,
    #         "affinityGraph": affinity_payload,
    #         "width": int(width),
    #         "height": int(height),
    #     }

    def ncolor_from_mask(self, payload: Mapping[str, Any]) -> dict[str, object]:
        try:
            mask_b64 = payload.get("mask")
            width = int(payload.get("width"))
            height = int(payload.get("height"))
        except Exception:
            return {"error": "invalid payload"}
        if not mask_b64 or width <= 0 or height <= 0:
            return {"error": "missing mask/shape"}
        try:
            raw = base64.b64decode(mask_b64)
            arr = np.frombuffer(raw, dtype=np.uint32)
            if arr.size != width * height:
                return {"error": "mask size mismatch"}
            mask = arr.reshape((height, width)).astype(np.int32, copy=False)
        except Exception as exc:
            return {"error": f"decode failed: {exc}"}
        expand = bool(payload.get("expand", True))
        ncm = _SEGMENTER._compute_ncolor_mask(mask, expand=expand)
        if ncm is None:
            return {"nColorMask": None}
        try:
            ngroups = int(np.unique(ncm[ncm > 0]).size)
            print(f"[ncolor-from-mask] groups={ngroups}")
        except Exception:
            pass
        encoded = base64.b64encode(np.ascontiguousarray(ncm.astype(np.uint32, copy=False)).tobytes()).decode("ascii")
        return {"nColorMask": encoded, "width": int(width), "height": int(height)}

    def format_labels(self, payload: Mapping[str, Any]) -> dict[str, object]:
        try:
            mask_b64 = payload.get("mask")
            width = int(payload.get("width"))
            height = int(payload.get("height"))
        except Exception:
            return {"error": "invalid payload"}
        if not mask_b64 or width <= 0 or height <= 0:
            return {"error": "missing mask/shape"}
        try:
            raw = base64.b64decode(mask_b64)
            arr = np.frombuffer(raw, dtype=np.uint32)
            if arr.size != width * height:
                return {"error": "mask size mismatch"}
            mask = arr.reshape((height, width)).astype(np.int32, copy=False)
        except Exception as exc:
            return {"error": f"decode failed: {exc}"}
        try:
            import ncolor  # type: ignore
            formatted = ncolor.format_labels(mask, clean=False, min_area=1, despur=False, verbose=False)
        except Exception as exc:
            return {"error": f"format_labels failed: {exc}"}
        encoded = base64.b64encode(np.ascontiguousarray(formatted.astype(np.uint32, copy=False)).tobytes()).decode("ascii")
        return {"mask": encoded, "width": int(width), "height": int(height)}




def build_html(config: Mapping[str, Any], *, inline_assets: bool = True) -> str:
    start = time.perf_counter()
    html = render_index(config, inline_assets=inline_assets, cache_buster=_CACHE_BUSTER)
    total_elapsed = time.perf_counter() - start
    print(
        f"[pywebview] build_html rendered in {total_elapsed*1000:.1f}ms (inline_assets={inline_assets})",
        flush=True,
    )
    return html


def _pick_free_port(host: str = "127.0.0.1") -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def _wait_for_port(host: str, port: int, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with closing(socket.create_connection((host, port), timeout=0.5)):
                return
        except OSError:
            time.sleep(0.05)
    raise RuntimeError(f"server at {host}:{port} did not become ready within {timeout} seconds")


def _start_uvicorn_thread(
    host: str,
    port: int,
    *,
    ssl_cert: str | None = None,
    ssl_key: str | None = None,
) -> tuple["uvicorn.Server", threading.Thread]:
    import uvicorn

    app = create_app()
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        ssl_certfile=ssl_cert,
        ssl_keyfile=ssl_key,
        log_level="info",
    )
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, name="UvicornThread", daemon=True)
    thread.start()
    while not server.started:
        if not thread.is_alive():
            raise RuntimeError("uvicorn server thread exited prematurely")
        time.sleep(0.05)
    return server, thread


def _start_uvicorn_subprocess(
    host: str,
    port: int,
    *,
    reload: bool = False,
    ssl_cert: str | None = None,
    ssl_key: str | None = None,
) -> subprocess.Popen:
    args = [
        sys.executable,
        "-m",
        "uvicorn",
        "gui.examples.pywebview_image_viewer:create_app",
        "--factory",
        "--host",
        host,
        "--port",
        str(port),
        "--log-level",
        "info",
    ]
    if reload:
        args.append("--reload")
        # Always watch app + web assets
        args.extend(["--reload-dir", str(Path(__file__).resolve().parent)])
        args.extend(["--reload-dir", str(WEB_DIR)])
        # Also watch editable third-party dirs we rely on (e.g., ncolor)
        try:
            import ncolor  # type: ignore
            args.extend(["--reload-dir", str(Path(ncolor.__file__).resolve().parent)])
        except Exception:
            pass
    if ssl_cert and ssl_key:
        args.extend(["--ssl-certfile", ssl_cert, "--ssl-keyfile", ssl_key])
    process = subprocess.Popen(args, stdout=sys.stdout, stderr=sys.stderr)
    return process


def run_desktop(
    *,
    host: str = "127.0.0.1",
    port: int | None = None,
    ssl_cert: str | None = None,
    ssl_key: str | None = None,
    reload: bool = False,
    snapshot_path: str | None = None,
    snapshot_timeout: float = 4.0,
    eval_js: str | None = None,
) -> None:
    app_start = time.perf_counter()

    def log_timing(label: str, reference: float = SCRIPT_START) -> None:
        elapsed = (time.perf_counter() - reference) * 1000.0
        print(f"[pywebview] {label} at {elapsed:.1f} ms", flush=True)

    serve_host = host or "127.0.0.1"
    serve_port = port if port and port > 0 else _pick_free_port(serve_host)
    scheme = "https" if ssl_cert and ssl_key else "http"

    server = None
    server_thread: Optional[threading.Thread] = None
    server_process: Optional[subprocess.Popen] = None

    try:
        if reload:
            server_process = _start_uvicorn_subprocess(
                serve_host,
                serve_port,
                reload=True,
                ssl_cert=ssl_cert,
                ssl_key=ssl_key,
            )
        else:
            server, server_thread = _start_uvicorn_thread(
                serve_host,
                serve_port,
                ssl_cert=ssl_cert,
                ssl_key=ssl_key,
            )
        _wait_for_port(serve_host, serve_port, timeout=10.0)
    except Exception:
        if server_process:
            server_process.terminate()
        raise

    window_url = f"{scheme}://{serve_host}:{serve_port}/"
    print(f"[pywebview] desktop UI loading {window_url}", flush=True)

    snapshot_target = Path(snapshot_path).expanduser() if snapshot_path else None
    automation_needed = bool(snapshot_target or eval_js)
    snapshot_timeout = max(0.1, snapshot_timeout)
    loaded_event = threading.Event()

    window = webview.create_window(
        "Omnipose PyWebView Viewer",
        url=window_url,
        width=1024,
        height=768,
        resizable=True,
        hidden=automation_needed,
    )

    if automation_needed:
        try:
            window.move(-20000, -20000)
            window.resize(1024, 768)
        except Exception:
            pass

    def _automation_worker():
        wait_for_load_sec = max(snapshot_timeout, 10.0)
        if not loaded_event.wait(timeout=wait_for_load_sec):
            print('[pywebview] automation timeout waiting for window load', flush=True)
            try:
                webview.destroy_window()
            except Exception:
                pass
            return
        if automation_needed:
            try:
                window.show()
            except Exception:
                pass
        if eval_js:
            try:
                result = window.evaluate_js(eval_js)
                print(f"[pywebview] eval-js result: {result!r}", flush=True)
            except Exception as exc:
                print(f"[pywebview] eval-js error: {exc}", file=sys.stderr)
        if snapshot_target:
            if snapshot_target.parent and not snapshot_target.parent.exists():
                snapshot_target.parent.mkdir(parents=True, exist_ok=True)
            try:
                prep_result = window.evaluate_js("(function(){try{if(typeof window.setImageVisible==='function'){window.setImageVisible(true,{silent:true});} if(typeof window.maskVisible==='boolean'){window.maskVisible=true;} if(typeof window.resetView==='function'){window.resetView();} else if(typeof window.fitViewToWindow==='function'){window.fitViewToWindow();} if(typeof window.draw==='function'){window.draw();} return {ok:true};}catch(e){return {ok:false, reason:String(e)};}})();")
                print(f"[pywebview] snapshot prep: {prep_result!r}", flush=True)
            except Exception as exc:
                print(f"[pywebview] snapshot prep error: {exc}", file=sys.stderr)
                prep_result = {"ok": False, "reason": str(exc)}
            time.sleep(0.25)
            try:
                capture_raw = window.evaluate_js("(function(){var canvas=document.getElementById('canvas'); if(!canvas){return {ok:false, reason:'no-canvas'};} if(!canvas.width||!canvas.height){return {ok:false, reason:'zero-size'};} try {var dataUrl=canvas.toDataURL('image/png'); return {ok:true, dataUrl:dataUrl};} catch(e){return {ok:false, reason:String(e)};}})();")
                print(f"[pywebview] snapshot raw: {capture_raw!r}", flush=True)
                capture_info = capture_raw if isinstance(capture_raw, dict) else (json.loads(capture_raw) if capture_raw else {'ok': False, 'reason': 'no-result'})
            except Exception as exc:
                capture_info = {'ok': False, 'reason': f'capture-eval-error: {exc}'}
            if capture_info.get('ok') and isinstance(capture_info.get('dataUrl'), str):
                data_url = capture_info['dataUrl']
                _, _, payload = data_url.partition(',')
                try:
                    snapshot_target.write_bytes(base64.b64decode(payload))
                    print(f"[pywebview] snapshot saved to {snapshot_target}", flush=True)
                except Exception as exc:
                    print(f"[pywebview] snapshot save failed: {exc}", file=sys.stderr)
            else:
                print(f"[pywebview] snapshot capture failed: {capture_info}", file=sys.stderr)
        try:
            webview.destroy_window()
        except Exception:
            pass

    def on_window_loaded() -> None:
        log_timing("window loaded")
        _SEGMENTER.preload_modules_async(delay=0.1)
        loaded_event.set()

    def on_window_shown() -> None:
        log_timing("window shown")

    def on_window_closing() -> None:
        log_timing("window closing")

    def on_window_closed() -> None:
        log_timing("window closed")

    window.events.loaded += on_window_loaded
    window.events.shown += on_window_shown
    window.events.closing += on_window_closing
    window.events.closed += on_window_closed

    def on_start() -> None:
        elapsed = (time.perf_counter() - app_start) * 1000.0
        print(f"[pywebview] event loop started after {elapsed:.1f} ms", flush=True)
        log_timing("event loop started")
        if automation_needed:
            threading.Thread(
                target=_automation_worker,
                name="ViewerAutomation",
                daemon=True,
            ).start()

    webview.start(on_start)

    if server_process:
        server_process.terminate()
        try:
            server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_process.kill()
    if server:
        server.should_exit = True
    if server_thread:
        server_thread.join(timeout=5)


def _get_system_info() -> dict[str, object]:
    total = None
    available = None
    used = None
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        total = int(vm.total)
        available = int(vm.available)
        used = int(vm.total - vm.available)
    except Exception:
        try:
            with open('/proc/meminfo', 'r', encoding='utf-8') as handle:
                data = handle.read().splitlines()
            meminfo = {}
            for line in data:
                parts = line.split(':', 1)
                if len(parts) != 2:
                    continue
                key = parts[0].strip()
                value = parts[1].strip().split()[0]
                meminfo[key] = int(value) * 1024
            total = meminfo.get('MemTotal')
            available = meminfo.get('MemAvailable')
            if total is not None and available is not None:
                used = int(total - available)
        except Exception:
            pass
    cpu_cores = os.cpu_count() or 1
    gpu_available = False
    gpu_name = None
    try:
        from omnipose import gpu as omni_gpu  # type: ignore

        device, gpu_ok = omni_gpu.use_gpu(0, use_torch=True)
        gpu_available = bool(gpu_ok)
        gpu_name = getattr(device, 'type', None)
        if gpu_available:
            try:
                import torch  # type: ignore

                gpu_name = torch.cuda.get_device_name(0)
            except Exception:
                gpu_name = 'GPU available'
    except Exception:
        gpu_available = False
        gpu_name = None
    return {
        'ram_total': total,
        'ram_available': available,
        'ram_used': used,
        'cpu_cores': cpu_cores,
        'gpu_available': gpu_available,
        'gpu_name': gpu_name,
        'use_gpu': _SEGMENTER.get_use_gpu(),
    }


def create_app() -> "FastAPI":
    from contextlib import asynccontextmanager
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles

    api = DebugAPI(log_path=WEBGL_LOG_PATH)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        _SEGMENTER.preload_modules_async(delay=0.0)
        yield

    app = FastAPI(title="Omnipose PyWebView Viewer", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        session_cookie = request.cookies.get(SESSION_COOKIE_NAME)
        state = SESSION_MANAGER.get_or_create(session_cookie)
        config = SESSION_MANAGER.build_config(state)
        html = build_html(config, inline_assets=False)
        response = HTMLResponse(html)
        response.set_cookie(
            SESSION_COOKIE_NAME,
            state.session_id,
            max_age=7 * 24 * 60 * 60,
            secure=False,
            httponly=False,
            samesite="Lax",
        )
        return response

    @app.post("/api/log", response_class=JSONResponse)
    async def api_log(payload: dict) -> JSONResponse:
        entries = payload.get('entries')
        if isinstance(entries, list):
            for entry in entries:
                try:
                    api.log(json.dumps(entry, ensure_ascii=False))
                except Exception:
                    api.log(str(entry))
            return JSONResponse({'status': 'ok'})
        messages = payload.get('messages')
        if isinstance(messages, list):
            for raw in messages:
                api.log(str(raw))
            return JSONResponse({'status': 'ok'})
        payload_type = payload.get('type')
        if payload_type == 'JS_ERROR':
            detail = payload.get('payload') or {}
            api.log('JS_ERROR')
            for key in ('message', 'filename', 'lineno', 'colno', 'stack'):
                api.log(f'    {key}: {detail.get(key)}')
        else:
            api.log(str(payload.get('message', '')))
        return JSONResponse({'status': 'ok'})

    @app.post("/api/open_image", response_class=JSONResponse)
    async def api_open_image(payload: dict) -> JSONResponse:
        if not isinstance(payload, dict):
            return JSONResponse({"error": "invalid payload"}, status_code=400)
        session_id = payload.get("sessionId")
        if not isinstance(session_id, str):
            return JSONResponse({"error": "sessionId required"}, status_code=400)
        try:
            state = SESSION_MANAGER.get(session_id)
        except KeyError:
            return JSONResponse({"error": "unknown session"}, status_code=404)
        path_value = payload.get("path")
        direction = payload.get("direction")
        try:
            if isinstance(path_value, str) and path_value:
                SESSION_MANAGER.set_image(state, Path(path_value))
            elif isinstance(direction, str) and direction in {"next", "prev"}:
                delta = 1 if direction == "next" else -1
                target = SESSION_MANAGER.navigate(state, delta)
                if target is None:
                    return JSONResponse({"error": "no_image"}, status_code=404)
                SESSION_MANAGER.set_image(state, target)
            else:
                return JSONResponse({"error": "path or direction required"}, status_code=400)
            return JSONResponse({"ok": True})
        except FileNotFoundError:
            return JSONResponse({"error": "file_not_found"}, status_code=404)
        except Exception as exc:  # pragma: no cover
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.get("/api/system_info", response_class=JSONResponse)
    async def api_system_info() -> JSONResponse:
        return JSONResponse(_get_system_info())

    @app.post("/api/use_gpu", response_class=JSONResponse)
    async def api_use_gpu(payload: dict | None = None) -> JSONResponse:
        payload = payload or {}
        enabled = payload.get("use_gpu")
        if enabled is None:
            return JSONResponse({"error": "use_gpu required"}, status_code=400)
        _SEGMENTER.set_use_gpu(bool(enabled))
        info = _get_system_info()
        return JSONResponse(info)

    @app.post("/api/save_state", response_class=JSONResponse)
    async def api_save_state(payload: dict) -> JSONResponse:
        if not isinstance(payload, dict):
            return JSONResponse({"error": "invalid payload"}, status_code=400)
        session_id = payload.get("sessionId")
        if not isinstance(session_id, str):
            return JSONResponse({"error": "sessionId required"}, status_code=400)
        try:
            state = SESSION_MANAGER.get(session_id)
        except KeyError:
            return JSONResponse({"error": "unknown session"}, status_code=404)
        viewer_state = payload.get("viewerState")
        if not isinstance(viewer_state, dict):
            return JSONResponse({"error": "viewerState required"}, status_code=400)
        image_path_raw = payload.get("imagePath")
        path_obj: Optional[Path] = None
        if isinstance(image_path_raw, str) and image_path_raw:
            path_obj = Path(image_path_raw).expanduser().resolve()
        SESSION_MANAGER.save_viewer_state(state, path_obj, viewer_state)
        return JSONResponse({"status": "ok"})

    @app.post("/api/segment", response_class=JSONResponse)
    async def api_segment(payload: dict | None = None) -> JSONResponse:
        if not isinstance(payload, dict):
            payload = {}
        session_id = payload.get("sessionId")
        if not isinstance(session_id, str):
            return JSONResponse({"error": "sessionId required"}, status_code=400)
        try:
            state = SESSION_MANAGER.get(session_id)
        except KeyError:
            return JSONResponse({"error": "unknown session"}, status_code=404)
        try:
            mode = payload.get("mode")
            if mode == "recompute":
                result = run_mask_update(payload, state=state)
            else:
                result = run_segmentation(payload, state=state)
            return JSONResponse(result)
        except Exception as exc:  # pragma: no cover - propagate error to client
            import traceback, sys
            print("[segment] EXCEPTION:", file=sys.stderr)
            traceback.print_exc()
            return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=500)

    @app.post("/api/resegment", response_class=JSONResponse)
    async def api_resegment(payload: dict | None = None) -> JSONResponse:
        if not isinstance(payload, dict):
            payload = {}
        session_id = payload.get("sessionId")
        if not isinstance(session_id, str):
            return JSONResponse({"error": "sessionId required"}, status_code=400)
        try:
            state = SESSION_MANAGER.get(session_id)
        except KeyError:
            return JSONResponse({"error": "unknown session"}, status_code=404)
        try:
            return JSONResponse(run_mask_update(payload, state=state))
        except Exception as exc:  # pragma: no cover - propagate error to client
            import traceback, sys
            print("[resegment] EXCEPTION:", file=sys.stderr)
            traceback.print_exc()
            return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=500)

    @app.post("/api/clear_cache", response_class=JSONResponse)
    async def api_clear_cache(request: Request) -> JSONResponse:
        session_cookie = request.cookies.get(SESSION_COOKIE_NAME)
        state: Optional[SessionState] = None
        if session_cookie:
            try:
                state = SESSION_MANAGER.get(session_cookie)
            except KeyError:
                state = None
        if state is not None:
            SESSION_MANAGER.clear_saved_states(state)
        _SEGMENTER.clear_cache()
        return JSONResponse({"status": "ok"})
    
    @app.post("/api/relabel_from_affinity", response_class=JSONResponse)
    async def api_relabel_from_affinity(payload: dict | None = None) -> JSONResponse:
        try:
            payload = payload or {}
            return JSONResponse(DebugAPI().relabel_from_affinity(payload))
        except Exception as exc:  # pragma: no cover
            import traceback, sys
            print("[relabel_from_affinity] EXCEPTION:", file=sys.stderr)
            traceback.print_exc()
            return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=500)

    @app.get("/api/ncolor", response_class=JSONResponse)
    async def api_ncolor() -> JSONResponse:
        try:
            return JSONResponse(DebugAPI().get_ncolor())
        except Exception as exc:  # pragma: no cover
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.post("/api/ncolor_from_mask", response_class=JSONResponse)
    async def api_ncolor_from_mask(payload: dict | None = None) -> JSONResponse:
        try:
            payload = payload or {}
            return JSONResponse(DebugAPI().ncolor_from_mask(payload))
        except Exception as exc:  # pragma: no cover
            return JSONResponse({"error": str(exc)}, status_code=500)
    @app.post("/api/format_labels", response_class=JSONResponse)
    async def api_format_labels(payload: dict | None = None) -> JSONResponse:
        try:
            return JSONResponse(DebugAPI().format_labels(payload or {}))
        except Exception:
            print("[format_labels] EXCEPTION:", file=sys.stderr)
            return JSONResponse({"error": "format_labels failed"})



    return app


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    ssl_cert: str | None = None,
    ssl_key: str | None = None,
    reload: bool = False,
    https_dev: bool = False,
) -> None:
    if https_dev and (ssl_cert or ssl_key):
        print("[pywebview] ignoring --https-dev because custom SSL cert or key provided", flush=True)
    if https_dev and not (ssl_cert and ssl_key):
        try:
            ssl_cert, ssl_key = _ensure_dev_certificate()
            print(f"[pywebview] using development TLS certificate at {ssl_cert}", flush=True)
        except Exception as exc:  # pragma: no cover - dev convenience
            print(f"[pywebview] failed to provision dev certificate: {exc}", flush=True)
            print("[pywebview] continuing without HTTPS", flush=True)
            ssl_cert = None
            ssl_key = None

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - convenience guard
        print(
            "FastAPI and uvicorn are required for --server mode. "
            "Install with 'pip install fastapi uvicorn'.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    scheme = "https" if ssl_cert and ssl_key else "http"
    presented_urls = []
    if host in {"0.0.0.0", "::"}:
        presented_urls.append(f"{scheme}://localhost:{port}")
        presented_urls.append(f"{scheme}://127.0.0.1:{port}")
    else:
        presented_urls.append(f"{scheme}://{host}:{port}")
    for url in presented_urls:
        print(f"[pywebview] serving at {url}", flush=True)

    if reload:
        # Include ncolor package directory in reload set if available (editable installs)
        reload_dirs = [str(Path(__file__).resolve().parent), str(WEB_DIR)]
        try:
            import ncolor  # type: ignore
            reload_dirs.append(str(Path(ncolor.__file__).resolve().parent))
        except Exception:
            pass
        uvicorn.run(
            "gui.examples.pywebview_image_viewer:create_app",
            factory=True,
            host=host,
            port=port,
            reload=True,
            reload_dirs=reload_dirs,
            ssl_certfile=ssl_cert,
            ssl_keyfile=ssl_key,
            log_level="info",
        )
        return

    app = create_app()
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        ssl_certfile=ssl_cert,
        ssl_keyfile=ssl_key,
        log_level="info",
    )
    server = uvicorn.Server(config)
    server.run()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Omnipose PyWebView Viewer")
    parser.add_argument(
        "--server",
        action="store_true",
        help="Launch as an HTTPS-capable FastAPI server instead of a desktop window.",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Server host when using --server (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port when using --server (default: 8000)",
    )
    parser.add_argument(
        "--ssl-cert",
        default=None,
        help="Path to SSL certificate for HTTPS server mode",
    )
    parser.add_argument(
        "--ssl-key",
        default=None,
        help="Path to SSL private key for HTTPS server mode",
    )
    # Server reload flags (default: OFF)
    parser.add_argument(
        "--reload",
        dest="reload",
        action="store_true",
        default=False,
        help="Enable uvicorn auto-reload for web server mode.",
    )
    parser.add_argument(
        "--no-reload",
        dest="reload",
        action="store_false",
        help="Disable uvicorn auto-reload for web server mode.",
    )
    parser.add_argument(
        "--https-dev",
        action="store_true",
        help="Serve over HTTPS using a temporary self-signed localhost certificate (requires openssl).",
    )
    parser.add_argument(
        "--desktop-host",
        default="127.0.0.1",
        help="Host interface for the embedded desktop server (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--desktop-port",
        type=int,
        default=0,
        help="Port for the embedded desktop server (default: auto)",
    )
    # Desktop-embedded server reload flags (default: OFF)
    parser.add_argument(
        "--desktop-reload",
        dest="desktop_reload",
        action="store_true",
        default=False,
        help="Enable uvicorn --reload for the embedded desktop server (development only).",
    )
    parser.add_argument(
        "--no-desktop-reload",
        dest="desktop_reload",
        action="store_false",
        help="Disable uvicorn --reload for the embedded desktop server.",
    )
    parser.add_argument(
        "--snapshot",
        metavar="PNG_PATH",
        help="Capture the viewer canvas to the given PNG file and exit.",
    )
    parser.add_argument(
        "--snapshot-timeout",
        type=float,
        default=4.0,
        help="Seconds to wait for the first draw before giving up on --snapshot (default: 4).",
    )
    parser.add_argument(
        "--eval-js",
        dest="eval_js",
        default=None,
        help="JavaScript snippet to evaluate after the viewer loads (testing/automation).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.server:
        run_server(
            host=args.host,
            port=args.port,
            ssl_cert=args.ssl_cert,
            ssl_key=args.ssl_key,
            reload=args.reload,
            https_dev=args.https_dev,
        )
    else:
        run_desktop(
            host=args.desktop_host,
            port=args.desktop_port if args.desktop_port and args.desktop_port > 0 else None,
            ssl_cert=args.ssl_cert,
            ssl_key=args.ssl_key,
            reload=args.desktop_reload,
            snapshot_path=args.snapshot,
            snapshot_timeout=args.snapshot_timeout,
            eval_js=args.eval_js,
        )


if __name__ == "__main__":
    main()
