"""Minimal PyWebView viewer replicating core Omnipose image interactions."""

from __future__ import annotations

import argparse
import base64
import io
import json
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import threading
from contextlib import closing
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

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
    )
else:
    from .sample_image import DEFAULT_BRUSH_RADIUS, get_instance_color_table, load_image_uint8


WEB_DIR = (Path(__file__).resolve().parent.parent / "web").resolve()
INDEX_HTML = WEB_DIR / "index.html"
STYLE_CSS = WEB_DIR / "style.css"
APP_JS = WEB_DIR / "app.js"
_CACHE_BUSTER = str(int(time.time()))
_DEV_CERT_DIR = Path(tempfile.gettempdir()) / "omnipose_pywebview_dev_ssl"


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


_IMAGE_CACHE: np.ndarray | None = None
_IMAGE_LOCK = threading.Lock()


def get_source_image() -> np.ndarray:
    global _IMAGE_CACHE
    if _IMAGE_CACHE is None:
        with _IMAGE_LOCK:
            if _IMAGE_CACHE is None:
                _IMAGE_CACHE = load_image_uint8(as_rgb=False)
    return np.asarray(_IMAGE_CACHE)


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

    def _ensure_model(self) -> None:
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    from cellpose_omni import models  # local import to avoid startup cost

                    self._model = models.CellposeModel(
                        gpu=False,
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
        ncolor_mask = self._compute_ncolor_mask(mask_uint32)
        if self._cache is not None:
            self._cache["mask"] = mask_uint32
            self._cache["ncolor_mask"] = ncolor_mask
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
        mask, p_out, _, bounds, augmented_affinity = core_module.compute_masks(
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
        ncolor_mask = self._compute_ncolor_mask(mask_uint32)
        cache["mask"] = mask_uint32
        cache["ncolor_mask"] = ncolor_mask
        cache["mask_shape"] = tuple(mask_uint32.shape)
        cache["last_mask_threshold"] = parsed["mask_threshold"]
        cache["last_flow_threshold"] = parsed["flow_threshold"]
        cache["last_options"] = merged_options
        if parsed["affinity_seg"]:
            cache["bounds"] = bounds
            cache["affinity"] = augmented_affinity
        else:
            cache.pop("bounds", None)
            cache.pop("affinity", None)
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

    def _compute_ncolor_mask(self, mask: np.ndarray) -> Optional[np.ndarray]:
        try:
            import ncolor
        except ImportError:
            return None
        if mask.size == 0:
            return None
        mask_int = np.asarray(mask, dtype=np.int32)
        try:
            labeled, _ = ncolor.label(mask_int, expand=True, return_n=True)
        except TypeError:
            labeled = ncolor.label(mask_int, expand=True)
        labeled_uint32 = np.ascontiguousarray(labeled.astype(np.uint32, copy=False))
        return labeled_uint32

    def _ensure_core(self):
        if self._core_module is None:
            from omnipose import core as core_module

            self._core_module = core_module
        return self._core_module

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

_SEGMENTER = Segmenter()


def render_index(
    config: dict[str, object],
    *,
    inline_assets: bool,
    cache_buster: str | None = None,
) -> str:
    html = INDEX_HTML.read_text(encoding="utf-8")
    config_script = f"<script>window.__OMNI_CONFIG__ = {json.dumps(config)}</script>"
    if inline_assets:
        css = STYLE_CSS.read_text(encoding="utf-8")
        js = APP_JS.read_text(encoding="utf-8")
        html = html.replace(
            '<link rel="stylesheet" href="/static/style.css" />',
            f"<style>{css}</style>",
        )
        html = html.replace(
            '<script src="/static/app.js"></script>',
            f"{config_script}\n<script>{js}</script>",
        )
    else:
        suffix = f"?v={cache_buster}" if cache_buster else ""
        html = html.replace(
            'href="/static/style.css"',
            f'href="/static/style.css{suffix}"',
        )
        html = html.replace(
            '<script src="/static/app.js"></script>',
            f"{config_script}\n<script src=\"/static/app.js{suffix}\"></script>",
        )
    return html


def run_segmentation(settings: Mapping[str, Any] | None = None) -> dict[str, object]:
    image = get_source_image()
    mask = _SEGMENTER.segment(image, settings=settings or {})
    mask_uint32 = np.ascontiguousarray(mask.astype(np.uint32, copy=False))
    encoded = base64.b64encode(mask_uint32.tobytes()).decode("ascii")
    height, width = mask_uint32.shape
    flow_overlay, dist_overlay = _SEGMENTER.get_overlays()
    ncolor_mask = _SEGMENTER.get_ncolor_mask()
    encoded_ncolor = None
    if ncolor_mask is not None:
        encoded_ncolor = base64.b64encode(np.ascontiguousarray(ncolor_mask).tobytes()).decode("ascii")
    return {
        "mask": encoded,
        "width": int(width),
        "height": int(height),
        "canRebuild": _SEGMENTER.has_cache,
        "flowOverlay": flow_overlay,
        "distanceOverlay": dist_overlay,
        "nColorMask": encoded_ncolor,
    }


def run_mask_update(settings: Mapping[str, Any] | None = None) -> dict[str, object]:
    mask = _SEGMENTER.resegment(settings=settings or {})
    mask_uint32 = np.ascontiguousarray(mask.astype(np.uint32, copy=False))
    encoded = base64.b64encode(mask_uint32.tobytes()).decode("ascii")
    height, width = mask_uint32.shape
    flow_overlay, dist_overlay = _SEGMENTER.get_overlays()
    ncolor_mask = _SEGMENTER.get_ncolor_mask()
    encoded_ncolor = None
    if ncolor_mask is not None:
        encoded_ncolor = base64.b64encode(np.ascontiguousarray(ncolor_mask).tobytes()).decode("ascii")
    return {
        "mask": encoded,
        "width": int(width),
        "height": int(height),
        "canRebuild": _SEGMENTER.has_cache,
        "flowOverlay": flow_overlay,
        "distanceOverlay": dist_overlay,
        "nColorMask": encoded_ncolor,
    }





class DebugAPI:
    def __init__(self, log_path: Path | None = None) -> None:
        self._log_path = log_path or Path("pywebview_debug.log")

    def log(self, message: str) -> None:
        message = str(message)
        with self._log_path.open("a", encoding="utf-8") as fh:
            fh.write(message + "\n")

    def segment(self, settings: Mapping[str, Any] | None = None) -> dict[str, object]:
        mode = None
        if isinstance(settings, Mapping):
            mode = settings.get("mode")
        if mode == "recompute":
            return run_mask_update(settings)
        return run_segmentation(settings)

    def resegment(self, settings: Mapping[str, Any] | None = None) -> dict[str, object]:
        return run_mask_update(settings)


def build_html(*, inline_assets: bool = True) -> str:
    start = time.perf_counter()
    image = get_source_image()
    after_load = time.perf_counter()
    buffer = io.BytesIO()
    imageio.imwrite(buffer, image, format="png")
    after_encode = time.perf_counter()
    data = base64.b64encode(buffer.getvalue()).decode("ascii")
    after_b64 = time.perf_counter()
    load_elapsed = after_load - start
    encode_elapsed = after_encode - after_load
    b64_elapsed = after_b64 - after_encode
    total_elapsed = after_b64 - start
    print(
        f"[pywebview] build_html timings: load={load_elapsed*1000:.1f}ms, "
        f"encode={encode_elapsed*1000:.1f}ms, b64={b64_elapsed*1000:.1f}ms, total={total_elapsed*1000:.1f}ms",
        flush=True,
    )
    height, width = image.shape[:2]
    config = {
        "width": int(width),
        "height": int(height),
        "imageDataUrl": f"data:image/png;base64,{data}",
        "colorTable": get_instance_color_table().tolist(),
        "brushRadius": DEFAULT_BRUSH_RADIUS,
        "maskOpacity": 0.8,
        "maskThreshold": -2.0,
        "flowThreshold": 0.0,
        "cluster": True,
        "affinitySeg": True,
    }
    return render_index(config, inline_assets=inline_assets, cache_buster=_CACHE_BUSTER)


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
        args.extend(["--reload-dir", str(Path(__file__).resolve().parent)])
        args.extend(["--reload-dir", str(WEB_DIR)])
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

    window = webview.create_window(
        "Omnipose PyWebView Viewer",
        url=window_url,
        width=1024,
        height=768,
        resizable=True,
    )

    def on_window_loaded() -> None:
        log_timing("window loaded")
        _SEGMENTER.preload_modules_async(delay=0.1)

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


def create_app() -> "FastAPI":
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles

    api = DebugAPI()

    app = FastAPI(title="Omnipose PyWebView Viewer")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return build_html(inline_assets=False)

    @app.post("/api/log", response_class=JSONResponse)
    async def api_log(payload: dict) -> JSONResponse:
        messages = payload.get("messages")
        if isinstance(messages, list):
            for raw in messages:
                api.log(str(raw))
        else:
            api.log(str(payload.get("message", "")))
        return JSONResponse({"status": "ok"})

    @app.post("/api/segment", response_class=JSONResponse)
    async def api_segment(payload: dict | None = None) -> JSONResponse:
        try:
            mode = None
            if isinstance(payload, dict):
                mode = payload.get("mode")
            if mode == "recompute":
                result = run_mask_update(payload)
            else:
                result = run_segmentation(payload)
            return JSONResponse(result)
        except Exception as exc:  # pragma: no cover - propagate error to client
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.post("/api/resegment", response_class=JSONResponse)
    async def api_resegment(payload: dict | None = None) -> JSONResponse:
        try:
            return JSONResponse(run_mask_update(payload))
        except Exception as exc:  # pragma: no cover - propagate error to client
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.on_event("startup")
    async def preload_modules() -> None:
        _SEGMENTER.preload_modules_async(delay=0.0)

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
        uvicorn.run(
            "gui.examples.pywebview_image_viewer:create_app",
            factory=True,
            host=host,
            port=port,
            reload=True,
            reload_dirs=[str(Path(__file__).resolve().parent), str(WEB_DIR)],
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
    parser.add_argument(
        "--desktop-reload",
        dest="desktop_reload",
        action="store_true",
        default=True,
        help="Use uvicorn --reload for the embedded desktop server (development only).",
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
        )


if __name__ == "__main__":
    main()
