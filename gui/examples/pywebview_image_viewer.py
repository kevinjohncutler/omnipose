"""Minimal PyWebView viewer replicating core Omnipose image interactions."""

from __future__ import annotations

import argparse
import base64
import io
import json
import shutil
import subprocess
import sys
import tempfile
import time
import threading
from pathlib import Path

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

    def segment(self, image: np.ndarray) -> np.ndarray:
        from omnipose.utils import normalize99

        self._ensure_model()
        arr = np.asarray(image, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr.mean(axis=-1)
        arr = normalize99(arr)
        with self._eval_lock:
            masks, *_ = self._model.eval(
                [arr],
                channels=None,
                rescale=None,
                mask_threshold=-2,
                flow_threshold=0.0,
                transparency=True,
                omni=True,
                cluster=True,
                resample=True,
                verbose=False,
                tile=False,
                niter=None,
                augment=False,
                affinity_seg=True,
            )
        return self._remap_labels(masks[0])

    @staticmethod
    def _remap_labels(mask: np.ndarray) -> np.ndarray:
        mask = np.asarray(mask)
        if mask.ndim != 2:
            raise ValueError("expected 2D mask from model")
        labels = np.unique(mask)
        labels = labels[labels > 0]
        if not len(labels):
            return np.zeros_like(mask, dtype=np.uint8)
        max_label = int(labels.max())
        mapping = np.zeros(max_label + 1, dtype=np.uint16)
        new_vals = ((np.arange(len(labels)) % 9) + 1).astype(np.uint16)
        mapping[labels.astype(np.int64)] = new_vals
        remapped = mapping[mask]
        return remapped.astype(np.uint8)


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


def run_segmentation() -> dict[str, object]:
    image = get_source_image()
    mask = _SEGMENTER.segment(image)
    encoded = base64.b64encode(mask.tobytes()).decode("ascii")
    height, width = mask.shape
    return {
        "mask": encoded,
        "width": int(width),
        "height": int(height),
    }





class DebugAPI:
    def __init__(self, log_path: Path | None = None) -> None:
        self._log_path = log_path or Path("pywebview_debug.log")

    def log(self, message: str) -> None:
        message = str(message)
        with self._log_path.open("a", encoding="utf-8") as fh:
            fh.write(message + "\n")

    def segment(self) -> dict[str, object]:
        return run_segmentation()


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
    }
    return render_index(config, inline_assets=inline_assets, cache_buster=_CACHE_BUSTER)


def run_desktop() -> None:
    app_start = time.perf_counter()

    def log_timing(label: str, reference: float = SCRIPT_START) -> None:
        elapsed = (time.perf_counter() - reference) * 1000.0
        print(f"[pywebview] {label} at {elapsed:.1f} ms", flush=True)

    html = build_html()
    api = DebugAPI()
    window = webview.create_window(
        "Omnipose PyWebView Viewer",
        html=html,
        width=1024,
        height=768,
        resizable=True,
        js_api=api,
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
    async def api_segment() -> JSONResponse:
        try:
            return JSONResponse(run_segmentation())
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
        "--reload",
        action="store_true",
        help="Enable uvicorn auto-reload for web server mode (development only).",
    )
    parser.add_argument(
        "--https-dev",
        action="store_true",
        help="Serve over HTTPS using a temporary self-signed localhost certificate (requires openssl).",
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
        run_desktop()


if __name__ == "__main__":
    main()
