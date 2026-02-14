"""FastAPI application factory and server launchers for the Omnipose GUI."""

from __future__ import annotations

import base64
import json
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import closing
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

import webview
from starlette.requests import Request

from .assets import WEB_DIR, GUI_DIR, build_html, append_gui_log
from .routes import DebugAPI, WEBGL_LOG_PATH, _choose_path_osascript
from .segmentation import _SEGMENTER, run_segmentation, run_mask_update
from .session import SESSION_MANAGER, SESSION_COOKIE_NAME, SessionState
from .system import get_system_info

if TYPE_CHECKING:
    import uvicorn

# Script start time for timing logs
SCRIPT_START = time.perf_counter()

# Dev certificate directory
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
        "gui.server.app:create_app",
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
        try:
            import ncolor  # type: ignore
            args.extend(["--reload-dir", str(Path(ncolor.__file__).resolve().parent)])
        except Exception:
            pass
    if ssl_cert and ssl_key:
        args.extend(["--ssl-certfile", ssl_cert, "--ssl-keyfile", ssl_key])
    process = subprocess.Popen(args, stdout=sys.stdout, stderr=sys.stderr)
    return process


def create_app() -> "Any":
    """Create and configure the FastAPI application."""
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
        t_start = time.perf_counter()
        session_cookie = request.cookies.get(SESSION_COOKIE_NAME)
        state = SESSION_MANAGER.get_or_create(session_cookie)
        t_session = time.perf_counter()
        config = SESSION_MANAGER.build_config(state)
        t_config = time.perf_counter()
        html = build_html(config, inline_assets=False)
        t_html = time.perf_counter()
        print(f"[perf] GET / session: {(t_session-t_start)*1000:.0f}ms, config: {(t_config-t_session)*1000:.0f}ms, html: {(t_html-t_config)*1000:.0f}ms, total: {(t_html-t_start)*1000:.0f}ms, size: {len(html)//1024}KB", flush=True)
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
                    line = json.dumps(entry, ensure_ascii=False)
                except Exception:
                    line = str(entry)
                api.log(line)
                append_gui_log(line)
            return JSONResponse({'status': 'ok'})
        messages = payload.get('messages')
        if isinstance(messages, list):
            for raw in messages:
                line = str(raw)
                api.log(line)
                append_gui_log(line)
            return JSONResponse({'status': 'ok'})
        payload_type = payload.get('type')
        if payload_type == 'JS_ERROR':
            detail = payload.get('payload') or {}
            api.log('JS_ERROR')
            append_gui_log('JS_ERROR')
            for key in ('message', 'filename', 'lineno', 'colno', 'stack'):
                line = f'    {key}: {detail.get(key)}'
                api.log(line)
                append_gui_log(line)
        else:
            line = str(payload.get('message', ''))
            api.log(line)
            append_gui_log(line)
        return JSONResponse({'status': 'ok'})

    @app.post("/api/open_image", response_class=JSONResponse)
    async def api_open_image(payload: dict) -> JSONResponse:
        t_start = time.perf_counter()
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
            print(f"[perf] open_image: {(time.perf_counter()-t_start)*1000:.0f}ms", flush=True)
            return JSONResponse({"ok": True})
        except FileNotFoundError:
            return JSONResponse({"error": "file_not_found"}, status_code=404)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.post("/api/open_image_folder", response_class=JSONResponse)
    async def api_open_image_folder(payload: dict) -> JSONResponse:
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
        if not isinstance(path_value, str) or not path_value:
            return JSONResponse({"error": "path required"}, status_code=400)
        try:
            folder_path = Path(path_value).expanduser().resolve()
            if folder_path.is_file():
                folder_path = folder_path.parent
            if not folder_path.exists() or not folder_path.is_dir():
                return JSONResponse({"error": "not_a_directory"}, status_code=400)
            files = SESSION_MANAGER._list_directory_images(folder_path)
            if not files:
                return JSONResponse({"error": "no_images"}, status_code=404)
            target = None
            if state.current_path and state.current_path.parent == folder_path:
                target = state.current_path
            else:
                target = files[0]
            SESSION_MANAGER.set_image(state, target)
            return JSONResponse({"ok": True})
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.post("/api/select_image_file", response_class=JSONResponse)
    async def api_select_image_file(payload: dict | None = None) -> JSONResponse:
        payload = payload or {}
        session_id = payload.get("sessionId")
        if not isinstance(session_id, str):
            return JSONResponse({"error": "sessionId required"}, status_code=400)
        try:
            state = SESSION_MANAGER.get(session_id)
        except KeyError:
            return JSONResponse({"error": "unknown session"}, status_code=404)
        if sys.platform == "darwin":
            file_path = _choose_path_osascript("file")
            if not file_path:
                return JSONResponse({"error": "cancelled"}, status_code=400)
        else:
            try:
                import tkinter as tk
                from tkinter import filedialog
            except Exception as exc:
                return JSONResponse({"error": f"tk_unavailable: {exc}"}, status_code=500)
            root = None
            try:
                root = tk.Tk()
                root.withdraw()
                root.attributes('-topmost', True)
                try:
                    root.update()
                except Exception:
                    pass
                file_path = filedialog.askopenfilename(
                    title="Select image",
                    filetypes=[
                        ("Images", "*.tif *.tiff *.png *.jpg *.jpeg *.bmp"),
                        ("All files", "*.*"),
                    ],
                    parent=root,
                )
            except Exception as exc:
                print(f"[select_image_file] dialog failed: {exc}")
                return JSONResponse({"error": str(exc)}, status_code=500)
            finally:
                if root is not None:
                    try:
                        root.destroy()
                    except Exception:
                        pass
        if not file_path:
            return JSONResponse({"error": "cancelled"}, status_code=400)
        try:
            path_obj = Path(file_path).expanduser().resolve()
            if not path_obj.exists() or not path_obj.is_file():
                return JSONResponse({"error": "file_not_found"}, status_code=404)
            SESSION_MANAGER.set_image(state, path_obj)
            return JSONResponse({"ok": True})
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.post("/api/select_image_folder", response_class=JSONResponse)
    async def api_select_image_folder(payload: dict | None = None) -> JSONResponse:
        payload = payload or {}
        session_id = payload.get("sessionId")
        if not isinstance(session_id, str):
            return JSONResponse({"error": "sessionId required"}, status_code=400)
        try:
            state = SESSION_MANAGER.get(session_id)
        except KeyError:
            return JSONResponse({"error": "unknown session"}, status_code=404)
        if sys.platform == "darwin":
            folder = _choose_path_osascript("folder")
            if not folder:
                return JSONResponse({"error": "cancelled"}, status_code=400)
        else:
            try:
                import tkinter as tk
                from tkinter import filedialog
            except Exception as exc:
                return JSONResponse({"error": f"tk_unavailable: {exc}"}, status_code=500)
            root = None
            try:
                root = tk.Tk()
                root.withdraw()
                root.attributes('-topmost', True)
                try:
                    root.update()
                except Exception:
                    pass
                folder = filedialog.askdirectory(parent=root)
            except Exception as exc:
                print(f"[select_image_folder] dialog failed: {exc}")
                return JSONResponse({"error": str(exc)}, status_code=500)
            finally:
                if root is not None:
                    try:
                        root.destroy()
                    except Exception:
                        pass
        if not folder:
            return JSONResponse({"error": "cancelled"}, status_code=400)
        try:
            folder_path = Path(folder).expanduser().resolve()
            if not folder_path.exists() or not folder_path.is_dir():
                return JSONResponse({"error": "not_a_directory"}, status_code=400)
            files = SESSION_MANAGER._list_directory_images(folder_path)
            if not files:
                return JSONResponse({"error": "no_images"}, status_code=404)
            target = files[0]
            SESSION_MANAGER.set_image(state, target)
            return JSONResponse({"ok": True})
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.get("/api/system_info", response_class=JSONResponse)
    async def api_system_info() -> JSONResponse:
        return JSONResponse(get_system_info(_SEGMENTER))

    @app.post("/api/use_gpu", response_class=JSONResponse)
    async def api_use_gpu(payload: dict | None = None) -> JSONResponse:
        payload = payload or {}
        enabled = payload.get("use_gpu")
        if enabled is None:
            return JSONResponse({"error": "use_gpu required"}, status_code=400)
        _SEGMENTER.set_use_gpu(bool(enabled))
        info = get_system_info(_SEGMENTER)
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
        except Exception as exc:
            import traceback
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
        except Exception as exc:
            import traceback
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
        except Exception as exc:
            import traceback
            print("[relabel_from_affinity] EXCEPTION:", file=sys.stderr)
            traceback.print_exc()
            return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=500)

    @app.get("/api/ncolor", response_class=JSONResponse)
    async def api_ncolor() -> JSONResponse:
        try:
            return JSONResponse(DebugAPI().get_ncolor())
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.post("/api/ncolor_from_mask", response_class=JSONResponse)
    async def api_ncolor_from_mask(payload: dict | None = None) -> JSONResponse:
        try:
            payload = payload or {}
            return JSONResponse(DebugAPI().ncolor_from_mask(payload))
        except Exception as exc:
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
    """Run the FastAPI server."""
    if https_dev and (ssl_cert or ssl_key):
        print("[pywebview] ignoring --https-dev because custom SSL cert or key provided", flush=True)
    if https_dev and not (ssl_cert and ssl_key):
        try:
            ssl_cert, ssl_key = _ensure_dev_certificate()
            print(f"[pywebview] using development TLS certificate at {ssl_cert}", flush=True)
        except Exception as exc:
            print(f"[pywebview] failed to provision dev certificate: {exc}", flush=True)
            print("[pywebview] continuing without HTTPS", flush=True)
            ssl_cert = None
            ssl_key = None

    try:
        import uvicorn
    except ImportError as exc:
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
        reload_dirs = [str(Path(__file__).resolve().parent), str(WEB_DIR)]
        try:
            import ncolor  # type: ignore
            reload_dirs.append(str(Path(ncolor.__file__).resolve().parent))
        except Exception:
            pass
        uvicorn.run(
            "gui.server.app:create_app",
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
    """Run the desktop PyWebView application."""
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
