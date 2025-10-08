# Omnipose PyWebView Image Viewer

This Markdown document describes the **`pywebview_image_viewer.py`** example used in the Omnipose GUI module.  
It demonstrates how to embed an interactive image editor and mask painter inside a PyWebView desktop application.

---

## Overview

The script opens a local PyWebView window with an HTML5-based canvas viewer.  
It mirrors the same interactive behaviors used in the Omnipose desktop GUI—brush painting, mask editing, undo/redo, and zoom/pan support.

Key points:

- Uses PyWebView to host an HTML/JS frontend locally.
- Loads image data encoded as base64 and color maps via Python.
- Implements live brush previews, fill tools, gamma adjustment, and label selection.
- Designed to be portable to a FastAPI or web-deployed version.

---

## Hotkeys and Controls

| Key / Action | Function |
|---------------|-----------|
| **B** | Brush tool |
| **G** | Flood Fill tool |
| **I** | Picker (color/sample) |
| **[ / ]** | Decrease / Increase brush size |
| **Digits 0–9** | Set current label |
| **M** | Toggle mask visibility |
| **Space** | Pan mode |
| **Scroll wheel** | Zoom in/out |
| **Ctrl+Z / Ctrl+Y** | Undo / Redo |

Mouse:
- **Left click + drag** → Paint
- **Alt/Option + click** → Erase temporarily
- **Right click** → Context menu (reserved for future)

---

## Structure

### Python (Backend)
- Loads a sample image via `load_image_uint8`.
- Initializes the color table and brush parameters.
- Embeds the HTML template using Python’s `Template` substitution.
- Launches a PyWebView window via `webview.create_window`.

### HTML / JavaScript (Frontend)
Implements the viewer, mask logic, and interactivity:

- `<canvas id="canvas">` for image rendering.  
- `<canvas id="brushPreview">` for transient overlay drawing.  
- Sidebar controls for gamma, brush diameter, tool info, and labels.
- JS functions handle painting, fill operations, undo/redo stacks, and cursor updates.

#### Slider Components

The viewer now renders custom sliders that ensure identical visuals across PyWebView, Safari/WebKit, Chromium, and Firefox. Each slider is declared with a minimal wrapper and one or two hidden range inputs:

```html
<div class="slider" data-slider-id="gamma" data-slider-type="single">
  <input type="range" id="gamma" min="10" max="600" value="100" />
</div>
```

To create a dual-handle slider (e.g., for window/level ranges), declare `data-slider-type="dual"` and supply two range inputs:

```html
<div class="slider" data-slider-id="window" data-slider-type="dual">
  <input type="range" min="0" max="255" value="32" step="1" />
  <input type="range" min="0" max="255" value="224" step="1" />
</div>
```

The JavaScript automatically injects track/fill/thumb nodes, synchronizes values, and dispatches the usual `input` events, so downstream logic can keep listening to the hidden range inputs. Dual sliders clamp the thumbs so the minimum never exceeds the maximum.

---

## Conversion for Web Deployment

To make the same UI accessible over HTTPS (multi-user or browser-based):
1. Extract the HTML/JS to static files.
2. Replace all calls to `window.pywebview.api.*` with REST or WebSocket calls.
3. Serve via FastAPI or another framework.
4. Keep all state handling and mask logic as a shared Python backend.

### Dual-Mode Launch Strategy

| Mode | Transport | Frontend | Backend Process | Notes |
|------|-----------|----------|-----------------|-------|
| **Desktop (PyWebView)** | Local IPC | Same HTML bundle embedded in the Python script | Runs in-process with the user | Ideal for single-user, offline workflows. |
| **Browser (HTTPS)** | REST/WebSocket over HTTPS | Static SPA served by ASGI app or CDN | Uvicorn/Gunicorn worker pool executing Python tasks | Enables remote access, multi-core execution, and headless operation. |

The goal is to share the UI bundle and business logic across both launch targets while swapping the transport layer.

### Recommended Web Architecture

1. **Backend Framework**: FastAPI (ASGI) with Uvicorn workers.  
   - Mount a `/ui` route to serve the static viewer assets.  
   - Expose REST endpoints for actions that currently call `window.pywebview.api.*` (logging, mask updates, Omnipose inference jobs, etc.).
2. **State Management**: Use per-session UUIDs (cookies or query params) to isolate mask buffers and Omnipose job state between users.  
   - Store transient state in Redis or an in-memory cache; persist final outputs to disk/S3.
3. **Execution Layer**: 
   - Short-running actions (brush strokes, mask previews) can execute inline on the web worker.
   - Long-running Omnipose jobs should be delegated to a task queue (Celery, RQ) backed by a worker pool sized for available CPU/GPU cores.
4. **Transport Choices**: 
   - Use REST `POST` for idempotent operations (load image, save mask, run inference).  
   - Add WebSocket (FastAPI’s `WebSocketRoute`) for streaming progress logs and real-time brush updates if latency becomes an issue.
5. **Security & Auth**: 
   - Require authentication before exposing arbitrary Python execution.  
   - Sandbox Omnipose commands by constraining working directories and enforcing resource limits (e.g., Docker/Kubernetes pod per session).
6. **HTTPS Termination**: 
   - Run behind a reverse proxy (nginx, Caddy, Traefik) for TLS and request throttling.  
   - Consider deploying via Kubernetes or Docker Compose for scaling multiple instances.

### Development Roadmap

1. **Refactor Frontend**: Move the inline HTML/JS into `gui/static/` and bundle via Vite/Rollup so both desktop and web builds share the same assets.
2. **API Layer**: Draft a minimal FastAPI app that mirrors the PyWebView `js_api` surface (`/log`, `/paint`, `/flood_fill`, `/undo`, `/run_omnipose`).
3. **Session Isolation**: Implement a session manager that tracks per-user masks and Omnipose jobs.  Test with multiple simultaneous clients.
4. **Arbitrary Python Execution**: Wrap Omnipose invocation in a task runner that accepts code or configuration, enforces runtime limits, and streams logs back to the client.
5. **Deployment Prototype**: Package the backend into a container image, run behind nginx with TLS, and validate latency over WAN.
6. **Observability**: Add structured logging and metrics (Prometheus/Grafana) so production instances can be monitored and autoscaled.

### Local Development Conveniences

- Provide a CLI flag (`--server`) that launches the FastAPI app bound to `https://127.0.0.1:8443` using a self-signed cert for quick testing.  
- Maintain `--desktop` (PyWebView) as today’s default for single-user exploration.  
- Share the same Python entry point to avoid divergence between deployment modes.

---

## Example Launch

```bash
python pywebview_image_viewer.py
```

The viewer window will open and load the example image in an interactive canvas.

---

## File Location

Place this file as:
```
omnipose/gui/pywebview_viewer.md
```

This ensures discoverability under the GUI examples directory.
