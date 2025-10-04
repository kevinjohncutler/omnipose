"""Minimal PyWebView viewer replicating core Omnipose image interactions."""

from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path
from string import Template

import webview
from imageio import v2 as imageio

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent))
    from sample_image import (  # type: ignore
        DEFAULT_BRUSH_RADIUS,
        get_instance_color_table,
        load_image_uint8,
    )
else:
    from .sample_image import DEFAULT_BRUSH_RADIUS, get_instance_color_table, load_image_uint8


HTML_TEMPLATE = Template(
    """<!DOCTYPE html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\" />
<title>Omnipose PyWebView Viewer</title>
<style>
html, body { margin: 0; height: 100%; background: #111; color: #eee; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
#app { display: flex; height: 100vh; width: 100vw; }
#viewer { flex: 1; position: relative; background: #111; overflow: hidden; }
#sidebar { width: 240px; padding: 20px; background: #181818; box-sizing: border-box; border-left: 1px solid #2a2a2a; display: flex; flex-direction: column; gap: 12px; }
#gamma { width: 100%; }
canvas { width: 100%; height: 100%; touch-action: none; cursor: grab; background: #111; }
canvas.painting { cursor: crosshair; }
.hint { margin-top: auto; font-size: 0.8rem; color: #aaa; line-height: 1.35; }
</style>
</head>
<body>
<div id=\"app\">
  <div id=\"viewer\">
    <canvas id=\"canvas\" width=\"$width\" height=\"$height\"></canvas>
  </div>
  <div id=\"sidebar\">
    <h2 style=\"margin:0;\">PyWebView</h2>
    <label for=\"gamma\">Gamma</label>
    <input type=\"range\" id=\"gamma\" min=\"10\" max=\"300\" value=\"100\" />
    <div id=\"gammaValue\">Gamma: 1.00</div>
    <div id=\"maskLabel\">Mask Label: 1</div>
    <div id=\"maskVisibility\">Mask Layer: On (toggle with 'M')</div>
    <div class=\"hint\">Hold Shift and drag to paint. Digits 0-9 set label. 'M' toggles mask. Scroll to zoom, drag to pan.</div>
  </div>
</div>
<script>
const imgWidth = $width;
const imgHeight = $height;
const imageDataUrl = "data:image/png;base64,$image_data";
const colorTable = $color_table;
const brushRadius = $brush_radius;

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const viewer = document.getElementById('viewer');
const dpr = window.devicePixelRatio || 1;

const offscreen = document.createElement('canvas');
offscreen.width = imgWidth;
offscreen.height = imgHeight;
const offCtx = offscreen.getContext('2d');

const maskCanvas = document.createElement('canvas');
maskCanvas.width = imgWidth;
maskCanvas.height = imgHeight;
const maskCtx = maskCanvas.getContext('2d');
const maskData = maskCtx.createImageData(imgWidth, imgHeight);
const maskValues = new Uint8Array(imgWidth * imgHeight);

function log(message) {
  const api = window.pywebview ? window.pywebview.api : null;
  if (api && api.log) {
    api.log(message);
  }
}

const brushOffsets = (() => {
  const offsets = [];
  const r2 = brushRadius * brushRadius;
  for (let dy = -brushRadius; dy <= brushRadius; dy += 1) {
    for (let dx = -brushRadius; dx <= brushRadius; dx += 1) {
      if ((dx * dx) + (dy * dy) <= r2) {
        offsets.push({ dx, dy });
      }
    }
  }
  return offsets;
})();

const viewState = { scale: 1.0, offsetX: 0.0, offsetY: 0.0 };
let maskVisible = true;
let currentLabel = 1;
let originalImageData = null;
let isPanning = false;
let isPainting = false;
let lastPoint = { x: 0, y: 0 };
let lastPaintPoint = null;

const gammaSlider = document.getElementById('gamma');
const gammaValue = document.getElementById('gammaValue');
const maskLabel = document.getElementById('maskLabel');
const maskVisibility = document.getElementById('maskVisibility');

function updateMaskLabel() {
  maskLabel.textContent = `Mask Label: $${currentLabel}`;
}

function updateMaskVisibilityLabel() {
  maskVisibility.textContent = `Mask Layer: $${maskVisible ? 'On' : 'Off'} (toggle with 'M')`;
}

function resizeCanvas() {
  const rect = viewer.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(rect.width * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));
  canvas.style.width = `$${rect.width}px`;
  canvas.style.height = `$${rect.height}px`;
  recenterView();
  log(`resize $${rect.width}x$${rect.height} scale=$${viewState.scale.toFixed(2)}`);
  draw();
}

function recenterView() {
  const rect = viewer.getBoundingClientRect();
  viewState.offsetX = (rect.width - imgWidth * viewState.scale) / 2;
  viewState.offsetY = (rect.height - imgHeight * viewState.scale) / 2;
}

function applyGamma(gamma) {
  if (!originalImageData) {
    return;
  }
  const source = originalImageData.data;
  const img = offCtx.createImageData(imgWidth, imgHeight);
  const target = img.data;
  for (let i = 0; i < source.length; i += 4) {
    const value = Math.pow(source[i] / 255, gamma) * 255;
    const v = Math.min(255, Math.max(0, Math.round(value)));
    target[i] = v;
    target[i + 1] = v;
    target[i + 2] = v;
    target[i + 3] = source[i + 3];
  }
  offCtx.putImageData(img, 0, 0);
  draw();
}

function redrawMaskCanvas() {
  const data = maskData.data;
  for (let i = 0; i < maskValues.length; i += 1) {
    const label = maskValues[i];
    const color = colorTable[label] || colorTable[label % colorTable.length];
    const idx = i * 4;
    data[idx] = color[0];
    data[idx + 1] = color[1];
    data[idx + 2] = color[2];
    data[idx + 3] = color[3];
  }
  maskCtx.putImageData(maskData, 0, 0);
}

function draw() {
  ctx.save();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.translate(viewState.offsetX * dpr, viewState.offsetY * dpr);
  ctx.scale(viewState.scale * dpr, viewState.scale * dpr);
  const smooth = viewState.scale < 1;
  ctx.imageSmoothingEnabled = smooth;
  ctx.imageSmoothingQuality = smooth ? 'high' : 'low';
  ctx.drawImage(offscreen, 0, 0);
  if (maskVisible) {
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(maskCanvas, 0, 0);
  }
  ctx.restore();
}

function getPointerPosition(evt) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: evt.clientX - rect.left,
    y: evt.clientY - rect.top,
  };
}

function screenToImage(point) {
  return {
    x: (point.x - viewState.offsetX) / viewState.scale,
    y: (point.y - viewState.offsetY) / viewState.scale,
  };
}

function stampAt(point) {
  const cx = Math.round(point.x);
  const cy = Math.round(point.y);
  for (const offset of brushOffsets) {
    const x = cx + offset.dx;
    const y = cy + offset.dy;
    if (x < 0 || y < 0 || x >= imgWidth || y >= imgHeight) {
      continue;
    }
    const idx = y * imgWidth + x;
    maskValues[idx] = currentLabel;
  }
}

function paintStroke(point) {
  const start = lastPaintPoint || point;
  const dx = point.x - start.x;
  const dy = point.y - start.y;
  const dist = Math.hypot(dx, dy);
  const step = Math.max(1, brushRadius * 0.5);
  const steps = Math.max(1, Math.ceil(dist / step));
  for (let i = 0; i <= steps; i += 1) {
    const t = steps === 0 ? 1 : i / steps;
    const px = start.x + dx * t;
    const py = start.y + dy * t;
    stampAt({ x: px, y: py });
  }
  redrawMaskCanvas();
  draw();
  lastPaintPoint = point;
}

canvas.addEventListener('wheel', (evt) => {
  evt.preventDefault();
  const pointer = getPointerPosition(evt);
  const pointerX = pointer.x;
  const pointerY = pointer.y;
  const imageX = (pointer.x - viewState.offsetX) / viewState.scale;
  const imageY = (pointer.y - viewState.offsetY) / viewState.scale;
  const factor = evt.deltaY < 0 ? 1.1 : 0.9;
  const newScale = Math.min(Math.max(viewState.scale * factor, 0.1), 12);
  viewState.scale = newScale;
  viewState.offsetX = pointer.x - imageX * viewState.scale;
  viewState.offsetY = pointer.y - imageY * viewState.scale;
  draw();
}, { passive: false });

canvas.addEventListener('pointerdown', (evt) => {
  const pointer = getPointerPosition(evt);
  lastPoint = pointer;
  if (evt.button === 0 && evt.shiftKey) {
    isPainting = true;
    canvas.classList.add('painting');
    canvas.setPointerCapture(evt.pointerId);
    const world = screenToImage(pointer);
    lastPaintPoint = null;
    paintStroke(world);
    return;
  }
  isPanning = true;
  canvas.setPointerCapture(evt.pointerId);
});

canvas.addEventListener('pointermove', (evt) => {
  const pointer = getPointerPosition(evt);
  if (isPainting) {
    const world = screenToImage(pointer);
    paintStroke(world);
    return;
  }
  if (!isPanning) {
    return;
  }
  const dx = pointer.x - lastPoint.x;
  const dy = pointer.y - lastPoint.y;
  viewState.offsetX += dx;
  viewState.offsetY += dy;
  lastPoint = pointer;
  draw();
});

function stopInteraction(evt) {
  if (isPainting) {
    canvas.classList.remove('painting');
  }
  isPainting = false;
  isPanning = false;
  lastPaintPoint = null;
  if (evt && evt.pointerId !== undefined) {
    try {
      canvas.releasePointerCapture(evt.pointerId);
    } catch (_) {
      /* ignore */
    }
  }
}

canvas.addEventListener('pointerup', stopInteraction);
canvas.addEventListener('pointerleave', stopInteraction);
canvas.addEventListener('pointercancel', stopInteraction);

window.addEventListener('keydown', (evt) => {
  if (evt.key === 'm' || evt.key === 'M') {
    maskVisible = !maskVisible;
    updateMaskVisibilityLabel();
    draw();
    evt.preventDefault();
    return;
  }
  if (evt.key >= '0' && evt.key <= '9') {
    currentLabel = parseInt(evt.key, 10);
    updateMaskLabel();
    evt.preventDefault();
  }
});

updateMaskLabel();
updateMaskVisibilityLabel();

function initialize() {
  log('initialize');
  const img = new Image();
  img.src = imageDataUrl;
  img.onload = () => {
    log('image loaded');
    offCtx.drawImage(img, 0, 0);
    originalImageData = offCtx.getImageData(0, 0, imgWidth, imgHeight);
    redrawMaskCanvas();
    resizeCanvas();
  };
}

window.addEventListener('resize', resizeCanvas);
gammaSlider.addEventListener('input', (evt) => {
  const value = Math.max(parseInt(evt.target.value, 10), 1);
  const gamma = value / 100.0;
  gammaValue.textContent = `Gamma: $${gamma.toFixed(2)}`;
  applyGamma(gamma);
});

window.addEventListener('load', () => {
  log('window load');
  initialize();
  setTimeout(() => {
    log('post-load resize');
    resizeCanvas();
  }, 100);
});
</script>
</body>
</html>
"""
)


class DebugAPI:
    def __init__(self, log_path: Path | None = None) -> None:
        self._log_path = log_path or Path("pywebview_debug.log")

    def log(self, message: str) -> None:
        message = str(message)
        with self._log_path.open("a", encoding="utf-8") as fh:
            fh.write(message + "\n")


def build_html() -> str:
    image = load_image_uint8(as_rgb=False)
    buffer = io.BytesIO()
    imageio.imwrite(buffer, image, format="png")
    data = base64.b64encode(buffer.getvalue()).decode("ascii")
    height, width = image.shape[:2]
    color_table = json.dumps(get_instance_color_table().tolist())
    return HTML_TEMPLATE.substitute(
        width=width,
        height=height,
        image_data=data,
        color_table=color_table,
        brush_radius=DEFAULT_BRUSH_RADIUS,
    )


def main() -> None:
    html = build_html()
    api = DebugAPI()
    webview.create_window(
        "Omnipose PyWebView Viewer",
        html=html,
        width=1024,
        height=768,
        resizable=True,
        js_api=api,
    )
    webview.start()


if __name__ == "__main__":
    main()
