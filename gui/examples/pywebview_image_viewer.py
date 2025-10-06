"""Minimal PyWebView viewer replicating core Omnipose image interactions."""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
import threading
from pathlib import Path
from string import Template

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


HTML_TEMPLATE = Template(
    """<!DOCTYPE html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\" />
<title>Omnipose PyWebView Viewer</title>
<style>
html, body { margin: 0; height: 100%; background: #111; color: #eee; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
#app { position: relative; width: 100vw; height: 100vh; overflow: hidden; }
#viewer { position: absolute; inset: 0; background: #111; overflow: hidden; }
#brushPreview { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; background: transparent; }
#loadingOverlay { position: absolute; inset: 0; background: rgba(17, 17, 17, 0.94); color: #c5ccd4; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 12px; font-size: 0.95rem; letter-spacing: 0.04em; z-index: 10; transition: opacity 180ms ease-out; }
#loadingOverlay.hidden { opacity: 0; pointer-events: none; }
#loadingOverlay .spinner { width: 36px; height: 36px; border-radius: 50%; border: 4px solid rgba(255, 255, 255, 0.16); border-top-color: rgba(255, 255, 255, 0.82); animation: spin 0.9s linear infinite; }
#loadingOverlay .message { text-transform: uppercase; font-size: 0.72rem; color: #8fa1b5; letter-spacing: 0.24em; }
#loadingOverlay.error { background: rgba(46, 18, 18, 0.94); color: #ff9d9d; }
#loadingOverlay.error .message { color: #ff9d9d; }
#sidebar { position: absolute; top: 0; right: 0; width: 260px; height: 100%; padding: 20px; background: rgba(24, 24, 24, 0.55); box-sizing: border-box; border-left: 1px solid rgba(255, 255, 255, 0.1); display: flex; flex-direction: column; gap: 14px; backdrop-filter: blur(18px); -webkit-backdrop-filter: blur(18px); color: var(--panel-text-color); }
.control { display: flex; flex-direction: column; gap: 6px; color: currentColor; }
.slider-row { display: flex; align-items: center; gap: 8px; }
.slider-row input[type="range"] { flex: 1; }
#gammaInput, #brushSizeInput { width: 64px; background: rgba(0, 0, 0, 0.3); border: 1px solid rgba(255, 255, 255, 0.2); color: currentColor; padding: 4px 6px; border-radius: 4px; }
#gamma, #brushSizeSlider { width: 100%; }
canvas.histogram { width: 100%; height: 80px; background: rgba(0, 0, 0, 0.4); border: 1px solid rgba(255, 255, 255, 0.15); border-radius: 6px; cursor: crosshair; touch-action: none; }
button { background: rgba(42, 106, 242, 0.85); border: none; color: #fff; padding: 10px 12px; border-radius: 6px; font-size: 0.95rem; cursor: pointer; transition: background 0.2s ease, transform 0.1s ease; }
button:disabled { opacity: 0.45; cursor: default; }
button:not(:disabled):hover { background: rgba(60, 123, 255, 0.95); transform: translateY(-1px); }
.status { font-size: 0.8rem; color: currentColor; min-height: 1.2rem; opacity: 0.8; }
canvas { width: 100%; height: 100%; touch-action: none; cursor: grab; }
#canvas { background: #111; }
.hint { margin-top: auto; font-size: 0.8rem; color: #aaa; line-height: 1.35; }
@keyframes spin { to { transform: rotate(360deg); } }
:root { --panel-text-color: #f4f4f4; }
@media (prefers-color-scheme: light) {
  :root { --panel-text-color: #1a1a1a; }
  #sidebar { background: rgba(255, 255, 255, 0.55); border-left: 1px solid rgba(0, 0, 0, 0.08); }
  canvas.histogram { background: rgba(255, 255, 255, 0.6); border: 1px solid rgba(0, 0, 0, 0.1); }
  button { background: rgba(42, 106, 242, 0.9); color: #fff; }
  #gammaInput, #brushSizeInput { background: rgba(255, 255, 255, 0.65); border: 1px solid rgba(0, 0, 0, 0.15); }
  .hint { color: rgba(0, 0, 0, 0.55); }
}
</style>
</head>
<body>
<div id=\"app\">
  <div id=\"viewer\">
    <canvas id=\"canvas\" width=\"$width\" height=\"$height\"></canvas>
    <canvas id=\"brushPreview\"></canvas>
    <div id=\"loadingOverlay\">
      <div class=\"spinner\"></div>
      <div class=\"message\">Loading viewer…</div>
    </div>
  </div>
  <div id=\"sidebar\">
    <h2 style=\"margin:0;\">PyWebView</h2>
    <div class=\"control\">
      <label>Intensity Window</label>
      <canvas id=\"histogram\" class=\"histogram\" width=\"220\" height=\"80\"></canvas>
      <div id=\"histRange\" class=\"status\">Window: [0, 255]</div>
    </div>
    <div class=\"control\">
      <label for=\"gamma\">Gamma</label>
      <div class=\"slider-row\">
        <input type=\"range\" id=\"gamma\" min=\"10\" max=\"600\" value=\"100\" />
        <input type=\"number\" id=\"gammaInput\" min=\"0.10\" max=\"6.00\" step=\"0.05\" value=\"1.00\" />
      </div>
      <div id=\"gammaValue\">Gamma: 1.00</div>
    </div>
    <div class=\"control\">
      <button id=\"segmentButton\">Segment (Omnipose)</button>
      <div id=\"segmentStatus\" class=\"status\"></div>
    </div>
    <div class=\"control\">
      <label for=\"brushSizeSlider\">Brush Diameter</label>
      <div class=\"slider-row\">
        <input type=\"range\" id=\"brushSizeSlider\" min=\"0\" max=\"4\" value=\"1\" />
        <input type=\"number\" id=\"brushSizeInput\" min=\"1\" max=\"9\" step=\"2\" value=\"3\" />
      </div>
    </div>
    <div class=\"control\">
      <label for=\"maskOpacity\">Mask Opacity</label>
      <div class=\"slider-row\">
        <input type=\"range\" id=\"maskOpacity\" min=\"0\" max=\"100\" value=\"80\" />
        <div id=\"maskOpacityValue\">80%</div>
      </div>
    </div>
    <div id=\"colorMode\">Mask Colors: Palette (toggle with 'N')</div>
    <div id=\"maskLabel\">Mask Label: 1</div>
    <div id=\"toolInfo\">Tool: Brush (B=Brush, G=Fill, I=Picker)</div>
    <div id=\"maskVisibility\">Mask Layer: On (toggle with 'M')</div>
    <div id=\"hoverInfo\" class=\"status\">Y: --, X: --, Val: --</div>
    <div class=\"hint\">Use B/G/I to switch tools, [ ] to resize brush, digits 0-9 set label. Hold Space to pan, scroll to zoom.</div>
  </div>
</div>
<script>
const imgWidth = $width;
const imgHeight = $height;
const imageDataUrl = "data:image/png;base64,$image_data";
const colorTable = $color_table;
const initialBrushRadius = $brush_radius;
const defaultPalette = generateSinebowPalette(Math.max(colorTable.length || 0, 256), 0.0);
const nColorPalette = generateSinebowPalette(Math.max(colorTable.length || 0, 256), 0.35);

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
const previewCanvas = document.getElementById('brushPreview');
const previewCtx = previewCanvas.getContext('2d');
previewCanvas.width = canvas.width;
previewCanvas.height = canvas.height;

const loadingOverlay = document.getElementById('loadingOverlay');
const loadingOverlayMessage = loadingOverlay ? loadingOverlay.querySelector('.message') : null;
let overlayDismissed = false;

function setLoadingOverlay(message, isError = false) {
  if (!loadingOverlay) {
    return;
  }
  if (typeof message === 'string' && loadingOverlayMessage) {
    loadingOverlayMessage.textContent = message;
  }
  loadingOverlay.classList.toggle('error', Boolean(isError));
}

function hideLoadingOverlay() {
  if (!loadingOverlay || overlayDismissed) {
    return;
  }
  overlayDismissed = true;
  loadingOverlay.classList.add('hidden');
  setTimeout(() => {
    if (loadingOverlay && loadingOverlay.parentElement) {
      loadingOverlay.parentElement.removeChild(loadingOverlay);
    }
  }, 240);
}

const pendingLogs = [];
let pywebviewReady = false;
let loggedPixelSample = false;
let drawLogCount = 0;
let logFlushTimer = null;
const startTime = (typeof performance !== 'undefined' && performance.now)
  ? performance.now()
  : Date.now();
let shuttingDown = false;
const pywebviewPoll = setInterval(() => {
  if (pywebviewReady) {
    clearInterval(pywebviewPoll);
    return;
  }
  const api = window.pywebview ? window.pywebview.api : null;
  if (api && api.log) {
    pywebviewReady = true;
    clearInterval(pywebviewPoll);
    log('pywebview api detected via poll');
    flushLogs();
  }
}, 50);

function pushToQueue(msg) {
  pendingLogs.push(msg);
  if (pendingLogs.length > 200) {
    pendingLogs.shift();
  }
  scheduleLogFlush();
}

function flushLogs() {
  if (logFlushTimer !== null) {
    clearTimeout(logFlushTimer);
    logFlushTimer = null;
  }
  if (!pendingLogs.length) {
    return;
  }
  if (pywebviewReady) {
    const api = window.pywebview && window.pywebview.api && window.pywebview.api.log
      ? window.pywebview.api
      : null;
    if (!api || !api.log) {
      return;
    }
    while (pendingLogs.length) {
      api.log(pendingLogs.shift());
    }
    return;
  }
  if (typeof fetch !== 'function') {
    return;
  }
  const payload = { messages: pendingLogs.splice(0, pendingLogs.length) };
  const body = JSON.stringify(payload);
  try {
    if (navigator && typeof navigator.sendBeacon === 'function') {
      const ok = navigator.sendBeacon('/api/log', new Blob([body], { type: 'application/json' }));
      if (!ok) {
        pendingLogs.unshift(...payload.messages);
        scheduleLogFlush();
      }
    } else {
      fetch('/api/log', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body,
        keepalive: true,
      }).catch(() => {
        pendingLogs.unshift(...payload.messages);
        scheduleLogFlush();
      });
    }
  } catch (_) {
    pendingLogs.unshift(...payload.messages);
    scheduleLogFlush();
  }
}

function scheduleLogFlush(delay = 200) {
  if (logFlushTimer !== null) {
    return;
  }
  logFlushTimer = setTimeout(() => {
    logFlushTimer = null;
    flushLogs();
  }, delay);
}

function formatMessage(message) {
  const now = (typeof performance !== 'undefined' && performance.now)
    ? (performance.now() - startTime)
    : (Date.now() - startTime);
  const timestamp = now.toFixed(1).padStart(7, ' ');
  return '[' + timestamp + ' ms] ' + message;
}

function log(message) {
  const formatted = formatMessage(String(message));
  try {
    console.log('[pywebview]', formatted);
  } catch (_) {
    /* console unavailable */
  }
  const api = window.pywebview ? window.pywebview.api : null;
  if (api && api.log) {
    api.log(formatted);
  } else {
    pushToQueue(formatted);
  }
}

const brushSizes = [1, 3, 5, 7, 9];
function nearestBrushIndex(diameter) {

  let best = 0;
  let diff = Infinity;
  for (let i = 0; i < brushSizes.length; i += 1) {
    const delta = Math.abs(brushSizes[i] - diameter);
    if (delta < diff) {
      diff = delta;
      best = i;
    }
  }
  return best;
}

const defaultDiameter = Math.max(1, Math.min(9, initialBrushRadius * 2 + 1));
let brushIndex = nearestBrushIndex(defaultDiameter);
let brushDiameter = brushSizes[brushIndex];
let brushOffsets = buildBrushOffsets(brushDiameter);

const brushSizeSlider = document.getElementById('brushSizeSlider');
const brushSizeInput = document.getElementById('brushSizeInput');
updateBrushControls();

const gammaSlider = document.getElementById('gamma');
const gammaValue = document.getElementById('gammaValue');
const maskLabel = document.getElementById('maskLabel');
const maskVisibility = document.getElementById('maskVisibility');
const toolInfo = document.getElementById('toolInfo');
const segmentButton = document.getElementById('segmentButton');
const segmentStatus = document.getElementById('segmentStatus');
const colorMode = document.getElementById('colorMode');
const gammaInput = document.getElementById('gammaInput');
const histogramCanvas = document.getElementById('histogram');
const histRangeLabel = document.getElementById('histRange');
const hoverInfo = document.getElementById('hoverInfo');
const maskOpacitySlider = document.getElementById('maskOpacity');
const maskOpacityValue = document.getElementById('maskOpacityValue');

const HISTORY_LIMIT = 200;
const undoStack = [];
const redoStack = [];
const viewState = { scale: 1.0, offsetX: 0.0, offsetY: 0.0 };
let maskVisible = true;
let currentLabel = 1;
let originalImageData = null;
let isPanning = false;
let isPainting = false;
let lastPoint = { x: 0, y: 0 };
let lastPaintPoint = null;
let strokeChanges = null;
let tool = 'brush';
let spacePan = false;

let hoverPoint = null;
let eraseActive = false;
let erasePreviousLabel = null;
let isSegmenting = false;
let useNColor = false;
let histogramData = null;
let windowLow = 0;
let windowHigh = 255;
let currentGamma = 1.0;
let histDragTarget = null;
let histDragOffset = 0;
let cursorInsideCanvas = false;
let cursorInsideImage = false;
let maskOpacity = 0.8;

const MIN_GAMMA = 0.1;
const MAX_GAMMA = 6.0;
const DEFAULT_GAMMA = 1.0;
const HIST_HANDLE_THRESHOLD = 8;

function resizePreviewCanvas() {
  previewCanvas.width = canvas.width;
  previewCanvas.height = canvas.height;
}

function clearPreview() {
  previewCtx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
}

function drawBrushPreview(point) {
  if (!point) {
    clearPreview();
    return;
  }
  clearPreview();
  const pixels = getBrushPixels(point.x, point.y);
  if (!pixels.length) {
    return;
  }
  previewCtx.save();
  previewCtx.scale(dpr, dpr);
  previewCtx.translate(viewState.offsetX, viewState.offsetY);
  previewCtx.scale(viewState.scale, viewState.scale);
  previewCtx.imageSmoothingEnabled = false;
  const strokeWidth = 1 / Math.max(viewState.scale, 1);
  previewCtx.lineWidth = strokeWidth;
  previewCtx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
  previewCtx.fillStyle = 'rgba(255, 255, 255, 0.18)';
  const size = 1;
  for (const pixel of pixels) {
    previewCtx.fillRect(pixel.x, pixel.y, size, size);
    previewCtx.strokeRect(pixel.x, pixel.y, size, size);
  }
  previewCtx.restore();
}

function updateCursor() {
  if (spacePan || isPanning) {
    canvas.style.cursor = isPanning ? 'grabbing' : 'grab';
  } else if (tool === 'brush' && cursorInsideImage) {
    canvas.style.cursor = 'none';
  } else {
    canvas.style.cursor = 'default';
  }
}

function buildBrushOffsets(diameter) {
  const radius = (diameter - 1) / 2;
  const r2 = radius * radius;
  const offsets = [];
  for (let dy = -radius; dy <= radius; dy += 1) {
    for (let dx = -radius; dx <= radius; dx += 1) {
      if (dx * dx + dy * dy <= r2) {
        offsets.push({ dx, dy });
      }
    }
  }
  return offsets;
}

function updateMaskLabel() {
  maskLabel.textContent = 'Mask Label: ' + currentLabel;
}

function updateMaskVisibilityLabel() {
  maskVisibility.textContent = 'Mask Layer: ' + (maskVisible ? 'On' : 'Off') + " (toggle with 'M')";
}

function updateToolInfo() {
  let description = 'Brush (B=Brush, G=Fill, I=Picker)';
  if (tool === 'fill') {
    description = 'Flood Fill (B=Brush, G=Fill, I=Picker)';
  } else if (tool === 'picker') {
    description = 'Picker (B=Brush, G=Fill, I=Picker)';
  }
  toolInfo.textContent = 'Tool: ' + description;
}

function startEraseOverride() {
  if (eraseActive) {
    return;
  }
  eraseActive = true;
  erasePreviousLabel = currentLabel;
  setTool('brush');
  currentLabel = 0;
  updateMaskLabel();
}

function stopEraseOverride() {
  if (!eraseActive) {
    return;
  }
  if (erasePreviousLabel !== null) {
    currentLabel = erasePreviousLabel;
  }
  eraseActive = false;
  erasePreviousLabel = null;
  updateMaskLabel();
}

function setTool(nextTool) {
  if (tool === nextTool) {
    return;
  }
  tool = nextTool;
  updateToolInfo();
  updateCursor();
}

function updateBrushControls() {
  brushSizeSlider.value = String(brushIndex);
  brushSizeInput.value = String(brushDiameter);
}

function setBrushIndex(index, fromUser = false) {
  const clamped = Math.max(0, Math.min(brushSizes.length - 1, index));
  if (brushIndex === clamped) {
    if (fromUser) {
      setTool('brush');
    }
    return;
  }
  brushIndex = clamped;
  brushDiameter = brushSizes[brushIndex];
  brushOffsets = buildBrushOffsets(brushDiameter);
  updateBrushControls();
  log('brush size set to ' + brushDiameter);
  if (fromUser) {
    setTool('brush');
  }
  drawBrushPreview(hoverPoint);
}

function pushHistory(indices, before, after) {
  undoStack.push({ indices, before, after });
  if (undoStack.length > HISTORY_LIMIT) {
    undoStack.shift();
  }
  redoStack.length = 0;
}

function applyHistoryEntry(entry, useAfter) {
  const values = useAfter ? entry.after : entry.before;
  const idxs = entry.indices;
  for (let i = 0; i < idxs.length; i += 1) {
    maskValues[idxs[i]] = values[i];
  }
}

function undo() {
  if (!undoStack.length) {
    return;
  }
  const entry = undoStack.pop();
  applyHistoryEntry(entry, false);
  redoStack.push(entry);
  redrawMaskCanvas();
  draw();
}

function redo() {
  if (!redoStack.length) {
    return;
  }
  const entry = redoStack.pop();
  applyHistoryEntry(entry, true);
  undoStack.push(entry);
  redrawMaskCanvas();
  draw();
}

function collectBrushIndices(target, centerX, centerY) {
  const pixels = getBrushPixels(centerX, centerY);
  for (const pixel of pixels) {
    target.add(pixel.y * imgWidth + pixel.x);
  }
}

function getBrushPixels(centerX, centerY) {
  const cx = Math.floor(centerX);
  const cy = Math.floor(centerY);
  const pixels = [];
  for (const offset of brushOffsets) {
    const x = cx + offset.dx;
    const y = cy + offset.dy;
    if (x < 0 || y < 0 || x >= imgWidth || y >= imgHeight) {
      continue;
    }
    pixels.push({ x, y });
  }
  return pixels;
}

function paintStroke(point) {
  if (!strokeChanges) {
    strokeChanges = new Map();
  }
  const start = lastPaintPoint || point;
  const dx = point.x - start.x;
  const dy = point.y - start.y;
  const dist = Math.hypot(dx, dy);
  const step = Math.max(1, brushDiameter * 0.5);
  const steps = Math.max(1, Math.ceil(dist / step));
  const local = new Set();
  for (let i = 0; i <= steps; i += 1) {
    const t = steps === 0 ? 1 : i / steps;
    const px = start.x + dx * t;
    const py = start.y + dy * t;
    collectBrushIndices(local, px, py);
  }
  let changed = false;
  local.forEach((idx) => {
    if (!strokeChanges.has(idx)) {
      const original = maskValues[idx];
      if (original === currentLabel) {
        return;
      }
      strokeChanges.set(idx, original);
    }
    if (maskValues[idx] !== currentLabel) {
      maskValues[idx] = currentLabel;
      changed = true;
    }
  });
  if (changed) {
    redrawMaskCanvas();
    draw();
  }
  lastPaintPoint = point;
}

function finalizeStroke() {
  if (!strokeChanges || strokeChanges.size === 0) {
    strokeChanges = null;
    return;
  }
  const keys = Array.from(strokeChanges.keys()).sort((a, b) => a - b);
  const count = keys.length;
  const indices = new Uint32Array(count);
  const before = new Uint8Array(count);
  const after = new Uint8Array(count);
  for (let i = 0; i < count; i += 1) {
    const idx = keys[i];
    indices[i] = idx;
    before[i] = strokeChanges.get(idx);
    after[i] = currentLabel;
  }
  pushHistory(indices, before, after);
  strokeChanges = null;
}

function floodFill(point) {
  const sx = Math.round(point.x);
  const sy = Math.round(point.y);
  if (sx < 0 || sy < 0 || sx >= imgWidth || sy >= imgHeight) {
    return;
  }
  const startIdx = sy * imgWidth + sx;
  const targetLabel = maskValues[startIdx];
  if (targetLabel === currentLabel) {
    return;
  }
  const visited = new Uint8Array(maskValues.length);
  const stack = [startIdx];
  const indices = [];
  while (stack.length) {
    const idx = stack.pop();
    if (visited[idx]) {
      continue;
    }
    visited[idx] = 1;
    if (maskValues[idx] !== targetLabel) {
      continue;
    }
    indices.push(idx);
    const x = idx % imgWidth;
    const y = (idx / imgWidth) | 0;
    if (x > 0) { stack.push(idx - 1); }
    if (x + 1 < imgWidth) { stack.push(idx + 1); }
    if (y > 0) { stack.push(idx - imgWidth); }
    if (y + 1 < imgHeight) { stack.push(idx + imgWidth); }
  }
  if (!indices.length) {
    return;
  }
  const sorted = Array.from(new Set(indices)).sort((a, b) => a - b);
  const count = sorted.length;
  const idxArr = new Uint32Array(count);
  const before = new Uint8Array(count);
  const after = new Uint8Array(count);
  for (let i = 0; i < count; i += 1) {
    const idx = sorted[i];
    idxArr[i] = idx;
    before[i] = maskValues[idx];
    after[i] = currentLabel;
    maskValues[idx] = currentLabel;
  }
  pushHistory(idxArr, before, after);
  redrawMaskCanvas();
  draw();
}

function pickColor(point) {
  const sx = Math.round(point.x);
  const sy = Math.round(point.y);
  if (sx < 0 || sy < 0 || sx >= imgWidth || sy >= imgHeight) {
    return;
  }
  const idx = sy * imgWidth + sx;
  currentLabel = maskValues[idx];
  updateMaskLabel();
  log('picker set label ' + currentLabel);
}

function redrawMaskCanvas() {
  const data = maskData.data;
  const palette = useNColor ? nColorPalette : defaultPalette;
  for (let i = 0; i < maskValues.length; i += 1) {
    const label = maskValues[i];
    const color = getColorForLabel(palette, label);
    const p = i * 4;
    data[p] = color[0];
    data[p + 1] = color[1];
    data[p + 2] = color[2];
    data[p + 3] = color[3];
  }
  maskCtx.putImageData(maskData, 0, 0);
}

function shouldLogDraw() {
  if (drawLogCount < 20) {
    drawLogCount += 1;
    return true;
  }
  drawLogCount += 1;
  return drawLogCount % 50 === 0;
}

function draw() {
  if (shuttingDown) {
    return;
  }
  if (shouldLogDraw()) {
    log('draw start scale=' + viewState.scale.toFixed(3) + ' offset=' + viewState.offsetX.toFixed(1) + ',' + viewState.offsetY.toFixed(1));
  }
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
  drawBrushPreview(hoverPoint);
  if (!loggedPixelSample && canvas.width > 0 && canvas.height > 0) {
    try {
      const cx = Math.floor(canvas.width / 2);
      const cy = Math.floor(canvas.height / 2);
      const sample = ctx.getImageData(cx, cy, 1, 1).data;
      log('center pixel rgba=' + Array.from(sample).join(','));
    } catch (err) {
      log('center pixel read failed: ' + (err && err.message ? err.message : err));
    }
    loggedPixelSample = true;
    hideLoadingOverlay();
  }
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

function resizeCanvas() {
  if (shuttingDown) {
    return;
  }
  const rect = viewer.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) {
    log('resize skipped: viewer size ' + rect.width.toFixed(1) + 'x' + rect.height.toFixed(1));
    if (!shuttingDown) {
      requestAnimationFrame(resizeCanvas);
    }
    return;
  }
  canvas.width = Math.max(1, Math.round(rect.width * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));
  canvas.style.width = rect.width + 'px';
  canvas.style.height = rect.height + 'px';
  resizePreviewCanvas();
  recenterView();
  draw();
  drawBrushPreview(hoverPoint);
}

function recenterView() {
  const rect = viewer.getBoundingClientRect();
  viewState.offsetX = (rect.width - imgWidth * viewState.scale) / 2;
  viewState.offsetY = (rect.height - imgHeight * viewState.scale) / 2;
  log('recenter to ' + viewState.offsetX.toFixed(1) + ',' + viewState.offsetY.toFixed(1));
}

function clampGammaValue(value) {
  if (Number.isNaN(value)) {
    return currentGamma;
  }
  return Math.min(MAX_GAMMA, Math.max(MIN_GAMMA, value));
}

function updateGammaLabel() {
  if (gammaValue) {
    gammaValue.textContent = 'Gamma: ' + currentGamma.toFixed(2);
  }
}

function syncGammaControls() {
  if (gammaSlider) {
    const sliderValue = Math.round(currentGamma * 100);
    gammaSlider.value = String(Math.min(600, Math.max(10, sliderValue)));
  }
  if (gammaInput) {
    gammaInput.value = currentGamma.toFixed(2);
  }
  updateGammaLabel();
}

function setGamma(gamma, { emit = true } = {}) {
  currentGamma = clampGammaValue(gamma);
  syncGammaControls();
  if (emit) {
    applyImageAdjustments();
  } else {
    renderHistogram();
  }
}

function applyGamma(gamma) {
  setGamma(gamma, { emit: true });
}

function sinebowColor(t) {
  const angle = 2 * Math.PI * (t - Math.floor(t));
  const r = Math.sin(angle) * 0.5 + 0.5;
  const g = Math.sin(angle + (2 * Math.PI) / 3) * 0.5 + 0.5;
  const b = Math.sin(angle + (4 * Math.PI) / 3) * 0.5 + 0.5;
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255), 200];
}

function generateSinebowPalette(size, offset = 0) {
  const count = Math.max(size, 2);
  const palette = new Array(count);
  palette[0] = [0, 0, 0, 0];
  const golden = 0.61803398875;
  for (let i = 1; i < count; i += 1) {
    const t = (offset + i * golden) % 1;
    palette[i] = sinebowColor(t);
  }
  return palette;
}

function applyImageAdjustments() {
  if (!originalImageData) {
    return;
  }
  const source = originalImageData.data;
  const img = offCtx.createImageData(imgWidth, imgHeight);
  const target = img.data;
  const low = windowLow / 255;
  const high = windowHigh / 255;
  const range = Math.max(high - low, 1 / 255);
  for (let i = 0; i < source.length; i += 4) {
    let value = source[i] / 255;
    value = Math.min(Math.max((value - low) / range, 0), 1);
    value = Math.pow(value, currentGamma);
    const v = Math.round(value * 255);
    target[i] = v;
    target[i + 1] = v;
    target[i + 2] = v;
    target[i + 3] = source[i + 3];
  }
  offCtx.putImageData(img, 0, 0);
  draw();
  renderHistogram();
}

function computeHistogram() {
  if (!originalImageData) {
    histogramData = null;
    return;
  }
  histogramData = new Uint32Array(256);
  const data = originalImageData.data;
  for (let i = 0; i < data.length; i += 4) {
    histogramData[data[i]] += 1;
  }
}

function renderHistogram() {
  if (!histogramCanvas || !histogramData) {
    return;
  }
  const ctx = histogramCanvas.getContext('2d');
  if (!ctx) {
    return;
  }
  const width = histogramCanvas.width;
  const height = histogramCanvas.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = '#161616';
  ctx.fillRect(0, 0, width, height);
  const maxCount = Math.max(...histogramData);
  if (maxCount > 0) {
    ctx.fillStyle = '#4c8dff';
    const binWidth = Math.max(width / 256, 1);
    for (let i = 0; i < 256; i += 1) {
      const value = histogramData[i] / maxCount;
      const barHeight = Math.max(1, Math.round(value * (height - 4)));
      const x = Math.floor(i * binWidth);
      ctx.fillRect(x, height - barHeight, Math.ceil(binWidth), barHeight);
    }
  }
  const lowX = (windowLow / 255) * width;
  const highX = (windowHigh / 255) * width;
  ctx.fillStyle = 'rgba(255, 255, 255, 0.08)';
  ctx.fillRect(lowX, 0, Math.max(highX - lowX, 1), height);
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(lowX, 0);
  ctx.lineTo(lowX, height);
  ctx.moveTo(highX, 0);
  ctx.lineTo(highX, height);
  ctx.stroke();
  if (windowHigh > windowLow) {
    ctx.strokeStyle = '#ffcf33';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    const startX = Math.max(0, Math.floor(lowX));
    const endX = Math.min(width, Math.ceil(highX));
    for (let x = startX; x <= endX; x += 1) {
      const intensity = (x / width) * 255;
      const y = gammaCurveY(intensity, width, height);
      if (x === startX) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();
    const midIntensity = windowLow + (windowHigh - windowLow) * 0.5;
    const handleX = (midIntensity / 255) * width;
    const handleY = gammaCurveY(midIntensity, width, height);
    ctx.fillStyle = '#ffcf33';
    ctx.beginPath();
    ctx.arc(handleX, handleY, 4, 0, Math.PI * 2);
    ctx.fill();
  }
  updateHistogramCursor();
}

function updateHistogramUI() {
  if (histRangeLabel) {
    histRangeLabel.textContent = 'Window: [' + windowLow + ', ' + windowHigh + ']';
  }
  renderHistogram();
}

function histogramValueFromEvent(evt) {
  const rect = histogramCanvas.getBoundingClientRect();
  const x = Math.min(Math.max(evt.clientX - rect.left, 0), rect.width);
  return Math.round((x / rect.width) * 255);
}

function gammaCurveY(intensity, width, height) {
  if (windowHigh <= windowLow) {
    return height - 2;
  }
  const clampedIntensity = Math.min(Math.max(intensity, windowLow), windowHigh);
  let t = (clampedIntensity - windowLow) / (windowHigh - windowLow);
  t = Math.min(Math.max(t, 0.0001), 0.9999);
  const mapped = Math.pow(t, 1 / currentGamma);
  const y = height - (mapped * (height - 4)) - 2;
  return Math.min(height - 2, Math.max(2, y));
}

function updateHistogramCursor(evt) {
  if (!histogramCanvas) {
    return;
  }
  if (histDragTarget) {
    histogramCanvas.style.cursor = (histDragTarget === 'range' || histDragTarget === 'gamma') ? 'grabbing' : 'ew-resize';
    return;
  }
  const rect = histogramCanvas.getBoundingClientRect();
  const x = evt ? evt.clientX - rect.left : NaN;
  const width = rect.width;
  const height = rect.height;
  const lowX = (windowLow / 255) * width;
  const highX = (windowHigh / 255) * width;
  const threshold = HIST_HANDLE_THRESHOLD;
  let cursor = 'crosshair';
  if (!Number.isNaN(x)) {
    if (Math.abs(x - lowX) < threshold || Math.abs(x - highX) < threshold) {
      cursor = 'ew-resize';
    } else if (x > lowX && x < highX) {
      cursor = 'grab';
      if (evt && windowHigh > windowLow) {
        const intensity = histogramValueFromEvent(evt);
        const y = evt.clientY - rect.top;
        const curveY = gammaCurveY(intensity, width, height);
        if (Math.abs(y - curveY) < threshold) {
          cursor = 'grab';
        }
      }
    }
  }
  histogramCanvas.style.cursor = cursor;
}

function setWindowBounds(low, high, { emit = true } = {}) {
  let clampedLow = Math.round(low);
  let clampedHigh = Math.round(high);
  if (Number.isNaN(clampedLow)) clampedLow = windowLow;
  if (Number.isNaN(clampedHigh)) clampedHigh = windowHigh;
  clampedLow = Math.max(0, Math.min(255, clampedLow));
  clampedHigh = Math.max(0, Math.min(255, clampedHigh));
  if (clampedHigh <= clampedLow) {
    if (histDragTarget === 'low') {
      clampedLow = Math.max(0, Math.min(254, clampedHigh - 1));
    } else if (histDragTarget === 'high') {
      clampedHigh = Math.min(255, Math.max(1, clampedLow + 1));
    } else {
      clampedHigh = Math.min(255, Math.max(1, clampedLow + 1));
    }
  }
  windowLow = clampedLow;
  windowHigh = clampedHigh;
  updateHistogramUI();
  if (emit) {
    applyImageAdjustments();
  }
}

function handleHistogramPointerDown(evt) {
  if (!histogramCanvas) {
    return;
  }
  evt.preventDefault();
  histogramCanvas.setPointerCapture(evt.pointerId);
  const rect = histogramCanvas.getBoundingClientRect();
  const width = rect.width;
  const height = rect.height;
  const x = Math.min(Math.max(evt.clientX - rect.left, 0), width);
  const y = Math.min(Math.max(evt.clientY - rect.top, 0), height);
  const intensity = (x / width) * 255;
  const lowX = (windowLow / 255) * width;
  const highX = (windowHigh / 255) * width;
  const threshold = HIST_HANDLE_THRESHOLD;
  histDragTarget = null;
  if (windowHigh > windowLow) {
    const curveY = gammaCurveY(intensity, width, height);
    if (intensity >= windowLow && intensity <= windowHigh && Math.abs(y - curveY) <= threshold) {
      histDragTarget = 'gamma';
    }
  }
  if (!histDragTarget && Math.abs(x - lowX) <= threshold) {
    histDragTarget = 'low';
  } else if (!histDragTarget && Math.abs(x - highX) <= threshold) {
    histDragTarget = 'high';
  } else if (!histDragTarget && x > lowX && x < highX) {
    histDragTarget = 'range';
    histDragOffset = histogramValueFromEvent(evt) - windowLow;
  } else if (!histDragTarget) {
    histDragTarget = Math.abs(x - lowX) < Math.abs(x - highX) ? 'low' : 'high';
  }
  if (histDragTarget !== 'range') {
    histDragOffset = 0;
  }
  updateHistogramCursor(evt);
  handleHistogramPointerMove(evt);
}

function handleHistogramPointerMove(evt) {
  if (!histogramCanvas) {
    return;
  }
  if (!histDragTarget) {
    updateHistogramCursor(evt);
    return;
  }
  evt.preventDefault();
  const value = histogramValueFromEvent(evt);
  if (histDragTarget === 'low') {
    setWindowBounds(Math.min(value, windowHigh - 1), windowHigh);
  } else if (histDragTarget === 'high') {
    setWindowBounds(windowLow, Math.max(value, windowLow + 1));
  } else if (histDragTarget === 'range') {
    const span = windowHigh - windowLow;
    let newLow = value - histDragOffset;
    newLow = Math.max(0, Math.min(255 - span, newLow));
    setWindowBounds(newLow, newLow + span);
  } else if (histDragTarget === 'gamma') {
    if (windowHigh > windowLow) {
      const rect = histogramCanvas.getBoundingClientRect();
      const height = rect.height;
      const width = rect.width;
      const clampedValue = Math.min(Math.max(value, windowLow + 0.5), windowHigh - 0.5);
      let t = (clampedValue - windowLow) / (windowHigh - windowLow);
      t = Math.min(Math.max(t, 0.0001), 0.9999);
      const yRatio = 1 - Math.min(Math.max((evt.clientY - rect.top) / height, 0.0001), 0.9999);
      let newGamma = Math.log(t) / Math.log(yRatio);
      if (!Number.isFinite(newGamma) || newGamma <= 0) {
        newGamma = currentGamma;
      }
      setGamma(newGamma);
    }
  }
  updateHistogramCursor(evt);
}

function handleHistogramPointerUp(evt) {
  if (!histogramCanvas) {
    return;
  }
  evt.preventDefault();
  histogramCanvas.releasePointerCapture(evt.pointerId);
  histDragTarget = null;
  histDragOffset = 0;
  updateHistogramCursor(evt);
}

function updateHoverInfo(point) {
  if (!hoverInfo) {
    cursorInsideImage = false;
    updateCursor();
    return;
  }
  if (!point || !originalImageData) {
    cursorInsideImage = false;
    hoverInfo.textContent = 'Y: --, X: --, Val: --';
    updateCursor();
    return;
  }
  const x = Math.round(point.x);
  const y = Math.round(point.y);
  if (x < 0 || y < 0 || x >= imgWidth || y >= imgHeight) {
    cursorInsideImage = false;
    hoverInfo.textContent = 'Y: --, X: --, Val: --';
    updateCursor();
    return;
  }
  const idx = (y * imgWidth + x) * 4;
  const value = originalImageData.data[idx];
  cursorInsideImage = true;
  hoverInfo.textContent = 'Y: ' + y + ', X: ' + x + ', Val: ' + value;
  updateCursor();
}

function getColorForLabel(palette, label) {
  if (label <= 0) {
    return [0, 0, 0, 0];
  }
  if (palette[label]) {
    const base = palette[label];
    return [base[0], base[1], base[2], Math.round(base[3] * maskOpacity)];
  }
  if (palette.length > 1) {
    const idx = ((label - 1) % (palette.length - 1)) + 1;
    const base = palette[idx];
    return [base[0], base[1], base[2], Math.round(base[3] * maskOpacity)];
  }
  return palette[0] || [0, 0, 0, 0];
}

function updateColorModeLabel() {
  if (!colorMode) {
    return;
  }
  const mode = useNColor ? 'N-Color' : 'Palette';
  colorMode.textContent = 'Mask Colors: ' + mode + " (toggle with 'N')";
}

function toggleColorMode() {
  useNColor = !useNColor;
  updateColorModeLabel();
  redrawMaskCanvas();
  draw();
}

function setSegmentStatus(message, isError = false) {
  if (!segmentStatus) {
    return;
  }
  segmentStatus.textContent = message || '';
  segmentStatus.style.color = isError ? '#ff8a8a' : '#9aa';
}

function applySegmentationMask(payload) {
  if (!payload || !payload.mask) {
    throw new Error('segment payload missing mask');
  }
  const binary = atob(payload.mask);
  if (binary.length !== maskValues.length) {
    throw new Error('mask size mismatch (' + binary.length + ' vs ' + maskValues.length + ')');
  }
  for (let i = 0; i < binary.length; i += 1) {
    maskValues[i] = binary.charCodeAt(i);
  }
  redrawMaskCanvas();
  draw();
  updateMaskLabel();
}

async function requestSegmentation() {
  if (window.pywebview && window.pywebview.api && window.pywebview.api.segment) {
    return window.pywebview.api.segment();
  }
  const response = await fetch('/api/segment', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
  });
  if (!response.ok) {
    throw new Error('HTTP ' + response.status);
  }
  return response.json();
}

async function runSegmentation() {
  if (!segmentButton || isSegmenting) {
    return;
  }
  isSegmenting = true;
  segmentButton.disabled = true;
  setSegmentStatus('Running segmentation…');
  try {
    const raw = await requestSegmentation();
    const payload = typeof raw === 'string' ? JSON.parse(raw) : raw;
    if (payload.error) {
      throw new Error(payload.error);
    }
    applySegmentationMask(payload);
    setSegmentStatus('Segmentation complete.');
  } catch (err) {
    console.error(err);
    const msg = err && err.message ? err.message : err;
    setSegmentStatus('Segmentation failed: ' + msg, true);
  } finally {
    isSegmenting = false;
    segmentButton.disabled = false;
  }
}

canvas.addEventListener('wheel', (evt) => {
  evt.preventDefault();
  const pointer = getPointerPosition(evt);
  const imageX = (pointer.x - viewState.offsetX) / viewState.scale;
  const imageY = (pointer.y - viewState.offsetY) / viewState.scale;
  const factor = evt.deltaY < 0 ? 1.1 : 0.9;
  const newScale = Math.min(Math.max(viewState.scale * factor, 0.1), 40);
  viewState.scale = newScale;
  viewState.offsetX = pointer.x - imageX * viewState.scale;
  viewState.offsetY = pointer.y - imageY * viewState.scale;
  draw();
}, { passive: false });

canvas.addEventListener('pointerdown', (evt) => {
  const pointer = getPointerPosition(evt);
  lastPoint = pointer;
  const world = screenToImage(pointer);
  updateHoverInfo(world);
  if (evt.button === 0) {
    if (spacePan) {
      isPanning = true;
      updateCursor();
      canvas.setPointerCapture(evt.pointerId);
      hoverPoint = null;
      drawBrushPreview(null);
      updateHoverInfo(null);
      return;
    }
    if (tool === 'fill') {
      floodFill(world);
      return;
    }
    if (tool === 'picker') {
      pickColor(world);
      return;
    }
    strokeChanges = new Map();
    isPainting = true;
    canvas.classList.add('painting');
    updateCursor();
    canvas.setPointerCapture(evt.pointerId);
    lastPaintPoint = null;
    paintStroke(world);
    hoverPoint = screenToImage(pointer);
    drawBrushPreview(hoverPoint);
    updateHoverInfo(hoverPoint);
    return;
  }
  isPanning = true;
  updateCursor();
  canvas.setPointerCapture(evt.pointerId);
  hoverPoint = null;
  drawBrushPreview(null);
  updateHoverInfo(null);
});

canvas.addEventListener('pointermove', (evt) => {
  const pointer = getPointerPosition(evt);
  const world = screenToImage(pointer);
  if (isPainting) {
    paintStroke(world);
    hoverPoint = world;
    drawBrushPreview(hoverPoint);
    updateHoverInfo(world);
    return;
  }
  if (!isPanning && !spacePan) {
    if (tool === 'brush') {
      hoverPoint = world;
      drawBrushPreview(hoverPoint);
    } else {
      hoverPoint = null;
      drawBrushPreview(null);
    }
    updateHoverInfo(world);
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
  updateHoverInfo(screenToImage(pointer));
});

function stopInteraction(evt) {
  if (isPainting) {
    canvas.classList.remove('painting');
    finalizeStroke();
  }
  isPainting = false;
  isPanning = false;
  spacePan = false;
  updateCursor();
  lastPaintPoint = null;
  if (hoverPoint) {
    drawBrushPreview(hoverPoint);
  } else {
    drawBrushPreview(null);
  }
  updateHoverInfo(hoverPoint);
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
canvas.addEventListener('mouseenter', () => {
  cursorInsideCanvas = true;
  updateCursor();
});
canvas.addEventListener('mouseleave', () => {
  cursorInsideCanvas = false;
  cursorInsideImage = false;
  updateCursor();
});

window.addEventListener('keydown', (evt) => {
  const tag = evt.target && evt.target.tagName ? evt.target.tagName.toLowerCase() : '';
  if (tag === 'input') {
    return;
  }
  const key = evt.key.toLowerCase();
  const modifier = evt.ctrlKey || evt.metaKey;
  if (!modifier && !evt.altKey && key === 'e') {
    startEraseOverride();
    evt.preventDefault();
  }
  if (modifier && key === 'z') {
    if (evt.shiftKey) {
      redo();
    } else {
      undo();
    }
    evt.preventDefault();
    return;
  }
  if (modifier && key === 'y') {
    redo();
    evt.preventDefault();
    return;
  }
  if (!modifier && !evt.altKey) {
    if (key === 'b') {
      setTool('brush');
      evt.preventDefault();
      return;
    }
    if (key === 'g') {
      setTool('fill');
      evt.preventDefault();
      return;
    }
    if (key === 'i') {
      setTool('picker');
      evt.preventDefault();
      return;
    }
    if (key === '[') {
      setBrushIndex(brushIndex - 1, true);
      evt.preventDefault();
      return;
    }
    if (key === ']') {
      setBrushIndex(brushIndex + 1, true);
      evt.preventDefault();
      return;
    }
    if (key === ' ') {
        spacePan = true;
      updateCursor();
      evt.preventDefault();
      return;
    }
  }
  if (key === 'm') {
    maskVisible = !maskVisible;
    updateMaskVisibilityLabel();
    draw();
    evt.preventDefault();
    return;
  }
  if (!modifier && !evt.altKey && key === 'n') {
    toggleColorMode();
    evt.preventDefault();
    return;
  }
  if (!modifier && key >= '0' && key <= '9') {
    const nextLabel = parseInt(key, 10);
    if (eraseActive) {
      erasePreviousLabel = nextLabel;
    } else {
      currentLabel = nextLabel;
      updateMaskLabel();
    }
    evt.preventDefault();
  }
});

window.addEventListener('keyup', (evt) => {
  if (evt.key === ' ') {
    spacePan = false;
    updateCursor();
  }
  if (evt.key && evt.key.toLowerCase() === 'e') {
    stopEraseOverride();
  }
});

function initialize() {
  log('initialize');
  const img = new Image();
  img.onload = () => {
    log('image loaded: ' + imgWidth + 'x' + imgHeight);
    offCtx.drawImage(img, 0, 0);
    originalImageData = offCtx.getImageData(0, 0, imgWidth, imgHeight);
    windowLow = 0;
    windowHigh = 255;
    currentGamma = DEFAULT_GAMMA;
    computeHistogram();
    setGamma(currentGamma, { emit: false });
    updateHistogramUI();
    applyImageAdjustments();
    redrawMaskCanvas();
    resizeCanvas();
    updateBrushControls();
  };
  img.onerror = (evt) => {
    const detail = evt?.message || 'unknown error';
    log('image load failed: ' + detail);
    setLoadingOverlay('Failed to load image', true);
  };
  img.src = imageDataUrl;
  updateCursor();
}

window.addEventListener('resize', resizeCanvas);
if (gammaSlider) {
  gammaSlider.addEventListener('input', (evt) => {
    const value = parseInt(evt.target.value, 10);
    const gamma = Number.isNaN(value) ? currentGamma : value / 100.0;
    setGamma(gamma);
  });
}

if (gammaInput) {
  gammaInput.addEventListener('change', () => {
    const value = parseFloat(gammaInput.value);
    if (Number.isNaN(value)) {
      gammaInput.value = currentGamma.toFixed(2);
      return;
    }
    setGamma(value);
  });
}

brushSizeSlider.addEventListener('input', (evt) => {
  const idx = parseInt(evt.target.value, 10);
  if (!Number.isNaN(idx)) {
    setBrushIndex(idx, true);
  }
});

brushSizeInput.addEventListener('change', (evt) => {
  let value = parseInt(evt.target.value, 10);
  if (Number.isNaN(value)) {
    brushSizeInput.value = String(brushDiameter);
    return;
  }
  value = Math.max(1, Math.min(9, value));
  if (value % 2 === 0) {
    value += value > brushDiameter ? 1 : -1;
    if (value < 1) value = 1;
    if (value > 9) value = 9;
  }
  setBrushIndex(nearestBrushIndex(value), true);
});

updateMaskLabel();
updateMaskVisibilityLabel();
updateToolInfo();
updateBrushControls();
updateColorModeLabel();
updateHoverInfo(null);
if (segmentButton) {
  segmentButton.addEventListener('click', () => {
    runSegmentation();
  });
}
if (histogramCanvas) {
  histogramCanvas.addEventListener('pointerdown', handleHistogramPointerDown);
  histogramCanvas.addEventListener('pointermove', handleHistogramPointerMove);
  histogramCanvas.addEventListener('pointerup', handleHistogramPointerUp);
  histogramCanvas.addEventListener('pointercancel', handleHistogramPointerUp);
  histogramCanvas.addEventListener('pointerleave', (evt) => {
    if (!histDragTarget) {
      updateHistogramCursor(evt);
    }
  });
  updateHistogramCursor();
}
if (maskOpacitySlider) {
  maskOpacitySlider.addEventListener('input', (evt) => {
    const value = parseInt(evt.target.value, 10);
    const fraction = Number.isNaN(value) ? maskOpacity : Math.min(1, Math.max(0, value / 100));
    maskOpacity = fraction;
    if (maskOpacityValue) {
      maskOpacityValue.textContent = Math.round(maskOpacity * 100) + '%';
    }
    redrawMaskCanvas();
    draw();
  });
  maskOpacitySlider.value = String(Math.round(maskOpacity * 100));
  if (maskOpacityValue) {
    maskOpacityValue.textContent = Math.round(maskOpacity * 100) + '%';
  }
}

let bootstrapped = false;
function boot() {
  if (bootstrapped) {
    return;
  }
  bootstrapped = true;
  log('boot (readyState=' + document.readyState + ')');
  initialize();
  setTimeout(resizeCanvas, 100);
  flushLogs();
}

if (document.readyState === 'complete' || document.readyState === 'interactive') {
  boot();
} else {
  window.addEventListener('load', boot);
}
window.addEventListener('pywebviewready', () => {
  pywebviewReady = true;
  log('pywebview ready event');
  flushLogs();
  boot();
});
window.addEventListener('beforeunload', () => {
  shuttingDown = true;
  pendingLogs.length = 0;
});
</script>
</body>
</html>
"""
)


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

    def _ensure_model(self) -> None:
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    from cellpose_omni import models  # local import to avoid startup cost

                    self._model = models.CellposeModel(
                        gpu=False,
                        model_type="bact_phase_affinity",
                    )

    def segment(self, image: np.ndarray) -> np.ndarray:
        from omnipose.utils import normalize99  # local import, cheap

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


def build_html() -> str:
    start = time.perf_counter()
    image = load_image_uint8(as_rgb=False)
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
    color_table = json.dumps(get_instance_color_table().tolist())
    return HTML_TEMPLATE.substitute(
        width=width,
        height=height,
        image_data=data,
        color_table=color_table,
        brush_radius=DEFAULT_BRUSH_RADIUS,
    )


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

    api = DebugAPI()

    app = FastAPI(title="Omnipose PyWebView Viewer")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return build_html()

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

    return app


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    ssl_cert: str | None = None,
    ssl_key: str | None = None,
    reload: bool = False,
) -> None:
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
    print(f"[pywebview] serving at {scheme}://{host}:{port}", flush=True)

    if reload:
        uvicorn.run(
            "gui.examples.pywebview_image_viewer:create_app",
            factory=True,
            host=host,
            port=port,
            reload=True,
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
        )
    else:
        run_desktop()


if __name__ == "__main__":
    main()
