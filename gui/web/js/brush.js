(function initOmniBrush(global) {
  'use strict';

  const DEFAULT_MODES = {
    SMOOTH: 'smooth',
    SNAPPED: 'snapped',
  };

  const state = {
    ctx: null,
    snappedKernelCache: new Map(),
  };

  function init(options) {
    state.ctx = Object.assign(
      {
        getBrushDiameter: () => 1,
        getBrushKernelMode: () => DEFAULT_MODES.SMOOTH,
        modes: DEFAULT_MODES,
        getImageDimensions: () => ({ width: 0, height: 0 }),
        previewCanvas: null,
        previewCtx: null,
        canvas: null,
        viewer: null,
        applyViewTransform: null,
        getViewState: () => ({ scale: 1 }),
        getDpr: () => (typeof global.devicePixelRatio === 'number' ? global.devicePixelRatio : 1),
        getTool: () => 'brush',
        previewToolTypes: new Set(['brush', 'erase']),
        crosshairToolTypes: new Set(['brush', 'erase', 'fill', 'picker']),
        isCrosshairEnabled: () => true,
        isDebugTouchOverlay: () => false,
        getTouchPointers: () => new Map(),
      },
      options || {},
    );
  }

  function ctx() {
    if (!state.ctx) {
      throw new Error('OmniBrush.init must be called before use');
    }
    return state.ctx;
  }

  function getModes() {
    const context = ctx();
    return context.modes || DEFAULT_MODES;
  }

  function toNumber(value, fallback = 0) {
    const numeric = Number(value);
    return Number.isFinite(numeric) ? numeric : fallback;
  }

  function getBrushKernelCenter(x, y) {
    return ctx().getBrushKernelMode() === getModes().SNAPPED
      ? { x: Math.round(x), y: Math.round(y) }
      : { x, y };
  }

  function enumerateBrushPixels(rawCenterX, rawCenterY) {
    const mode = ctx().getBrushKernelMode();
    const centerX = rawCenterX - 0.5;
    const centerY = rawCenterY - 0.5;
    if (mode === getModes().SNAPPED) {
      return enumerateSnappedBrushPixels(centerX, centerY);
    }
    return enumerateSmoothBrushPixels(centerX, centerY);
  }

  function enumerateSmoothBrushPixels(centerX, centerY) {
    const pixels = [];
    const diameter = toNumber(ctx().getBrushDiameter(), 1);
    const radius = (diameter - 1) / 2;
    const { width, height } = ctx().getImageDimensions() || {};
    const imgWidth = toNumber(width, 0);
    const imgHeight = toNumber(height, 0);
    if (radius <= 0) {
      const x = Math.round(centerX);
      const y = Math.round(centerY);
      if (x >= 0 && x < imgWidth && y >= 0 && y < imgHeight) {
        pixels.push({ x, y });
      }
      return pixels;
    }
    const threshold = (radius + 0.35) * (radius + 0.35);
    const minX = Math.max(0, Math.floor(centerX - radius - 1));
    const maxX = Math.min(imgWidth - 1, Math.ceil(centerX + radius + 1));
    for (let x = minX; x <= maxX; x += 1) {
      const dx = x - centerX;
      if (dx * dx > threshold) {
        continue;
      }
      const dy = Math.sqrt(Math.max(0, threshold - dx * dx));
      let yMin = Math.ceil(centerY - dy);
      let yMax = Math.floor(centerY + dy);
      if (yMax < yMin) {
        continue;
      }
      if (yMin < 0) {
        yMin = 0;
      }
      if (yMax >= imgHeight) {
        yMax = imgHeight - 1;
      }
      for (let y = yMin; y <= yMax; y += 1) {
        pixels.push({ x, y });
      }
    }
    return pixels;
  }

  function enumerateSnappedBrushPixels(centerX, centerY) {
    const center = getBrushKernelCenter(centerX, centerY);
    const offsets = getSnappedKernelOffsets(toNumber(ctx().getBrushDiameter(), 1));
    const { width, height } = ctx().getImageDimensions() || {};
    const imgWidth = toNumber(width, 0);
    const imgHeight = toNumber(height, 0);
    const pixels = [];
    for (const offset of offsets) {
      const x = center.x + offset.x;
      const y = center.y + offset.y;
      if (x >= 0 && x < imgWidth && y >= 0 && y < imgHeight) {
        pixels.push({ x, y });
      }
    }
    return pixels;
  }

  function getSnappedKernelOffsets(diameter) {
    const key = diameter | 0;
    if (state.snappedKernelCache.has(key)) {
      return state.snappedKernelCache.get(key);
    }
    const radius = (diameter - 1) / 2;
    const threshold = (radius + 0.35) * (radius + 0.35);
    const span = Math.ceil(radius + 1);
    const offsets = [];
    for (let y = -span; y <= span; y += 1) {
      for (let x = -span; x <= span; x += 1) {
        if (x * x + y * y <= threshold) {
          offsets.push({ x, y });
        }
      }
    }
    state.snappedKernelCache.set(key, offsets);
    return offsets;
  }

  function traverseLine(x0, y0, x1, y1, callback) {
    let ix0 = Math.round(x0);
    let iy0 = Math.round(y0);
    const ix1 = Math.round(x1);
    const iy1 = Math.round(y1);
    const dx = Math.abs(ix1 - ix0);
    const sx = ix0 < ix1 ? 1 : -1;
    const dy = -Math.abs(iy1 - iy0);
    const sy = iy0 < iy1 ? 1 : -1;
    let err = dx + dy;
    while (true) {
      callback(ix0, iy0);
      if (ix0 === ix1 && iy0 === iy1) break;
      const e2 = 2 * err;
      if (e2 >= dy) {
        err += dy;
        ix0 += sx;
      }
      if (e2 <= dx) {
        err += dx;
        iy0 += sy;
      }
    }
  }

  function collectBrushIndices(target, centerX, centerY) {
    const pixels = enumerateBrushPixels(centerX, centerY);
    const { width } = ctx().getImageDimensions() || {};
    const imgWidth = toNumber(width, 0);
    for (const pixel of pixels) {
      target.add(pixel.y * imgWidth + pixel.x);
    }
  }

  function resizePreviewCanvas() {
    const context = ctx();
    const { canvas, previewCanvas, viewer } = context;
    if (!canvas || !previewCanvas) {
      return;
    }
    previewCanvas.width = canvas.width;
    previewCanvas.height = canvas.height;
    if (canvas.style.width) {
      previewCanvas.style.width = canvas.style.width;
    } else if (viewer && typeof viewer.getBoundingClientRect === 'function') {
      const rect = viewer.getBoundingClientRect();
      if (rect) {
        previewCanvas.style.width = rect.width + 'px';
      }
    }
    if (canvas.style.height) {
      previewCanvas.style.height = canvas.style.height;
    } else if (viewer && typeof viewer.getBoundingClientRect === 'function') {
      const rect = viewer.getBoundingClientRect();
      if (rect) {
        previewCanvas.style.height = rect.height + 'px';
      }
    }
    previewCanvas.style.transform = canvas.style.transform || '';
  }

  function clearPreview() {
    const previewCtx = ctx().previewCtx;
    const previewCanvas = ctx().previewCanvas;
    if (!previewCtx || !previewCanvas) {
      return;
    }
    previewCtx.setTransform(1, 0, 0, 1, 0, 0);
    previewCtx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
  }

  function drawTouchOverlay() {
    const previewCtx = ctx().previewCtx;
    const canvas = ctx().canvas;
    if (!previewCtx || !canvas || !ctx().isDebugTouchOverlay()) {
      return;
    }
    const rect = canvas.getBoundingClientRect();
    const pointers = ctx().getTouchPointers();
    if (!pointers || typeof pointers.forEach !== 'function') {
      return;
    }
    previewCtx.save();
    previewCtx.setTransform(1, 0, 0, 1, 0, 0);
    previewCtx.lineWidth = 2;
    previewCtx.strokeStyle = 'rgba(0, 200, 255, 0.9)';
    previewCtx.fillStyle = 'rgba(0, 200, 255, 0.2)';
    pointers.forEach((data) => {
      const x = data.x - rect.left;
      const y = data.y - rect.top;
      if (!Number.isFinite(x) || !Number.isFinite(y)) {
        return;
      }
      previewCtx.beginPath();
      previewCtx.arc(x, y, 16, 0, Math.PI * 2);
      previewCtx.fill();
      previewCtx.stroke();
    });
    previewCtx.restore();
  }

  function drawBrushPreview(point, { crosshairOnly = false } = {}) {
    clearPreview();
    const context = ctx();
    const previewCtx = context.previewCtx;
    if (!previewCtx || !context.applyViewTransform) {
      drawTouchOverlay();
      return;
    }
    if (point) {
      let circleCenter = null;
      const snappedMode = context.getBrushKernelMode() === getModes().SNAPPED;
      const snappedCenter = snappedMode
        ? { x: Math.round(point.x - 0.5) + 0.5, y: Math.round(point.y - 0.5) + 0.5 }
        : null;
      if (!crosshairOnly && context.previewToolTypes && context.previewToolTypes.has(context.getTool())) {
        const pixels = enumerateBrushPixels(point.x, point.y);
        if (pixels.length) {
          const kernelCenter = getBrushKernelCenter(point.x, point.y);
          circleCenter = snappedMode ? snappedCenter : { x: kernelCenter.x, y: kernelCenter.y };
          const radius = (toNumber(context.getBrushDiameter(), 1) - 1) / 2;
          previewCtx.save();
          context.applyViewTransform(previewCtx, { includeDpr: true });
          previewCtx.imageSmoothingEnabled = false;
          previewCtx.fillStyle = 'rgba(255, 255, 255, 0.24)';
          for (const pixel of pixels) {
            previewCtx.fillRect(pixel.x, pixel.y, 1, 1);
          }
          previewCtx.restore();
          if (radius >= 0) {
            previewCtx.save();
            context.applyViewTransform(previewCtx, { includeDpr: true });
            const scale = Math.max(context.getViewState().scale || 1, 0.0001);
            previewCtx.lineWidth = 1 / Math.max(scale * context.getDpr(), 1);
            previewCtx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
            previewCtx.beginPath();
            const radiusPixels = radius + 0.5;
            previewCtx.arc(circleCenter.x, circleCenter.y, radiusPixels, 0, Math.PI * 2);
            previewCtx.stroke();
            previewCtx.restore();
          }
        }
      }
      const wantCrosshair = Boolean(context.isCrosshairEnabled && context.isCrosshairEnabled()) || crosshairOnly;
      if (wantCrosshair) {
        const crosshairHalf = 0.25;
        const crossPoint = (circleCenter && !crosshairOnly)
          ? circleCenter
          : (snappedMode && snappedCenter ? snappedCenter : { x: point.x, y: point.y });
        previewCtx.save();
        context.applyViewTransform(previewCtx, { includeDpr: true });
        const scale = Math.max(context.getViewState().scale || 1, 0.0001);
        previewCtx.lineWidth = 1 / Math.max(scale * context.getDpr(), 1);
        previewCtx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
        previewCtx.beginPath();
        previewCtx.moveTo(crossPoint.x - crosshairHalf, crossPoint.y);
        previewCtx.lineTo(crossPoint.x + crosshairHalf, crossPoint.y);
        previewCtx.moveTo(crossPoint.x, crossPoint.y - crosshairHalf);
        previewCtx.lineTo(crossPoint.x, crossPoint.y + crosshairHalf);
        previewCtx.stroke();
        previewCtx.restore();
      }
    }
    drawTouchOverlay();
  }

  const api = global.OmniBrush || {};
  Object.assign(api, {
    init,
    getBrushKernelCenter,
    enumerateBrushPixels,
    collectBrushIndices,
    traverseLine,
    resizePreviewCanvas,
    drawBrushPreview,
  });
  global.OmniBrush = api;
})(typeof window !== 'undefined' ? window : globalThis);
